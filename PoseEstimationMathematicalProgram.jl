module PoseEstimationMathematicalProgram

using JuMP
using Gurobi
using StaticArrays
using Rotations
using GeometryTypes
using FileIO
using CoordinateTransformations
using MultilinearOpt

export PoseEstimationSolveData, constructModel

type PoseEstimationSolveData
    m
    R
    T
    R_abs
    R_diff
    phi
    alpha
    C
    f
    f_outlier
end

function constructModel(scenePoints, points, faces;
    use_relaxed_form = false, rotation_relaxation = :Fixed, other_solver=nothing, closeness=10.0, R0_in=zeros(3, 3), optPhiMax = 0.1, optBigNumber = 10, kwargs...)
    if other_solver == nothing
        m = Model(solver=GurobiSolver(;kwargs...))
    else
        m = Model(solver=other_solver)
    end

    N_scene = size(scenePoints, 2)
    N_model_verts = size(points, 1)
    N_model_faces = size(faces, 1)

    # Set up transform
    @variable(m, -1 <= R[1:3, 1:3] <= 1 )
    @variable(m, -optBigNumber <= T[1:3] <= optBigNumber )

    R_abs = nothing
    R_diff = nothing
    if rotation_relaxation == :Fixed
        @constraint(m, R .== R0_in)
    elseif rotation_relaxation == :Domain
        # Bound R to have row and column L1 norm <= sqrt(3), since
        # row and column L2 norm = 2.
        # (We can't do lower bound L1 norm >= 1 b/c that's not
        # a convex constraint)
        @variable(m, R_abs[1:3, 1:3] >= 0 )
        @constraint(m, R_abs .>= R)
        @constraint(m, R_abs .>= -R)
        for i in 1:3
            @constraint(m, [1 1 1] * R_abs[:, i] .<= sqrt(3))
            @constraint(m, [1 1 1] * R_abs[i, :] .<= sqrt(3))
        end


        # Bound R to be close to R0, in terms of L1 norm
        @variable(m, R_diff[1:3, 1:3] >= 0 )
        @constraint(m, R_diff .>= (R - R0_in))
        @constraint(m, R_diff .>= -(R - R0_in))
        for i in 1:3
            @constraint(m, dot([1; 1; 1], R_diff[:, i]) <= closeness)
            @constraint(m, dot([1; 1; 1], R_diff[i, :]) <= closeness)
        end
    elseif rotation_relaxation == :RtRBilinear
        # Orthogonality
        @constraint(m, transpose(R) * R .== eye(3, 3))
        #@constraint(m, R * transpose(R) .== eye(3, 3))
        # Forcing right-handedness of coordinates via cross products
        # (which should result in det(R) = +1)
        for i in 0:2
          @constraint(m, cross(R[(i+0)%3+1, :], R[(i+1)%3+1, :]) .== R[(i+2)%3+1, :])
          #@constraint(m, cross(R[:, (i+0)%3+1], R[:, (i+1)%3+1]) .== R[:, (i+2)%3+1])
        end
        relaxbilinear!(m, method=:Logarithmic2D, disc_level=4)
    end

    # Terms required for setting up objective:
    # Slack for distance-from-each-scene-point
    @variable(m, phi[1:N_scene] >= 0)

    # Model-scene affine coefficients
    @variable(m, 0 <= C[1:N_scene, 1:N_model_verts] <= 1.0)

    # Face selection
    if (use_relaxed_form)
        @variable(m, 0 <= f[1:N_scene, 1:N_model_faces] <= 1.0)
        @variable(m, 0 <= f_outlier[1:N_scene] <= 1.0)
    else
        @variable(m, 0 <= f[1:N_scene, 1:N_model_faces] <= 1.0, Bin)
        @variable(m, 0 <= f_outlier[1:N_scene] <= 1.0, Bin)
    end
    for i in 1:N_scene
        @constraint(m, sum(f[i, :]) + f_outlier[i] == 1)
    end

    # Coefficients must be affine coefficients if outlier isn't taken
    for i in 1:N_scene
        @constraint(m, sum(C[i, :]) + f_outlier[i] == 1)
    end

    # Build vertex-membership matrix F
    # where F[i, j] == 1 --> 
    #         vertex j is a member of face i
    F = Array{Float64}(size(faces, 1), size(points, 1))
    F .= 0.0
    for i = 1:size(faces, 1)
        for j = faces[i]
            F[i, j] = 1
        end
    end

    # Force members of C to be 0 if their respective
    # face is not active
    for i in 1:N_scene, j in 1:N_model_verts
        @constraint(m, C[i, j] <= dot(F[:, j], f[i, :]))
    end

    # Convert points into modelVerts, which is just a
    # 3xN array of the same data. (Allows vectorization
    # in just a bit.)
    modelVerts = Array{Float64}(3, size(points, 1))
    for i=1:size(points, 1)
        for k=1:3
            modelVerts[k, i] = points[i][k]
        end
    end

    # Create L1-norm-helper-members alpha for computing
    # the elementwise absolute value of the correspondence error
    @variable(m, alpha[1:3, 1:N_scene] >= 0)
    
    for i in 1:N_scene
        # Phi = Sum of Abs of Error X/Y/Z Components
        @constraint(m, phi[i] == dot([1; 1; 1],  alpha[:, i]))

        l1ErrorPos = R * scenePoints[:, i] + T - modelVerts * C[i, :]
        selector = optBigNumber * (f_outlier[i])
        @constraint(m, alpha[:, i] .>= l1ErrorPos - selector)
        @constraint(m, alpha[:, i] .>= -l1ErrorPos - selector)
    end

    @objective(m, Min, sum(phi)/N_scene + sum(f_outlier)*optPhiMax/N_scene);
    
    return PoseEstimationSolveData(m, R, T, R_abs, R_diff,
        phi, alpha, C, f, f_outlier,)
end

end #module