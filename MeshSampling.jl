module MeshSampling

using StaticArrays
using GeometryTypes
using FileIO

export samplePointsFromMesh

function getFaceArea(points, face::Face)
    a = points[face[1]]
    b = points[face[2]]
    c = points[face[3]]
    return norm(cross(b - a, c - a)) / 2.
end

function samplePointsFromMesh(N_points, points, faces)
    # Get normalized cumulative area of faces
    cum_area = Array{Float64}(size(faces, 1))
    for i = 1:size(faces, 1)
        cum_area[i] = getFaceArea(points, faces[i])
    end
    cum_area = cumsum(cum_area)
    cum_area /= cum_area[end]
    
    sampled_pts = Array{Float64}(3, N_points)
    sampled_faces = zeros(N_points, size(faces, 1))
    for i=1:N_points
        # Pick the face we'll sample from
        sample = rand()
        k = findfirst(cum_area .>= sample)
        a = points[faces[k][1]]
        b = points[faces[k][2]]
        c = points[faces[k][3]]
        
        sampled_faces[i, k] = 1
            
        s1 = rand()
        s2 = rand()
        while (s1 + s2 > 1.0)
            s1 = rand()
            s2 = rand()
        end
        pt = a + s1*(b - a) + s2*(c - a)
        for k=1:3
            sampled_pts[k, i] = pt[k]
        end
    end
    return (sampled_pts, sampled_faces)
end

end # module