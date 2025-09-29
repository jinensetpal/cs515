using NearestNeighbors, Distributions, SparseArrays, LinearAlgebra
using Plots

function spatial_graph_edges(n::Integer,d::Integer;degreedist=LogNormal(log(4),1))
	xy = rand(d,n)
	T = BallTree(xy)
	# form the edges for sparse
	ei = Int[]
	ej = Int[]
	for i=1:n
		deg = min(ceil(Int,rand(degreedist)),n-1)
		idxs, dists = knn(T, xy[:,i], deg+1)
		for j in idxs
			if i != j
				push!(ei,i)
				push!(ej,j)
			end
		end
	end
	return xy, ei, ej
end
function spatial_network(n::Integer, d::Integer; degreedist=LogNormal(log(3),1))
	xy, ei, ej = spatial_graph_edges(n, d;degreedist=degreedist)
	A = sparse(ei,ej,1,n,n)
	return max.(A,A'), xy
end
using Random
Random.seed!(10) # ensure repeatable results...
A,xy = spatial_network(10, 2)

println(A)
println(xy)
