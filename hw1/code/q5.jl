## Generate a simple spatial graph model to look at.
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

function plotgraph(A::SparseMatrixCSC,xy::AbstractArray{T,2};kwargs...) where T
	px,py = zeros(T,0),zeros(T,0)
	3
	P = [px,py]
	rows = rowvals(A)
	skip = NaN.*xy[:,begin] # first row
	for j=1:size(A,2) # for each column
		for nzi in nzrange(A, j)
			i = rows[nzi]
			if i > j
				push!.(P, @view xy[:,i])
				push!.(P, @view xy[:,j])
				push!.(P, skip)
			end
		end
	end
	plot(px,py;framestyle=:none,legend=false,kwargs...)
end

plotgraph(A,xy,alpha=0.25)
scatter!(xy[1,:],xy[2,:], markersize=2, markerstrokewidth=0, color=1)

function evolve(x::Vector, p::Real, A::AbstractMatrix)
	log_not_infected = log.(1 .- p.*x)
	y = 1 .- exp.(A*log_not_infected)
end
"""
Run k steps of the evolution and return the results as a matrix.
Each column of the matrix has the probabilities that the node
is infected under the `wrong` evolve function.
The first column of X is the initial vector x0.
At each iteration, we make sure the probabilities are at least x0 and these
are fixed.
"""
function evolve_steps(x0::Vector, p::Real, A::AbstractMatrix, k::Int)
	X = zeros(length(x0),k+1)
	X[:, 1] = x0
	for i=1:k
		X[:,i+1] = max.(evolve(X[:,i], p, A), X[:,1]) # fix the initial probability x0
	end
	return X
end

p = .2
x0 = zeros(size(A,1))
x0[1] = 1
X = evolve_steps(x0, p, A, 10)
X_plot = scatter(sum(X[2:end, :], dims=1)', label="net infection probability")
savefig(X_plot, "q5-p4.png")

"""
Run k steps of the approximate evolution and return the results as a matrix.
Each column of the matrix has the probabilities that the node
is infected under the `wrong` evolve function.
The first column of X is the initial vector x0.
At each iteration, we make sure the probabilities are at least x0 and these
are fixed.
"""
function approx_evolve_steps(x0::Vector, p::Real, A::AbstractMatrix, k::Int)
	X = zeros(length(x0),k+1)
	X[:, 1] = x0
	for i=1:k
		X[:,i+1] = max.(p.*(A*X[:, i]), X[:, i])
	end
	return X
end

X_approx = approx_evolve_steps(x0, p, A, 10)
X_approx_plot = scatter(sum(X_approx[2:end, :], dims=1)', label="approx net infection probability")
X_approx_plot = scatter!(sum(X[2:end, :], dims=1)', label="exact net infection probability")
savefig(X_approx_plot, "q5-p5.png")
X_approx_plot

p = .075
X = evolve_steps(x0, p, A, 10)
X_approx = approx_evolve_steps(x0, p, A, 10)
small_p_plot = scatter(sum(X_approx[2:end, :], dims=1)', label="approx net infection probability")
small_p_plot = scatter!(sum(X[2:end, :], dims=1)', label="exact net infection probability")
savefig(small_p_plot, "q5-p5b.png")

Random.seed!(10)
A,xy = spatial_network(1000, 2)
x0 = zeros(size(A,1))
x0[1] = 1

println("===")
for p in [.05, .1, .15, .2]
	println(p, ":\t", sum(approx_evolve_steps(x0, p, A, 10), dims=1)[end])
end
println("---")

for p in [.05, .1, .15, .2]
	println(p, ":\t", sum(evolve_steps(x0, p, A, 10), dims=1)[end])
end
println("===")

""" return a new "social" graph where we have implemented
social distancing by removing an f-fraction of your neighbors
based on spatial proximity. So f=0 is the original network
and f=1 is the empty network."""
function social_distance(A::SparseMatrixCSC, xy::Matrix, f::Real)
	# access the CSC arrays directly, see help on nzrange
	rowval = rowvals(A)
	n = size(A,1)
	new_ei = Vector{Int}()
	new_ej = Vector{Int}()
	for j = 1:n
		neighbors = Vector{Int}()
		dists = Vector{Float64}()
		myxy = @view xy[:,j] # don't make a copy
		for nzi in nzrange(A, j)
			# edge from (i,j)
			i = rowval[nzi]
			push!(neighbors, i)
			push!(dists, norm(xy[:,i]-myxy))
		end
		p = sortperm(dists) # sort distances
		nkeep = ceil(Int, (1-f)*length(dists))
		for i=1:nkeep
			push!(new_ei, neighbors[p[i]])
			push!(new_ej, j)
		end
	end
	A = sparse(new_ei,new_ej,1, size(A,1),size(A,2))
	return max.(A,A')
end

println("---")
for f in [.1, .2, .3, .4, .5, .6]
	local sparse_X = social_distance(A, xy, f)
	println("f = ", f)

	println("Exact -")
	for p in [.05, .1, .15, .2]
		println(p, ":\t", sum(evolve_steps(x0, p, sparse_X, 10), dims=1)[end])
	end
	println()

	println("Approximate -")
	for p in [.05, .1, .15, .2]
		println(p, ":\t", sum(approx_evolve_steps(x0, p, sparse_X, 10), dims=1)[end])
	end
	println("---")
end
