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

function approx_evolve_steps(x0::Vector, p::Real, A::AbstractMatrix, k::Int)
	X = zeros(length(x0),k+1)
	X[:, 1] = x0
	for i=1:k
		X[:,i+1] = max.(p.*(A*X[:, i]), X[:, i])
	end
	return X
end

function power_method(x0::Vector, p::Real, A::AbstractMatrix, k::Int)
	res = (p * A)^k * x0
	if norm(res) < 1e-6
		return zeros(length(x0))
	end
	return res / norm(res)
end

using Random
Random.seed!(10) # ensure repeatable results...

p = 0.2
for nodes in [10, 1000]
	println("nodes: ", nodes)

	global A, xy = spatial_network(nodes, 2)
	if nodes == 10
		global x0 = zeros(size(A,1)); x0[1] = 1
	else
		global x0 = zeros(size(A,1)); x0[end] = 1
	end

	true_eig = maximum(eigvals(Matrix(p * A)))

	for steps in [10, 50, 100]
		global final_state = approx_evolve_steps(x0, p, A, steps)[:, end]
		# global final_state = power_method(x0, p, A, steps)

		global eigenvalue = (final_state' * (p * A) * final_state) / norm(final_state)^2
		println("steps: ", steps, ", eigenvalue: ", eigenvalue, ", approximation err:", true_eig - eigenvalue)
	end
	println("-----")
end


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

A, xy = spatial_network(1000, 2)
x0 = zeros(size(A,1)); x0[end] = 1

step = 0.0001
prev_eigenvalue = 0
for f in 0.9999:-step:0
	global final_state = power_method(x0, p, social_distance(A, xy, f), 100)
	global eigenvalue = (final_state' * (p * A) * final_state) / norm(final_state)^2
	if eigenvalue > 1
		println("minimum social distancing needed: ", f+step, " with eigenvalue: ", prev_eigenvalue)
		break
	end
	global prev_eigenvalue = eigenvalue
end

step = 0.0001
prev_eigenvalue = 0
for p in 0.071:step:0.2  # we know 0.2 has eigenvalue > 1
	global final_state = power_method(x0, p, A, 100)
	global eigenvalue = (final_state' * (p * A) * final_state) / norm(final_state)^2
	println("p: ", p, " with eigenvalue: ", eigenvalue)
	if eigenvalue > 1
		println("most powerful failing virus has p: ", p+step, " with eigenvalue: ", prev_eigenvalue)
		break
	end
	global prev_eigenvalue = eigenvalue
end

using StatsBase
function vaccinate(A::SparseMatrixCSC, f::Real)
	A = 1 * A   # copy by value
	for rc in StatsBase.sample(1:size(A)[1]-1, Int(f * size(A)[1]), replace=false)   # can't vaccinate patient zero
		A[:, rc] .= 0
		A[rc, :] .= 0
	end

	return dropzeros(A)
end

eigenvals = []
for trial in 1:1000
	global A_vac = vaccinate(A, .72)
	global final_state = power_method(x0, p, A_vac, 10)
	if final_state != zeros(length(final_state))  # in this case, all the neighbors of patient zero are vaccinated
		global eigenvalue = (final_state' * (p * A_vac) * final_state) / norm(final_state)^2
		push!(eigenvals, eigenvalue)
	end
end

println("Mean: ", mean(eigenvals), ", Std. Dev: ", std(eigenvals))
