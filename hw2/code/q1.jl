#!/usr/bin/env julia

using LinearAlgebra, SparseArrays, Plots

function map_index(i::Integer, j::Integer, n::Integer)
	if 1 < i < n+1 && 1 < j < n+1
		return 4n + (i - 2)*(n-1) + j-1
	elseif i == 1
		return j
	elseif i == n+1
		return n + 1 + j
	elseif j == 1
		return 2(n+1) + i - 1
	elseif j == n+1
		return 2(n+1) + n - 2 + i
	end
end

function laplacian(n::Integer, f::Function)
	A = sparse(1I, (n+1)^2, (n+1)^2)
	A[diagind(A)[4n+1:end]] .= -4

	fvec = zeros((n+1)^2)

	global row_index = 4n + 1
	for i in 2:n
		for j in 2:n
			A[row_index, map_index(i-1, j, n)] = 1
			A[row_index, map_index(i+1, j, n)] = 1
			A[row_index, map_index(i, j-1, n)] = 1
			A[row_index, map_index(i, j+1, n)] = 1
			fvec[row_index] = f(i, j)

			global row_index += 1
		end
	end

	return A, fvec/n^2
end

# n = 20
# A, fv = laplacian(n, (x, y) -> 1)
# 
# A = A[4n+1:end,4n+1:end]
# fv = fv[4n+1:end]
# 
# # equivalent system of equations
# A = -A
# fv = -fv
# 
# # check if new A is PD
# is_pd = all((2 .* Array(diag(A)) - sum(A, dims=2)) .>= 0)  # diagonally dominant implies PD
# println("Is `A` definitely PD? ", is_pd)
# 
# alpha = 1
# T = 100
# oracle_sol = A\fv
# norm_oracle_sol = norm(oracle_sol)
# 
# function richardson(A::SparseMatrixCSC, fv::Vector, T::Integer, alpha::Number, oracle_sol::Vector)
# 	norm_oracle_sol = norm(oracle_sol)
# 	x_hat = 1 * fv  # copy-by-value, not reference (https://stackoverflow.com/a/76109083/10671309)
# 	err = [norm(x_hat - oracle_sol) / norm_oracle_sol]   # initial error
# 	for itr in 1:T
# 		x_hat = (I - (alpha * A)) * x_hat + (alpha * fv)
# 		push!(err, norm(x_hat - oracle_sol) / norm_oracle_sol)
# 	end
# 	return (x_hat, err)
# end
# 
# function richardson(A::SparseMatrixCSC, fv::Vector, T::Integer, alpha::Number)
# 	x_hat = 1 * fv  # copy-by-value, not reference (https://stackoverflow.com/a/76109083/10671309)
# 	res = [norm(A * x_hat - fv) / norm(fv)]
# 	for itr in 1:T
# 		x_hat = (I - (alpha * A)) * x_hat + (alpha * fv)
# 		push!(res, norm(A * x_hat - fv) / norm(fv))
# 	end
# 	return (x_hat, res, minimum(push!(findall(x->x < 1e-5, res), T+1)))
# end
# 
# x_hat, err = richardson(A, fv, T, alpha, oracle_sol)
# println(norm(x_hat - A\fv))
# # p = plot(err, xlabel="iteration", label="error", yscale=:log10)
# # savefig(p, "q1-1.pdf")
# 
# T = 1000
# 
# alphs = []
# errs = []
# ress = []
# tols = []
# for alph_pow in 0:2
# 	for alph_coeff in 9:-.1:1
# 		local alpha = alph_coeff * 10^(-1. *alph_pow)
# 		local (x_hat, res, min_tol_iter) = richardson(A, fv, T, alpha)
# 		local err = norm(x_hat - oracle_sol) / norm_oracle_sol
# 
# 		push!(alphs, alpha)
# 		push!(errs, err)
# 		push!(ress, res)
# 		push!(tols, min_tol_iter)
# 	end
# end
# 
# p = scatter(alphs, errs, label="error", yscale=:log10, xscale=:log10, xlabel="alpha")
# savefig(p, "q1-3.pdf")
# 
# tol = 1e-5
# largest_alpha = maximum(alphs[errs .< tol])
# println("largest feasible alpha: ", largest_alpha)
# println("relative error: ", errs[alphs .== largest_alpha][end])
# println("relative residual: ", ress[alphs .== largest_alpha][end][end])
# println("fastest alpha: ", alphs[findall(x-> x == minimum(tols), tols)][end])
