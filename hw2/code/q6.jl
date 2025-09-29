using SparseMatricesCSR, DelimitedFiles, LinearAlgebra, SparseArrays, Plots, Random

function partial_row_projection(rowptr, colval, nzval, row_idx, skip_col_idx, x)
	res = .0
	skip_val = 1

	for row_ptr in rowptr[row_idx]:(rowptr[row_idx+1]-1)  # only work on k non-zero entries
		if colval[row_ptr] != skip_col_idx
			res += x[colval[row_ptr]] * nzval[row_ptr]
		else
			skip_val = nzval[row_ptr]
		end
	end

	return res, skip_val
end

function random_coordinate_descent(rowptr, colval, nzval, y::Vector, T_max::Integer, tol::Number)
	x_hat = 1 * y  # copy-by-value, not reference (https://stackoverflow.com/a/76109083/10671309)
	for epoch in 1:T_max
		x_hat_new = similar(x_hat)

		for _ in 1:length(y)
			itr = rand(1:length(y))
			proj_val, skip_val = partial_row_projection(rowptr, colval, nzval, itr, itr, x_hat)
			x_hat_new[itr] = (y[itr] - proj_val) / skip_val
		end

		if norm(x_hat_new - x_hat) < tol
			return (x_hat, epoch * size(y)[1])
		end
		x_hat = 1 * x_hat_new
	end
	return (x_hat, T_max * size(y)[1])
end

function cyclic_coordinate_descent(rowptr, colval, nzval, y::Vector, T_max::Integer, tol::Number)
	x_hat = 1 * y  # copy-by-value, not reference (https://stackoverflow.com/a/76109083/10671309)
	for epoch in 1:T_max
		x_hat_new = similar(x_hat)

		for itr in 1:length(y)
			proj_val, skip_val = partial_row_projection(rowptr, colval, nzval, itr, itr, x_hat)
			x_hat_new[itr] = (y[itr] - proj_val) / skip_val
		end

		if norm(x_hat_new - x_hat) < tol
			return (x_hat, epoch * size(y)[1])
		end
		x_hat = 1 * x_hat_new
	end
	return (x_hat, T_max * size(y)[1])
end


# coords = readdlm("chutes-and-ladders-coords.csv",',')
# data = readdlm("chutes-and-ladders-matrix.csv",',')
# 
# xc = coords[:, 1]
# yc = coords[:, 2]
# 
# TI = Int.(data[:,1])
# TJ = Int.(data[:,2])
# TV = data[:,3]
# T = sparse(TI, TJ, TV, 101, 101)
# A = SparseMatrixCSR(I - T')
# 
# is_dd = all((2 .* Array(diag(A)) - sum(A, dims=2)) .>= 0)
# println("Is `A` diagonally dominant? ", is_dd)
# 
# y = ones(101)
# y[100] = 0
# 
# Random.seed!(0)
# T_max = 1000000
# tol = 1e-10
# 
# x_oracle = A\y
# 
# x_r, itr_r = random_coordinate_descent(A.rowptr, A.colval, A.nzval, y, T_max, tol)
# x_c, itr_c = cyclic_coordinate_descent(A.rowptr, A.colval, A.nzval, y, T_max, tol)
# 
# println(norm(x_oracle - x_r))
# println(norm(x_oracle - x_c))
