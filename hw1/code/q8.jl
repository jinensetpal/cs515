using SparseArrays

C = sparse([0 1 0 0 1; 2 0 0 6 0; 0 0 0 0 3])
x = [1, 2, 3]
@show C.colptr
@show C.rowval
@show C.nzval

""" Returns y = A'*x where A is given by the CSC arrays
colptr, rowval, nzval, m, n and x is the vector. """
function csc_transpose_matvec(colptr, rowval, nzval, m, n, x)
	res = zeros(n)

	for col_idx in 1:n
		for col_ptr in colptr[col_idx]:(colptr[col_idx+1]-1)
			res[col_idx] += x[rowval[col_ptr]] * nzval[col_ptr]
		end
	end

	return res
end

# println(transpose(C)x)
# println(csc_transpose_matvec(C.colptr, C.rowval, C.nzval, size(C)[1], size(C)[2], x))

""" Returns = A[:,i]'*x where A is given by the CSC arrays
colptr, rowval, nzval, m, n and x is the vector. """
function csc_column_projection(colptr, rowval, nzval, m, n, i, x)
	res = 0
	for col_ptr in colptr[i]:(colptr[i+1]-1)
		res += x[rowval[col_ptr]] * nzval[col_ptr]
	end

	return res
end

for i in 1:5
	println(transpose(C[:, i]) * x)
	println(csc_column_projection(C.colptr, C.rowval, C.nzval, size(C)[1], size(C)[2], i, x))
end

""" Returns rho = A[:,i]'*A[:,j] where A is given by the CSC arrays
colptr, rowval, nzval, m, n and i, and j are the column indices. """
function csc_col_col_prod(colptr, rowval, nzval, m, n, i, j)
	res = 0
	for col_ptr_i in colptr[i]:(colptr[i+1]-1)
		for col_ptr_j in colptr[j]:(colptr[j+1]-1)
			if rowval[col_ptr_i] == rowval[col_ptr_j]
				res += nzval[col_ptr_i] * nzval[col_ptr_j]
			end
		end
	end

	return res
end

# for i in 1:5
# 	for j in 1:5
# 		println(i,j, " ", transpose(C[:, i]) * C[:, j] == csc_col_col_prod(C.colptr, C.rowval, C.nzval, size(C)[1], size(C)[2], i, j))
# 	end
# end

""" Returns rho = A[i,j] where A is given by the CSC arrays
colptr, rowval, nzval, m, n and i, and j are the column indices. """
function csc_lookup(colptr, rowval, nzval, m, n, i, j)
	for col_ptr in colptr[j]:(colptr[j+1]-1)
		if rowval[col_ptr] == i
			return nzval[col_ptr]
		end
	end

	return 0
end

# for i in 1:3
# 	for j in 1:5
# 		println(i,j, " ", C[i, j] == csc_lookup(C.colptr, C.rowval, C.nzval, size(C)[1], size(C)[2], i, j))
# 	end
# end

""" Returns x = A[i,:] where A is given by the CSC arrays
colptr, rowval, nzval, m, n and i is the row index . """
function csc_lookup_row(colptr, rowval, nzval, m, n, i)
	res = zeros(n)
	for col_idx in 1:n
		for col_ptr in colptr[col_idx]:(colptr[col_idx+1]-1)
			if rowval[col_ptr] == i
				res[col_idx] = nzval[col_ptr]
			end
		end
	end

	return res
end

# for i in 1:3
# 	println(i, " ", C[i, :] == csc_lookup_row(C.colptr, C.rowval, C.nzval, size(C)[1], size(C)[2], i))
# end
