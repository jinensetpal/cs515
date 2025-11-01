using SparseArrays, LinearAlgebra

# important: copied and modified from textbook
function oracle(A::Matrix)
	A = copy(A)#saveacopy
	n = size(A,1)
	F = Matrix(1.0I,n,n)
	d = zeros(n)
	for i=1:n−1
		alpha = A[i,i]
		d[i] = sqrt(alpha)
		F[i+1:end,i] = A[i+1:end,i]/alpha
		A[i+1:end,i+1:end] −= A[i+1:end,i]*A[i,i+1:end]'/alpha
	end
	d[n] = sqrt(A[n,n])
	return F*Diagonal(d),d
end

function tridiag_cholesky(A::Matrix)
	L = spzeros(size(A))
	n, m = size(A)

	if n == 1
		return sqrt(A)
	else
		L[1, 1] = sqrt(A[1, 1])  # set gamma
		L[2, 1] = A[2, 1] / L[1, 1]  # rest of column vector is already zero

		L_1 = tridiag_cholesky(A[2:end, 2:end])
		L[2:end, 2:end] = L_1
		L[2, 2] = sqrt(L_1[1, 1]^2 - L[2, 1]^2)
		if n > 2
			L[3, 2] = L_1[1, 1] * L_1[2, 1] / L[2, 2]
		end
	end

	return L
end

A = Tridiagonal(ones(9), Vector(1:10), ones(9))
F = tridiag_cholesky(Matrix(A))
println("Decomposition Error: ", norm(A - F*F'))
