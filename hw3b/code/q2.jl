using SparseArrays, LinearAlgebra

# important: copied and modified from textbook
function check_definiteness(X::Matrix, check_positive::Bool)  # if check_positive = true, check if positive definite, else check if negative definite
	A = copy(Float64.(X) + X')
	if !check_positive
		A = -A
	end

	n = size(A,1)
	F = Matrix(1.0I,n,n)
	d = zeros(n)
	for i=1:n−1
		alpha = A[i,i]
		if alpha < 0
			return false
		end
		d[i] = sqrt(alpha)
		F[i+1:end,i] = A[i+1:end,i]/alpha
		A[i+1:end,i+1:end] −= A[i+1:end,i]*A[i,i+1:end]'/alpha
	end
	return true
end

A = randn(10, 10)
PD = A'*A
println("Sanity Check 1 (expect true): ", check_definiteness(PD, true))
println("Sanity Check 2 (expect false): ", check_definiteness(A, true))


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

n = 10
A, fv = laplacian(n, (x, y) -> 1)
println()
println("Laplacian with boundary conditions (Positive): ", check_definiteness(Matrix(A), true))
println("Laplacian with boundary conditions (Negative): ", check_definiteness(Matrix(A), false))

A_smol = A[4n+1:end,4n+1:end]
println()
println("Laplacian without boundary conditions (Positive): ", check_definiteness(Matrix(A_smol), true))
println("Laplacian without boundary conditions (Negative): ", check_definiteness(Matrix(A_smol), false))
