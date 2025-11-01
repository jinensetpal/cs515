using SparseArrays, LinearAlgebra, Plots

function solve(A::Matrix, B::Matrix)
	n, k = size(B)
	factor = div(n, 2)

	if n == 1
		return B ./ A
	else
		A_1 = A[1:factor, 1:factor]
		A_2 = A[1:factor, end-factor+1:end]
		A_3 = A[end-factor+1:end, 1:factor]
		A_4 = A[end-factor+1:end, end-factor+1:end]
		
		x_2 = solve(A_4 - A_3 * solve(A_1, A_2), B[end-factor+1:end, :] - A_3 * solve(A_1, B[1:factor, :]))
		x_1 = solve(A_1, B[1:factor, :] - A_2 * x_2)
		return vcat(x_1, x_2)
	end
end

Z = randn(2^4, 2^4)
A = Z'Z
b = ones(size(Z)[1], 1)

x_hat = solve(A, b)
println(norm(x_hat - A\b))
