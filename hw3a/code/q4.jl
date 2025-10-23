
function check_lu(A, L, U, p, tol)
	@assert all(diag(L) .== 1)  #  check if L has a diagonal of 1
	@assert tril(L) == L  # check if L is lower triangular
	@assert triu(U) == U  # check if U is upper triangular
	@assert sort(p) == 1:length(p)   # check if it's actually a permutation

	@assert norm(A[p, :] - L * U) < tol  # check reconstruction
end

using Plots

A = [2 1 ; 1 2]
b = [5.5 ; 0.5]

function f(A, b, x)
	return x' * A * x / 2 .- x' * b
end

n = 1000
max = 10
min = -10
U = zeros(n, n)

for (idx_i, val_i) in enumerate(range(min, max, length=n))
	for (idx_j, val_j) in enumerate(range(min, max, length=n))
		U[idx_i, idx_j] = f(A, b, [val_i ; val_j])
	end
end

p = surface(U, camera=(40, 60))
savefig(p, "q4-1.pdf")

A = [4]
b = [-4.5]

u = zeros(n)
for (idx_i, val_i) in enumerate(range(min, max, length=n))
	u[idx_i] = f(A, b, [val_i])[1]
end

p = plot(u)
savefig(p, "q4-2.pdf")
