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

function map_index_inv(k::Integer, n::Integer)
	if k <= n+1
		return 1, k
	elseif k <= 2(n+1)
		return n + 1, k - (n+1)
	elseif k < 2(n+1) + n
		return k - 2(n + 1) + 1, 1
	elseif k <= 4n
		return k - 2(n + 1) - n + 2, n + 1
	else
		return div(k-1 - 4n, n-1) + 2, (k-1 - 4n)%(n-1) + 2
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
u = A\fv

U = spzeros(n+1, n+1)
for k in 4n+1:(n+1)^2
	i, j = map_index_inv(k, n)
	U[i, j] = u[k]
end

for i in 1:n+1
	for j in 1:n+1
		pred_i, pred_j = map_index_inv(map_index(i, j, n), n)
		if !(pred_i == i && pred_j == j)
			println(i, " ", j)
		end
	end
end

p = surface(U)
savefig(p, "q7.png")

A_smol = A[4n+1:end,4n+1:end]
fv_smol = fv[4n+1:end]
u_smol = A_smol\fv_smol

norm(u_smol - u[4n+1:end])
