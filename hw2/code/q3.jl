using SparseMatricesCSR, LinearAlgebra, SparseArrays, Random
using ZipFile, DelimitedFiles

function row_projection(rowptr, colval, nzval, row_idx, x)
	res = .0

	for row_ptr in rowptr[row_idx]:(rowptr[row_idx+1]-1)
		res += x[colval[row_ptr]] * nzval[row_ptr]
	end

	return res
end

function pagerank(rowptr, colval, nzval, n::Integer, v::Vector, alpha_richardson::Number, alpha_pagerank::Number, T_max::Integer, resid_tol::Number)
	v *= alpha_pagerank
	x_hat = 1 * v

	# create alpha * P
	nzval *= alpha_pagerank

	# csr_matmul
	p_matmul = (x) -> map((i) -> row_projection(rowptr, colval, nzval, i, x), 1:n)

	itr = 0
	for itr in 1:T_max
		p_matmul_xhat = p_matmul(x_hat)
		if norm(x_hat - p_matmul_xhat - (1 - alpha_pagerank) * v)/norm((1 - alpha_pagerank) * v) < resid_tol
			return (x_hat, itr)
		end
		x_hat = (1 - alpha_richardson) * x_hat + alpha_richardson * (p_matmul_xhat + (1 - alpha_pagerank) * v)
	end
	return (x_hat, T_max)
end


# Random.seed!(0)
# P = sprand(1_000_000,1_000_000, 15/1_000_000) |>
# 		A->begin fill!(A.nzval,1); A; end |>
# 		A->begin ei,ej,ev = findnz(A); d = sum(A;dims=2);
# 			return sparse(ej,ei,ev./d[ei], size(A)...); end
# P = SparseMatrixCSR(P)  # I want CSR!
# rowptr,colval,nzval = P.rowptr, P.colval, P.nzval

# pagerank(rowptr, colval, nzval, size(P)[1], v, alpha_richardson, alpha_pagerank, T_max, tol)


function load_data()
	r = ZipFile.Reader("wikipedia-2005.zip")
	try
		@assert length(r.files) == 1
		f = first(r.files)
		data = readdlm(f,'\n',Int)
		n = data[1]
		colptr = data[2:2+n] # colptr has length n+1
		rowval = data[3+n:end] # n+2 elements before start of rowval
		A = SparseMatrixCSC(n,n,colptr,rowval,ones(length(rowval))) |>
		A->begin ei,ej,ev = findnz(A); d = sum(A;dims=2);
		return sparse(ej,ei,ev./d[ei], size(A)...) end
	finally
		close(r)
	end
end

alpha_pagerank = 0.85
alpha_richardson = 0.5
T_max = 1000
tol = 1e-5

P = SparseMatrixCSR(load_data())
rowptr,colval,nzval = P.rowptr, P.colval, P.nzval

v = ones(size(P)[1])/size(P)[1]
x_hat, itrs = pagerank(rowptr, colval, nzval, size(P)[1], v, alpha_richardson, alpha_pagerank, T_max, tol)
println(itrs)
