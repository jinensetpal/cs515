
include("q1.jl")
include("q5.jl")
include("q6.jl")

# 2d laplacian
n = 40
A, y = laplacian(n, (x, y) -> 1)

A = SparseMatrixCSR(A[4n+1:end,4n+1:end])
y = y[4n+1:end]

# chutes and ladders
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
# y = ones(101)
# y[100] = 0


T_max = 1000
alpha = 1.4
tol = 1e-4

tols = []
works_stpd = []
works_coord = []
for tol_pow in 1:4
	for tol_coeff in 9:-.1:1
		local tol = tol_coeff * 10^(-1. * tol_pow)
		local x_stpd, itr_stpd = steepest_descent(A, y, T_max, alpha, tol)
		local work_stpd = itr_stpd * nnz(A)

		local x_coord, itr_coord = cyclic_coordinate_descent(A.rowptr, A.colval, A.nzval, y, T_max, tol)
		local work_coord = itr_coord / length(y) * nnz(A)

		push!(tols, tol)
		push!(works_stpd, work_stpd)
		push!(works_coord, work_coord)
	end
end

p = scatter(tols, works_stpd, label="Steepest Descent")
p = scatter!(tols, works_coord, label="Coordinate Descent", xlabel="tolerance", ylabel="work", yscale=:log10, xscale=:log10)
p = xflip!(true)
savefig(p, "q8-2.pdf")
