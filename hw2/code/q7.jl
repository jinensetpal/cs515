using DelimitedFiles, LinearAlgebra, SparseArrays, Plots


function gen_coord_descent(A, y::Vector, T_max::Integer, tol::Number, n_coords::Integer)
	x_hat = 1 * y  # copy-by-value, not reference (https://stackoverflow.com/a/76109083/10671309)
	for itr in 1:T_max
		g_k = A * x_hat - y   # matrix-vector product
		update = ((g_k' * g_k) / (g_k' * A * g_k)) * g_k

		for start in 1:n_coords:length(y)-n_coords
			for coord in start:start+n_coords
				x_hat[coord] -= update[coord]
			end
		end

		if norm(g_k) < tol
			return (x_hat, itr * length(y) / n_coords)
		end
	end
	return (x_hat, T_max * length(y) / n_coords)
end

coords = readdlm("chutes-and-ladders-coords.csv",',')
data = readdlm("chutes-and-ladders-matrix.csv",',')

xc = coords[:, 1]
yc = coords[:, 2]

TI = Int.(data[:,1])
TJ = Int.(data[:,2])
TV = data[:,3]
T = sparse(TI, TJ, TV, 101, 101)
A = I - T'

is_pd = all((2 .* Array(diag(A)) - sum(A, dims=2)) .>= 0)  # diagonally dominant implies PD
println("Is `A` definitely PD? ", is_pd)

y = ones(101)
y[100] = 0

T_max = 1000
tol = 1e-5

x, itr = gen_coord_descent(A, y, T_max, tol, 2)
