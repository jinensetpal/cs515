using DelimitedFiles, LinearAlgebra, SparseArrays, Plots


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

function richardson(A, fv::Vector, T_max::Integer, alpha::Number, tol::Number)
	x_hat = 1 * fv  # copy-by-value, not reference (https://stackoverflow.com/a/76109083/10671309)
	for itr in 1:T_max
		x_hat = (I - (alpha * A)) * x_hat + (alpha * fv)
		if norm(A * x_hat - fv) / norm(fv) < tol
			return (x_hat, itr)
		end
	end
	return (x_hat, T_max)
end

T_max = 1000
alpha = .2
tol = 1e-5

alphs = []
itrs = []
for alph_pow in 0:1
	for alph_coeff in 9:-.05:1
		local alpha = alph_coeff * 10^(-1. *alph_pow)
		local (x, itr) = richardson(A, y, T_max, alpha, tol)

		push!(alphs, alpha)
		push!(itrs, itr)
	end
end

p = scatter(alphs, itrs, label="#itr to convergence (or give up at 1000)", xlabel="alpha")
savefig(p, "q4-1.pdf")
println(alphs[findall(x-> x == minimum(itrs), itrs)][end])

# failed midterm experimend
function get_alpha(gamma::Number)
	return 1 / abs(gamma - 1) - 1e-2
end

b = ones(2)
for gamma in 0:.1:10
	if gamma == 1
		continue
	end

	local A = [gamma -1; -1 gamma]
	local alpha = get_alpha(gamma)

	x_hat, n_iter = richardson(A, b, T_max, alpha, tol)
	x_true = A \ b
	println(gamma, ", ", alpha, ", ", x_hat, ", ", n_iter, ", ", norm(A * x_hat - b), ", ", norm(x_true - x_hat))
end

