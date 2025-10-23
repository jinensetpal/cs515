using SparseArrays, LinearAlgebra, BenchmarkTools, Plots

function forwardsolve(L, y)  # Lower triangular L
	x = zeros(length(y))  # set to unset values to zero, allows us to directly call dot and compute a partial sum

	for i in 1:length(y)
		x[i] = (y[i] - dot(L[i, :], x))/L[i, i]  # iterate and solve sequentially
	end
	return x
end

function backsolve(U, y)  # Upper Triangular U
	x = zeros(length(y))

	for i in length(y):-1:1  # reverse iteration order
		x[i] = (y[i] - dot(U[i, :], x))/U[i, i]
	end
	return x
end

n = 10
# A = sparse(2:n, 1:n-1, -1.0, n, n) + I
# b = ones(n)

function lt_solve(A, b)
	return forwardsolve(A, b)
end


tol = 1e-5
n_trials = 5

lt_sol_mean = []
lt_sol_std = []
lt_opt_mean = []
lt_opt_std = []
for n_pow in 1:4
	local n = 10^n_pow
	local lt_sol_time = []
	local lt_opt_time = []
	for n_trial in 1:n_trials
		global A = sparse(2:n, 1:n-1, -1.0, n, n) + I
		global b = ones(n)
	
		@assert norm(A\b - lt_solve(A, b)) < tol
		push!(lt_opt_time, @belapsed A\b)
		push!(lt_sol_time, @belapsed lt_solve(A, b))
	end
	push!(lt_opt_mean, mean(lt_opt_time))
	push!(lt_opt_std, std(lt_opt_time))
	push!(lt_sol_mean, mean(lt_sol_time))
	push!(lt_sol_std, std(lt_sol_time))
end

plot(lt_opt_mean, err=lt_opt_std, label="julia's solver")
p = plot!(lt_sol_mean, err=lt_sol_std, label="my custom solver", xlabel="n = 10^")
savefig(p, "q2-1.pdf")

function solve(A, b)
	L, U, p = lu(A)
	return backsolve(U, forwardsolve(L, b[p]))
end


sol_mean = []
sol_std = []
opt_mean = []
opt_std = []
for n_pow in 1:5
	local n = 10^n_pow
	local sol_time = []
	local opt_time = []
	for n_trial in 1:n_trials
	 	global A = rand(n, n)
	 	global b = rand(n)
	
		@assert norm(A\b - solve(A, b)) < tol
		push!(opt_time, @belapsed A\b)
		push!(sol_time, @belapsed solve(A, b))
	end
	push!(opt_mean, mean(opt_time))
	push!(opt_std, std(opt_time))
	push!(sol_mean, mean(sol_time))
	push!(sol_std, std(sol_time))
end

plot(opt_mean, err=opt_std, label="julia's solver")
p = plot!(sol_mean, err=sol_std, label="my custom solver", xlabel="n = 10^")
savefig(p, "q2-2.pdf")
