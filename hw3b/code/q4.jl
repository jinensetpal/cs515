using LinearAlgebra

function solve1_pivot2(A::Matrix, b::Vector)
	m,n = size(A)
	@assert(m==n, "the system is not square")
	@assert(n==length(b), "vector b has the wrong length")
	if n==1
		display(A)
		@show(b)
		return [b[1]/A[1]]
	else
		# let's make sure we have an equation
		# that we can eliminate!
		# let's try that again, where we pick the
		# largest magnitude entry!
		maxval = abs(A[1,1])
		newrow = 1
		for j=2:n
			if abs(A[j,1]) > maxval
				newrow = j
				maxval = abs(A[j,1])
			end
		end
		if maxval < eps(1.0)
			error("the system is singular")
		end
		@show newrow
		# swap rows 1, and newrow
		if newrow != 1
			tmp = A[1,:]
			A[1,:] .= A[newrow,:]
			A[newrow,:] .= tmp
			b[1], b[newrow] = b[newrow], b[1]
		end
		D = A[2:end,2:end]
		c = A[1,2:end]
		d = A[2:end,1]
		α = A[1,1]
		y = solve1_pivot2(D-d*c'/α, b[2:end]-b[1]/α*d)
		γ = (b[1] - c'*y)/α
		return pushfirst!(y,γ)
	end
end

function solve1_pivot2_fixed(A::Matrix, b::Vector)
	m,n = size(A)
	@assert(m==n, "the system is not square")
	@assert(n==length(b), "vector b has the wrong length")
	if n==1
		@show(b)
		display(A)
		return [1.]
	else
		# let's make sure we have an equation
		# that we can eliminate!
		# let's try that again, where we pick the
		# largest magnitude entry!
		maxval = abs(A[1,1])
		newrow = 1
		for j=2:n
			if abs(A[j,1]) > maxval
				newrow = j
				maxval = abs(A[j,1])
			end
		end
		if maxval < eps(1.0)
			error("the system is singular")
		end
		@show newrow
		# swap rows 1, and newrow
		if newrow != 1
			tmp = A[1,:]
			A[1,:] .= A[newrow,:]
			A[newrow,:] .= tmp
			b[1], b[newrow] = b[newrow], b[1]
		end
		D = A[2:end,2:end]
		display(D)
		c = A[1,2:end]
		d = A[2:end,1]
		α = A[1,1]
		y = solve1_pivot2_fixed(D-d*c'/α, b[2:end]-b[1]/α*d)
		γ = (b[1] - c'*y)/α
		return pushfirst!(y,γ)
	end
end

A = [1 2 2; 0 2 1; -1 2 2]
lambda = 1

Y = A - lambda * I
b = zeros(size(Y)[1])

println("Valid problem? ", Bool(rank(A) - rank(Y)))
x_hat = solve1_pivot2_fixed(Y, b)
println("Correct Solution? ", all((A * x_hat) ./ x_hat .== lambda))
