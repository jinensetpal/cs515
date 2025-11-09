function f(x::Float32, a::Float32, b::Float32, c::Float32)
	return a*x^2 + b*x + c
end

function roots(a::Float32, b::Float32, c::Float32)
	if a > 0
		# then it's convex, f'(x') = 0, f(x') should be negative
		if f(-b/(2a), a, b, c) > 0
			return (NaN, NaN)
		end

		x_pos = -b/(2a) + maximum([abs(a), abs(1/a), abs(b), abs(1/b), abs(c), abs(1/c)])
		x_neg = -b/(2a)
	elseif a == 0  # linear function
		return (-c/b, NaN)
	elseif a < 0
		# then it's concave, f'(x') = 0, f(x') should be positive
		if f(-b/(2a), a, b, c) < 0
			return (NaN, NaN)
		end

		x_neg = -b/(2a) + maximum([abs(a), abs(1/a), abs(b), abs(1/b), abs(c), abs(1/c)])
		x_pos = -b/(2a)
	end

	tol = 1e-10
	max_iter = 100000

	iter = 1
	root = [NaN, NaN]
	while iter < max_iter
		midpoint = (x_pos + x_neg) / 2
		f_mid = f(midpoint, a, b, c)

		if abs(f_mid) < tol
			root[1] = midpoint
		elseif sign(f_mid) == 1
			x_pos = midpoint
		elseif sign(f_mid) == -1
			x_neg = midpoint
		end

		iter += 1
	end

	if !isnan(root[1])
		root[2] = -b/a - root[1]
	end

	return root
end

function citardauq(a::Float32, b::Float32, c::Float32)
	return (2c/(-b + sqrt(b^2 - 4a*c)), 2c/(-b - sqrt(b^2 - 4a*c)))
end

function lumbric_vonbrand(a::Float32, b::Float32, c::Float32)
	if b < 0
		r1 = (- b + sqrt(b^2 - 4a*c)) / 2a
	elseif b > 0
		r1 = (- b - sqrt(b^2 - 4a*c)) / 2a
	else
		return (sqrt(c/a), -sqrt(c/a))
	end

	r2 = - b/a - r1
	return (r1, r2)
end

# to get extreme examples, we need 4ac \approx = 0, because then b - b \approx 0 will incur floating point errors
a = Float32(1e-4)
b = Float32(10 * randn(1)[1])
c = Float32(1e-4)

sol = lumbric_vonbrand(a, b, c)
@show(sol)
@show(f(sol[1], a, b, c))
@show(f(sol[2], a, b, c))

sol = citardauq(a, b, c)
@show(sol)
@show(f(sol[1], a, b, c))
@show(f(sol[2], a, b, c))

sol = roots(a, b, c)
@show(sol)
@show(f(sol[1], a, b, c))
@show(f(sol[2], a, b, c))
