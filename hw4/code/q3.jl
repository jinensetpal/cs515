function mysum(x::Vector{Float64})
	s = zero(Float64)
	for i=1:length(x)
		s += x[i]
	end
	return s
end

function mystablesum(x::Vector{Float64})
	x = sort(x)
	s = zero(Float64)
	for i=1:length(x)
		s += x[i]
	end
	return s
end

function kahansum(x::Vector{Float64})
    s = 0.0
    c = 0.0

	for i=1:length(x)
        y = x[i] - c
        t = s + y
        c = (t - s) - y
        s = t
	end

    return s
end

function oracle(x::Vector{Float64})  # test on arithmetic series
	return length(x) * (minimum(x) + maximum(x)) / 2
end


x = Vector{Float64}(1e9:-1e1:1)
println("Original: ", mysum(x))
println("Permuted: ", mystablesum(x))
println("Kahan: ", kahansum(x))
println("Oracle: ", oracle(x))
