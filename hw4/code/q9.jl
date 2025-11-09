#include("gpt2.jl")

function read_token_sequence(file="wikitext_tokens.txt")
	open(file, "r") do io
		tokens = Vector{Vector{Int}}()
		for line in eachline(io)
			if startswith(line, "#")
				continue
			end
			if length(strip(line)) == 0
				continue
			end
			vals = split(line, ",")
			toks = parse.(Int, vals)
			push!(tokens, toks)
		end

		return tokens
	end
end

#using LaTeXStrings

#tokens = read_token_sequence()
#Y = reduce(vcat, [gpt2func(tokens[i], gpt2model) for i in 1:100])
#s_Y = svdvals(Y)
#
#n = size(Y)[1]
#Z = reduce(vcat, [gpt2func(rand(0:50256, length(tokens[i])), gpt2model) for i in 1:100])
#s_Z = svdvals(Z)

#p = Plots.plot(s_Y - s_Z, yaxis=:log, ylabel=L"\mathbf{s}_{\mathbf{text}, i} - \mathbf{s}_{\mathbf{random}, i}", xlabel=L"i \in \{0, 1, \ldots, d\}")
#savefig(p, "q9.pdf")

U_E, S_E, Vt_E = svd(Float64.(gpt2model.E))
U_Y, S_Y, Vt_Y = svd(Float64.(Y))

Q_E, R_E = qr(Float64.(gpt2model.E))
Q_Y, R_Y = qr(Float64.(Y))

U, S, Vt = svd(R_Y * R_E')

Q_A = Q_Y * U
S_A = diagm(S)
Vt_A = Vt * Q_E'
