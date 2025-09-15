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

y = ones(101)
y[100] = 0

x = A\y

p = scatter(xc, yc, zcolor=x,
			markersize=16, label="", marker=:square, markerstrokewidth=0, size=(400,400),
			xlims=(0.5,10.5),ylims=(0.5,10.5),aspect_ratio=:equal)
function draw_chutes_and_annotate(p)
	CL = [ 1 4 9 21 28 36 51 71 80 98 95 93 87 64 62 56 49 48 16
		  38 14 31 42 84 44 67 91 100 78 75 73 24 60 19 53 11 26 6]
	for col=1:size(CL,2)
		i = CL[1,col]
		j = CL[2,col]
		if i > j # this is a chute
			plot!(p,[xc[i],xc[j]],[yc[i],yc[j]],color=2,label="")
		else
			plot!(p,[xc[i],xc[j]],[yc[i],yc[j]],color=1,label="")
		end
	end
	map(i->annotate!(p,xc[i],yc[i], text("$i", :white, 8)), 1:100)
	p
end
p = draw_chutes_and_annotate(p)
savefig(p, "q6.png")

expected_length = 0.0
delta = 1.
flag = true
k = 3
while delta > 1e-6 || flag
	global delta = k * (T^(k-1)*T[:, 101])[100]
	global expected_length += delta
	global k += 1

	if delta > 1e-6
		global flag = false # trip flag
	end
end

println(expected_length)
