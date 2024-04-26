JL = julia

fmt:
	$(JL) --project=. -e 'using JuliaFormatter; format(["src", "test","examples"]; verbose=true)'

nsys: 
	nsys launch julia --project=/home/exaclior/projects/CudaTraining.jl/examples/ --color=yes
