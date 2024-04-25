JL = julia

fmt:
	$(JL) --project=. -e 'using JuliaFormatter; format(["src", "test","examples"]; verbose=true)'
