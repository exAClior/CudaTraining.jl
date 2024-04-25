JT = jupytext

# Convert all code under src/ to notebooks
nb: src/*.jl
	$(JT) --to notebook src/*.jl --output notebooks/

# invnb1:
# 	$(JT) --to jl:percent notebooks/naive_rev.ipynb --output naive_rev.jl
