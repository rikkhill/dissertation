# dissertation
Repo for UCL CSML MSc dissertation project

Requires numpy and pandas.

MovieLens 1M dataset needs to live in a directory `./data/1M`

MovieLens Genome dataset needs to live in `./data/genome`

To produce a k=10 factorisation of the MovieLens dataset, create a directory `./output/factorisations/wnmf`, and run `factorise.py 10` - your factor matrices will be produced in the directory.

To produce responsibility keywords and exemplar films from this output, run `./semantics.py 10 ./output/factorisations/wnmf`

Proprietary data is not available for reproduction
