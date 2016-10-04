# dissertation
Repo for UCL CSML MSc dissertation project

Requires numpy and pandas, as well as up-to-date submodule pymf

It is worth pointing out this is a collection of ad-hoc scripts for producing test metrics and visualisations, rather than a well-tested software product. Readers are welcome to use these scripts to reproduce the results, but they are not tested on all platforms, and may be subject to environmental irregularities. Proprietary data used in the dissertation project is unavailable for inspection.

The MovieLens 1M dataset needs to live in a directory `./data/1M` and MovieLens Genome dataset needs to live in `./data/genome` - running `./scripts/data_download.sh` may do this for you, but is not tested on all platforms. 

To produce a k=10 factorisation of the MovieLens dataset, create a directory `./output/factorisations/wnmf`, and run `./factorise.py 10` - your factor matrices will be produced in the directory.

To produce responsibility keywords and exemplar films from this output, run `./semantics.py 10 ./output/factorisations/wnmf`

The script `sample_factorisation.py` will produce training and test error results for the relevant method of matrix factorisation in a file in `./outpiut/results/`. Inspection of the script is recommended.

The script `factorise_augmented.py` is an ad-hoc script which wrangles relevant side-information and constructs the appropriate matrix for factorisation with an appropriate method. Inspection of this script is also recommended.
