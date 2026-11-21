# Native slow-EMA experiment

This branch makes the final readout selectable in the Python binding. It fits
the same Clump model five times per seed, once for the closing-pass mean and
once per candidate EMA speed. The readout consumes each batch’s counts and
sums, shares the trainer’s donor/child split indices, and never assigns rows.

From this checkout, in an active Python 3.12 virtual environment:

```sh
python -m pip install maturin faiss-cpu==1.16.0 scikit-learn==1.9.1 numpy==2.5.5
maturin develop --release -m clump-python/Cargo.toml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/bench_emeans_native_readout.py
```

Each dataset uses a fixed permutation seeded with 20261121. Of 1,797 digit
images, 1,024 train the codebook, 386 select its speed and 387 score it. Of
10,000 word vectors, the corresponding counts are 4,096, 2,952 and 2,952.
The five fitting seeds are 7, 13, 23, 37 and 53. All runs use three epochs.
For each dataset and codebook size, the lowest median validation error
selects one speed from 1/16, 1/4, 1 and 4 times the schedule floor (0.1 ×
the initial step). In these four native comparisons validation chose 1.

MSE is the average squared Euclidean distance from each row to its nearest
final centroid, independently reassigned and scored in float64. Each percentage
is the median of five paired changes, 100 × (slow / closing − 1).

| Dataset | Centers | Train change | Test change | Test wins |
| --- | ---: | ---: | ---: | ---: |
| Digits | 8 | +0.142% | +0.033% | 2/5 |
| Digits | 64 | +0.925% | −0.052% | 4/5 |
| Word2Vec | 32 | +0.257% | −0.023% | 5/5 |
| Word2Vec | 256 | +2.384% | −1.036% | 5/5 |

The [20 paired observations](emeans_native_readout.csv) record both train
and test error for each seed. These are quality measurements, with one data
split per domain. The prototype also computes ordinary closing means before
returning its separate readout, so its runtime would overstate the cost of a
dedicated implementation.
