# Inference Optimization Report

## Context

The production API serves single-row predictions to a Gradio UI; tail latency matters more than throughput. The Part 1 model was trained as a scikit-learn `LGBMClassifier` and exported to ONNX for serving. This report measures the gain and identifies where time is now spent.

## Method

- **Profiling** ([optimization/profile_inference.py](../optimization/profile_inference.py)) — 1000 calls to `predict_credit_default` under `cProfile`, with 20 warm-up calls to settle imports and lazy initialization.
- **Benchmark** ([optimization/benchmark.py](../optimization/benchmark.py)) — 1000 single-row `predict_proba` / `session.run` calls per backend on the same preprocessed feature array, 50 warm-up calls each. Latency measured with `time.perf_counter_ns()`.
- **Baseline artifact** — `optimization/baseline_lgbm.pkl`, the sklearn `LGBMClassifier` exported from the Part 1 MLflow run (100 features, classes [0, 1]).
- **Production artifact** — `app/artefacts/model.onnx`, served via `onnxruntime` with `ORT_ENABLE_ALL` graph optimization and `intra_op_num_threads=1`.
- **Hardware** — local machine, single CPU thread. Numbers are reproducible; absolute values will shift on Hugging Face Spaces.

## Results

### Pure inference latency (single row)

| Backend          | Mean    | p50     | p95     | p99     | Throughput   |
| ---------------- | ------- | ------- | ------- | ------- | ------------ |
| sklearn LightGBM | 1.93 ms | 1.55 ms | 4.58 ms | 6.85 ms | ~518 req/s   |
| ONNX Runtime     | 0.009 ms| 0.009 ms| 0.010 ms| 0.020 ms| ~108,000 req/s |

**ONNX is ~210× faster than the sklearn baseline on the inference path, with a 32% smaller artifact (2.0 MB vs 2.9 MB).** Predictability also improves: ONNX's p95 is essentially identical to its mean, while sklearn's tail is 3× its median.

### End-to-end profile (1000 iterations of `predict_credit_default`)

The cProfile run shows total elapsed time of **4.94 s for 1000 calls (~4.94 ms per request end-to-end)**. Time decomposition by cumulative cost:

| Stage                                                   | Share of total | Notes |
| ------------------------------------------------------- | -------------- | ----- |
| `preprocess_application` (pandas DataFrame build + sklearn preprocessor `transform`) | ~98%           | DataFrame construction and `validate_data` checks dominate |
| `onnx_session.run`                                      | ~0.2%          | Negligible; matches the standalone benchmark |
| Threshold + logging                                     | <0.1%          | |

The preprocessing pipeline rebuilds a 100-column pandas DataFrame from a Python dict and runs it through `SimpleImputer + StandardScaler` on every request. `pandas.DataFrame.__init__` and `sklearn.utils.validation.check_array` together account for the majority of the time.

## Decision: keep ONNX in production

The 210× speedup makes ONNX the obvious choice on every axis we care about:

- **Latency** — the model is essentially free; tail latency is bounded by preprocessing.
- **Artifact size** — 2.0 MB vs 2.9 MB matters for the Docker image we push to Hugging Face Spaces.
- **Portability** — `onnxruntime` has no Python-side LightGBM dependency, simplifying the runtime image.
- **Determinism** — identical predictions to the sklearn baseline within float32 tolerance (verified during Part 1 export).

## Next bottleneck

End-to-end latency is now bounded by preprocessing, not inference. Concrete follow-ups, in priority order:

1. **Replace pandas DataFrame construction with a pre-allocated NumPy array.** The feature order is fixed in `app/artefacts/feature_list.csv`; we can write directly to a NumPy buffer and skip pandas entirely. Expected saving: 2–3 ms per request.
2. **Pre-cache the imputed default row.** 71 of the 100 features are NaN-filled-with-median on every request. Cache the post-imputation default vector at startup and only overwrite the cells the API actually populates. Expected saving: most of the remaining preprocessing time.
3. **Skip sklearn `validate_data` overhead** — once we bypass pandas, the `check_array` cost in the `SimpleImputer` / `StandardScaler` pipeline drops sharply because the input is already a contiguous float32 array.

These changes are out of scope for this report — they require touching the production `preprocess_application` path and warrant a follow-up benchmark to confirm no regression on the prediction outputs.

## Reproducing

```bash
python optimization/profile_inference.py   # writes optimization/profile.prof
python optimization/benchmark.py            # writes optimization/benchmark_results.json
```

The profile file can be inspected interactively with `snakeviz optimization/profile.prof` or via `python -m pstats optimization/profile.prof`.
