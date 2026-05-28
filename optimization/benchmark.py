"""Benchmark sklearn LightGBM (Part 1 baseline) vs ONNX runtime (production).

Both backends consume the same preprocessed 100-feature array. We measure
per-request latency for single-row inference, which is the production access
pattern (one customer at a time).

Run:
    python optimization/benchmark.py

Writes a summary to optimization/benchmark_results.json.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as rt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.preprocessing import preprocess_application  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

BASELINE_MODEL_PATH = Path(__file__).parent / "baseline_lgbm.pkl"
ONNX_MODEL_PATH = PROJECT_ROOT / "app" / "artefacts" / "model.onnx"
RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"

ITERATIONS = 1000
WARMUP = 50
#hello

SAMPLE_INPUT = {
    "application_id": 100001,
    "code_gender": "M",
    "flag_own_car": "N",
    "name_contract_type": "Cash loans",
    "name_family_status": "Married",
    "name_education_type": "Higher education",
    "organization_type": "Self-employed",
    "amt_income_total": 150000.0,
    "amt_credit": 500000.0,
    "amt_annuity": 25000.0,
    "amt_goods_price": 450000.0,
    "birth_date": "1993-03-01",
    "employment_start_date": "2017-11-01",
    "registration_date": "2012-07-01",
    "id_publish_date": "2015-01-01",
    "last_phone_change_date": "2023-05-01",
    "ext_source_2": 0.5,
    "ext_source_3": 0.4,
    "region_population_relative": 0.02,
    "region_rating_client_w_city": 2,
    "obs_30_cnt_social_circle": 1.0,
    "def_30_cnt_social_circle": 0.0,
    "amt_req_credit_bureau_qrt": 0.0,
    "reg_city_not_live_city": 0,
    "floorsmax_avg": 0.2,
    "totalarea_mode": 0.1,
    "years_beginexpluatation_medi": 0.97,
    "flag_document_3": 1,
    "cnt_fam_members": 2.0,
}


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct))


def _summarize(label: str, latencies_ms: list[float]) -> dict:
    mean = statistics.fmean(latencies_ms)
    summary = {
        "backend": label,
        "iterations": len(latencies_ms),
        "mean_ms": round(mean, 4),
        "p50_ms": round(_percentile(latencies_ms, 50), 4),
        "p95_ms": round(_percentile(latencies_ms, 95), 4),
        "p99_ms": round(_percentile(latencies_ms, 99), 4),
        "throughput_rps": round(1000.0 / mean, 2),
    }
    return summary


def benchmark_sklearn(processed: np.ndarray) -> dict:
    model = joblib.load(BASELINE_MODEL_PATH)
    for _ in range(WARMUP):
        model.predict_proba(processed)

    latencies: list[float] = []
    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        model.predict_proba(processed)
        latencies.append((time.perf_counter_ns() - start) / 1_000_000)
    return _summarize("sklearn LightGBM", latencies)


def benchmark_onnx(processed: np.ndarray) -> dict:
    opts = rt.SessionOptions()
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    session = rt.InferenceSession(str(ONNX_MODEL_PATH), opts)

    feed = {"float_input": processed}
    for _ in range(WARMUP):
        session.run(None, feed)

    latencies: list[float] = []
    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        session.run(None, feed)
        latencies.append((time.perf_counter_ns() - start) / 1_000_000)
    return _summarize("ONNX Runtime", latencies)


def main() -> None:
    _, processed = preprocess_application(dict(SAMPLE_INPUT))

    sklearn_result = benchmark_sklearn(processed)
    onnx_result = benchmark_onnx(processed)

    speedup = sklearn_result["mean_ms"] / onnx_result["mean_ms"]
    artifact_sizes_kb = {
        "baseline_lgbm.pkl": round(BASELINE_MODEL_PATH.stat().st_size / 1024, 1),
        "model.onnx": round(ONNX_MODEL_PATH.stat().st_size / 1024, 1),
    }

    results = {
        "iterations": ITERATIONS,
        "warmup": WARMUP,
        "speedup_onnx_over_sklearn": round(speedup, 2),
        "artifact_sizes_kb": artifact_sizes_kb,
        "backends": [sklearn_result, onnx_result],
    }

    print("\nInference latency (single-row, ms)")
    print(f"{'backend':<18} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'rps':>10}")
    for r in results["backends"]:
        print(
            f"{r['backend']:<18} "
            f"{r['mean_ms']:>8.3f} {r['p50_ms']:>8.3f} {r['p95_ms']:>8.3f} "
            f"{r['p99_ms']:>8.3f} {r['throughput_rps']:>10.1f}"
        )
    print(f"\nONNX speedup over sklearn: {speedup:.2f}x")
    print(f"Artifact sizes: {artifact_sizes_kb}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
