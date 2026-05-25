"""Profile the end-to-end ONNX prediction pipeline with cProfile.

"""

from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.prediction_service import predict_credit_default 

ITERATIONS = 1000
OUTPUT_PROFILE = Path(__file__).parent / "profile.prof"

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


def warm_up() -> None:
    for _ in range(20):
        predict_credit_default(dict(SAMPLE_INPUT))


def run_profile() -> None:
    warm_up()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(ITERATIONS):
        predict_credit_default(dict(SAMPLE_INPUT))
    profiler.disable()

    profiler.dump_stats(OUTPUT_PROFILE)

    print(f"\nProfiled {ITERATIONS} predictions; profile saved to {OUTPUT_PROFILE}\n")
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)


if __name__ == "__main__":
    run_profile()
