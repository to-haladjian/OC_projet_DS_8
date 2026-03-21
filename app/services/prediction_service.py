"""Prediction service: ONNX model loading and credit scoring inference."""

import logging
from pathlib import Path

import onnxruntime as rt

from app.core.config import OPTIMAL_THRESHOLD
from app.services.preprocessing import preprocess_application

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "artefacts" / "model.onnx"

# Load ONNX model once at startup
_session_options = rt.SessionOptions()
_session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
_session_options.intra_op_num_threads = 1

_onnx_session = rt.InferenceSession(str(MODEL_PATH), _session_options)
logger.info("ONNX model loaded from %s", MODEL_PATH.name)


def predict_credit_default(features_dict: dict) -> tuple[int | None, float, bool]:
    """Run the full credit scoring prediction pipeline.

    Args:
        features_dict: Raw application features (from schema.model_dump()).

    Returns:
        (application_id, default_probability, credit_approved)
    """
    application_id, processed_array = preprocess_application(features_dict)

    # ONNX outputs: [0] = label (int), [1] = probabilities (list of dicts)
    result = _onnx_session.run(None, {"float_input": processed_array})
    probabilities = result[1][0]  # dict: {0: prob_no_default, 1: prob_default}
    default_probability = float(probabilities[1])
    credit_approved = bool(default_probability < OPTIMAL_THRESHOLD)

    logger.info(
        "Prediction completed: application_id=%s, probability=%.4f, approved=%s",
        application_id,
        default_probability,
        credit_approved,
    )

    return application_id, default_probability, credit_approved
