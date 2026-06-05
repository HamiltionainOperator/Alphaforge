"""ML-based pre-filter pipeline for BRAIN-bound alpha expressions.

NN-1 (failure gate) is the only model live right now. It exists in two
implementations that share the same FEATURE_NAMES / extract_features inputs
and the same predict-probability contract:

    judge.nn1.predict_fail_probability(expr, settings)        -> RandomForest
    judge.nn1_torch.predict_fail_probability_torch(expr, ...)  -> PyTorch MLP

The MLP wins on AUC (0.777 vs 0.690) but is more aggressive at the default
0.5 threshold; the RF wins on FAIL-class precision at that threshold. Both
artifacts ship — choose by your operating point.

Other layers (NN-2/3/4) live in this package as they come online.
"""
from judge.labels import LABEL_FAIL, LABEL_OK, classify_outcome
from judge.features import FEATURE_VERSION, FEATURE_NAMES, extract_features

__all__ = [
    "LABEL_FAIL",
    "LABEL_OK",
    "classify_outcome",
    "FEATURE_VERSION",
    "FEATURE_NAMES",
    "extract_features",
]
