import pickle

import pandas as pd

from src.feature.feature_builder import create_features
from src.utils.config import FEATURE_COLUMNS_PATH, MODEL_PATH
from src.utils.matchpoint import derive_matchpoint

_model = None
_feature_columns = None


def _ensure_model_loaded():
    global _model, _feature_columns
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(FEATURE_COLUMNS_PATH, "rb") as f:
            _feature_columns = pickle.load(f)


def predict_similarity(email1: str, email2: str, matchpoint: str = "") -> float:
    _ensure_model_loaded()
    scores = predict_scores_batch(email1, [email2], matchpoints=[matchpoint] if matchpoint else None)
    return scores[0][1]


def predict_scores_batch(
    input_email: str,
    candidate_emails: list[str],
    chunk_size: int = 500,
    matchpoints: list[str] | None = None,
) -> list[tuple[str, float]]:
    _ensure_model_loaded()

    input_email = input_email.strip().lower()
    results: list[tuple[str, float]] = []

    for start in range(0, len(candidate_emails), chunk_size):
        chunk = candidate_emails[start : start + chunk_size]
        chunk_matchpoints = None
        if matchpoints is not None:
            chunk_matchpoints = matchpoints[start : start + chunk_size]
        else:
            chunk_matchpoints = [
                derive_matchpoint(input_email, email) for email in chunk
            ]

        df = pd.DataFrame(
            {
                "inputEmail": input_email,
                "matchEmail": chunk,
                "Matchpoint": chunk_matchpoints,
            }
        )
        features = df.apply(create_features, axis=1)[_feature_columns]
        scores = _model.predict(features)

        for email, score in zip(chunk, scores):
            results.append((email, round(float(score), 2)))

    return results
