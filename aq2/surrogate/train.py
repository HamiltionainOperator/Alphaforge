"""Surrogate model v2 trainer.

Loads feedback.json + memory.json, extracts 41-dim features from each
entry, trains two regressors (sharpe_model, fitness_model), reports 5-fold
CV RMSE, and saves a joblib bundle to surrogate_v1.joblib.

Usage
-----
    python -m aq2.surrogate.train
    python aq2/surrogate/train.py
    python aq2/surrogate/train.py --data-dir /path/to/data --out /path/to/out.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is importable regardless of cwd.
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import joblib

from judge.features import extract_features, to_vector, FEATURE_NAMES  # noqa: E402

# Default paths (relative to repo root).
_DEFAULT_DATA_DIR = _REPO / "data"
_DEFAULT_OUT = Path(__file__).parent / "surrogate_v1.joblib"


def _load_entries(data_dir: Path) -> list[dict]:
    """Load and merge entries from feedback.json and memory.json."""
    entries: list[dict] = []
    for fname in ("feedback.json", "memory.json"):
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"[train] Warning: {fpath} not found — skipping.")
            continue
        try:
            doc = json.loads(fpath.read_text())
        except json.JSONDecodeError as exc:
            print(f"[train] Warning: could not parse {fpath}: {exc} — skipping.")
            continue
        # Support both {"entries": [...]} and bare list formats.
        if isinstance(doc, list):
            entries.extend(doc)
        elif isinstance(doc, dict):
            entries.extend(doc.get("entries", []))
        print(f"[train] Loaded {fname}: {len(doc.get('entries', doc) if isinstance(doc, dict) else doc)} entries")
    return entries


def _build_dataset(
    entries: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and build (X, y_sharpe, y_fitness) arrays.

    Non-numeric or missing sharpe/fitness values are treated as 0.0 (failed
    or untested alphas contribute negative signal).
    """
    X: list[list[float]] = []
    y_sharpe: list[float] = []
    y_fitness: list[float] = []
    n_skipped = 0

    for e in entries:
        expr = e.get("expression")
        if not expr:
            n_skipped += 1
            continue
        settings = e.get("settings")
        try:
            feat = extract_features(expr, settings)
            X.append(to_vector(feat))
        except Exception as exc:
            print(f"[train] Feature extraction failed for expr={expr!r:.60}: {exc}")
            n_skipped += 1
            continue

        sh = e.get("sharpe")
        fit = e.get("fitness")
        y_sharpe.append(float(sh) if isinstance(sh, (int, float)) else 0.0)
        y_fitness.append(float(fit) if isinstance(fit, (int, float)) else 0.0)

    if n_skipped:
        print(f"[train] Skipped {n_skipped} entries (missing expression or feature error).")

    return np.array(X), np.array(y_sharpe), np.array(y_fitness)


def _build_models(backend: str = "auto"):
    """Instantiate regressor pair. Try LightGBM first, fall back to sklearn GBR."""
    if backend in ("auto", "lgbm"):
        try:
            import lightgbm as lgb  # noqa: PLC0415

            sharpe_model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42,
                verbose=-1,
            )
            fitness_model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42,
                verbose=-1,
            )
            used = "LightGBM"
            return sharpe_model, fitness_model, used
        except ImportError:
            if backend == "lgbm":
                raise RuntimeError(
                    "LightGBM requested (--backend lgbm) but not installed."
                )

    # Fallback: sklearn GradientBoostingRegressor.
    from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415

    sharpe_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    fitness_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    used = "sklearn GBR"
    return sharpe_model, fitness_model, used


def train(
    data_dir: Path | None = None,
    out_path: Path | None = None,
    backend: str = "auto",
    cv_folds: int = 5,
) -> dict:
    """Main training routine. Returns the saved bundle dict."""
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    out_path = Path(out_path) if out_path else _DEFAULT_OUT

    # 1. Load data.
    print(f"[train] Loading data from {data_dir} ...")
    all_entries = _load_entries(data_dir)
    print(f"[train] Total entries loaded: {len(all_entries)}")

    if not all_entries:
        print("[train] No entries found. Nothing to train.")
        sys.exit(1)

    # 2. Extract features.
    print("[train] Extracting features ...")
    X, y_sharpe, y_fitness = _build_dataset(all_entries)
    n = len(X)
    print(f"[train] Dataset: {n} samples x {X.shape[1] if n else 0} features")

    if n < cv_folds:
        print(
            f"[train] Warning: only {n} samples — not enough for {cv_folds}-fold CV. "
            "Skipping CV, training on full data."
        )
        cv_folds = 0

    # 3. Instantiate models.
    sharpe_model, fitness_model, used_backend = _build_models(backend)
    print(f"[train] Backend: {used_backend}")

    # 4. 5-fold cross-validated RMSE.
    if cv_folds > 0:
        from sklearn.model_selection import cross_val_score  # noqa: PLC0415

        print(f"[train] Running {cv_folds}-fold CV ...")

        sharpe_cv = cross_val_score(
            sharpe_model, X, y_sharpe,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
        )
        fitness_cv = cross_val_score(
            fitness_model, X, y_fitness,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
        )
        print(f"  Sharpe  CV RMSE: {-sharpe_cv.mean():.4f} ± {sharpe_cv.std():.4f}")
        print(f"  Fitness CV RMSE: {-fitness_cv.mean():.4f} ± {fitness_cv.std():.4f}")

    # 5. Final fit on all data.
    print("[train] Fitting final models on full dataset ...")
    sharpe_model.fit(X, y_sharpe)
    fitness_model.fit(X, y_fitness)

    # 6. Save bundle.
    bundle = {
        "sharpe_model": sharpe_model,
        "fitness_model": fitness_model,
        "backend": used_backend,
        "n_train": n,
        "feature_names": list(FEATURE_NAMES),
        "feature_version": "v3",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"[train] Saved surrogate bundle to {out_path}")
    return bundle


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m aq2.surrogate.train",
        description=(
            "Train the AlphaQuant v2 surrogate (LightGBM/GBR) from feedback + memory data."
        ),
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help=(
            f"Directory containing feedback.json and memory.json "
            f"(default: {_DEFAULT_DATA_DIR})"
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help=f"Output path for the joblib bundle (default: {_DEFAULT_OUT})",
    )
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "lgbm", "gbr"],
        help="ML backend: auto (LightGBM → GBR fallback), lgbm, or gbr (default: auto)",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        metavar="N",
        help="Number of cross-validation folds (default: 5; set 0 to skip CV)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_dir=args.data_dir,
        out_path=args.out,
        backend=args.backend,
        cv_folds=args.cv_folds,
    )
