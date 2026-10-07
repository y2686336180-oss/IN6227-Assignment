from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 6227
FOLDS = 3
THRESHOLD = 0.5


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def audit_frame(frame):
    numeric = frame.select_dtypes(include="number")
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3 - q1
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "missing_cells": int(frame.isna().sum().sum()),
        "label_counts": {str(k): int(v) for k, v in frame.label.value_counts().items()},
        "unlabelled_rows": int(frame.label.isna().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_feature_rows": int(frame.drop(columns="label").duplicated().sum()),
        "numeric_summary": numeric.describe().round(6).to_dict(),
        "iqr_flagged_counts": ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum().astype(int).to_dict(),
        "categorical_counts": {
            c: {str(k): int(v) for k, v in frame[c].value_counts(dropna=False).items()}
            for c in frame.select_dtypes(exclude="number").columns if c != "label"
        },
    }


def make_pipeline(model, numeric, categorical, scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline(num_steps), numeric),
        ("categorical", categorical_pipeline, categorical),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def compute_metrics(y, probabilities, predictions=None):
    predictions = (probabilities >= THRESHOLD).astype(int) if predictions is None else predictions
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    return {
        "AP": float(average_precision_score(y, probabilities)),
        "ROC_AUC": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision_yes": float(precision_score(y, predictions, zero_division=0)),
        "recall_yes": float(recall_score(y, predictions, zero_division=0)),
        "F1_yes": float(f1_score(y, predictions, zero_division=0)),
        "TN": int(matrix[0, 0]), "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]), "TP": int(matrix[1, 1]),
    }


def paired_bootstrap(y, lr_probs, rf_probs, iterations):
    """Paired IID test-row bootstrap; conditional on the fitted models."""
    rng = np.random.default_rng(SEED)
    samples = np.empty((iterations, 2))
    for i in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        samples[i, 0] = average_precision_score(y[idx], lr_probs[idx])
        samples[i, 1] = average_precision_score(y[idx], rf_probs[idx])
    return {
        "iterations": iterations,
        "method": "Paired IID bootstrap of test rows; 2.5th/97.5th percentiles; fixed fitted models",
        "LR_AP_95CI": np.quantile(samples[:, 0], [.025, .975]).tolist(),
        "RF_AP_95CI": np.quantile(samples[:, 1], [.025, .975]).tolist(),
        "RF_minus_LR_AP_95CI": np.quantile(samples[:, 1] - samples[:, 0], [.025, .975]).tolist(),
    }


def make_figures(y, probabilities, metrics, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    colors = {"Logistic regression": "#24649a", "Random forest": "#cb7335"}
    for name, p in probabilities.items():
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(recall, precision, color=colors[name], lw=1.7,
                     label=f"{name} (AP {metrics[name]['AP']:.3f})")
    axes[0].axhline(float(y.mean()), color="#777777", linestyle="--", lw=1, label=f"Prevalence ({y.mean():.3f})")
    axes[0].set(xlabel="Recall (yes)", ylabel="Precision (yes)", xlim=(0, 1), ylim=(0, 1.03), title="Test precision-recall curves")
    axes[0].legend(fontsize=8.3, loc="lower left", frameon=True,
                   facecolor="white", edgecolor="none", framealpha=1)
    for ax, name in zip(axes[1:], probabilities):
        matrix = np.array([[metrics[name]["TN"], metrics[name]["FP"]],
                           [metrics[name]["FN"], metrics[name]["TP"]]])
        fractions = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{matrix[row, col]:,}\n({fractions[row, col]:.1%})", ha="center", va="center",
                        fontsize=10, color="white" if fractions[row, col] > .55 else "#1d2730")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no", "yes"], yticklabels=["no", "yes"],
               xlabel="Predicted", ylabel="Actual", title=name)
        ax.tick_params(length=0)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(output / "comparison.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "comparison.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--jobs", type=int, default=1, help="Parallel trees; use 1 on restricted Windows environments")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train_path, test_path = args.data_dir / "train.csv", args.data_dir / "test.csv"
    raw_train = pd.read_csv(train_path)
    assert "label" in raw_train.columns and raw_train.columns.is_unique
    assert set(raw_train.label.dropna().unique()) == {"no", "yes"}
    train = raw_train.dropna(subset=["label"]).copy()
    X = train.drop(columns="label")
    y = train.label.map({"no": 0, "yes": 1}).astype(int)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    assert np.isfinite(X[numeric].to_numpy()[~X[numeric].isna().to_numpy()]).all()
    audit = {"train": audit_frame(raw_train), "numeric_features": numeric,
             "categorical_features": categorical,
             "train_numeric_correlations": X[numeric].corr().round(6).to_dict()}
    save_json(out / "data_audit.json", audit)

    # Fix all modelling decisions before reading test.csv for final evaluation.
    cv = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X, y))
    specs = {
        "Logistic regression": (
            make_pipeline(LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000,
                          tol=1e-4, random_state=SEED), numeric, categorical, scale=True),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "Random forest": (
            make_pipeline(RandomForestClassifier(n_estimators=200, max_features="sqrt",
                          bootstrap=True, random_state=SEED, n_jobs=args.jobs), numeric, categorical, scale=False),
            {"model__max_depth": [12, None], "model__min_samples_leaf": [1, 5]},
        ),
    }
    fitted, details, cv_frames = {}, {}, []
    # Treat failure to converge as an experiment failure rather than silently report it.
    warnings.filterwarnings("error", category=ConvergenceWarning)
    for name, (pipeline, grid) in specs.items():
        print(f"Tuning {name}: {FOLDS}-fold stratified CV, average precision", flush=True)
        search = GridSearchCV(pipeline, grid, scoring="average_precision", cv=cv, refit=False,
                              n_jobs=1, error_score="raise", return_train_score=True)
        tick = time.perf_counter()
        with threadpool_limits(limits=args.jobs):
            search.fit(X, y)
        search_seconds = time.perf_counter() - tick
        best = clone(pipeline).set_params(**search.best_params_)
        tick = time.perf_counter()
        with threadpool_limits(limits=args.jobs):
            best.fit(X, y)
        refit_seconds = time.perf_counter() - tick
        best_index = int(search.best_index_)
        cv_scores = [float(search.cv_results_[f"split{i}_test_score"][best_index]) for i in range(FOLDS)]
        details[name] = {
            "best_parameters": search.best_params_,
            "all_model_parameters": best.named_steps["model"].get_params(),
            "cv_AP_mean": float(np.mean(cv_scores)),
            "cv_AP_sd": float(np.std(cv_scores, ddof=1)),
            "cv_AP_folds": cv_scores,
            "cv_train_AP_mean": float(search.cv_results_["mean_train_score"][best_index]),
            "search_seconds": search_seconds, "refit_seconds": refit_seconds,
            "transformed_feature_count": len(best.named_steps["preprocess"].get_feature_names_out()),
            "train_metrics": compute_metrics(y, best.predict_proba(X)[:, 1]),
        }
        if name == "Logistic regression":
            details[name]["n_iter"] = best.named_steps["model"].n_iter_.tolist()
        fitted[name] = best
        frame = pd.DataFrame(search.cv_results_)
        frame.insert(0, "model", name)
        cv_frames.append(frame)
        print(f"  Selected {search.best_params_}; CV AP={np.mean(cv_scores):.6f}; refit={refit_seconds:.2f}s", flush=True)
    pd.concat(cv_frames, ignore_index=True).to_csv(out / "cv_results.csv", index=False)
    selected_model = max(details, key=lambda name: details[name]["cv_AP_mean"])

    # A diagnostic sensitivity check, not a test-driven feature-selection step.
    ablations = []
    for name, fitted_pipeline in fitted.items():
        print(f"Sensitivity check without composite_rank: {name}", flush=True)
        reduced_numeric = [c for c in numeric if c != "composite_rank"]
        reduced = make_pipeline(clone(fitted_pipeline.named_steps["model"]), reduced_numeric,
                                categorical, scale=name == "Logistic regression")
        with threadpool_limits(limits=args.jobs):
            scores = cross_val_score(reduced, X.drop(columns="composite_rank"), y,
                                     cv=cv, scoring="average_precision", n_jobs=1, error_score="raise")
        ablations.append({"model": name, "full_cv_AP": details[name]["cv_AP_mean"],
                          "without_composite_rank_cv_AP": float(scores.mean()),
                          "without_composite_rank_sd": float(scores.std(ddof=1)),
                          "without_minus_full_AP": float(scores.mean() - details[name]["cv_AP_mean"]),
                          "fold_AP": scores.tolist()})
    pd.DataFrame(ablations).to_csv(out / "ablation.csv", index=False)

    print("Model configurations fixed. Evaluating the supplied test split.", flush=True)
    raw_test = pd.read_csv(test_path)
    assert list(raw_train.columns) == list(raw_test.columns), "Train/test schema mismatch"
    assert set(raw_test.label.dropna().unique()) == {"no", "yes"}
    test = raw_test.dropna(subset=["label"]).copy()
    Xt = test.drop(columns="label")
    yt = test.label.map({"no": 0, "yes": 1}).astype(int).to_numpy()
    assert np.isfinite(Xt[numeric].to_numpy()[~Xt[numeric].isna().to_numpy()]).all()
    feature_overlap = len(pd.merge(raw_train.drop(columns="label"), raw_test.drop(columns="label"), how="inner").drop_duplicates())
    assert feature_overlap == 0, "Exact feature rows overlap across supplied splits"
    audit.update({"test": audit_frame(raw_test), "cross_split_exact_feature_overlap": feature_overlap,
                  "test_unseen_categories": {c: sorted(set(Xt[c].dropna()) - set(X[c].dropna())) for c in categorical}})
    save_json(out / "data_audit.json", audit)
    probabilities, test_metrics = {}, {}
    pred_frame = pd.DataFrame({"csv_row_1based_including_header": test.index + 2, "label": test.label, "y_true": yt})
    for name, model in fitted.items():
        probs = model.predict_proba(Xt)[:, 1]
        assert len(probs) == len(yt) and np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
        probabilities[name] = probs
        test_metrics[name] = compute_metrics(yt, probs)
        stem = "lr" if name == "Logistic regression" else "rf"
        pred_frame[f"{stem}_probability_yes"] = probs
        pred_frame[f"{stem}_prediction"] = (probs >= THRESHOLD).astype(int)
        assert sum(test_metrics[name][k] for k in ["TN", "FP", "FN", "TP"]) == len(yt)
        print(f"  {name}: {test_metrics[name]}", flush=True)
    # Majority-class baseline: constant training prevalence for ranking, all-no decisions.
    test_metrics["Majority baseline"] = compute_metrics(yt, np.full(len(yt), float(y.mean())), np.zeros(len(yt), dtype=int))
    pred_frame.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(test_metrics).T.rename_axis("model").to_csv(out / "test_metrics.csv")
    uncertainty = paired_bootstrap(yt, probabilities["Logistic regression"], probabilities["Random forest"], args.bootstrap_iterations)
    make_figures(yt, probabilities, test_metrics, out)
    results = {
        "seed": SEED, "folds": FOLDS, "threshold": THRESHOLD,
        "primary_metric": "Average precision (AP), not trapezoidal PR-AUC",
        "positive_label": "yes", "selected_by_training_CV": selected_model,
        "labelled_train_rows": len(train), "labelled_test_rows": len(test),
        "train_positive_prevalence": float(y.mean()), "test_positive_prevalence": float(yt.mean()),
        "models": details, "ablation": ablations, "test_metrics": test_metrics,
        "bootstrap": uncertainty,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_path, test_path]},
        "elapsed_seconds": time.perf_counter() - start,
    }
    save_json(out / "results.json", results)
    print(f"Finished in {results['elapsed_seconds']:.1f}s; results saved in {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
