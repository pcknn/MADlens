"""Image-level validation metrics from fixed predictions, clustered by patient.

No models are trained here. CIs describe validation-sample uncertainty only.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 42
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUROC",
    "average_precision": "Average Precision (AP)",
    "precision": "Precision",
    "recall_sensitivity": "Sensitivity/Recall",
    "specificity": "Specificity",
    "f1_score": "F1",
    "cohen_kappa": "Cohen's kappa",
}
PREDICTION_COLUMNS = [
    "path", "patient_id", "body_part", "label", "abnormal_probability",
    "prediction", "threshold",
]


def path_metadata(path):
    """Require the existing MURA .../XR_*/patientNNN/studyN_label/image path."""
    parts = Path(path).parts
    patients = [i for i, part in enumerate(parts) if re.fullmatch(r"patient\d+", part)]
    if len(patients) != 1:
        raise ValueError(f"Cannot extract a unique MURA patient ID from {path!r}")
    i = patients[0]
    if (i < 1 or i + 2 != len(parts) - 1
            or not parts[i - 1].startswith("XR_")
            or not re.fullmatch(r"study\d+_(positive|negative)", parts[i + 1])):
        raise ValueError(f"Unexpected MURA image path structure: {path!r}")
    return parts[i], parts[i - 1][3:], int(parts[i + 1].endswith("_positive"))


def collect_predictions(model, dataset):
    """Collect predictions in the existing (unshuffled validation) dataset order."""
    y_true, y_probs = [], []
    for images, labels in dataset:
        y_probs.extend(model.predict(images, verbose=0).flatten())
        y_true.extend(labels.numpy().astype(int))
    return np.asarray(y_true), np.asarray(y_probs)


def prediction_frame(validation_df, y_true, y_probs, threshold):
    frame = validation_df.reset_index(drop=True).copy()
    y_true, y_probs = np.asarray(y_true), np.asarray(y_probs)
    if y_true.shape != (len(frame),) or y_probs.shape != (len(frame),):
        raise ValueError("Validation rows, labels, and scores must have matching lengths.")
    if not np.array_equal(frame["label"].to_numpy(), y_true):
        raise ValueError("Dataset labels are not aligned with validation dataframe rows.")
    frame["patient_id"] = [path_metadata(path)[0] for path in frame["path"]]
    frame["abnormal_probability"] = y_probs
    frame["threshold"] = threshold
    frame["prediction"] = (y_probs >= threshold).astype(int)
    frame["correct"] = frame["label"] == frame["prediction"]
    validate_predictions(frame)
    return frame


def validate_predictions(frame):
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction columns missing: {sorted(missing)}")
    if frame.empty or frame[PREDICTION_COLUMNS].isna().any().any():
        raise ValueError("Prediction data must be nonempty and contain no missing fields.")
    duplicates = frame.loc[frame["path"].duplicated(False), "path"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate image paths: {duplicates[:5]}")
    for row in frame.itertuples(index=False):
        patient, anatomy, label = path_metadata(row.path)
        if (row.patient_id, row.body_part, row.label) != (patient, anatomy, label):
            raise ValueError(f"Patient, body part, or true label disagrees with path: {row.path}")
    probs = frame["abnormal_probability"].to_numpy(dtype=float)
    thresholds = frame["threshold"].to_numpy(dtype=float)
    if (not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any()
            or not np.isfinite(thresholds).all()
            or ((thresholds < 0) | (thresholds > 1)).any()):
        raise ValueError("Scores and thresholds must be finite probabilities in [0, 1].")
    if not np.array_equal(frame["prediction"].to_numpy(), (probs >= thresholds).astype(int)):
        raise ValueError("Predicted labels disagree with scores and recorded thresholds.")


def export_predictions(frame, output_path):
    validate_predictions(frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep existing extra columns (e.g. correct) alongside the common schema.
    columns = PREDICTION_COLUMNS + [c for c in frame if c not in PREDICTION_COLUMNS]
    frame[columns].to_csv(output_path, index=False)


def point_metrics(y_true, y_probs, y_pred):
    """Preserve legacy metric keys and zero_division=0 conventions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    if np.unique(y_true).size != 2:
        raise ValueError("Both true classes are required for the requested evaluation metrics.")
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_probs)),
        "average_precision": float(average_precision_score(y_true, y_probs)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=[0, 1])),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }
    return metrics, cm


def patient_bootstrap_indices(patient_ids, rng):
    """Sample N patients uniformly with replacement and repeat each full cluster."""
    _, inverse = np.unique(patient_ids, return_inverse=True)
    clusters = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
    while True:
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        yield np.concatenate([clusters[i] for i in sampled])


def _bootstrap(frames, n_resamples, seed):
    """Use identical patient draws for all aligned models; reject invalid draws."""
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    arrays = [
        (f["label"].to_numpy(), f["abnormal_probability"].to_numpy(), f["prediction"].to_numpy())
        for f in frames
    ]
    points = [point_metrics(*a)[0] for a in arrays]
    samples = np.empty((n_resamples, len(frames), len(METRIC_LABELS)))
    valid = attempts = 0
    draws = patient_bootstrap_indices(frames[0]["patient_id"].to_numpy(), np.random.default_rng(seed))
    while valid < n_resamples:
        indices = next(draws)
        attempts += 1
        # AUROC and the requested class-specific evaluation need both true classes.
        # Reject the whole draw, including in paired comparisons, and draw again.
        if np.unique(arrays[0][0][indices]).size != 2:
            continue
        values = [point_metrics(*(a[indices] for a in model_arrays))[0] for model_arrays in arrays]
        values = np.array([[m[key] for key in METRIC_LABELS] for m in values])
        if not np.isfinite(values).all():
            continue
        samples[valid] = values
        valid += 1
    metadata = {
        "bootstrap_unit": "patient", "evaluation_unit": "image",
        "confidence_level": 0.95, "ci_method": "percentile (2.5, 97.5)",
        "bootstrap_valid_resamples": valid, "bootstrap_attempts": attempts,
        "bootstrap_skipped_resamples": attempts - valid, "bootstrap_seed": seed,
        "patients": int(frames[0]["patient_id"].nunique()),
        "uncertainty": "Validation-sample uncertainty; fixed predictions, no model retraining",
        "invalid_resample_policy": "Skip draws missing a true class or yielding nonfinite metrics; redraw until target is reached",
    }
    return points, samples, metadata


def evaluate_predictions(frame, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    validate_predictions(frame)
    points, samples, metadata = _bootstrap([frame], n_resamples, seed)
    report = points[0]
    for i, key in enumerate(METRIC_LABELS):
        lower, upper = np.percentile(samples[:, 0, i], [2.5, 97.5])
        report[f"{key}_ci_lower"] = float(lower)
        report[f"{key}_ci_upper"] = float(upper)
    thresholds = frame["threshold"].unique()
    report["threshold"] = float(thresholds[0]) if len(thresholds) == 1 else "per-row recorded thresholds"
    report.update(metadata)
    return report


def print_metrics(report):
    for key, label in METRIC_LABELS.items():
        print(f"  {label}: {report[key]:.4f} (95% CI {report[key + '_ci_lower']:.4f}, {report[key + '_ci_upper']:.4f})")
    print(f"  Patient bootstrap: {report['bootstrap_valid_resamples']} valid draws; "
          f"{report['bootstrap_skipped_resamples']} invalid draws skipped. Validation-sample uncertainty only.")


def evaluate_by_anatomy(frame, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Keep the existing body-part table columns and add stronger metrics and CIs."""
    rows = []
    legacy_names = {
        "accuracy": "Accuracy", "auc": "AUC", "precision": "Precision",
        "recall_sensitivity": "Recall", "specificity": "Specificity", "f1_score": "F1 Score",
        "average_precision": "Average Precision (AP)", "cohen_kappa": "Cohen's kappa",
    }
    for anatomy, group in frame.groupby("body_part", sort=True):
        report = evaluate_predictions(group, n_resamples, seed)
        row = {
            "Body Part": anatomy, "Total Images": len(group),
            "Normal Images": int((group["label"] == 0).sum()),
            "Abnormal Images": int((group["label"] == 1).sum()),
        }
        for key, label in legacy_names.items():
            row[label] = report[key]
            row[f"{label} CI Lower"] = report[f"{key}_ci_lower"]
            row[f"{label} CI Upper"] = report[f"{key}_ci_upper"]
        row.update({k: v for k, v in report.items() if k.startswith("bootstrap_")
                    or k in {"confidence_level", "ci_method", "evaluation_unit", "uncertainty", "patients"}})
        rows.append(row)
    result = pd.DataFrame(rows)
    print("\nImage-level validation metrics by body part (AUC = AUROC; 95% patient bootstrap CIs):")
    print(result.to_string(index=False))
    return result


def save_validation_report(frame, output_dir):
    output_dir = Path(output_dir)
    export_predictions(frame, output_dir / "validation_predictions.csv")
    report = evaluate_predictions(frame)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nImage-level validation metrics:")
    print_metrics(report)
    return report


def report_thresholds(model, dataset, validation_df, thresholds):
    """Reuse one prediction pass; keep every sweep threshold fixed in its bootstrap."""
    y_true, y_probs = collect_predictions(model, dataset)
    print("\nThreshold Results:")
    for threshold in thresholds:
        frame = prediction_frame(validation_df, y_true, y_probs, threshold)
        print(f"\nThreshold: {threshold}")
        print_metrics(evaluate_predictions(frame))


def align_predictions(pooled, specialist, expected):
    """Require exact coverage of an explicit validation manifest; never inner-join."""
    validate_predictions(pooled)
    validate_predictions(specialist)
    if expected.empty or expected["path"].duplicated().any():
        raise ValueError("Expected validation paths must be nonempty and unique.")
    expected_paths = set(expected["path"])
    for name, frame in [("pooled", pooled), ("specialist", specialist)]:
        paths = set(frame["path"])
        if paths != expected_paths:
            raise ValueError(f"{name} validation coverage mismatch: "
                             f"missing={sorted(expected_paths - paths)[:5]}, "
                             f"unexpected={sorted(paths - expected_paths)[:5]}")
    # Reindex only after proving equal sets; nothing is dropped.
    ordered = expected.sort_values("path").reset_index(drop=True)
    frames = [f.set_index("path").loc[ordered["path"]].reset_index() for f in [pooled, specialist]]
    for name, frame in zip(["pooled", "specialist"], frames):
        for column in ["patient_id", "body_part", "label"]:
            if not np.array_equal(frame[column].to_numpy(), ordered[column].to_numpy()):
                raise ValueError(f"{name} {column} does not match validation manifest.")
    return frames


def paired_comparison(pooled, specialist, expected, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    pooled, specialist = align_predictions(pooled, specialist, expected)
    # Paired model comparisons use the requested fixed 0.50 threshold.
    for name, frame in [("pooled", pooled), ("specialist", specialist)]:
        if not (frame["threshold"] == 0.50).all():
            raise ValueError(f"Paired comparison requires the existing fixed 0.50 threshold: {name}")
    points, samples, metadata = _bootstrap([pooled, specialist], n_resamples, seed)
    rows = []
    for i, (key, label) in enumerate(METRIC_LABELS.items()):
        pooled_ci = np.percentile(samples[:, 0, i], [2.5, 97.5])
        specialist_ci = np.percentile(samples[:, 1, i], [2.5, 97.5])
        delta_ci = np.percentile(samples[:, 1, i] - samples[:, 0, i], [2.5, 97.5])
        rows.append({
            "metric": label, "pooled": points[0][key], "specialist": points[1][key],
            "pooled_ci_lower": float(pooled_ci[0]), "pooled_ci_upper": float(pooled_ci[1]),
            "specialist_ci_lower": float(specialist_ci[0]), "specialist_ci_upper": float(specialist_ci[1]),
            "specialist_minus_pooled": points[1][key] - points[0][key],
            "delta_ci_lower": float(delta_ci[0]), "delta_ci_upper": float(delta_ci[1]),
            "threshold": 0.50, "images": len(pooled), **metadata,
        })
    return pd.DataFrame(rows)


def validation_manifest(csv_path):
    """Use exactly the same path fix as the existing dataset-loading helpers."""
    frame = pd.read_csv(csv_path, header=None, names=["path"])
    frame["path"] = frame["path"].str.replace("MURA-v1.1/", "MURA-v1.1_files/", regex=False)
    metadata = [path_metadata(path) for path in frame["path"]]
    frame[["patient_id", "body_part", "label"]] = pd.DataFrame(metadata, index=frame.index)
    return frame
