"""Synthetic evaluation cases; no neural-network training or dataset mutation."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from evaluation import (
    BOOTSTRAP_RESAMPLES, METRIC_LABELS, PREDICTION_COLUMNS,
    align_predictions, evaluate_predictions, export_predictions, paired_comparison,
    path_metadata, patient_bootstrap_indices, prediction_frame, validate_predictions,
    validation_manifest,
)


def synthetic_frame():
    # Unequal cluster sizes distinguish patient bootstrap from image bootstrap.
    patients = [1, 1, 1, 2, 3, 3]
    labels = np.array([0, 0, 0, 1, 1, 1])
    paths = [f"MURA-v1.1_files/valid/XR_ELBOW/patient{p:05d}/"
             f"study1_{'positive' if label else 'negative'}/image{i + 1}.png"
             for i, (p, label) in enumerate(zip(patients, labels))]
    df = pd.DataFrame({"path": paths, "label": labels, "body_part": "ELBOW"})
    return prediction_frame(df, labels, np.array([0.2, 0.5, 0.1, 0.9, 0.4, 0.6]), 0.5)


class EvaluationTests(unittest.TestCase):
    def test_path_metadata_and_export_roundtrip(self):
        f = synthetic_frame()
        self.assertEqual(path_metadata(f.iloc[0]["path"]), ("patient00001", "ELBOW", 0))
        for bad in ["image.png", "XR_ELBOW/patient1/image.png",
                    "XR_ELBOW/patient1/patient2/study1_positive/image.png"]:
            with self.assertRaises(ValueError):
                path_metadata(bad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            export_predictions(f, path)
            loaded = pd.read_csv(path)
            validate_predictions(loaded)
            self.assertEqual(list(loaded.columns[:7]), PREDICTION_COLUMNS)
            pd.testing.assert_frame_equal(f, loaded[f.columns])
            manifest_path = Path(tmp) / "valid.csv"
            f["path"].str.replace("MURA-v1.1_files/", "MURA-v1.1/", regex=False).to_csv(
                manifest_path, index=False, header=False)
            expected = validation_manifest(manifest_path)
            self.assertEqual(expected["path"].tolist(), f["path"].tolist())

    def test_full_cluster_sampling_and_multiplicity(self):
        f = synthetic_frame()
        draws = patient_bootstrap_indices(f["patient_id"].to_numpy(), np.random.default_rng(7))
        sizes = f.groupby("patient_id").size().to_dict()
        saw_duplicate = False
        for _ in range(50):
            indices = next(draws)
            counts = np.bincount(indices, minlength=len(f))
            sampled_patients = 0
            for patient, group in f.groupby("patient_id"):
                multiplicities = counts[group.index]
                self.assertTrue((multiplicities == multiplicities[0]).all())
                sampled_patients += int(multiplicities[0])
                saw_duplicate |= multiplicities[0] > 1
                self.assertEqual(int(multiplicities.sum()), int(multiplicities[0]) * sizes[patient])
            self.assertEqual(sampled_patients, len(sizes))
        self.assertTrue(saw_duplicate)

    def test_5000_valid_draws_reference_metrics_and_percentiles(self):
        f = synthetic_frame()
        report = evaluate_predictions(f)  # Exercise the production 5,000-valid default.
        self.assertEqual(report["bootstrap_valid_resamples"], 5_000)
        self.assertEqual(BOOTSTRAP_RESAMPLES, 5_000)
        self.assertGreater(report["bootstrap_skipped_resamples"], 0)
        self.assertEqual(report["bootstrap_attempts"], 5_000 + report["bootstrap_skipped_resamples"])
        y, probs, pred = (f[c].to_numpy() for c in ["label", "abnormal_probability", "prediction"])
        reference = {
            "accuracy": accuracy_score(y, pred), "auc": roc_auc_score(y, probs),
            "average_precision": average_precision_score(y, probs),
            "precision": precision_score(y, pred, zero_division=0),
            "recall_sensitivity": recall_score(y, pred, zero_division=0),
            "specificity": 2 / 3, "f1_score": f1_score(y, pred, zero_division=0),
            "cohen_kappa": cohen_kappa_score(y, pred),
        }
        for key, value in reference.items():
            self.assertAlmostEqual(report[key], value)
        # Independent oracle for the draw/rejection stream and percentile CIs.
        rng = np.random.default_rng(42)
        clusters = [np.array([0, 1, 2]), np.array([3]), np.array([4, 5])]
        reference_samples = {key: [] for key in METRIC_LABELS}
        attempts = 0
        while len(reference_samples["accuracy"]) < 5_000:
            indices = np.concatenate([clusters[i] for i in rng.choice(3, 3, replace=True)])
            attempts += 1
            truth, score, predicted = y[indices], probs[indices], pred[indices]
            if np.unique(truth).size != 2:
                continue
            tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
            values = {
                "accuracy": accuracy_score(truth, predicted), "auc": roc_auc_score(truth, score),
                "average_precision": average_precision_score(truth, score),
                "precision": precision_score(truth, predicted, zero_division=0),
                "recall_sensitivity": recall_score(truth, predicted, zero_division=0),
                "specificity": tn / (tn + fp), "f1_score": f1_score(truth, predicted, zero_division=0),
                "cohen_kappa": cohen_kappa_score(truth, predicted),
            }
            for key in reference_samples:
                reference_samples[key].append(values[key])
        self.assertEqual(attempts, report["bootstrap_attempts"])
        for key, values in reference_samples.items():
            lower, upper = np.percentile(values, [2.5, 97.5])
            self.assertAlmostEqual(report[key + "_ci_lower"], lower)
            self.assertAlmostEqual(report[key + "_ci_upper"], upper)

    def test_alignment_is_one_to_one_and_does_not_drop_cases(self):
        f = synthetic_frame()
        expected = f[["path", "patient_id", "body_part", "label"]]
        left, right = align_predictions(f, f.sample(frac=1, random_state=9), expected)
        pd.testing.assert_frame_equal(left, right)
        cases = [pd.concat([f, f.iloc[[0]]]), f.iloc[1:], f.iloc[:-1]]
        for column, value in [("patient_id", "patient99999"), ("body_part", "WRIST"), ("label", 1),
                              ("prediction", 1), ("abnormal_probability", float("nan")), ("threshold", -1)]:
            bad = f.copy()
            bad.loc[0, column] = value
            cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                align_predictions(f, bad, expected)
        # Missing from both exports is still caught by the external manifest.
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            align_predictions(f.iloc[1:], f.iloc[1:], expected)

    def test_paired_identical_models_have_exact_zero_delta_intervals(self):
        f = synthetic_frame()
        result = paired_comparison(f, f.sample(frac=1, random_state=8), f, n_resamples=100)
        np.testing.assert_array_equal(result["specialist_minus_pooled"], 0)
        np.testing.assert_array_equal(result["delta_ci_lower"], 0)
        np.testing.assert_array_equal(result["delta_ci_upper"], 0)
        np.testing.assert_array_equal(result["pooled_ci_lower"], result["specialist_ci_lower"])

    def test_paired_delta_matches_shared_draw_oracle(self):
        pooled = synthetic_frame()
        specialist = prediction_frame(pooled, pooled["label"].to_numpy(),
                                      np.array([0.05, 0.2, 0.1, 0.8, 0.9, 0.95]), 0.5)
        result = paired_comparison(pooled, specialist, pooled, n_resamples=100)
        rng = np.random.default_rng(42)
        clusters = [np.array([0, 1, 2]), np.array([3]), np.array([4, 5])]
        deltas = []
        while len(deltas) < 100:
            idx = np.concatenate([clusters[i] for i in rng.choice(3, 3, replace=True)])
            y = pooled["label"].to_numpy()[idx]
            if np.unique(y).size < 2:
                continue
            deltas.append(accuracy_score(y, specialist["prediction"].to_numpy()[idx]) -
                          accuracy_score(y, pooled["prediction"].to_numpy()[idx]))
        row = result[result["metric"] == "Accuracy"].iloc[0]
        np.testing.assert_allclose([row["delta_ci_lower"], row["delta_ci_upper"]], np.percentile(deltas, [2.5, 97.5]))

    def test_undefined_evaluation_and_threshold_mismatch_fail_explicitly(self):
        f = synthetic_frame()
        with self.assertRaisesRegex(ValueError, "Both true classes"):
            evaluate_predictions(f[f["label"] == 0], n_resamples=10)
        other = prediction_frame(f, f["label"].to_numpy(), f["abnormal_probability"].to_numpy(), 0.45)
        with self.assertRaisesRegex(ValueError, "fixed 0.50"):
            paired_comparison(f, other, f, n_resamples=10)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            prediction_frame(f, 1 - f["label"].to_numpy(), f["abnormal_probability"].to_numpy(), 0.5)

    def test_notebooks_and_scripts_share_evaluation_helpers(self):
        root = Path(__file__).resolve().parents[1]
        for name in ["main.py", "main_anatomy_specific.py", "main_anatomy_group.py"]:
            ast.parse((root / name).read_text())
        for name in ["main.ipynb", "main_anatomy_specific.ipynb"]:
            nb = json.loads((root / name).read_text())
            source = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
            tree = ast.parse(source)
            imports = {a.name for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "evaluation" for a in n.names}
            if name == "main.ipynb":
                self.assertTrue({"prediction_frame", "evaluate_by_anatomy", "save_validation_report", "report_thresholds"} <= imports)
            else:
                script = ast.parse((root / "main_anatomy_specific.py").read_text())
                self.assertEqual(ast.dump(tree), ast.dump(script))


if __name__ == "__main__":
    unittest.main()
