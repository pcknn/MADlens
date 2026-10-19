"""Compare regression trees and random forests on frozen CNN image features."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

from main_anatomy_group import (
    ANATOMY_GROUPS,
    BODY_PARTS,
    DECISION_THRESHOLD,
    IMG_SIZE,
    SEED,
    TRAIN_IMAGE_CSV,
    VALID_IMAGE_CSV,
    calculate_metrics,
    compute_class_weights,
    make_dataframe,
    make_dataset,
    save_confusion_matrix,
)


RESULTS_ROOT = Path("anatomy_tree_results")


def build_feature_extractor():
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False,
        weights="imagenet", pooling="avg",
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    return tf.keras.Model(inputs, backbone(x, training=False))


def extract_features(extractor, df):
    # No shuffle or augmentation: feature rows must match dataframe rows.
    features = extractor.predict(make_dataset(df, training=False), verbose=1)
    return np.asarray(features, dtype=np.float32)


def build_estimators():
    return {
        "regression_tree": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=10, random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            max_features="sqrt", n_jobs=-1, random_state=SEED,
        ),
    }


def validate_dataframe(df, split, groups=ANATOMY_GROUPS):
    if df.empty:
        raise ValueError(f"{split} split is empty.")
    unexpected = sorted(set(df["body_part"]) - set(BODY_PARTS))
    if unexpected:
        raise ValueError(f"Unexpected body parts in {split}: {unexpected}")
    missing = df.loc[~df["path"].map(lambda path: Path(path).is_file()), "path"]
    if not missing.empty:
        raise FileNotFoundError(
            f"{len(missing)} missing {split} images; first: {missing.iloc[0]}"
        )
    for name, parts in groups.items():
        labels = df.loc[df["body_part"].isin(parts), "label"]
        if set(labels) != {0, 1}:
            raise ValueError(f"{split}/{name} must contain normal and abnormal images.")


def train_group(group_name, body_parts, train_df, valid_df, train_x, valid_x,
                results_root=RESULTS_ROOT, methods=None):
    train_mask = train_df["body_part"].isin(body_parts).to_numpy()
    valid_mask = valid_df["body_part"].isin(body_parts).to_numpy()
    train_part = train_df.loc[train_mask]
    valid_part = valid_df.loc[valid_mask]
    train_y = train_part["label"].to_numpy(dtype=int)
    valid_y = valid_part["label"].to_numpy(dtype=int)
    weights = compute_class_weights(train_part)
    sample_weights = np.asarray([weights[label] for label in train_y])
    summaries = []

    for method, estimator in build_estimators().items():
        if methods is not None and method not in methods:
            continue
        print(f"Training {group_name}: {method}", flush=True)
        # Balance classes using training labels only for both estimators.
        estimator.fit(train_x[train_mask], train_y, sample_weight=sample_weights)
        if method == "regression_tree":
            scores = estimator.predict(valid_x[valid_mask])
        else:
            positive_column = list(estimator.classes_).index(1)
            scores = estimator.predict_proba(valid_x[valid_mask])[:, positive_column]
        metrics, cm = calculate_metrics(valid_y, scores)
        run_dir = results_root / group_name / method
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "anatomy_group": group_name,
            "body_parts": ",".join(body_parts),
            "method": method,
            "train_images": len(train_part),
            "valid_images": len(valid_part),
            **metrics,
        }
        pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2)
        predictions = valid_part.reset_index(drop=True).copy()
        predictions["abnormal_score"] = scores
        predictions["prediction"] = (scores >= DECISION_THRESHOLD).astype(int)
        predictions["correct"] = predictions["prediction"] == predictions["label"]
        predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
        save_confusion_matrix(cm, f"{group_name}: {method}", run_dir)
        joblib.dump({
            "estimator": estimator,
            "feature_extractor": "../../feature_extractor.keras",
            "threshold": DECISION_THRESHOLD,
            "body_parts": body_parts,
            "image_size": IMG_SIZE,
            "preprocessing": "RGB float32 pixels in [0, 255]; preprocessing is inside feature extractor",
        }, run_dir / "model.joblib")
        summaries.append(summary)
    return summaries


def main(workflow="grouped", methods=None):
    groups = {
        "grouped": ANATOMY_GROUPS,
        "specific": {part: (part,) for part in BODY_PARTS},
        "pooled": {"ALL": tuple(BODY_PARTS)},
    }[workflow]
    results_root = {
        "grouped": RESULTS_ROOT,
        "specific": Path("anatomy_specific_tree_results"),
        "pooled": Path("pooled_tree_results"),
    }[workflow]
    tf.keras.utils.set_random_seed(SEED)
    train_df = make_dataframe(TRAIN_IMAGE_CSV)
    valid_df = make_dataframe(VALID_IMAGE_CSV)
    validate_dataframe(train_df, "train", groups)
    validate_dataframe(valid_df, "validation", groups)
    included_parts = set().union(*groups.values())
    train_df = train_df.loc[train_df["body_part"].isin(included_parts)].reset_index(drop=True)
    valid_df = valid_df.loc[valid_df["body_part"].isin(included_parts)].reset_index(drop=True)

    results_root.mkdir(parents=True, exist_ok=True)
    extractor = build_feature_extractor()
    extractor.save(results_root / "feature_extractor.keras")
    # Extract each image once, including wrist images shared between groups.
    train_x = extract_features(extractor, train_df)
    valid_x = extract_features(extractor, valid_df)
    summaries = []
    for group_name, body_parts in groups.items():
        summaries.extend(train_group(
            group_name, body_parts, train_df, valid_df, train_x, valid_x,
            results_root=results_root, methods=methods,
        ))
        pd.DataFrame(summaries).to_csv(results_root / "all_tree_results.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
