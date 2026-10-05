import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import json

# configuration

DATA_ROOT = Path("MURA-v1.1_files")

TRAIN_IMAGE_CSV = DATA_ROOT / "train_image_paths.csv"
VALID_IMAGE_CSV = DATA_ROOT / "valid_image_paths.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DECISION_THRESHOLD = 0.50
#####
SEED = 42

BODY_PARTS = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]

RESULTS_ROOT = Path("anatomy_specific_results")
RESULTS_ROOT.mkdir(exist_ok=True, parents=True)

# data loading


def get_label(path):
    if "positive" in path:
        return 1  # abnormal
    else:
        return 0  # normal


def fix_path(path):
    return path.replace("MURA-v1.1/", "MURA-v1.1_files/")


def get_body_part(path):
    for part in Path(path).parts:
        if part.startswith("XR_"):
            return part.replace("XR_", "")
    return "UNKNOWN"


def make_dataframe(csv_path):
    df = pd.read_csv(csv_path, header=None, names=["path"])
    df["label"] = df["path"].apply(get_label)
    df["path"] = df["path"].apply(fix_path)
    df["body_part"] = df["path"].apply(get_body_part)
    return df


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(df, training=False):
    paths = df["path"].values
    labels = df["label"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def compute_class_weights(df):
    counts = df["label"].value_counts()

    if 0 not in counts or 1 not in counts:
        raise ValueError(
            "Both classes must be present in the dataframe to compute class weights."
        )

    normal_count = counts[0]
    abnormal_count = counts[1]
    total_count = normal_count + abnormal_count

    class_weight = {
        0: float(total_count / (2 * normal_count)),
        1: float(total_count / (2 * abnormal_count)),
    }
    
    return class_weight


# model


def build_model():
    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(10 / 360),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def make_callbacks(run_dir):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            min_delta=0.001,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-7
        ),
        keras.callbacks.CSVLogger(run_dir / "epoch_log.csv"),
    ]


# evaluations


def collect_predictions(model, dataset):
    y_true = []
    y_probs = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0).flatten()
        y_true.extend(labels.numpy().astype(int))
        y_probs.extend(probs)

    return np.asarray(y_true), np.asarray(y_probs)


def calculate_metrics(y_true, y_probs, threshold=DECISION_THRESHOLD):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_probs),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics, cm


# plots


def save_history_plots(history, body_part, run_dir):
    plot_specs = [
        ("loss", "val_loss", "Loss"),
        ("auc", "val_auc", "AUC"),
        ("accuracy", "val_accuracy", "Accuracy"),
    ]

    for train_key, val_key, metric_name in plot_specs:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[train_key], label=f"Training {metric_name}")
        plt.plot(history.history[val_key], label=f"Validation {metric_name}")
        plt.title(f"{body_part}: Training vs Validation {metric_name}")
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(run_dir / f"{metric_name.lower()}_history.png", dpi=300)
        plt.close()


def save_confusion_matrix(cm, body_part, run_dir):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Abnormal"],
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{body_part}: Confusion Matrix (threshold={DECISION_THRESHOLD})")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=300)
    plt.close()


# one anatomy specific experiment


def train_anatomy_model(body_part, train_df, valid_df):
    print(f"Training model for body part: {body_part}")

    run_dir = RESULTS_ROOT / body_part
    run_dir.mkdir(exist_ok=True, parents=True)

    train_part = train_df[train_df["body_part"] == body_part].copy()
    valid_part = valid_df[valid_df["body_part"] == body_part].copy()

    if train_part.empty or valid_part.empty:
        raise ValueError(f"No data available for body part: {body_part}")

    train_normal = int((train_part["label"] == 0).sum())
    train_abnormal = int((train_part["label"] == 1).sum())
    valid_normal = int((valid_part["label"] == 0).sum())
    valid_abnormal = int((valid_part["label"] == 1).sum())

    print(f"Train images: {len(train_part)}")
    print(f"Normal: {train_normal}")
    print(f"Abnormal: {train_abnormal}")
    print(f"Validation images: {len(valid_part)}")
    print(f"Normal: {valid_normal}")
    print(f"Abnormal: {valid_abnormal}")

    class_weights = compute_class_weights(train_part)
    print(f"Class weights: {class_weights}")

    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    train_ds = make_dataset(train_part, training=True)
    valid_ds = make_dataset(valid_part, training=False)

    model = build_model()

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(run_dir),
        verbose=1,
    )

    y_true, y_probs = collect_predictions(model, valid_ds)
    metrics, cm = calculate_metrics(y_true, y_probs)

    best_epoch = int(np.argmax(history.history["val_auc"]) + 1)
    best_val_auc = float(np.max(history.history["val_auc"]))

    summary = {
        "body_part": body_part,
        "train_images": len(train_part),
        "train_normal": train_normal,
        "train_abnormal": train_abnormal,
        "valid_images": len(valid_part),
        "valid_normal": valid_normal,
        "valid_abnormal": valid_abnormal,
        "class_weights_normal": class_weights[0],
        "class_weights_abnormal": class_weights[1],
        "epochs_ran": len(history.history["loss"]),
        "best_epoch_by_val_auc": best_epoch,
        "best_val_auc_during_training": best_val_auc,
        **metrics,
    }

    pd.DataFrame(history.history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions_df = valid_part.reset_index(drop=True).copy()
    predictions_df["abnormal_probability"] = y_probs
    predictions_df["prediction"] = (y_probs >= DECISION_THRESHOLD).astype(int)
    predictions_df["correct"] = predictions_df["label"] == predictions_df["prediction"]
    predictions_df.to_csv(run_dir / "validation_predictions.csv", index=False)

    save_history_plots(history, body_part, run_dir)
    save_confusion_matrix(cm, body_part, run_dir)

    model.save(run_dir / f"{body_part.lower()}_specialist.keras")

    print("\nFinal specialist metrics:")
    for key in [
        "accuracy",
        "auc",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
    ]:
        print(f"  {key}: {summary[key]:.4f}")

    del model
    keras.backend.clear_session()

    return summary


# main


def main():
    tf.keras.utils.set_random_seed(SEED)

    train_df = make_dataframe(TRAIN_IMAGE_CSV)
    valid_df = make_dataframe(VALID_IMAGE_CSV)

    print("\nTraining images by body part:")
    print(train_df["body_part"].value_counts().sort_index())

    print("\nValidation images by body part:")
    print(valid_df["body_part"].value_counts().sort_index())

    unexpected_train = sorted(set(train_df["body_part"]) - set(BODY_PARTS))
    unexpected_valid = sorted(set(valid_df["body_part"]) - set(BODY_PARTS))
    if unexpected_train or unexpected_valid:
        raise ValueError(
            f"Unexpected body parts found. Train={unexpected_train}, Valid={unexpected_valid}"
        )

    train_missing = train_df["path"].apply(lambda p: not Path(p).exists()).sum()
    valid_missing = valid_df["path"].apply(lambda p: not Path(p).exists()).sum()

    print("\nMissing train image files:", train_missing)
    print("Missing valid image files:", valid_missing)

    if train_missing > 0 or valid_missing > 0:
        raise FileNotFoundError(
            "Some image paths are wrong. Check DATA_ROOT and fix_path()."
        )

    all_results = []

    for body_part in BODY_PARTS:
        result = train_anatomy_model(body_part, train_df, valid_df)
        all_results.append(result)

        pd.DataFrame(all_results).to_csv(
            RESULTS_ROOT / "all_specialist_results.csv",
            index=False,
        )

    results_df = pd.DataFrame(all_results)

    display_columns = [
        "body_part",
        "valid_images",
        "accuracy",
        "auc",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "best_epoch_by_val_auc",
    ]

    print("ALL 7 ANATOMY SPECIFIC MODELS COMPLETE")
    print(results_df[display_columns].to_string(index=False))
    print(f"\nSaved master results to: {RESULTS_ROOT / 'all_specialist_results.csv'}")


if __name__ == "__main__":
    main()
