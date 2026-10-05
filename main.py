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

DATA_ROOT = Path("MURA-v1.1_files")

TRAIN_IMAGE_CSV = DATA_ROOT / "train_image_paths.csv"
VALID_IMAGE_CSV = DATA_ROOT / "valid_image_paths.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_label(path):
    if "positive" in path:
        return 1  # abnormal
    else:
        return 0  # normal


def fix_path(path):
    return path.replace("MURA-v1.1/", "MURA-v1.1_files/")


def make_dataframe(csv_path):
    df = pd.read_csv(csv_path, header=None, names=["path"])

    df["label"] = df["path"].apply(get_label)
    df["path"] = df["path"].apply(fix_path)

    return df


train_df = make_dataframe(TRAIN_IMAGE_CSV)
valid_df = make_dataframe(VALID_IMAGE_CSV)


def get_body_part(path):
    for part in Path(path).parts:
        if part.startswith("XR_"):
            return part.replace("XR_", "")
    return "UNKNOWN"


train_df["body_part"] = train_df["path"].apply(get_body_part)
valid_df["body_part"] = valid_df["path"].apply(get_body_part)

print("\nValidation images per body part:")
print(valid_df["body_part"].value_counts())


train_missing = train_df["path"].apply(lambda p: not Path(p).exists()).sum()
valid_missing = valid_df["path"].apply(lambda p: not Path(p).exists()).sum()

print("\nMissing train image files:", train_missing)
print("Missing valid image files:", valid_missing)

if train_missing > 0 or valid_missing > 0:
    raise FileNotFoundError(
        "Some image paths are wrong. Check DATA_ROOT and fix_path()."
    )


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    return image, label


train_paths = train_df["path"].values
train_labels = train_df["label"].values

valid_paths = valid_df["path"].values
valid_labels = valid_df["label"].values

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))

train_ds = (
    train_ds.shuffle(
        buffer_size=len(train_paths), seed=42, reshuffle_each_iteration=True
    )
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
valid_ds = (
    valid_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# look at one batch before training

for images, labels in train_ds.take(1):
    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First labels:", labels[:10].numpy())

    plt.figure(figsize=(8, 8))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy() / 255.0)

        label = int(labels[i].numpy())
        title = "Abnormal" if label == 1 else "Normal"

        plt.title(title)
        plt.axis("off")

    plt.show()


for images, labels in valid_ds.take(1):
    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First labels:", labels[:10].numpy())

    plt.figure(figsize=(8, 8))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy() / 255.0)

        label = int(labels[i].numpy())
        title = "Abnormal" if label == 1 else "Normal"

        plt.title(title)
        plt.axis("off")

    plt.show()


# weights from test.py
counts = train_df["label"].value_counts()

normal_count = counts[0]
abnormal_count = counts[1]
total_count = normal_count + abnormal_count

class_weight = {
    0: float(total_count / (2 * normal_count)),
    1: float(total_count / (2 * abnormal_count)),
}

print("\nClass weights:")
print(class_weight)


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

# model.summary()


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=3,
    min_delta=0.001,
    restore_best_weights=True,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-7
)


history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20,
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr],
    shuffle=False,
)


final_results = model.evaluate(valid_ds, verbose=1, return_dict=True)

print("\nFinal restored model evaluation:")
for name, value in final_results.items():
    print(f"{name}: {value:.4f}")


def plot_training_history(history):
    # loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("Training Loss vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # AUC graph
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["auc"], label="Training AUC")
    plt.plot(history.history["val_auc"], label="Validation AUC")
    plt.title("Training AUC vs Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.show()

    # accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_training_history(history)


# threshold tuning
def test_thresholds(model, dataset, thresholds=[0.30, 0.35, 0.40, 0.45, 0.50]):
    y_true = []
    y_probs = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)

        y_true.extend(labels.numpy().astype(int))
        y_probs.extend(predictions.flatten())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    print("\nThreshold Results:")

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\nThreshold: {threshold}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")


test_thresholds(model, valid_ds)


def get_validation_predictions(model, dataset, valid_df, threshold=0.50):
    y_true = []
    y_probs = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0).flatten()

        y_true.extend(labels.numpy().astype(int))
        y_probs.extend(predictions)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    pred_df = valid_df.reset_index(drop=True).copy()
    pred_df["y_true"] = y_true
    pred_df["abnormal_probability"] = y_probs
    pred_df["y_pred"] = y_pred
    pred_df["correct"] = pred_df["y_true"] == pred_df["y_pred"]

    return pred_df


def evaluate_body_part_accuracies(pred_df):
    results = []

    for body_part, group in pred_df.groupby("body_part"):
        y_true = group["y_true"].values
        y_pred = group["y_pred"].values
        y_probs = group["abnormal_probability"].values

        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results.append(
            {
                "Body Part": body_part,
                "Total Images": len(group),
                "Normal Images": int((group["y_true"] == 0).sum()),
                "Abnormal Images": int((group["y_true"] == 1).sum()),
                "Accuracy": accuracy,
                "AUC": auc,
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity,
                "F1 Score": f1,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nValidation accuracy by body part:")
    print(results_df.to_string(index=False))

    return results_df


# confusion matrix input desired threshold
def make_confusion_matrix(model, dataset, threshold):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)

        y_true.extend(labels.numpy().astype(int))
        y_pred.extend((predictions >= threshold).astype(int).flatten())

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Abnormal"]
    )

    disp.plot(cmap=plt.cm.Blues)

    plt.title(f"Confusion Matrix, threshold={threshold}")
    plt.savefig(
        f"confusion_matrix_threshold_{threshold}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))


make_confusion_matrix(model, valid_ds, threshold=0.50)


threshold = 0.50

valid_predictions_df = get_validation_predictions(
    model, valid_ds, valid_df, threshold=threshold
)

body_part_results = evaluate_body_part_accuracies(valid_predictions_df)


def gradcam_heatmap(image, model, base_model, last_conv_layer_name="out_relu"):
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)

    image = tf.cast(image, tf.float32)

    grad_based_model = keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )

    base_model_index = None

    for index, layer in enumerate(model.layers):
        if layer is base_model:
            base_model_index = index
            break
    if base_model_index is None:
        raise ValueError("Base model not found in the model's layers.")

    classifier_layers = model.layers[base_model_index + 1 :]

    with tf.GradientTape() as tape:
        preprocessed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        conv_outputs, base_output = grad_based_model(preprocessed_image, training=False)
        tape.watch(conv_outputs)
        x = base_output

        for layer in classifier_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=False)
            else:
                x = layer(x)
        abnormal_score = x[:, 0]

    grads = tape.gradient(abnormal_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + keras.backend.epsilon())

    return heatmap.numpy(), float(abnormal_score.numpy()[0])


def overlay_heatmap(image, heatmap, alpha=0.25, power=0.8):
    if tf.is_tensor(image):
        image = image.numpy()
    image = image.astype("float32") / 255.0
    heatmap = np.power(heatmap, power)
    heatmap = np.uint8(255 * heatmap)
    cmap = plt.cm.turbo(np.arange(256))[:, :3]
    color_heatmap = cmap[heatmap]
    color_heatmap = tf.image.resize(color_heatmap, IMG_SIZE).numpy()
    overlay = (1 - alpha) * image + alpha * color_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)

    return overlay


def get_correct_examples(model, dataset, threshold=0.50, num_normal=4, num_abnormal=5):
    normal_examples = []
    abnormal_examples = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0).flatten()
        true_labels = labels.numpy().astype(int)
        pred_labels = (probs >= threshold).astype(int)

        for i in range(len(true_labels)):
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            prob = probs[i]

            if true_label != pred_label:
                continue

            if pred_label == 1:
                confidence = prob
            else:
                confidence = 1 - prob

            example = {
                "image": images[i].numpy(),
                "true_label": true_label,
                "pred_label": pred_label,
                "prob": prob,
                "confidence": confidence,
            }

            if true_label == 0:
                normal_examples.append(example)
            else:
                abnormal_examples.append(example)

    normal_examples = sorted(
        normal_examples, key=lambda x: x["confidence"], reverse=True
    )

    abnormal_examples = sorted(
        abnormal_examples, key=lambda x: x["confidence"], reverse=True
    )

    selected_examples = normal_examples[:num_normal] + abnormal_examples[:num_abnormal]

    print(f"Selected {len(selected_examples)} correct examples.")
    print(f"Normal examples: {len(normal_examples[:num_normal])}")
    print(f"Abnormal examples: {len(abnormal_examples[:num_abnormal])}")

    return selected_examples


def show_selected_originals(examples):
    plt.figure(figsize=(12, 12))

    for i, example in enumerate(examples):
        image = example["image"]
        true_label = example["true_label"]
        pred_label = example["pred_label"]
        prob = example["prob"]

        true_name = "Abnormal" if true_label == 1 else "Normal"
        pred_name = "Abnormal" if pred_label == 1 else "Normal"

        plt.subplot(3, 3, i + 1)
        plt.imshow(image / 255.0)

        plt.title(
            f"True: {true_name}\nPred: {pred_name}\nScore: {prob:.2f}", fontsize=9
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_selected_gradcam(examples, model, base_model, alpha=0.25):
    plt.figure(figsize=(12, 12))

    for i, example in enumerate(examples):
        image = example["image"]
        true_label = example["true_label"]
        pred_label = example["pred_label"]
        prob = example["prob"]

        heatmap, abnormal_score = gradcam_heatmap(
            image, model, base_model, last_conv_layer_name="out_relu"
        )

        gradcam_image = overlay_heatmap(image, heatmap, alpha=alpha)

        true_name = "Abnormal" if true_label == 1 else "Normal"
        pred_name = "Abnormal" if pred_label == 1 else "Normal"

        plt.subplot(3, 3, i + 1)
        plt.imshow(gradcam_image)

        plt.title(
            f"True: {true_name}\nPred: {pred_name}\nScore: {prob:.2f}", fontsize=9
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()


selected_examples = get_correct_examples(
    model, valid_ds, threshold=0.50, num_normal=4, num_abnormal=5
)

show_selected_originals(selected_examples)

show_selected_gradcam(selected_examples, model, base_model, alpha=0.25)

# looks at test batch from intial testing (current commented out)
"""
def show_heatmap(model, base_model, dataset, threshold=0.50, max_images=9):
    for images, labels in dataset.take(1):
        predictions = model.predict(images, verbose=0).flatten()

        plt.figure(figsize=(12, 12))

        for i in range(max_images):
            heatmap, abnormal_score = gradcam_heatmap(
                images[i],
                model,
                base_model,
                last_conv_layer_name="out_relu"
            )

            gradcam_image = overlay_heatmap(images[i], heatmap, alpha=0.25)

            true_label = int(labels[i].numpy())
            pred_label = 1 if predictions[i] >= threshold else 0

            true_name = "Abnormal" if true_label == 1 else "Normal"
            pred_name = "Abnormal" if pred_label == 1 else "Normal"

            plt.subplot(3, 3, i + 1)
            plt.imshow(gradcam_image)

            plt.title(
                f"True: {true_name}\nPred: {pred_name}\nScore: {predictions[i]:.2f}",
                fontsize=9
            )

            plt.axis("off")

        plt.tight_layout()
        plt.show()

show_heatmap(model, base_model, valid_ds, threshold=0.50)
"""
