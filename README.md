# MADlens X-Ray Image Classifier

## Musculoskeletal Abnormality Detection CNN Project

MADlens is a prototype X-ray image classifier built to detect **normal vs. abnormal** musculoskeletal X-ray cases using a convolutional neural network with TensorFlow.

The project compares three training strategies for abnormality detection in musculoskeletal radiographs: a pooled model, individual anatomy specialists, and models trained on groups of anatomies.

---

## Model / Architecture

* Used **MobileNetV2**, a lightweight pretrained CNN, as a frozen feature extractor.
* Trained the model to distinguish between **normal** and **abnormal** X-ray cases.
* Applied **class weighting** to reduce bias toward the majority class.
* Used slight **data augmentation** to help reduce overfitting.
* Implemented **Early Stopping** and **Learning Rate Reduction** while monitoring validation AUC.
* Reported threshold-based metrics alongside AUROC and Average Precision (AP).
* Included a tunable **decision threshold** to adjust the trade-off between recall and precision.

---

## Dataset

This project uses **MURA: MSK X-rays** from the Stanford University Center for Artificial Intelligence in Medicine and Imaging (AIMI).

MURA contains 40,561 radiographs from 14,863 studies, with a normal or abnormal label assigned to each study. MADlens applies that study label to each image and evaluates predictions at the image level. [MURA dataset paper](https://arxiv.org/abs/1712.06957)

* Elbow
* Finger
* Forearm
* Hand
* Humerus
* Shoulder
* Wrist

The local manifests used for these runs contain **36,808 training images from 11,184 patients** and **3,197 validation images from 783 patients**. Patient IDs are disjoint across the manifests. The validation set contains 1,667 normal-labeled images and 1,530 abnormal-labeled images.

The raw images and CSV files are **not included** in this repository.

To access the dataset, visit: [MURA: MSK X-rays](https://aimi.stanford.edu/datasets/mura-msk-xrays)

> Access requires agreeing to Stanford's Research Use Agreement.

---

## Results and Findings

The **pooled model** trains on all seven anatomies. The **individual specialists** train one model per anatomy, using the anatomy recorded in the MURA path to select the model for each validation image. The **grouped models** train on three predefined anatomy subsets.

### Pooled versus individual specialists

Both approaches were evaluated on the same **3,197 images from 783 patients**. Threshold-based metrics use **0.50**. AUROC and AP use the abnormal-class scores. The intervals below are for the paired differences, using 5,000 valid patient bootstrap draws per comparison. **pp** means percentage points.

| Metric | Pooled | Individual specialists | Specialist − pooled (paired 95% CI) |
| --- | --- | --- | --- |
| Accuracy | 74.13% | 75.91% | +1.78 pp (0.00 to +3.65) |
| AUROC | 0.8168 | 0.8319 | +0.0151 (+0.0012 to +0.0298) |
| Average Precision (AP) | 0.8261 | 0.8348 | +0.0087 (−0.0060 to +0.0238) |
| Precision | 77.35% | 76.46% | −0.89 pp (−3.17 to +1.40) |
| Sensitivity/Recall | 64.97% | 71.76% | +6.80 pp (+4.01 to +9.74) |
| Specificity | 82.54% | 79.72% | −2.82 pp (−5.19 to −0.57) |
| F1 | 0.7062 | 0.7404 | +0.0342 (+0.0131 to +0.0571) |
| Cohen's kappa | 0.4783 | 0.5162 | +0.0379 (+0.0025 to +0.0752) |

The individual specialists detected **104 more abnormal-labeled images** (1,098 versus 994) and flagged **47 more normal-labeled images** as abnormal (338 versus 291). The largest sensitivity shift was for hand, from **27.0% to 73.0%**, while specificity fell from **95.2% to 64.9%**.

| Anatomy | Images | Δ AUROC (paired 95% CI) | Δ sensitivity | Δ specificity |
| --- | --- | --- | --- | --- |
| Elbow | 465 | +0.0228 (−0.0086 to +0.0533) | +4.35 pp | +1.28 pp |
| Finger | 461 | −0.0180 (−0.0493 to +0.0126) | +15.79 pp | −16.82 pp |
| Forearm | 301 | −0.0196 (−0.0466 to +0.0078) | −1.32 pp | −3.33 pp |
| Hand | 460 | +0.0181 (−0.0161 to +0.0557) | +46.03 pp | −30.26 pp |
| Humerus | 288 | +0.0236 (−0.0068 to +0.0573) | +3.57 pp | +3.38 pp |
| Shoulder | 563 | −0.0030 (−0.0238 to +0.0174) | −7.55 pp | +14.04 pp |
| Wrist | 659 | +0.0248 (+0.0110 to +0.0393) | −4.75 pp | +7.69 pp |

Results file generated locally: `comparison_results/pooled_vs_specialists.csv`. It includes all eight metrics, both models' point estimates and intervals, and paired differences overall and for each anatomy.

### Pooled versus grouped models

Each grouped model was compared with the pooled model on the same images within its own group. The two groups containing wrist overlap and are reported separately. Elbow is outside these three groups.

| Group | Images | Pooled → grouped AUROC | Δ AUROC (95% CI) | Δ sensitivity (95% CI, pp) | Δ specificity (95% CI, pp) |
| --- | --- | --- | --- | --- | --- |
| Wrist–hand–finger | 1,580 | 0.8101 → 0.8161 | +0.0059 (−0.0035 to +0.0159) | +3.42 pp (+1.29 to +5.73) | −2.71 pp (−4.38 to −1.00) |
| Forearm–wrist | 960 | 0.8582 → 0.8638 | +0.0055 (−0.0064 to +0.0184) | −2.69 pp (−5.16 to −0.24) | +2.92 pp (+0.73 to +5.24) |
| Shoulder–humerus | 851 | 0.8373 → 0.8365 | −0.0007 (−0.0199 to +0.0182) | −4.07 pp (−7.27 to −0.85) | +8.31 pp (+3.91 to +12.59) |

Every group-level AUROC difference interval includes zero. At threshold 0.50, shoulder–humerus had the largest specificity gain (+8.31 pp) and a 4.07 pp sensitivity decrease. The three separate CSVs are generated under `comparison_results/` with filenames `pooled_vs_wrist_hand_finger.csv`, `pooled_vs_forearm_wrist.csv`, and `pooled_vs_shoulder_humerus.csv`.

### Pooled model threshold sweep

These point estimates use the same saved pooled predictions at each threshold. Lower thresholds detected more abnormal-labeled images and flagged more normal-labeled images. The paired comparisons use the fixed **0.50** row.

| Threshold | Accuracy | Precision | Sensitivity | Specificity | F1 |
| --- | --- | --- | --- | --- | --- |
| 0.30 | 64.4% | 58.1% | 92.0% | 39.0% | 0.712 |
| 0.35 | 68.1% | 61.9% | 87.1% | 50.7% | 0.723 |
| 0.40 | 71.3% | 66.5% | 80.3% | 62.9% | 0.728 |
| 0.45 | 73.4% | 71.7% | 73.5% | 73.4% | 0.726 |
| 0.50 | 74.1% | 77.4% | 65.0% | 82.5% | 0.706 |

### Interpretation and limitations

These are **exploratory image-level validation results**. The validation split also guided early stopping and learning-rate reduction. The patient bootstrap estimates validation-sample uncertainty for fixed trained models. The results describe this split and do not establish performance on an independent test set or at the study or patient level. The per-anatomy findings are descriptive and the reported intervals are individual, without adjustment for multiple comparisons.

---

## Future Improvements

* Evaluate on an independent patient-disjoint test set and assess study-level predictions.
* Fine-tune the model by unfreezing part of the MobileNetV2 backbone.
* Add a model or preprocessing step to classify the seven body parts before abnormality detection.
* Select and evaluate anatomy-specific thresholds using separate tuning and test data.
* Explore additional medical imaging datasets.
* Expand beyond X-rays to other scan types, such as CT or MRI, if appropriate datasets are available.

---

## Reproducing the Project

### Requirements

* Python **3.11.15**
* TensorFlow
* Pandas
* NumPy
* Other dependencies listed in `requirements.txt`

### Setup

1. Obtain the MURA dataset from Stanford AIMI.
2. Extract the dataset into the project root.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

The recommended option is to use the notebook for readability:

```bash
jupyter notebook main.ipynb
```

You can also run the Python script directly:

```bash
python main.py
```

### Validation metrics and prediction exports

The existing training/evaluation scripts and notebooks now report image-level
AUROC, **Average Precision (AP)**, accuracy, precision, sensitivity/recall,
specificity, F1, and Cohen's kappa, with 95% patient-clustered bootstrap CIs.
The existing TensorFlow training/evaluation metrics are retained. Final AUROC
and AP use raw abnormal-class probabilities. AP uses scikit-learn's
[`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
which uses changes in recall as precision weights, rather than trapezoidal
interpolation. Existing `auc`/`AUC` result columns mean AUROC.

Run the existing pipelines to generate predictions; the evaluation code does not
reconstruct missing predictions from aggregate results:

```bash
python main.py
python main_anatomy_specific.py
# Optional existing grouped-anatomy experiments:
python main_anatomy_group.py
```

Exports are written to:

| Pipeline | Prediction file | Report threshold |
| --- | --- | --- |
| `main.py` or `main.ipynb` | `pooled_results/validation_predictions.csv` | `DECISION_THRESHOLD` (0.50) |
| `main_anatomy_specific.py` or `main_anatomy_specific.ipynb` | `anatomy_specific_results/<BODY_PART>/validation_predictions.csv` | `DECISION_THRESHOLD` (0.50) |
| `main_anatomy_group.py` | `anatomy_group_results/<GROUP>/validation_predictions.csv` | `DECISION_THRESHOLD` (0.50) |

The anatomy-specific notebook copies the associated Python script's pipeline:
it trains a separate model for each entry in `BODY_PARTS`, filtering both training
and validation rows to that anatomy. Both versions use the same metrics,
0.50 threshold, and output locations. Running either version writes to the same
per-anatomy result files. The pooled command-line script saves plots as PNGs in
`pooled_results/`, with the confusion matrix saved in the project root.

Every export contains the same columns: `path`, `patient_id`, `body_part`,
`label`, `abnormal_probability`, `prediction`, and `threshold`. Existing extra
columns are retained. Patient IDs are extracted from the `patientNNNNN` path
component; anatomy and true labels are checked against the MURA path as well.
Validation datasets remain unshuffled and predictions retain dataframe order.
Scripts retain their existing `metrics.json` and summary CSV files with added
CI columns. The pooled script and notebook also save `metrics.json` and
`body_part_metrics.csv` beside their prediction export. Grouped models save an
additional report for each anatomy within their validation group.

Each report samples the unique patients in its own validation subset uniformly
with replacement. All images of a selected patient are retained, including all
copies when the same patient is sampled repeatedly. Metrics continue to weight
individual images. The evaluator collects **5,000 valid resamples** and takes
the 2.5th and 97.5th percentiles for each CI, using bootstrap seed 42.
Draws missing either true class or yielding nonfinite metrics are rejected and
replaced until 5,000 valid draws are collected. Reports record attempts and
skipped draws. An original subset without both true classes raises an explicit
error because it cannot support this full metric set. The existing
`zero_division=0` convention for precision/recall/F1 is retained.

Bootstrap draws use already-generated predictions with their recorded, fixed
thresholds. No model is retrained, no threshold is optimized within a draw, and
no study/patient prediction aggregation is introduced. These intervals describe
**validation-sample uncertainty**, not model-training or random-seed uncertainty.
Bootstrap reporting adds computation time, including at each existing sweep
threshold. The sweep's thresholds are unchanged; final reports use 0.50.

### Paired pooled-versus-specialist comparison

Once both models' exports exist, compare them without training or loading a
neural network:

```bash
python compare_predictions.py \
  --pooled pooled_results/validation_predictions.csv \
  --specialist anatomy_specific_results/*/validation_predictions.csv \
  --output comparison_results/pooled_vs_specialists.csv
```

The specialist files are combined in sorted filename order. Files must be
disjoint. The command validates unique image paths, patient IDs, anatomy,
true labels, probabilities, and thresholded predictions. It requires exact
one-to-one coverage of `MURA-v1.1_files/valid_image_paths.csv` (or the file passed
via `--validation-csv`). Missing/unexpected paths or conflicting metadata raise
errors. There is no inner join and no silent dropping of unmatched cases.
Old prediction exports lacking the new fields must be regenerated.

For one anatomy or an existing anatomy group, explicitly declare its subset:

```bash
python compare_predictions.py \
  --pooled pooled_results/validation_predictions.csv \
  --specialist anatomy_group_results/FOREARM_WRIST/validation_predictions.csv \
  --body-parts FOREARM WRIST \
  --output comparison_results/pooled_vs_forearm_wrist.csv
```

Only explicitly excluded anatomies are filtered out; coverage must still exactly
match the validation manifest within the selected anatomies. Do not combine
overlapping grouped-model exports: WRIST appears in two existing groups, so those
files intentionally fail duplicate-image validation when combined.

Both models must have recorded threshold 0.50 for this comparison. The same
patient draws are applied to both aligned prediction sets; every draw computes
specialist-minus-pooled metric differences. The CSV contains both model point
estimates and CIs, the point difference and its paired 95% percentile CI, and
bootstrap diagnostics, overall and within each anatomy. Nothing is inferred
when prediction files are absent.

The comparison computes the bootstrap from saved predictions and can take several minutes. To check the evaluation implementation without training:

```bash
python -m unittest discover -s tests -v
```
