# MADlens X-Ray Image Classifier

## Musculoskeletal Abnormality Detection CNN Project

MADlens is a prototype X-ray image classifier built to detect **normal vs. abnormal** musculoskeletal X-ray cases using a convolutional neural network with Tensorflow.

The purpose of this project is to explore how computer vision could help save time, support radiologists in identifying abnormalities, and potentially flag cases that may need closer review.

---

## Model / Architecture

* Used **MobileNetV2**, a lightweight pretrained CNN, as a frozen feature extractor.
* Trained the model to distinguish between **normal** and **abnormal** X-ray cases.
* Applied **class weighting** to reduce bias toward the majority class.
* Used slight **data augmentation** to help reduce overfitting.
* Implemented **Early Stopping** and **Learning Rate Reduction** while monitoring validation AUC.
* Used **AUC instead of accuracy** as a key metric because it gives a more informative view of performance when classes are imbalanced.
* Included a tunable **decision threshold** to adjust the trade-off between recall and precision.

---

## Dataset

This project uses **MURA: MSK X-rays** from the Stanford University Center for Artificial Intelligence in Medicine and Imaging (AIMI).

MURA contains over **40,000 bone X-ray images** labeled as either **normal** or **abnormal**. The dataset includes the following body parts:

* Elbow
* Finger
* Forearm
* Hand
* Humerus
* Shoulder
* Wrist

Abnormalities may include fractures, dislocations, post-surgical hardware, and other findings.

The raw images and CSV files are **not included** in this repository.

To access the dataset, visit: [MURA: MSK X-rays](https://aimi.stanford.edu/datasets/mura-msk-xrays)

> Access requires agreeing to Stanford's Research Use Agreement.

---

## Results and Findings

### Final Evaluation

Validation set results using the best weights restored by early stopping with a default threshold of **0.50**:

| Metric    | Value |
| --------- | ----: |
| Accuracy  | 73.7% |
| AUC       | 0.815 |
| Loss      | 0.529 |
| Precision | 0.798 |
| Recall    | 0.603 |

### Decision Threshold Sweep

Lowering the decision threshold increased recall but reduced precision. This means the model caught more abnormal cases, but also produced more false alarms.

| Threshold |  Accuracy | Precision |    Recall |        F1 |
| --------: | --------: | --------: | --------: | --------: |
|      0.30 |     65.7% |     0.594 |     0.893 |     0.714 |
|      0.35 |     69.6% |     0.640 |     0.832 |     0.724 |
|  **0.40** | **72.9%** | **0.700** | **0.760** | **0.729** |
|      0.45 |     74.2% |     0.755 |     0.682 |     0.717 |
|      0.50 |     73.7% |     0.798 |     0.603 |     0.687 |

### Classification Report

Classification report at threshold **0.40**, the value used for final reporting:

| Class                | Precision | Recall | F1-score | Support |
| -------------------- | --------: | -----: | -------: | ------: |
| Normal               |      0.76 |   0.70 |     0.73 |   1,667 |
| Abnormal             |      0.70 |   0.76 |     0.73 |   1,530 |
| **Overall accuracy** |           |        | **0.73** |   3,197 |

As a first prototype, the model performed fairly evenly across both classes. At the default threshold of **0.50**, the model missed about **4 out of 10** truly abnormal cases. Lowering the threshold to **0.40** reduced the missed abnormal cases to about **2.4 out of 10**, but at the cost of lower precision and more false positives.

One important limitation is that performance varied by body part. Some body parts, especially **hand** and **shoulder**, appeared to reduce the model's overall performance.

---

## Body Part Performance

| Body Part    | Images | Normal | Abnormal |  Accuracy | Precision |    Recall |        F1 |
| ------------ | -----: | -----: | -------: | --------: | --------: | --------: | --------: |
| Forearm      |    301 |    150 |      151 |     80.4% |     0.843 |     0.748 |     0.793 |
| Humerus      |    288 |    148 |      140 |     76.7% |     0.716 |     0.864 |     0.783 |
| Wrist        |    659 |    364 |      295 |     76.2% |     0.725 |     0.753 |     0.739 |
| Elbow        |    465 |    235 |      230 |     75.3% |     0.737 |     0.778 |     0.757 |
| Finger       |    461 |    214 |      247 |     73.1% |     0.737 |     0.773 |     0.755 |
| **Hand**     |    460 |    271 |      189 |     71.1% |     0.769 | **0.423** | **0.546** |
| **Shoulder** |    563 |    285 |      278 | **67.1%** | **0.612** |     0.917 |     0.734 |

The **hand** category had especially low recall at **0.423**, meaning the model missed many actual abnormal hand cases. The **shoulder** category had the lowest accuracy at **67.1%** and a lower precision score of **0.612**, meaning abnormal predictions for shoulder images were less reliable.

These differences suggest that a single binary classifier may not perform equally well across all body parts. A future version may benefit from first identifying the body part, then using a specialized model or threshold for that specific category.

---

## Future Improvements

* Fine-tune the model by unfreezing part of the MobileNetV2 backbone.
* Add a model or preprocessing step to classify the seven body parts before abnormality detection.
* Test separate thresholds for each body part.
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
