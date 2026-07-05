# Musculoskeletal Abnormality Detection (MADlens) CNN Project

* **Goal was to create a mock tool that could assist radiologist, not replace in a simulated space**
   * Saving resources such as time by assisting radiologists in identifying abnormalities and potentially catching things missed by the human eye

## Model/Architecture
Brief Description about our CNN (frozen model), gradcam, no fine tuning yet, etc

## Dataset
* Using *MURA: MSK Xrays* from the Stanford University Center for Artificial Intelligence in Medicine and Imaging (AIMI).
   * MURA contains a large dataset of over 40 thousand bone X-rays, labeled Abnormal or Normal
   * Containing elbow, finger, forearm, hand, humerus, shoulder, and wrist samples
   * Abnormalities can range from: Dislocations, Fractures, Post-surgical hardware, and more
* The dataset's raw images and CSVs are not included in this repository
   * Please visit the link to inquire about the contents of the dataset, [MURA: MSK Xrays](https://aimi.stanford.edu/datasets/mura-msk-xrays)
   * **Access requires agreeing to Stanford's Research Use Agreement**

## Our Results and Findings
...


## Requirements for Reproducing
* python version used 3.11.15
1. Obtain dataset and extract to project root
2. pip install -r requirements.txt
   * Tensorflow, Pandas, numpy, and more
3. Recommended to use main.ipynb for readability
   * or use python main.py


## Future Additions and Improvements
....
