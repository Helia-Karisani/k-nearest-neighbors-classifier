# Telecom Customer Segmentation (K-Nearest Neighbors Classifier)

Classifies telecom customers into one of 4 service categories from demographic and service-usage features, using a K-Nearest Neighbors (KNN) classifier.

---

## Problem

A telecom provider segments its customers into four groups based on service usage. Predicting a customer's group from demographic and service data lets the company personalize offers.

This is a **supervised multi-class classification** task.

Target label: `custcat` (4 classes)
1. Basic Service
2. E-Service
3. Plus Service
4. Total Service

---

## Dataset

- Source: `teleCust1000t.csv` (IBM dataset)
- Target: `custcat`
- Features: all other columns (`X = df.drop('custcat', axis=1)`)

---

## Model: K-Nearest Neighbors

KNN is an instance-based (non-parametric) classifier. Training mostly means storing the data, and a new point is classified by the labels of its **k closest** training points.

### Scaling

KNN relies on distances, so features with large values (like income) would dominate binary flags. Features are standardized with `StandardScaler()`:

`x_scaled = (x - mean) / std`

### Math

Euclidean distance:

`d(x, xi) = sqrt( sum_j (x_j - xi_j)^2 )`

Prediction: find the k nearest training points and take the most frequent class among them:

`y_hat = argmax_c sum_{xi in Nk(x)} 1[ yi = c ]`

---

## Workflow

1. Load the dataset into a Pandas DataFrame.
2. Check class distribution with `df['custcat'].value_counts()`.
3. Split into `X` (all columns except `custcat`) and `y` (`custcat`).
4. Standardize features with `StandardScaler`.
5. Train/test split: 80/20, `random_state = 4`.
6. Train a baseline KNN with k = 3.
7. Evaluate accuracy on the test set.
8. Sweep k = 1..10 (and a larger sweep up to 100) and pick the k with the best test accuracy.

---

## Result

- Best accuracy in the k = 1..10 sweep: **0.34**
- Best k: **9**

Accuracy is measured on the held-out test split.

---

## Tech Stack

- Python
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn

---

## How to Run

```bash
git clone https://github.com/Helia-Karisani/k-nearest-neighbors-classifier.git
cd k-nearest-neighbors-classifier
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook k-nearest-neighbors-classifier.ipynb
```

---

## Possible Improvements

- Use cross-validation to choose k more reliably.
- Try other distance metrics (Manhattan, Minkowski).
- Report a confusion matrix and per-class precision/recall.
- Compare with logistic regression, decision tree, and random forest.
