# Telecom Customer Segmentation (K-Nearest Neighbors Classifier)

Classify telecom customers into one of 4 service categories using demographic + service-usage features, using a K-Nearest Neighbors (KNN) classifier.

---

## Project idea (problem statement)

A telecommunications provider segments its customer base into four groups based on service-usage patterns.  
If we can predict a customer's group from demographic/service data, the company can personalize offers.

This is a **supervised multi-class classification** task: given labeled examples, train a model to predict the class of a new (unknown) customer.

Target label: `custcat` (4 classes)
1. Basic Service
2. E-Service
3. Plus Service
4. Total Service

---

## Dataset

- Source: `teleCust1000t.csv` (IBM dataset)
- Target column: `custcat`
- Features: all other columns in the dataset (the notebook uses `X = df.drop('custcat', axis=1)`)

---

## Model: K-Nearest Neighbors (KNN)

KNN is an instance-based (non-parametric) classifier:
- “Training” mostly means **storing** the training data.
- Prediction for a new point is based on the labels of the **k closest** training points.

### Why scaling is required

KNN relies on distances. If features have different scales (e.g., income vs. binary flags), large-scale features dominate distances.

This project uses **standardization**:
- For each feature: x_scaled = (x - mean) / std
- Implemented via `StandardScaler()`.

---

## Math (plain, inline)

Let a customer be a feature vector x in R^d.

**Distance (Euclidean):**  
d(x, xi) = sqrt( sum_{j=1..d} (x_j - xi_j)^2 )

**KNN prediction rule (classification):**
1) Find Nk(x) = the set of k training points with smallest distance to x  
2) Predict the most frequent class among them:

y_hat = argmax_{c in {1,2,3,4}} sum_{xi in Nk(x)} 1[ yi = c ]

(Ties depend on the library’s tie-breaking; scikit-learn resolves this deterministically.)

---

## What the notebook does (technical workflow)

1. Load the dataset into a Pandas DataFrame.
2. Quick sanity check / class distribution using `df['custcat'].value_counts()`.
3. Split into:
   - X = all columns except `custcat`
   - y = `custcat`
4. Normalize features with `StandardScaler`:
   - X_norm = StandardScaler().fit_transform(X)
5. Train/test split:
   - 80% train, 20% test
   - `random_state = 4`
6. Train a baseline KNN model:
   - Example: k = 3
7. Evaluate on the test set using accuracy:
   - accuracy = (# correct) / (# test samples)
8. Hyperparameter sweep over k:
   - k = 1..10 (and also a larger sweep up to 100 in the notebook)
   - pick k with best test accuracy in the sweep

---

## Result (from the notebook run)

- Best accuracy in the k = 1..10 sweep: **0.34**
- Best k in that sweep: **k = 9**

(Accuracy is reported on the held-out test split.)

---

## Tech stack

- Python
- NumPy, Pandas
- scikit-learn
- Matplotlib
- (Seaborn imported in the notebook)

---

## How to run

1. Clone the repo:
   - git clone <your-repo-url>
   - cd <repo-folder>

2. Create/activate an environment (example):
   - python -m venv .venv
   - source .venv/bin/activate  (macOS/Linux)
   - .venv\Scripts\activate     (Windows)

3. Install dependencies:
   - pip install numpy pandas scikit-learn matplotlib seaborn jupyter

4. Open the notebook:
   - jupyter notebook

5. Run all cells in:
   - `k-nearest-neighbors-classifier.ipynb`

---

## Repo structure (suggested)

- `k-nearest-neighbors-classifier.ipynb`  -> main analysis + model training
- `README.md`                              -> this file

---

## Notes / possible improvements

- Use cross-validation (instead of a single train/test split) to select k more reliably.
- Try alternative distance metrics (Manhattan, Minkowski) and compare.
- Report confusion matrix + per-class precision/recall (accuracy alone can hide class imbalance effects).
- Compare against simple baselines (logistic regression, decision tree, random forest).

---
