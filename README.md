# K-Nearest Neighbors (KNN) from Scratch

This project implements the **K-Nearest Neighbors (KNN)** algorithm from scratch using Python and NumPy, without using any machine learning libraries like scikit-learn.

It demonstrates the core idea behind instance-based learning and distance-based classification.

---

## What is KNN?

K-Nearest Neighbors is a simple supervised learning algorithm used for classification (and regression).

It works by:

* Storing all training data
* Calculating distance between a test point and all training points
* Selecting the **K closest neighbors**
* Predicting the class by majority vote

---

## Formula Used

Euclidean Distance:
$$
d=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$
---

## Project Structure

```
knn-from-scratch/
│── main.py        # Main implementation
│── README.md     # Documentation
```

---

## Features

* Pure Python + NumPy implementation
* No external ML libraries
* Supports multi-class classification
* Easy to understand and extend
* Beginner-friendly ML project

---

## How It Works

1. Store training dataset
2. For each test sample:

   * Compute distance to all training points
   * Sort distances
   * Pick top K nearest neighbors
3. Return most common class

---

## Code Implementation

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict_single(self, x):
        distances = []

        for x_train in self.X_train:
            distance = self.euclidean_distance(x, x_train)
            distances.append(distance)

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])
```

---

## Example Usage

```python
import numpy as np
from knn import KNN

X_train = np.array([
    [1,2],
    [2,3],
    [3,4],
    [4,5],
    [5,6],
    [7,8],
    [8,9]
])

y_train = np.array([0,0,0,0,1,1,1])

X_test = np.array([
    [2,2],
    [7,7]
])

model = KNN(k=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)
```

Output:

```
[0 1]
```

---

## Time Complexity

For each prediction:

* Distance computation: O(n)
* Sorting: O(n log n)

Overall:

```
O(n log n)
```

---

## Limitations

* Slow for large datasets
* Sensitive to feature scaling
* Requires storing entire dataset
* No training phase (lazy learning)

---

## Improvements You Can Add

* Weighted KNN
* Feature scaling (Standardization)
* KD-Tree optimization
* KNN regression
* Distance metrics (Manhattan, Minkowski)

---

## Why This Project Matters

This project helps you understand:

* Instance-based learning
* Distance metrics
* Decision boundaries
* Lazy learning algorithms

It is a strong foundation for understanding real-world ML systems.

---
