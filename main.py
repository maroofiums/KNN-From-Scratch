import numpy as np
from collections import Counter

class KNN:
  def __init__(self,k=3):
    self.k = k
  def fit(self,X,y):
    self.X_train = X
    self.y_train = y
  def euclidean_distance(self,x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
  def predict_single(self,x):
    distances = []

    for x_train in self.X_train:
      distance = self.euclidean_distance(x,x_train)
      distances.append(distance)
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

  def predict(self,X):
    predictions = []

    for x in X:
      prediction = self.predict_single(x)
      predictions.append(prediction)
    return np.array(predictions)


if __name__ == "__main__":
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
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print(predictions)