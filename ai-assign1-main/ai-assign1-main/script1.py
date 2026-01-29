import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Task 1: Binary classification using KNN classifier

data = pd.read_csv("data.csv")
print(data.head())
print(data.shape)
data = data.drop(columns=['id', 'Unnamed: 32'])
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
X = data.drop(columns=['diagnosis']).values
y = data['diagnosis'].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
np.random.seed(42)
indices = np.random.permutation(len(X))

split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def minkowski(a, b, p=3):
    return np.sum(np.abs(a - b) ** p) ** (1/p)

def cosine(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hamming(a, b):
    return np.sum(a != b)

def knn_predict(X_train, y_train, x_test, k, distance_func):
    distances = []

    for i in range(len(X_train)):
        dist = distance_func(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    labels = [label for _, label in neighbors]
    return Counter(labels).most_common(1)[0][0]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


K_values = [3, 4, 9, 20, 47]

distance_functions = {
    "Euclidean": euclidean,
    "Manhattan": manhattan,
    "Minkowski": minkowski,
    "Cosine": cosine,
    "Hamming": hamming
}

results = {}

for name, func in distance_functions.items():
    acc_list = []

    for k in K_values:
        predictions = []

        for x in X_test:
            pred = knn_predict(X_train, y_train, x, k, func)
            predictions.append(pred)

        acc = accuracy(y_test, np.array(predictions))
        acc_list.append(acc)

    results[name] = acc_list


plt.figure(figsize=(10,6))

for name, acc in results.items():
    plt.plot(K_values, acc, marker='o', label=name)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K for Different Distance Measures")
plt.legend()
plt.grid()
plt.show()
# Save results to a CSV file
results_df = pd.DataFrame(results, index=K_values)

results_df.to_csv("knn_results.csv", index_label="K Value")
print("Results saved to knn_results.csv")       

best_acc = 0
best_k = None
best_distance = None

for dist, acc_list in results.items():
    for k, acc in zip(K_values, acc_list):
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_distance = dist

print("Best K:", best_k)
print("Best Distance:", best_distance)
print("Best Accuracy:", best_acc)

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

best_func = distance_functions[best_distance]

final_predictions = [
    knn_predict(X_train, y_train, x, best_k, best_func)
    for x in X_test
]

TP, TN, FP, FN = confusion_matrix(y_test, np.array(final_predictions))

precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("Confusion Matrix")
print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
print("Precision:", precision)
print("Recall:", recall)


X_vis = X[:, :2]

X_train_v = X_vis[train_idx]
X_test_v = X_vis[test_idx]
xx, yy = np.meshgrid(
    np.linspace(X_vis[:,0].min(), X_vis[:,0].max(), 200),
    np.linspace(X_vis[:,1].min(), X_vis[:,1].max(), 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = [
    knn_predict(X_train_v, y_train, point, best_k, euclidean)
    for point in grid
]

Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_test_v[:,0], X_test_v[:,1], c=y_test, edgecolor='k')
plt.title("Decision Boundary (2D Projection)")
plt.show()
