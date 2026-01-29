import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter


DATA_DIR = "cifar-10-python/cifar-10-batches-py"

def load_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch




X_train = []
y_train = []

for i in range(1, 6):
    batch = load_batch(f"{DATA_DIR}/data_batch_{i}")
    X_train.append(batch[b'data'])
    y_train.extend(batch[b'labels'])

X_train = np.vstack(X_train)
y_train = np.array(y_train)




test_batch = load_batch(f"{DATA_DIR}/test_batch")
X_test = test_batch[b'data']
y_test = np.array(test_batch[b'labels'])



print("Train:", X_train.shape)
print("Test:", X_test.shape)




np.random.seed(42)

train_idx = np.random.choice(len(X_train), 5000, replace=False)
test_idx = np.random.choice(len(X_test), 1000, replace=False)

X_train = X_train[train_idx]
y_train = y_train[train_idx]

X_test = X_test[test_idx]
y_test = y_test[test_idx]


X_train = X_train / 255.0
X_test = X_test / 255.0


def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def minkowski(a, b, p=3):
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

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


K_values = [1, 3, 5, 9, 15]

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

        print(f"{name} | K={k} | Accuracy={acc:.4f}")

    results[name] = acc_list


plt.figure(figsize=(10, 6))

for name, acc in results.items():
    plt.plot(K_values, acc, marker='o', label=name)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("CIFAR-10 KNN: Accuracy vs K for Different Distance Metrics")
plt.legend()
plt.grid()
plt.show()


best_acc = 0
best_k = None
best_dist = None

for dist, acc_list in results.items():
    for k, acc in zip(K_values, acc_list):
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_dist = dist

print("\nBest Model")
print("Best K:", best_k)
print("Best Distance Metric:", best_dist)
print("Best Accuracy:", best_acc)


def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


best_func = distance_functions[best_dist]

final_predictions = [
    knn_predict(X_train, y_train, x, best_k, best_func)
    for x in X_test
]

cm = confusion_matrix(y_test, final_predictions)

precision = []
recall = []

for i in range(10):
    TP = cm[i, i]
    FP = np.sum(cm[:, i]) - TP
    FN = np.sum(cm[i, :]) - TP

    precision.append(TP / (TP + FP + 1e-9))
    recall.append(TP / (TP + FN + 1e-9))

print("\nEvaluation Metrics")
print("Average Precision:", np.mean(precision))
print("Average Recall:", np.mean(recall))

