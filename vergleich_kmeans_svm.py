import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
import matplotlib
matplotlib.use("TkAgg")

# 1. Daten laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_flat = x_train.reshape(-1, 784) / 255.0
x_test_flat = x_test.reshape(-1, 784) / 255.0

# Optional: Subsets zur Beschleunigung
x_train_small = x_train_flat[:10000]
y_train_small = y_train[:10000]
x_test_small = x_test_flat[:2000]
y_test_small = y_test[:2000]

print("Daten vorbereitet.")

# 2. k-Means Clustering (unsupervised)
print("\n--- k-Means ---")
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(x_train_small)
cluster_labels = kmeans.labels_

# Cluster zu tatsächlichen Labels mappen
def map_clusters_to_labels(clusters, true_labels):
    label_map = {}
    for cluster_id in range(10):
        mask = (clusters == cluster_id)
        if np.any(mask):
            cluster_labels = true_labels[mask]
            most_common = mode(cluster_labels)
            label_value = most_common.mode if isinstance(most_common.mode, np.ndarray) else np.array([most_common.mode])
            label_map[cluster_id] = int(label_value[0])
        else:
            label_map[cluster_id] = -1
    return label_map

cluster_to_label = map_clusters_to_labels(cluster_labels, y_train_small)
predicted_labels_kmeans = np.array([cluster_to_label[c] for c in cluster_labels])
acc_kmeans = accuracy_score(y_train_small, predicted_labels_kmeans)
print(f"Accuracy (k-Means auf Training): {acc_kmeans:.2f}")

# 3. SVM Klassifikation (supervised)
print("\n--- SVM ---")
svm = SVC()
svm.fit(x_train_small, y_train_small)
svm_predictions = svm.predict(x_test_small)
acc_svm = accuracy_score(y_test_small, svm_predictions)
print(f"Accuracy (SVM auf Testdaten): {acc_svm:.2f}")
print(classification_report(y_test_small, svm_predictions))

# 4. Confusion Matrix anzeigen (für SVM)
cm = confusion_matrix(y_test_small, svm_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Confusion Matrix (SVM)")
plt.tight_layout()
plt.show()
