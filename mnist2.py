
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

# 1. Daten laden und normalisieren
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# 2. k-Means Clustering
kmeans = KMeans(n_clusters=10, random_state=42).fit(x_train)
labels = kmeans.labels_

# Cluster-Labels zu echten Ziffern zuordnen
def map_clusters(clusters, labels):
    return {i: mode(labels[clusters == i]).mode[0] for i in range(10)}

mapping = map_clusters(kmeans.labels_, y_train)
pred_kmeans = np.array([mapping[c] for c in kmeans.labels_])
print("k-Means Accuracy:", accuracy_score(y_train, pred_kmeans))

# 3. SVM-Klassifikation
svm = SVC().fit(x_train, y_train)
pred_svm = svm.predict(x_test)
print("SVM Accuracy:", accuracy_score(y_test, pred_svm))
print(classification_report(y_test, pred_svm))

# 4. Confusion Matrix
cm = confusion_matrix(y_test, pred_svm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Confusion Matrix (SVM)")
plt.show()
