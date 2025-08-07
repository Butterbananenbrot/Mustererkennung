import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from scipy.stats import mode
from tensorflow.keras.datasets import mnist

# 1. Daten laden, kürzen, normalisieren
(trainingsbilder, trainingslabels), (testbilder, testlabels) = mnist.load_data()
trainingsbilder, trainingslabels = shuffle(trainingsbilder, trainingslabels)
trainingsbilder = trainingsbilder[:10000].reshape(-1, 784) / 255.0
trainingslabels = trainingslabels[:10000]
testbilder = testbilder.reshape(-1, 784) / 255.0

# 2. k-Means Clustering und Mapping
kmeans = KMeans(n_clusters=10, random_state=42).fit(trainingsbilder)
mapping = {i: mode(trainingslabels[kmeans.labels_ == i], keepdims=True)[0][0] for i in range(10)}
pred_kmeans = np.array([mapping[c] for c in kmeans.predict(testbilder)])
print("k-Means Accuracy:", accuracy_score(testlabels, pred_kmeans))
# print("Genutzter Algorithmus:", kmeans.algorithm)

# 3. SVM Klassifikation
svm = SVC().fit(trainingsbilder, trainingslabels)
pred_svm = svm.predict(testbilder)
print("SVM Accuracy:", accuracy_score(testlabels, pred_svm))
print(classification_report(testlabels, pred_svm))

# 4. Confusion Matrix anzeigen
sns.heatmap(confusion_matrix(testlabels, pred_svm), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Confusion Matrix (SVM)")
plt.savefig("confusion_matrix.png")
