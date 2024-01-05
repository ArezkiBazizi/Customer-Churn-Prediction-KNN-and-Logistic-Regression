# Importer les bibliothèques nécessaires
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données à partir d'un fichier CSV
data = pd.read_csv("dataset.csv",delimiter=";")

# Séparer les attributs X et les étiquettes y
X = data.iloc[:, :-1]
X = X.to_numpy()
y = data.iloc[:, -1]
y = y.to_numpy()


# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Définir la fonction de distance de Hamming pour les variables catégorielles
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    return math.sqrt(distance)

# Implémenter l'algorithme des k plus proches voisins
def k_nearest_neighbors(X_train, y_train, X_test, k):
    y_pred = []
    for test_instance in X_test:
        distances = []
        for train_instance in X_train:
            dist = euclidean_distance(test_instance, train_instance)
            distances.append(dist)

        # Trouver les k voisins les plus proches
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]

        # Prédire l'étiquette en se basant sur le  majoritaire
        unique_labels, label_counts = np.unique(nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(label_counts)]

        y_pred.append(predicted_label)

    return y_pred

# Appliquer l'algorithme des k plus proches voisins
k = 5
y_pred = k_nearest_neighbors(X_train, y_train, X_test, k)

# Définir la fonction d'évaluation de l'exactitude
def exactitude(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

# Évaluer les performances du modèle
prec = exactitude(y_test, y_pred)
print("Précision:", prec*100)