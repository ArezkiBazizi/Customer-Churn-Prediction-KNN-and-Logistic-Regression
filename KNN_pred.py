 #Importer les bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Charger les données à partir d'un fichier CSV
data = pd.read_csv("dataset.csv", delimiter=";",nrows=2000)


# Séparer les attributs X et les étiquettes y
X = data.iloc[:, :-1]
X = X.to_numpy()
y = data.iloc[:, -1]
y = y.to_numpy()


# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Définir la distance personnalisée en utilisant la distance de Hamming
def hamming_dist(x1, x2):
    return distance.hamming(x1, x2)

# Créer un objet de classificateur k-NN et ajuster le modèle aux données d'entraînement
k = 5
error_rate = []
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i, metric=hamming_dist)
    knn.fit(X_train, y_train)

# Prédire les étiquettes pour les données de test
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,10),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)