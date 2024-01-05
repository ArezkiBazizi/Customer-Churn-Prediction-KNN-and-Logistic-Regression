import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Chargement des données
data = pd.read_csv("dataset.csv", delimiter=";")
X = data.iloc[:, :-1]
X = X.to_numpy()
y = data.iloc[:, -1]
y = y.to_numpy()

# Spliter des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajout de la colonne de biais aux caractéristiques d'entraînement
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
# Initialisation des paramètres du modèle
theta = np.zeros(X_train.shape[1])

# Définition de la fonction sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Calcul de la fonction de coût
def loss(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return J

# Implémentation de la descente de gradient
def descente_grad(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        h = sigmoid(X.dot(theta))
        theta = theta - (alpha / m) * X.T.dot(h - y)
        J_history[iter] = loss(X, y, theta)
    return theta, J_history
    
# Calcul des paramètres optimaux du modèle
alpha = 0.01
num_iters = 1000
theta, J_history = descente_grad(X_train, y_train, theta, alpha, num_iters)

# Ajout de la colonne de biais aux caractéristiques de test
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Prédiction des étiquettes pour les données de test
probabilities = sigmoid(X_test.dot(theta))
y_pred = (probabilities >= 0.5).astype(int)

# Évaluation des performances du modèle
prec = accuracy_score(y_test, y_pred)
print("Précision:", prec*100)
# Évaluation des performances du modèle avec le F1 score 
# Traçage de l'évolution de la fonction de coût
plt.plot(J_history)
plt.xlabel('itérations')
plt.ylabel('loss')
plt.show()


# Calculer la matrice de confusion
confusion_mat = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion avec une heatmap
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs réelles')
plt.title('Matrice de confusion')
plt.show()