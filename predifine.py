
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# charger les attributes et les targets
data = pd.read_csv("dataset.csv",delimiter=";")
X = data.iloc[:, :-1]
X = X.to_numpy()
y = data.iloc[:, -1]
y = y.to_numpy()

# Separer les donnees en train 80% et 20% test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creéer un object de regression lineaire
logreg = LogisticRegression()

# entrainement du model
logreg.fit(X_train, y_train)

# la prédiction
y_pred = logreg.predict(X_test)

# Evaluation des performances
prec = accuracy_score(y_test, y_pred)
print("Précision:", prec*100)


# Calculer la matrice de confusion
confusion_mat = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion avec une heatmap
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs réelles')
plt.title('Matrice de confusion')
plt.show()