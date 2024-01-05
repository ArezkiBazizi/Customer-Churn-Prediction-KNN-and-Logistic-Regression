import math
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset.csv", delimiter=';')
data = data.to_numpy()

X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)
x = [X_train[i][0] for i in range(len(X_train))]
y = [X_train[i][1] for i in range(len(X_train))]

colors = ['red' if X_train[i][-1] == 0 else 'blue' for i in range(len(X_train))]
plt.scatter(x, y, c=colors)
for row in X_test :
    plt.scatter(row[0], row[1], marker='x', color='green')
plt.show()

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    return math.sqrt(distance)

def knn(training_set, test_point, k):
    distances = []
    for row in training_set:
        dist = euclidean_distance(row[:-1], test_point)
        distances.append((row, dist))
    
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]#recuperer les voisins
    output_values = [row[-1] for row in neighbors]#recuperer les classes
    prediction = max(set(output_values), key=output_values.count)#predir la classe du test_point en prenant le max
    return prediction

prediction = []
for row in X_test :
    pred = knn(X_train, row[:-1], k=33)
    prediction.append((row[-1], pred))

#print("Predicted class:", prediction)
m = np.zeros((2, 2))
for row in prediction :
   if row[0] == 1.0 and row[1] == 1.0 :
       m[0][0] +=1
   elif row[0] == 1.0 and row[1] == 0.0 :
       m[0][1] +=1
   elif row[0] == 0.0 and row[1] == 1.0 :
       m[1][0] +=1
   elif row[0] == 0.0 and row[1] == 0.0 :
      m[1][1] +=1

print(m)

from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings("ignore")

y_train = [row[7] for row in X_train]
X_train = [row[,1:6] for row in X_train]


knn = KNeighborsClassifier(n_neighbors=33)
knn.fit(X_train, y_train)
prediction = []
for row in X_test :
    test_point = [row[:-1]]
    pred = knn.predict(test_point)
    prediction.append((row[-1], pred[0]))
    
#print("Predicted class:", prediction)
m = np.zeros((2, 2))
for row in prediction :
   if row[0] == 1.0 and row[1] == 1.0 :
       m[0][0] +=1
   elif row[0] == 1.0 and row[1] == 0.0 :
       m[0][1] +=1
   elif row[0] == 0.0 and row[1] == 1.0 :
       m[1][0] +=1
   elif row[0] == 0.0 and row[1] == 0.0 :
      m[1][1] +=1

print(m)
