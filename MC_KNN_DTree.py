from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix



# Chargement du dataset Iris
data = pd.read_csv("Static/IRIS.csv")

# Séparation des caractéristiques et de la cible
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']


#Division du dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Création et entraînement des modèles KNN et Decision Tree

# Modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Modèle Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

#Prédiction sur l'ensemble de test et génération des matrices de confusion

# Prédictions du modèle KNN
y_pred_knn = knn.predict(X_test)

# Matrice de confusion pour KNN
confusion_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# Prédictions du modèle Decision Tree
y_pred_tree = tree.predict(X_test)

# Matrice de confusion pour Decision Tree
confusion_matrix_tree = confusion_matrix(y_test, y_pred_tree)

#Affichage des matrices de confusion
# print("Matrice de confusion pour KNN:")
# print(confusion_matrix_knn)

# print("\nMatrice de confusion pour Decision Tree:")
# print(confusion_matrix_tree)


# Créer une figure et un axe pour la heatmap
fig, ax = plt.subplots()

# Générer la heatmap à partir de la matrice de confusion
# sns.heatmap(confusion_matrix_knn, ax=ax)

sns.heatmap(confusion_matrix_tree, ax=ax)


# Ajouter des étiquettes aux axes
ax.set_xlabel('Prédictions')
ax.set_ylabel('Vérités')

# Afficher les titres des axes
# ax.set_title('Matrice de Confusion KNN')
ax.set_title('Matrice de Confusion Decision Tree')

# Afficher la heatmap
plt.show()