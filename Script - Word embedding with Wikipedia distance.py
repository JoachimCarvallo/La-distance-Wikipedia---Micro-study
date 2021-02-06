# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 23:05:56 2021
@author: joach
"""
# Chargement des librairies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from scipy.spatial import distance_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

# Chargement des données
n_search = pd.read_csv("D:/Scolarité/MS IA Télécom Paris/Complexity & Intelligence/n_search_10_classes.csv", 
                       header=0, index_col=0, encoding='latin-1')
mots = n_search.index

# Création de la variable des catégories
categ = np.concatenate([np.repeat("Colors", 11), np.repeat("Numbers", 11), np.repeat("Animals", 11), 
                        np.repeat("Jobs", 11), np.repeat("Clothes", 11), np.repeat("Rooms", 11), 
                        np.repeat("Fruits", 11), np.repeat("Vegetables", 11), np.repeat("Feelings", 11), 
                        np.repeat("Names", 11)])

# Suppression des valeurs nulles (si aucune page ne contient un couple de mots)
n_search[n_search.isnull()] = 1

# Création de la matrice de distances Wikipédia à partir des données
dist_mat = pd.DataFrame(np.zeros(n_search.shape), columns = mots, index = mots)
for row in n_search.index:
    for col in n_search.columns:
        dist_mat[row][col] = (np.max([np.log(n_search[row][row]), np.log(n_search[col][col])]) - np.log(n_search[row][col]))/(np.log(2*10**12) - np.min([np.log(n_search[row][row]), np.log(n_search[col][col])]))

# Création de la matrice M et calcul des valeurs propres
M = np.zeros(dist_mat.shape)
for i in range(dist_mat.shape[0]):
    for j in range(dist_mat.shape[1]):
        M[i,j] = (dist_mat.iloc[0,j]**2 + dist_mat.iloc[i,0]**2 - dist_mat.iloc[i,j]**2)/2

eigVals, eigVecs = np.linalg.eig(M)

# Calcul des coordonnés des mots
X = np.dot(eigVecs, np.sqrt(np.diag(np.abs(eigVals))))
X = pd.DataFrame(X, index = mots)

# Calcul de l'erreur moyenne par rapport à la matrice des distances
print("Erreur moyenne par mot (en %) :")
print(np.round((pd.DataFrame(distance_matrix(X, X), columns = mots, index = mots) - dist_mat).abs().mean()*100, 1))
print("Erreur moyenne globale (en %) :")
print(np.round((pd.DataFrame(distance_matrix(X, X), columns = mots, index = mots) - dist_mat).abs().mean().mean()*100, 1))

# Réduction de la dimension avec l'algorithme t-SNE pour la visualisation
X_embedded = TSNE(n_components=2, perplexity=11, early_exaggeration=15, n_iter=5000).fit_transform(X)

# Visualisation de nos mots 
X_temp = pd.DataFrame(X_embedded, index = mots, columns = ["X1","X2"])
X_temp["Catégories"] = categ

plt.figure(figsize = (17,12))
sns.scatterplot(data = X_temp, x = "X1", y = "X2", hue = "Catégories", s=200)
plt.grid()
texts = []
for x, y, s in zip(X_temp['X1'], X_temp['X2'], mots):
    texts.append(plt.text(x, y, s, size=15))
plt.title('%s iterations' % adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5)))

# Clustering par k-means sur notre plongement lexical 
kmeans = KMeans(n_clusters=10, random_state=0, n_init = 100, algorithm="elkan").fit(X)
class_k_means = kmeans.predict(X)

# Visualisation du clustering 
X_temp = pd.DataFrame(X_embedded, index = mots, columns = ["X1","X2"])
X_temp["Catégories"] = list(map(str, class_k_means))

plt.figure(figsize = (17,12))
sns.scatterplot(data = X_temp, x = "X1", y = "X2", hue = "Catégories", s=200)
#plt.axis('equal')
plt.grid()
texts = []
for x, y, s in zip(X_temp['X1'], X_temp['X2'], mots):
    texts.append(plt.text(x, y, s, size=15))
plt.title('%s iterations' % adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5)))

# Estimation de la qualité du clustering
n = len(mots)
count = 0
for i in range(n):
    for j in range(i+1,n):
        if((class_k_means[i] == class_k_means[j]) & (categ[i] == categ[j])): count += 1
        if((class_k_means[i] != class_k_means[j]) & (categ[i] != categ[j])): count += 1
            
print("Indice de Rand : {}".format(count/(n*(n-1)/2)))

print("Indice de Rand ajusté du hasard : {}".format(ari(categ, class_k_means)))