#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:39:20 2024

@author: rodrigo
"""

# script para cargar y plotear dígitos


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# %% Cargar datos
carpeta = 'C:/Users/nicow/Desktop/Facultad/LaboDeDatos/Archivos TP2/'
# un array para las imágenes, otro para las etiquetas (por qué no lo ponen en el mismo array #$%@*)
data_imgs = np.load(carpeta+'mnistc_images.npy')
data_chrs = np.load(carpeta+'mnistc_labels.npy')[:, np.newaxis]

# mostrar forma del array:
# 1ra dimensión: cada una de las imágenes en el dataset
# 2da y 3ra dimensión: 28x28 píxeles de cada imagen
print(data_imgs.shape)
print(data_chrs.shape)


# %% Grafico imagen

# Elijo la imagen correspondiente a la letra que quiero graficar
n_digit = 10
image_array = data_imgs[n_digit, :, :, 0]
image_label = data_chrs[n_digit]


# Ploteo el grafico
plt.figure(figsize=(10, 8))
plt.imshow(image_array, cmap='gray')
plt.title('caracter: ' + str(image_label))
plt.axis('off')
plt.show()


# %%
imgs = []
for i in range(0, len(data_chrs)):
    imgs.append(data_imgs[i, :, :, 0].reshape(-1))

imgs = np.asarray(imgs)

imgs = []
chrs = []
for i in range(0, len(data_chrs)):
    if data_chrs[i].item() in {1, 2, 3, 4, 9}:
        imgs.append(data_imgs[i, :, :, 0].reshape(-1))
        chrs.append(data_chrs[i].item())
imgs = np.asarray(imgs)
chrs = np.asarray(chrs)
# %%
X_dev, X_holdOut, y_dev, y_holdOut = train_test_split(imgs, chrs, test_size=0.3)
accuracy = []
for i in range(1, 26):
    clf = DecisionTreeClassifier(max_depth=i)
    clf.fit(X_dev, y_dev)
    y_pred = clf.predict(X_dev)
    accuracy.append((accuracy_score(y_dev, y_pred)))
data = pd.DataFrame({'max_depth':range(1,26),'accuracy':accuracy})
#%%
fig, axes = plt.subplots()
fig.suptitle('Rendimiento con profundidades maximas', fontsize=16)
data.plot(kind='line',x ='max_depth',y ='accuracy',legend = False, ax = axes )
plt.plot((10, 10), (0, 1.0001), linestyle='--', color='orange')
plt.xlabel('Profundidad', fontsize=15)
plt.ylabel('Rendimiento', fontsize=15)


#%%
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
depth = 10

dtcGini = DecisionTreeClassifier(criterion='gini')
dtcEntr = DecisionTreeClassifier(criterion='entropy')
dtcLog = DecisionTreeClassifier(criterion='log_loss')
dtcRand = DecisionTreeClassifier(splitter='random')
dtcLog2 = DecisionTreeClassifier(max_features='log2')
dtcBalanced = DecisionTreeClassifier(criterion='log_loss',class_weight='balanced')


scoresGini = cross_val_score(dtcGini, X_dev, y_dev, cv=kf)
scoresEntr = cross_val_score(dtcEntr, X_dev, y_dev, cv=kf)
scoresLog = cross_val_score(dtcLog, X_dev, y_dev, cv=kf)
scoresRand = cross_val_score(dtcRand, X_dev, y_dev, cv=kf)
scoresLog2 = cross_val_score(dtcLog2, X_dev, y_dev, cv=kf)
scoresBalanced =cross_val_score(dtcBalanced, X_dev, y_dev, cv=kf)

table = pd.DataFrame({'Gini':scoresGini,'Entropy':scoresEntr,'Log':scoresLog,'Random':scoresRand,'Log2':scoresLog2,'Balanced':scoresBalanced})
table = table.mean(axis = 0)


#%%
training = []
holdOut = []
for i in range(1, 50):
    clf = DecisionTreeClassifier(min_samples_leaf= i,max_depth=7,random_state=42)
    clf.fit(X_dev, y_dev)
    y_pred = clf.predict(X_dev)
    y_predHoldOut = clf.predict(X_holdOut)
    training.append((accuracy_score(y_dev, y_pred)))
    holdOut.append((accuracy_score(y_holdOut, y_predHoldOut)))
data = pd.DataFrame({'max_depth':range(1,50),'desarrollo':training,'holdOut':holdOut})
fig, axes = plt.subplots()
fig.suptitle('Rendimiento con profundidades maximas', fontsize=16)
data.plot(kind='line',x ='max_depth',y =['desarrollo','holdOut'],xlabel='Profundidad',ylabel='Rendimiento', ax = axes)

#%% EMPIEZO HOLDOUT PUNTO 4
clf = DecisionTreeClassifier(criterion='log_loss',max_depth=5,random_state=42)
clf.fit(X_dev, y_dev)
y_pred = clf.predict(X_holdOut)
#%% Rendimiento 

fig, axes = plt.subplots()
fig.suptitle('Rendimiento con profundidades maximas', fontsize=16)
data.plot(kind='line',x ='max_depth',y =['accuracy','holdOut'],legend = False, ax = axes )
plt.plot((3, 3), (0, 1.0001), linestyle='--', color='red')
plt.xlabel('Profundidad', fontsize=15)
plt.ylabel('Rendimiento', fontsize=15)
#%% Matriz confusion
cf_matrix = confusion_matrix(y_holdOut, y_pred)
sns.heatmap(cf_matrix,xticklabels=[1,2,3,4,9],yticklabels=[1,2,3,4,9], annot=True, fmt='d', cmap='Blues', cbar=False)