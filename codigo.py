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
carpeta = '/home/Estudiante/Escritorio/LaboDatos_TP2/'
# un array para las imágenes, otro para las etiquetas (por qué no lo ponen en el mismo array #$%@*)
data_imgs = np.load(carpeta+'mnistc_images.npy')
data_chrs = np.load(carpeta+'mnistc_labels.npy')[:, np.newaxis]

cero = np.zeros((28,28),float)
uno = np.zeros((28,28),float)
dos = np.zeros((28,28),float)
tres = np.zeros((28,28),float)
cuatro = np.zeros((28,28),float) 
cinco = np.zeros((28,28),float)
seis = np.zeros((28,28),float) 
siete = np.zeros((28,28),float) 
ocho = np.zeros((28,28),float) 
nueve = np.zeros((28,28),float)
promedios = [cero,uno,dos,tres,cuatro,cinco,seis,siete,ocho,nueve]
promedio_general = np.zeros((28,28),float)
desviacion = np.zeros((28,28),float)
contador = np.zeros(10)
for i in range(0,len(data_imgs)):
    promedios[data_chrs[i].item()] += (data_imgs[i,:,:,0])
    contador[data_chrs[i].item()] += 1
for j in range(0,10):
    promedios[j] = promedios[j]/contador[j]
    promedio_general += promedios[j]/10
    
for n in range(0,10):
    for j in range(0,28):
        for k in range(0,28):
            desviacion[j][k] += pow(promedios[n][j][k] - promedio_general[j][k],2)/10
            desviacion[j][k] = np.sqrt(desviacion[j][k]) 
data = np.zeros((10,10),float)


for i in range(0,10):
    for j in range(0,10):
        data[i][j] = np.linalg.norm(promedios[i]/256 - promedios[j]/256)
df = pd.DataFrame(data)


sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(df,xticklabels=[0,1,2,3,4,5,6,7,8,9],yticklabels=[0,1,2,3,4,5,6,7,8,9], annot=True, fmt='.3g', cmap='Blues', cbar=False)