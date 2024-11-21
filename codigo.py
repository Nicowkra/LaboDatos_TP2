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
carpeta = '/home/valentin/Escritorio/LaboDatos_TP2/'
# un array para las imágenes, otro para las etiquetas (por qué no lo ponen en el mismo array #$%@*)
data_imgs = np.load(carpeta+'mnistc_images.npy')
data_chrs = np.load(carpeta+'mnistc_labels.npy')[:, np.newaxis]

# %% Ej 2.1.1
#Defino 10 matrices distintas, una por digito, donde voy a sumar todas las imagenes respectivas al conjunto
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
#Creo una lista de matrices donde voy a tener el  promedio de valor de pixel por cada pixel de clase de digito
promedios = [cero,uno,dos,tres,cuatro,cinco,seis,siete,ocho,nueve]
#defino matriz vacia para construir el promedio de todos los digitos juntos, la desviacion estandar y una lista donde pondre la cantidad de elementos por digito
promedio_general = np.zeros((28,28),float)
desviacion = np.zeros((28,28),float)
contador = np.zeros(10)
#Hago iteraciones para sumar los digitos en su matriz designada para promedio correspondiente y sumo al contador tambieb
for i in range(0,len(data_imgs)):
    promedios[data_chrs[i].item()] += (data_imgs[i,:,:,0])
    contador[data_chrs[i].item()] += 1
#divido por 10 y el promedio estaria listo
for j in range(0,10):
    promedios[j] = promedios[j]/contador[j]
    promedio_general += promedios[j]/10

#Aplico formula de desviacion estandar para cada digito para cada promedio y sumo luego 
for n in range(0,10):
    for j in range(0,28):
        for k in range(0,28):
            desviacion[j][k] += pow(promedios[n][j][k] - promedio_general[j][k],2)/10
            desviacion[j][k] = np.sqrt(desviacion[j][k]) 
data = np.zeros((10,10),float)

#%% Ej 2.1.2
#calculo la norma de la resta de dos promedios, para saber que tan parecidos son dos clases, ya que mientras mas se parece la norma de la resta tiende a 0, 
for i in range(0,10):
    for j in range(0,10):
        data[i][j] = np.linalg.norm(promedios[i]/256 - promedios[j]/256)
df = pd.DataFrame(data)

# Lo que imprimo es una matriz de comparacion, donde mientras mayor sea el numero,menos se parecen las clases 
sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(df,xticklabels=[0,1,2,3,4,5,6,7,8,9],yticklabels=[0,1,2,3,4,5,6,7,8,9], annot=True, fmt='.3g', cmap='Blues', cbar=False)
#%% Ej 2.1.3
#Veo un numero arbitrario, por ejemplo el 4, veo su promedio y calculo desviacion estandar para saber que tanto se parecen entre si el conjunto de las imagenes que representan el numero 4
desviacionCuatro = np.zeros((28,28),float)
for k in range(0,len(data_chrs)):
            if data_chrs[k].item() == 4:
                for i in range(0,28):
                    for j in range(0,28):
                        desviacionCuatro[i][j] += np.sqrt((pow((data_imgs[k,:,:,0][i][j])-(promedios[4])[i][j],2))/contador[4])
# %% Ej 2.2.1
dataImgs = [] #uso reshape para tener todas las matrices como un array
for i in range(0, len(data_chrs)):
    dataImgs.append(data_imgs[i, :, :, 0].reshape(-1))

dataImgs = np.asarray(dataImgs)

imgs = []
chrs = []
for i in range(0, len(data_chrs)): #Filtro los numeros a analizar
    if data_chrs[i].item() in {1, 2, 3, 4, 9}:
        imgs.append(data_imgs[i, :, :, 0].reshape(-1))
        chrs.append(data_chrs[i].item())
imgs = np.asarray(imgs)
chrs = np.asarray(chrs)

#%% Ej 2.2.2
# Exploro el rendimiento con distintas profundidades maximas
X_dev, X_holdOut, y_dev, y_holdOut = train_test_split(imgs, chrs, test_size=0.3) #separo en dev y holdout
accuracy = []
for i in range(1, 26):
    clf = DecisionTreeClassifier(max_depth=i, random_state=42)
    clf.fit(X_dev, y_dev) #Entrena
    y_pred = clf.predict(X_dev) #Prediccion
    accuracy.append((accuracy_score(y_dev, y_pred))) #Veo la prediccion contra el valor real
    
data = pd.DataFrame({'max_depth':range(1,26),'accuracy':accuracy})
plt.rcParams["figure.figsize"] = (6,4)
fig, axes = plt.subplots()
fig.suptitle('Rendimiento con profundidades maximas', fontsize=16)
data.plot(kind='line',x ='max_depth',y ='accuracy',xlabel='Profundidad',ylabel='Rendimiento',legend = False, ax = axes)
plt.plot((3, 3), (0, 1.0001), linestyle='--', color='orange')
plt.plot((10, 10), (0, 1.0001), linestyle='--', color='orange');

# %% Ej 2.2.3
# Exploro hiperparametros y comparo con K-Folding
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

dtcGini = DecisionTreeClassifier(criterion='gini', random_state=42)
dtcEntr = DecisionTreeClassifier(criterion='entropy', random_state=42)
dtcLog = DecisionTreeClassifier(criterion='log_loss', random_state=42)
dtcRand = DecisionTreeClassifier(splitter='random', random_state=42)
dtcLog2 = DecisionTreeClassifier(max_features='log2', random_state=42)
dtcBalanced = DecisionTreeClassifier(criterion='entropy',class_weight='balanced')

scoresGini = cross_val_score(dtcGini, X_dev, y_dev, cv=kf)
scoresEntr = cross_val_score(dtcEntr, X_dev, y_dev, cv=kf)
scoresLog = cross_val_score(dtcLog, X_dev, y_dev, cv=kf)
scoresRand = cross_val_score(dtcRand, X_dev, y_dev, cv=kf)
scoresLog2 = cross_val_score(dtcLog2, X_dev, y_dev, cv=kf)
scoresBalanced =cross_val_score(dtcBalanced, X_dev, y_dev, cv=kf)

table = pd.DataFrame({'Gini':scoresGini,'Entropy':scoresEntr,'Log':scoresLog,'Random':scoresRand,'Log2':scoresLog2,'Balanced':scoresBalanced})
table = table.mean(axis = 0)

# Exploro posibles valores para min_samples_split y min_samples_leaf
trainingSplit = []
for i in range(2, 100):
    clf = DecisionTreeClassifier(min_samples_split= i,random_state=42)
    clf.fit(X_dev, y_dev)
    y_pred = clf.predict(X_dev)
    trainingSplit.append((accuracy_score(y_dev, y_pred)))
dataSplit = pd.DataFrame({'minSplit':range(2,100),'accuracy':trainingSplit})

trainingLeaf = []
for i in range(1, 50):
    clf = DecisionTreeClassifier(min_samples_leaf= i,random_state=42)
    clf.fit(X_dev, y_dev)
    y_pred = clf.predict(X_dev)
    trainingLeaf.append((accuracy_score(y_dev, y_pred)))
dataLeaf = pd.DataFrame({'minLeaf':range(1,50),'accuracy':trainingLeaf})

plt.rcParams["figure.figsize"] = (20,7)
fig, axes = plt.subplots(1,2)
fig.suptitle('Rendimiento con min_samples_split y  min_samples_leaf', fontsize=16)
dataSplit.plot(kind='line',x ='minSplit',y ='accuracy',xlabel='valor min_samples_split',ylabel='Rendimiento',legend = False, ax = axes[0])
dataLeaf.plot(kind='line',x ='minLeaf',y ='accuracy',xlabel='valor min_samples_leaf',ylabel='Rendimiento',legend = False, ax = axes[1]);

# %% Ej 2.2.4
# Entrenamos el modelo y lo evaluamos
clf = DecisionTreeClassifier(criterion='log_loss',max_depth = 10,random_state=42)
clf.fit(X_dev, y_dev)
y_pred = clf.predict(X_holdOut)

training = []
holdOut = []
for i in range(1, 26):
    clf = DecisionTreeClassifier(criterion='log_loss',max_depth=i,random_state=42)
    clf.fit(X_dev, y_dev)
    y_pred = clf.predict(X_dev)
    y_predHoldOut = clf.predict(X_holdOut)
    training.append((accuracy_score(y_dev, y_pred)))
    holdOut.append((accuracy_score(y_holdOut, y_predHoldOut)))
data = pd.DataFrame({'max_depth':range(1,26),'desarrollo':training,'holdOut':holdOut})
fig, axes = plt.subplots()
fig.suptitle('Comparación entre desarrollo y validación', fontsize=16)
data.plot(kind='line',x ='max_depth',y =['desarrollo','holdOut'],xlabel='Profundidad',ylabel='Rendimiento', ax = axes)
plt.plot((7, 7), (0, 1.0001), linestyle='--', color='red')


y_pred = clf.predict(X_holdOut)
cf_matrix = confusion_matrix(y_holdOut, y_pred)
sns.heatmap(cf_matrix,xticklabels=[1,2,3,4,9],yticklabels=[1,2,3,4,9], annot=True, fmt='d', cmap='Blues', cbar=False);
