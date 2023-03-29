'''

Per lavorare con una serie di slice con le rispettive maschere 3D che contengono le informazioni di 3 organi in 3 colori diversi, 
è possibile utilizzare una variante della rete U-Net chiamata 3D U-Net.

La 3D U-Net è una rete neurale convoluzionale progettata per l'elaborazione di dati 3D, che è stata sviluppata per la segmentazione di 
immagini biomediche volumetriche, come ad esempio le immagini di risonanza magnetica o di tomografia computerizzata. 
La rete è basata sulla struttura della U-Net, ma è estesa per lavorare con immagini volumetriche invece che con immagini bidimensionali.

Nel caso specifico in cui si vogliano segmentare tre organi con colori diversi, la 3D U-Net può essere addestrata per produrre in output una 
maschera tridimensionale che assegna un valore specifico ad ogni voxel dell'immagine originale, in base all'organo di appartenenza.

Per addestrare la rete, è necessario fornire un set di immagini volumetriche di addestramento e le rispettive maschere di segmentazione 3D, 
ciascuna delle quali assegna un valore specifico ad ogni voxel dell'immagine volumetrica originale, in base all'organo di appartenenza.

Una volta addestrata la rete, sarà possibile utilizzarla per segmentare nuove immagini volumetriche e isolare le regioni di interesse corrispondenti 
ai tre organi di interesse, utilizzando le maschere di segmentazione prodotte in output dalla rete.

'''

import os
import numpy as np

from sklearn.model_selection import train_test_split
from keras.callbacks import  ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K

from unet_model import get_unet_3d

'''

Il coefficiente di Dice può essere utilizzato come funzione di perdita per la rete, poiché cerca di massimizzare la sovrapposizione 
tra la maschera predetta e quella reale. L'uso del coefficiente di Dice come funzione di perdita può migliorare la precisione della rete e la qualità 
delle maschere segmentate.

Invece, la distanza di Hausdorff può essere utilizzata come metrica di valutazione per la rete, poiché cerca di valutare la precisione della maschera 
predetta rispetto alla maschera reale. L'uso della distanza di Hausdorff come metrica di valutazione può aiutare a valutare la qualità delle maschere 
segmentate prodotte dalla rete.

'''

# la funzione dice_coefficient calcola il coefficiente di Dice tra le maschere predette e quelle reali
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    size_true = K.sum(y_true, axis=[1,2,3])
    size_pred = K.sum(y_pred, axis=[1,2,3])
    dice = (2. * intersection + smooth) / (size_true + size_pred + smooth)
    return K.mean(dice, axis=0)

# la funzione dice_loss restituisce l'errore di Dice per la funzione di perdita della rete
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# la funzione hausdorff_distance calcola la distanza di Hausdorff tra le maschere predette e quelle reali
def hausdorff_distance(y_true, y_pred):
    d1 = K.sqrt(K.sum(K.square(y_true - y_pred), axis=[1,2,3]))
    d2 = K.sqrt(K.sum(K.square(y_pred - y_true), axis=[1,2,3]))
    hausdorff = K.maximum(d1, d2)
    return K.max(hausdorff, axis=0)

# Carica i dati di training
# train_data = np.load('training_data.ninpy')
train_data = np.load('training_data.npy', allow_pickle=True)

# Prende solo le slice e le rispettive maschere
X = train_data[:,0,:,:,:]
Y = train_data[:,1,:,:,:]

'''

La funzione train_test_split della libreria sklearn viene utilizzata per dividere i dati in un set di addestramento e uno di validazione. 
La funzione prende in input due insiemi di dati X e Y (caratteristiche e etichette, rispettivamente) e divide casualmente i dati in due insiemi, 
uno per il training e uno per la validazione.

Nel codice, X contiene le slice delle immagini e Y le relative maschere di segmentazione. 
La funzione divide questi dati in quattro parti: 

    -   X_train e Y_train, che vengono utilizzati per addestrare il modello 
    -   X_val e Y_val, che vengono utilizzati per valutare le prestazioni del modello durante l'addestramento. 

La dimensione del set di validazione è impostata al 10% del dataset totale, utilizzando test_size=0.2, mentre random_state=42 garantisce 
che la divisione casuale sia la stessa in ogni esecuzione del codice.

'''

# Dividi i dati in training e validation set
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# Crea il modello UNet
# model = get_unet_3d(input_shape=(None, None, None, 1), n_labels=3)

model = get_unet_3d(input_shape = (None, None, None, 1))

# Compila il modello
# model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy', MeanIoU(num_classes=3)])

# la rete viene compilata utilizzando l'ottimizzatore Adam, l'errore di Dice come funzione di perdita e il coefficiente di Dice
# e la distanza di Hausdorff come metriche di valutazione.
model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient, hausdorff_distance])

# Prepara i checkpoint per salvare il modello
checkpoint_path = "models/model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_loss', save_best_only=True, verbose=1)

# Prepara l'early stopping per evitare l'overfitting
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

# Fai il training del modello
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=1, callbacks=[checkpoint_callback, early_stopping_callback])
