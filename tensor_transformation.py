'''

Il file training_data.npy deve contenere i dati di addestramento della 3D U-Net. In particolare, il file deve contenere un array numpy che 
rappresenta le immagini di training in formato tensoriale. Questo array deve avere una forma di (numero_di_immagini, altezza, larghezza, 
profondità, numero_di_canali), dove numero_di_immagini è il numero totale di immagini di training, altezza, larghezza e profondità rappresentano 
le dimensioni delle immagini di training e numero_di_canali rappresenta il numero di canali dell'immagine, che solitamente è 1 per immagini in 
scala di grigi o 3 per immagini a colori.

Per trasformare le tue slice e maschere 3D in un array numpy tensoriale, puoi utilizzare la funzione np.stack(), specificando l'argomento axis per indicare 
l'asse della profondità. 

'''

import numpy as np
import pandas as pd

df = pd.read_csv("to_train.csv")

# carica le tue slice e maschere 3D e convertile in array numpy

slice_array = np.array(df['slice_path'].values)
mask_array = np.array(df['mask_path'].values)

path1 = '../BD-Image-Segmentation-Comp/train/case2/case2_day1/scans/slice_0001_266_266_1.50_1.50.png'
path2 = '../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0001_266_266_1.50_1.50.png'

# impila le slice e maschere 3D lungo l'asse della profondità
data = np.stack([slice_array, mask_array], axis=-1)

# salva l'array numpy in training_data.npy
np.save('training_data.npy', data)



'''

Le slice e maschere 3D sono tutte della stessa dimensione e che hanno lo stesso numero di canali.

                                (PER CONFERMA)

import cv2

slice = cv2.imread(path1)       # (266, 266, 3) 
mask = cv2.imread(path2)        # (266, 266, 3)

print(slice.shape, mask.shape)

'''
