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
from glob import glob

SLICES_PATH = './training/scans/'
MASKS_PATH = './training/masks/'

df = pd.read_csv("merged_df.csv")

# carica le tue slice e maschere 3D e convertile in array numpy

list_slices = glob.glob(SLICES_PATH + '*.png')
list_masks = glob.glob(MASKS_PATH + '*.png')

slice_array = np.array(list_slices)
mask_array = np.array(list_masks)

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
