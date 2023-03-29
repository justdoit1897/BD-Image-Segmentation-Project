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
import glob
import cv2 as cv

SLICES_PATH = './training/scans/'
MASKS_PATH = './training/masks/'

# carica le tue slice e maschere 3D e convertile in array numpy

list_slices = glob.glob(SLICES_PATH + '*.png')
list_masks = glob.glob(MASKS_PATH + '*.png')

list_slices = sorted(list_slices)
list_masks = sorted(list_masks)

# creo 3 vettori colonna che conterranno rispettivamente la slice, la maschera e la profondità

slice_array = []
mask_array = []
depth_array = []

for slice_path in list_slices:
    # apri l'immagine e ottieni la profondità dal nome del file
    
    img = cv.imread(slice_path)
    depth = float(slice_path.split('/')[-1].split('_')[-2]) # assume che il nome del file sia nel formato: "slice_XXX_depth_1.5.png"
    
    # converto l'immagine in array numpy e aggiungi alla lista
    # slice_np = np.array(img)
    slice_array.append(img)
    depth_array.append(depth)

for mask_path in list_masks:
    
    # converto la maschera in array numpy e la aggiungo alla lista
    mask = cv.imread(mask_path)
    # mask_np = np.array(mask)
    mask_array.append(mask)

# impilo le slice e maschere 3D lungo l'asse della profondità
data = np.stack([slice_array, mask_array, depth_array], axis=-1)

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
