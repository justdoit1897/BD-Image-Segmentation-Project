'''

Il file training_data.npy deve contenere i dati di addestramento della 3D U-Net. In particolare, il file deve contenere un array numpy che 
rappresenta le immagini di training in formato tensoriale. Questo array deve avere una forma di (numero_di_immagini, altezza, larghezza, 
profondità, numero_di_canali), dove numero_di_immagini è il numero totale di immagini di training, altezza, larghezza e profondità rappresentano 
le dimensioni delle immagini di training e numero_di_canali rappresenta il numero di canali dell'immagine, che solitamente è 1 per immagini in 
scala di grigi o 3 per immagini a colori.

Per trasformare le tue slice e maschere 3D in un array numpy tensoriale, puoi utilizzare la funzione np.stack(), specificando l'argomento axis per indicare 
l'asse della profondità. 

'''

# import numpy as np
# import pandas as pd
# import glob
# import cv2 as cv

# from tqdm import tqdm

# SLICES_PATH = './training/scans/'
# MASKS_PATH = './training/masks/'

# # carica le tue slice e maschere 3D e convertile in array numpy

# list_slices = glob.glob(SLICES_PATH + '*.png')
# list_masks = glob.glob(MASKS_PATH + '*.png')

# list_slices = sorted(list_slices)
# list_masks = sorted(list_masks)

# # creo 3 vettori colonna che conterranno rispettivamente la slice, la maschera e la profondità

# slice_array = []
# mask_array = []
# depth_array = []

# print(f"\nConversione slice...\n")

# for slice_path in tqdm(list_slices):
#     # apri l'immagine e ottieni la profondità dal nome del file
    
#     img = cv.imread(slice_path)
#     depth = float(slice_path.split('/')[-1].split('_')[-2]) # assume che il nome del file sia nel formato: "slice_XXX_depth_1.5.png"
    
#     # converto l'immagine in array numpy e aggiungi alla lista
#     slice_np = np.array(img)
#     slice_array.append(slice_np, dtype=object)
#     depth_array.append(depth)

# print(f"\nFine conversione slice\n")

# print(f"\nConversione maschere...\n")

# for mask_path in tqdm(list_masks):
    
#     # converto la maschera in array numpy e la aggiungo alla lista
#     mask = cv.imread(mask_path)
#     mask_np = np.array(mask, dtype=object)
#     mask_array.append(mask)

# print(f"\nFine conversione maschere\n")

# print("Inizio creazione tensore...")

# # impilo le slice e maschere 3D lungo l'asse della profondità
# data = np.stack([slice_array, mask_array, depth_array], axis=-1)

# print("Fine creazione tensore.")

# # salva l'array numpy in training_data.npy
# np.save('training_data.npy', data)
'''

import numpy as np
import pandas as pd
import glob
import cv2 as cv

from multiprocessing import Pool
from tqdm import tqdm

SLICES_PATH = './training/scans/'
MASKS_PATH = './training/masks/'

# carica le tue slice e maschere 3D e convertile in array numpy

list_slices = glob.glob(SLICES_PATH + '*.png')
list_masks = glob.glob(MASKS_PATH + '*.png')

list_slices = sorted(list_slices)
list_masks = sorted(list_masks)

# determina le dimensioni delle immagini
height, width, channels = cv.imread(list_slices[0]).shape

# creo un array numpy vuoto in cui impilare le slice
slice_array = np.empty((len(list_slices), height, width, channels), dtype=np.uint8)

# creo un vettore colonna che conterrà la profondità
depth_array = np.empty((len(list_slices), 1), dtype=float)

print(f"\nConversione slice...\n")

for i, slice_path in enumerate(tqdm(list_slices)):
    # apri l'immagine e ottieni la profondità dal nome del file
    img = cv.imread(slice_path)
    depth = float(slice_path.split('/')[-1].split('_')[-2]) # assume che il nome del file sia nel formato: "slice_XXX_depth_1.5.png"
    
    # aggiungi l'immagine all'array
    slice_array[i] = img
    depth_array[i] = depth

print(f"\nFine conversione slice\n")

# creo un array numpy vuoto in cui impilare le maschere
mask_array = np.empty((len(list_masks), height, width, channels), dtype=np.uint8)

print(f"\nConversione maschere...\n")

for i, mask_path in enumerate(tqdm(list_masks)):
    # converto la maschera in array numpy e la aggiungo all'array
    mask = cv.imread(mask_path)
    mask_array[i] = mask

print(f"\nFine conversione maschere\n")

print("Inizio creazione tensore...")

# impilo le slice e maschere 3D lungo l'asse della profondità
data = np.concatenate([slice_array, mask_array], axis=-1)

print("Fine creazione tensore.")

# salva l'array numpy in training_data.npy
np.save('training_data.npy', data)
np.save('training_depth.npy', depth_array)

'''

'''

Le slice e maschere 3D sono tutte della stessa dimensione e che hanno lo stesso numero di canali.

                                (PER CONFERMA)

import cv2

slice = cv2.imread(path1)       # (256, 256, 3) 
mask = cv2.imread(path2)        # (256, 256, 3)

print(slice.shape, mask.shape)

'''

import numpy as np
import pandas as pd
import glob
import cv2 as cv

from joblib import Parallel, delayed
from tqdm import tqdm

SLICES_PATH = './training/scans/'
MASKS_PATH = './training/masks/'

bar_format_yellow = "{l_bar}\x1b[33m{bar}\x1b[0m{r_bar}]"
bar_format_blue = "{l_bar}\x1b[34m{bar}\x1b[0m{r_bar}]"

# carica le tue slice e maschere 3D e convertile in array numpy

list_slices = glob.glob(SLICES_PATH + '*.png')
list_masks = glob.glob(MASKS_PATH + '*.png')

list_slices = sorted(list_slices)
list_masks = sorted(list_masks)

# determina le dimensioni delle immagini
height, width, channels = cv.imread(list_slices[0]).shape

# creo un array numpy vuoto in cui impilare le slice
slice_array = np.empty((len(list_slices), height, width, channels), dtype=np.uint8)

# creo un vettore colonna che conterrà la profondità
depth_array = np.empty((len(list_slices), 1), dtype=float)

print(f"\nConversione slice...\n")

def process_slice(slice_path):
    # apri l'immagine e ottieni la profondità dal nome del file
    img = cv.imread(slice_path)
    depth = float(slice_path.split('/')[-1].split('_')[-2]) # assume che il nome del file sia nel formato: "slice_XXX_depth_1.5.png"
    
    return (img, depth)

with Parallel(n_jobs=-1, prefer="processes") as parallel:
    results = parallel(delayed(process_slice)(slice_path) for slice_path in tqdm(list_slices, bar_format=bar_format_yellow))

for i, result in enumerate(results):
    # aggiungi l'immagine all'array
    slice_array[i] = result[0]
    depth_array[i] = result[1]

print(f"\nFine conversione slice\n")

# creo un array numpy vuoto in cui impilare le maschere
mask_array = np.empty((len(list_masks), height, width, channels), dtype=np.uint8)

print(f"\nConversione maschere...\n")

def process_mask(mask_path):
    # converto la maschera in array numpy e la aggiungo all'array
    mask = cv.imread(mask_path)
    return mask

with Parallel(n_jobs=-1, prefer="processes") as parallel:
    results = parallel(delayed(process_mask)(mask_path) for mask_path in tqdm(list_masks, bar_format=bar_format_blue))

for i, result in enumerate(results):
    mask_array[i] = result

print(f"\nFine conversione maschere\n")

print("Inizio creazione tensore...")

# impilo le slice e maschere 3D lungo l'asse della profondità
data = np.concatenate([slice_array, mask_array], axis=-1)

print("Fine creazione tensore.")

# salva l'array numpy in training_data.npy
np.save('training_data.npy', data)
np.save('training_depth.npy', depth_array)
