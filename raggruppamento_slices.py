import os
import cv2 as cv
import pandas as pd
import random
import numpy as np
import shutil
import tifffile as tiff

from glob import glob
from tqdm import tqdm

PATH = './training'
MASKS_PATH = PATH + '/masks/'
SCANS_PATH = PATH + '/scans/'

bar_format_red = "{l_bar}\x1b[31m{bar}\x1b[0m{r_bar}]"
bar_format_green = "{l_bar}\x1b[32m{bar}\x1b[0m{r_bar}]"
bar_format_yellow = "{l_bar}\x1b[33m{bar}\x1b[0m{r_bar}]"
bar_format_blue = "{l_bar}\x1b[34m{bar}\x1b[0m{r_bar}]"
bar_format_magenta = "{l_bar}\x1b[35m{bar}\x1b[0m{r_bar}]"
bar_format_cyan = "{l_bar}\x1b[36m{bar}\x1b[0m{r_bar}]"

# if not os.path.exists(PATH):
#     os.mkdir(PATH) 
#     print("\nCartella training creata con successo!\n")
    
#     if not os.path.exists(MASKS_PATH):
#         os.mkdir(MASKS_PATH) 
#         print("\nCartella training/masks creata con successo!\n")
        
#     if not os.path.exists(SCANS_PATH):
#         os.mkdir(SCANS_PATH) 
#         print("\nCartella training/scans creata con successo!\n")

TRAIN_PATH = "../BD-Image-Segmentation-Comp/train/"

MERGED_DF = "../BD-Image-Segmentation-Comp/dataframes/merged_df.csv"

folders = [name for name in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, name))]

folders = sorted(folders, key=lambda x: int(x[4:]))

print(folders)

merged_df = pd.read_csv(MERGED_DF)

df = merged_df[merged_df['segmentation'] != '(nan, nan, nan)']

# print(merged_df['segmentation'])

slice_counts = df.groupby(['case_id', 'day_id', 'width_px'])['segmentation'].count()
# slice_counts = df.groupby(['case_id', 'day_id', 'slice_id'])['segmentation'].count()
print(slice_counts)       #slice_counts[case_id][day_id]

count = []

for slice in slice_counts:
    count.append(slice)

count = np.array(count)    
print(count)

slice_counts = df.groupby(['case_id', 'day_id'])['width_px'].unique()
print(slice_counts)

depth_array = np.concatenate(slice_counts.values)

# for slice in slice_counts:
#     depth.append(slice.values)
    
print(depth_array)

import math

array_profondita = []

for i in range(len(count)):
    array_profondita.append(math.ceil(count[i] * depth_array[i]))

array_profondita = np.array(array_profondita)

print(array_profondita)

print(df)

ALTEZZA = 256
LARGHEZZA = 256

slices = []
start = 0

# for length in count:
#     end = start + length
#     slices.append(df[start:end].reset_index(drop=True))
#     start = end

# for df in slices:
#     for index, row in df.iterrows():
#         print(row[6], row[7])
    

# print(len(slices))

# import matplotlib.pyplot as plt



# img = cv.imread(slices[0]['mask_path'])

# plt.imshow(img)
# plt.show()

# import tifffile

# for slice in slices:
    
#     print(slice['case_id'].unique())
#     print(slice['mask_path'])
    
#     # Salva i dati in un file TIF
#     tifffile.imsave('output.tif', data, metadata={'axes': 'ZCYX', 'z': 10, 'c': 1, 'y': 512, 'x': 512})
    
#     with tiff.TiffWriter(f"{TRAIN_PATH}/maschere_3D.tif", bigtiff=True) as tif:
#         for mask in tqdm(slice['mask_path'], bar_format=bar_format_green):
#             img = cv.imread(mask, cv.IMREAD_COLOR)
#             img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Conversione da BGR a RGB
#             img_stack = [img] * slice['case_id'].unique()  # Impilamento delle immagini
#             img_stack = np.stack(img_stack, axis=0)
#             tif.write(img_stack, photometric='rgb')
    
    # print(len(slice))

# print(float(slices[2]['width_px'].unique()))
# print(len(slices[2]))

print(f"\n--- Inizio eliminazione, cropping e resize di maschere e slices ---")

for index, riga in tqdm(merged_df.iterrows(), total=len(merged_df), bar_format=bar_format_green):
    
    if riga['segmentation'] != '(nan, nan, nan)':
    
        # se le immagini sono rettangolari: croppa
        if(riga['width'] == 360 & riga['height'] == 310):
            
            scan = cv.imread(riga['path'])[:, 25:335]
            maschera = cv.imread(riga['mask_path'])[:, 25:335]
            
            cv.imwrite(riga['path'], scan)
            cv.imwrite(riga['mask_path'], maschera)
        
        # a prescindere fai il resize
        scan = cv.imread(riga['path'])
        maschera = cv.imread(riga['mask_path'])
        
        resized_scan = cv.resize(scan, [LARGHEZZA, ALTEZZA])
        resized_mask = cv.resize(maschera, [LARGHEZZA, ALTEZZA]) 
        
        cv.imwrite(riga['path'], resized_scan)
        cv.imwrite(riga['mask_path'], resized_mask)
    
    else:
        
        # se non presentano segmentazione, vado a rimuovere la slice e la corrispondente maschera che, banalmente, sarà tutta nera
        os.remove(riga['path'])
        os.remove(riga['mask_path'])

print(f"\nOperazione avvenuta con successo!\n")

# # Numero di immagini bidimensionali impilate per formare la maschera tridimensionale
# n_slices = 10

# # Caricamento dei file temporanei e creazione del file tif
# with tiff.TiffWriter(f"{TRAIN_PATH}/masks.tif", bigtiff=True) as tif:
#     for mask in tqdm(list_masks, bar_format=bar_format_green):
#         img = cv.imread(mask, cv.IMREAD_COLOR)
#         img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Conversione da BGR a RGB
#         img_stack = [img] * n_slices  # Impilamento delle immagini
#         img_stack = np.stack(img_stack, axis=0)
#         tif.write(img_stack, photometric='rgb')
