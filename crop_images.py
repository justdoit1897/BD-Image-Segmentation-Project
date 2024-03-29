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

if not os.path.exists(PATH):
    os.mkdir(PATH) 
    print("\nCartella training creata con successo!\n")
    
    if not os.path.exists(MASKS_PATH):
        os.mkdir(MASKS_PATH) 
        print("\nCartella training/masks creata con successo!\n")
        
    if not os.path.exists(SCANS_PATH):
        os.mkdir(SCANS_PATH) 
        print("\nCartella training/scans creata con successo!\n")

merged_df = pd.read_csv('./merged_df.csv')

merged_df_360x310 = merged_df[(merged_df['width'] == 360) & (merged_df['height'] == 310)].reset_index().drop(columns='index')

# print(merged_df_360x310.head())

# print(merged_df_360x310)        

# for element in merged_df_360x310['segmentation']:
#     if element == '(nan, nan, nan)':
#         print(element)
#         print("\n")
    
for index, riga in tqdm(merged_df_360x310.iterrows(), total=len(merged_df_360x310), bar_format=bar_format_magenta):
    
    if riga['segmentation'] != '(nan, nan, nan)':
        
        scan = cv.imread(riga['path'])[:, 25:335]
        maschera = cv.imread(riga['mask_path'])[:, 25:335]
    
        cv.imwrite(SCANS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + riga['slice_name'], scan)
        cv.imwrite(MASKS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + 'mask_' + riga['slice_name'], maschera)

print(f"\nFine cropping.\n")

ALTEZZA = 256
LARGHEZZA = 256

def flip_coin():
    return random.randint(0, 1)

def do_skewing(slice, mask):
    
    rows, cols = slice.shape[:2]
    
    coin = flip_coin()
    
    if coin == 0:
        # angolazione in gradi
        angle = random.randint(-10, 10)
        while angle == 0:
            angle = random.randint(-10, 10)
        # definisco la matrice di trasformazione affine
        M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    
    else:
        
        moneta = flip_coin()
        
        if(moneta == 0):
            M = np.float32([[-1, 0, cols-1], [0, 1, 0]])
        
        else:
            M = np.float32([[1, 0, 0], [0, -1, rows-1]])
    
    # applico lo skewing alla slice
    skewed_slice = cv.warpAffine(slice, M, (cols, rows))
    
    # applico lo skewing alla maschera corrispondente
    skewed_mask = cv.warpAffine(mask, M, (cols, rows))
    
    return skewed_slice, skewed_mask

merged_df = pd.read_csv('./merged_df.csv')

count_quadrate = 0
count_rettangolari = 0

for index, tupla in merged_df.iterrows():
    
    if (tupla['segmentation'] == '(nan, nan, nan)'):
        
        if(tupla['width'] == 360):
            count_rettangolari += 1
        else:
            count_quadrate += 1

soglia_quadrate = round(count_quadrate * 0.1)
soglia_rettangolari = round(count_rettangolari * 0.1)

counter_skewing = 0

print(f"\n--- Inizio resize di maschere e slices. ---")

print(f"\n\t1. Inizio resize di maschere e slices quadrate.\n")

for index, riga in tqdm(merged_df[(merged_df['width'] != 360) & (merged_df['height'] != 310)].reset_index().drop(columns='index').iterrows(),\
    total=len(merged_df[(merged_df['width'] != 360) & (merged_df['height'] != 310)]), bar_format=bar_format_yellow):
    
    if riga['segmentation'] != '(nan, nan, nan)':
    
        scan = cv.imread(riga['path'])
        maschera = cv.imread(riga['mask_path'])
        
        resized_scan = cv.resize(scan, [LARGHEZZA, ALTEZZA])
        resized_mask = cv.resize(maschera, [LARGHEZZA, ALTEZZA])   
        
        coin = flip_coin()    
        if (coin == 0):
            if(counter_skewing != soglia_quadrate):
                resized_scan, resized_mask = do_skewing(resized_scan, resized_mask)
                counter_skewing += 1
        
        cv.imwrite(SCANS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + riga['slice_name'], resized_scan)
        cv.imwrite(MASKS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + 'mask_' + riga['slice_name'], resized_mask)

print(f"\nTotale immagini quadrate a cui è stato effettuato lo skewing: {counter_skewing}/{count_quadrate}\n")
    
print(f"\n\t2. Inizio resize di maschere e slices rettangolari.\n")

counter_skewing = 0

for index, riga in tqdm(merged_df[(merged_df['width'] == 360) & (merged_df['height'] == 310)].reset_index().drop(columns='index').iterrows(),\
    total=len(merged_df[(merged_df['width'] == 360) & (merged_df['height'] == 310)]), bar_format=bar_format_blue):
    
    if riga['segmentation'] != '(nan, nan, nan)':
        scan = cv.imread(SCANS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + riga['slice_name'])
        maschera = cv.imread(MASKS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + 'mask_' + riga['slice_name'])
        
        resized_scan = cv.resize(scan, [LARGHEZZA, ALTEZZA])
        resized_mask = cv.resize(maschera, [LARGHEZZA, ALTEZZA])  
        
        coin = flip_coin() 
        
        if (coin == 0):
            if(counter_skewing != soglia_rettangolari):
                resized_scan, resized_mask = do_skewing(resized_scan, resized_mask) 
                counter_skewing += 1
        
        cv.imwrite(SCANS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + riga['slice_name'], resized_scan)
        cv.imwrite(MASKS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + 'mask_' + riga['slice_name'], resized_mask)
        
print(f"\nFine resize di maschere e slices.\n")

print(f"\nTotale immagini rettangolari a cui è stato effettuato lo skewing: {counter_skewing}/{count_rettangolari}\n")


list_masks = sorted(glob(MASKS_PATH + '*.png'))

print("Creazione file masks.tif")

# caricamento dei file temporanei e creazione del file tif
with tiff.TiffWriter(f"{PATH}/masks.tif", bigtiff=True) as tif:
    for mask in tqdm(list_masks, bar_format=bar_format_green):
        img = cv.imread(mask, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # conversione da BGR a RGB
        tif.write(img, photometric='rgb')
        
# eliminazione dei file temporanei
shutil.rmtree(MASKS_PATH)

list_scans = sorted(glob(SCANS_PATH + '*.png'))

print("Creazione file scans.tif")

# caricamento dei file temporanei e creazione del file tif
with tiff.TiffWriter(f"{PATH}/scans.tif", bigtiff=True) as tif:
    for scan in tqdm(list_scans):
        img = cv.imread(scan, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # conversione da BGR a RGB
        tif.write(img, photometric='rgb')
        
# eliminazione dei file temporanei
shutil.rmtree(SCANS_PATH)
    
print("Fine creazione file scans.tif")
