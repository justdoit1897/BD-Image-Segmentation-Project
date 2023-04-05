import os
import cv2 as cv
import tifffile as tiff

from glob import glob
from tqdm import tqdm

PATH = './training'
MASKS_PATH = './training/masks/'
SCANS_PATH = './training/scans/'

list_masks = sorted(glob(MASKS_PATH + '*.png'))

print("Creazione file masks.tif")

# caricamento dei file temporanei e creazione del file tif
with tiff.TiffWriter(f"{PATH}/masks.tif", bigtiff=True) as tif:
    for mask in tqdm(list_masks):
        img = cv.imread(mask, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # conversione da BGR a RGB
        tif.write(img, photometric='rgb')
        
# eliminazione dei file temporanei
os.remove(MASKS_PATH)
    
# print("Fine creazione file masks.tif")

# import tifffile as tiff
# import numpy as np


# # Apertura del file tif
# with tiff.TiffFile('scans.tif') as tif:
#     # Lettura della prima immagine nel file tif
#     img = tif.pages[0].asarray()
    
#     # Stampa dei valori dei pixel
#     print(np.unique(img))

list_scans = sorted(glob(SCANS_PATH + '*.png'))

print("Creazione file scans.tif")

# caricamento dei file temporanei e creazione del file tif
with tiff.TiffWriter(f"{PATH}/scans.tif", bigtiff=True) as tif:
    for scan in tqdm(list_scans):
        img = cv.imread(scan, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # conversione da BGR a RGB
        tif.write(img, photometric='rgb')
        
# eliminazione dei file temporanei
os.remove(SCANS_PATH)
    
print("Fine creazione file scans.tif")
