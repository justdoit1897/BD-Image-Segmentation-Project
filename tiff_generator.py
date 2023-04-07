import pandas as pd
import numpy as np
import cv2 
import tifffile as tiff

from joblib import Parallel, delayed
from tqdm import tqdm

merged_df = pd.read_csv('./dataframes/merged_df.csv')

segmented_rows = merged_df.loc[merged_df['segmentation'] != '(nan, nan, nan)']

print(f"{segmented_rows.head()}\n\n--- Inizio creazione files .tif ---\n")

bar_format_green = "{l_bar}\x1b[32m{bar}\x1b[0m{r_bar}]"
bar_format_yellow = "{l_bar}\x1b[33m{bar}\x1b[0m{r_bar}]"

scans = []

masks = []

def generate_scans_tiff(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # conversione da BGR a RGB
    return img
        
def generate_masks_tiff(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # conversione da BGR a RGB
    return img

print(f"\t1- Inizio Generazione Lista Scan\n")
        
with Parallel(n_jobs=-1, prefer="processes") as parallel:
    results = parallel(delayed(generate_scans_tiff)(row[1]['path']) for row in\
        tqdm(segmented_rows.iterrows(), total=(len(segmented_rows)), bar_format=bar_format_green))

for img in results:
    scans.append(img)

scans = np.array(scans)

print(f"\n\tFine Generazione Lista Scan\n")

with tiff.TiffWriter(f"./training/scans.tif", bigtiff=True) as tif:    
    tif.write(scans, photometric='rgb', compression=('DEFLATE', 4))
    
print(f"\t2- Inizio Generazione Lista Maschere\n")
with Parallel(n_jobs=-1, prefer="processes") as parallel:
    results = parallel(delayed(generate_masks_tiff)(row[1]['mask_path']) for row in\
        tqdm(segmented_rows.iterrows(), total=(len(segmented_rows)), bar_format=bar_format_yellow))

for img in results:
    masks.append(img)    

masks = np.array(masks)

print(f"\n\tFine Generazione Lista Maschere\n")
        
with tiff.TiffWriter(f"./training/masks.tif", bigtiff=True) as tif:    
    tif.write(masks, photometric='rgb', compression=('DEFLATE', 4))
        
print(f"--- Fine creazione files .tif ---")