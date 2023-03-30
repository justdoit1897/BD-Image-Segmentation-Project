import os
import cv2
import pandas as pd
from tqdm import tqdm

PATH = './training'
MASKS_PATH = PATH + '/masks/'
SCANS_PATH = PATH + '/scans/'

bar_format_yellow = "{l_bar}\x1b[33m{bar}\x1b[0m{r_bar}]"
bar_format_blue = "{l_bar}\x1b[34m{bar}\x1b[0m{r_bar}]"

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

print(f"\n--- Inizio resize di maschere e slices. ---")

print(f"\n\t1. Inizio resize di maschere e slices quadrate.\n")

for index, riga in tqdm(merged_df[(merged_df['width'] != 360) & (merged_df['height'] != 310)].reset_index().drop(columns='index').iterrows(),\
    total=len(merged_df[(merged_df['width'] != 360) & (merged_df['height'] != 310)]), bar_format=bar_format_yellow):
    
    if riga['segmentation'] != '(nan, nan, nan)':
    
        scan = cv2.imread(riga['path'])
        maschera = cv2.imread(riga['mask_path'])
        
        resized_scan = cv2.resize(scan, [200,200])
        resized_mask = cv2.resize(maschera, [200,200])   
        
        cv2.imwrite(SCANS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + riga['slice_name'], resized_scan)
        cv2.imwrite(MASKS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + 'mask_' + riga['slice_name'], resized_mask)
    
print(f"\n\t2. Inizio resize di maschere e slices 360x310.\n")

for index, riga in tqdm(merged_df[(merged_df['width'] == 360) & (merged_df['height'] == 310)].reset_index().drop(columns='index').iterrows(),\
    total=len(merged_df[(merged_df['width'] == 360) & (merged_df['height'] == 310)]), bar_format=bar_format_blue):
    
    if riga['segmentation'] != '(nan, nan, nan)':
        scan = cv2.imread(SCANS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + riga['slice_name'])
        maschera = cv2.imread(MASKS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + 'mask_' + riga['slice_name'])
        
        resized_scan = cv2.resize(scan, [200,200])
        resized_mask = cv2.resize(maschera, [200,200])   
        
        cv2.imwrite(SCANS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + riga['slice_name'], resized_scan)
        cv2.imwrite(MASKS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + 'mask_' + riga['slice_name'], resized_mask)
        
print(f"\nFine resize di maschere e slices.\n")