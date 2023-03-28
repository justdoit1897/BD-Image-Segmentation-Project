import os
import cv2
import pandas as pd
from tqdm import tqdm

PATH = './training'
MASKS_PATH = PATH + '/masks/'
SCANS_PATH = PATH + '/scans/'

if not os.path.exists(PATH):
    os.mkdir(PATH) 
    print("\nCartella training creata con successo!\n")
    
    if not os.path.exists(MASKS_PATH):
        os.mkdir(MASKS_PATH) 
        print("\nCartella training/masks creata con successo!\n")
        
    if not os.path.exists(SCANS_PATH):
        os.mkdir(SCANS_PATH) 
        print("\nCartella training/scans creata con successo!\n")

image_details = pd.read_csv('./image_details_ordered.csv')

image_details_360x310 = image_details[(image_details['width'] == 360) & (image_details['height'] == 310)].reset_index().drop(columns='index')

print(image_details_360x310.head())

for index, riga in tqdm(image_details_360x310.iterrows(), total=len(image_details_360x310)):
    
    riga['case_id']
    
    scan = cv2.imread(riga['path'])[:, 25:335]
    maschera = cv2.imread(riga['mask_path'])[:, 25:335]
    
    cv2.imwrite(SCANS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + riga['slice_name'], scan)
    cv2.imwrite(MASKS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + 'mask_' + riga['slice_name'], maschera)

print(f"\nFine cropping.\n")
