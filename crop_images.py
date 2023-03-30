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



merged_df = pd.read_csv('./merged_df.csv')

merged_df_360x310 = merged_df[(merged_df['width'] == 360) & (merged_df['height'] == 310)].reset_index().drop(columns='index')

# print(merged_df_360x310.head())

# print(merged_df_360x310)        

# for element in merged_df_360x310['segmentation']:
#     if element == '(nan, nan, nan)':
#         print(element)
#         print("\n")
    
for index, riga in tqdm(merged_df_360x310.iterrows(), total=len(merged_df_360x310)):
    
    if riga['segmentation'] != '(nan, nan, nan)':
        
        scan = cv2.imread(riga['path'])[:, 25:335]
        maschera = cv2.imread(riga['mask_path'])[:, 25:335]
    
        cv2.imwrite(SCANS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + riga['slice_name'], scan)
        cv2.imwrite(MASKS_PATH + 'case' + str(riga['case_id']) + '_day' + str(riga['day_id']) + '_' + 'mask_' + riga['slice_name'], maschera)

print(f"\nFine cropping.\n")
