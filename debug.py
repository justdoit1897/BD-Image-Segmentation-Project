import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import modules.mask_ops as mop
import math
from colorama import init, Fore
init()


from tqdm import tqdm
from itertools import zip_longest 
os.environ["QT_QPA_PLATFORM"] = "xcb"


BASE_DIR = "../BD-Image-Segmentation-Comp/" 
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

train_df = pd.read_csv(TRAIN_CSV)
print(train_df)

# Sostituisco le classi con valori numerici per una miglior leggibilità
class_mapping = {'large_bowel': 0, 'small_bowel': 1, 'stomach': 2}

train_df['class'] = train_df['class'].replace(class_mapping)

# print(train_df)

splits = train_df['id'].str.split("_", n = 4, expand = True)
# print(splits)

train_df['case_id'] = splits[0]
train_df['day_id'] = splits[1]
train_df['slice_id'] = splits[3].astype(int)
train_df['case_id'] = train_df['case_id'].str.replace('case', '').astype(int) 
train_df['day_id'] = train_df['day_id'].str.replace('day', '').astype(int)

print(train_df)

# Elimino le colonne "class" e "segmentation" e ne salvo il contenuto in due liste quasi omonime
classe = train_df.pop('class')
segmentation = train_df.pop('segmentation')

# Raggruppo gli elementi delle liste in gruppi di tre
grouped_segmentation = list(zip_longest(*[iter(segmentation)]*3, fillvalue=None))
grouped_class = list(zip_longest(*[iter(classe)]*3, fillvalue=None))

# Elimino i duplicati sulla colonna "id"
train_df = train_df.drop_duplicates(subset=['id']).reset_index(drop=True)

# Reinserisco le colonne alla fine del dataframe
train_df.insert(len(train_df.columns), 'class', grouped_class)
train_df.insert(len(train_df.columns), 'segmentation', grouped_segmentation)

print(train_df)

list_slices = glob.glob(TRAIN_DIR+'/*/*/scans/*.png')

image_details = pd.DataFrame({'path':list_slices})

splits = image_details['path'].str.split("/", n = 7, expand = True)

image_details[['case_id', 'day_id']] = splits[4].str.split("_", expand = True)

image_details['case_id'] = image_details['case_id'].str.replace('case', '').astype(int) 
image_details['day_id'] = image_details['day_id'].str.replace('day', '').astype(int)

image_details['slice_name'] = splits[6]

print(image_details)

slice_info = image_details['slice_name'].str.split(n=6, expand=True, pat="_")

image_details['slice_id'] = slice_info[1].astype(int)

image_details['width'] = slice_info[2].astype(int)
image_details['height'] = slice_info[3].astype(int)

image_details['width_px'] = slice_info[4].astype(float)
#.round(2).apply(lambda x: '{:.2f}'.format(x))
image_details['height_px'] = slice_info[5].str.replace('.png', '', regex=False).astype(float)

print(image_details)

# Creo i path delle cartelle che conterranno le maschere

splits[5] = splits[5].str.replace('scans', 'masks')
splits[6] = splits[6].str.replace('slice', 'mask_slice')

percorsi_cartelle = splits.drop(columns=[6])
percorsi_cartelle = percorsi_cartelle.apply(lambda x: '/'.join(x.astype(str)), axis=1)

percorsi_maschere = splits.apply(lambda x: '/'.join(x.astype(str)), axis=1)

image_details.insert(1, 'mask_path', percorsi_maschere)

for path in percorsi_cartelle:
    # Creo la nuova cartella se non esiste già
    if not os.path.exists(path):
        os.mkdir(path) 
        print("\nCartella creata con successo!\n")


# ordino in ordine crescente in base a case_id, day_id e slice_id
image_details = image_details.sort_values(by=['case_id', 'day_id', 'slice_id'], ascending=True).reset_index(drop=True)
print(image_details)


# salvo il dataframe
image_details.to_csv('image_details_ordered.csv', index=False)

train_df = train_df.sort_values(by=['case_id', 'day_id', 'slice_id'], ascending=True).reset_index(drop=True)
print(train_df)

# salvo il dataframe
train_df.to_csv('train_ordered.csv', index=False)

merged_df = pd.merge(train_df, image_details, on=['case_id', 'day_id', 'slice_id'])

merged_df.to_csv('merged_df.csv', index=False)

def crea_maschera_vuota(path, width, height):

    # Creo una maschera vuota con le dimensioni dei pixel calcolate
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # cv.imwrite(path, mask)
    cv.imwrite(path, mask)

def rle_to_image(rle_code, height, width):
    
    # Verifica se la codifica è 'nan'
    
    if isinstance(rle_code, float) and math.isnan(rle_code):
        return np.zeros((height, width), dtype=np.uint8)
    
    # decodifica la codifica RLE
    rle_numbers = [int(i) for i in rle_code.split()]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)

    # crea un'immagine vuota
    img = np.zeros(height*width, dtype=np.uint8)

    # colora i pixel coperti dalla maschera
    for index, length in rle_pairs:
        index -= 1
        img[index:index+length] = 255

    # ridimensiona l'immagine e la restituisce
    return img.reshape((height, width))

merged_df['is_created_mask'] = [True] * merged_df.shape[0]
# merged_df['is_created_mask'] = [False] * merged_df.shape[0]

print(f"\nInizio creazione maschere vuote...\n")    

for index, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    
    mask_path = row['mask_path']
    
    if row['is_created_mask'] == False:
        
        crea_maschera_vuota(mask_path, row['width'], row['height'])
        
        row['is_created_mask'] = True
        
    else:
        #print("Maschere vuote già create")
        continue

print(f"\nFine creazione maschere vuote.\n")

'''
print(f"Inizio colorazione maschere vuote...\n")   

for index, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    
    mask_path = row['mask_path']
    
    if row['is_created_mask'] == True:
        
        red_segment = rle_to_image(row['segmentation'][0], row['height'], row['width'])
        green_segment = rle_to_image(row['segmentation'][1], row['height'], row['width'])
        blue_segment = rle_to_image(row['segmentation'][2], row['height'], row['width'])

        merged_mask = cv.merge([red_segment, green_segment, blue_segment])
        
        cv.imwrite(mask_path, merged_mask)  # Save merged mask to mask_path
        
    else:
        print("Maschera vuota inesistente.")
        continue
    
print(f"\nFine colorazione maschere vuote.\n")

'''

'''

La funzione equalizza_immagini legge un'immagine in scala di grigi dal percorso specificato, 
ne equalizza l'istogramma e sovrascrive l'immagine originale con quella equalizzata. 
Infine, viene stampato un messaggio che indica che l'immagine equalizzata è stata salvata 
al percorso specificato. 

'''

def equalizza_immagini(path: str):
    
    """Funzione usata per equalizzare e salvare un'immagine 
    delle dimensioni specificate

    Args:
        path (str): percorso in cui salvare l'immagine
        width (int): larghezza dell'immagine
        height (int): altezza dell'immagine
    """
    
    # Leggo l'immagine situata in 'path' in scala di grigi
    
    img = cv.imread(path, 0) 

    equ = cv.equalizeHist(img)
    
    cv.imwrite(path, equ)
    
    #print(f"\nEqualizzazione immagine in: {path}")

merged_df['is_equalized'] = [True] * merged_df.shape[0]
# merged_df['is_equalized'] = [False] * merged_df.shape[0]    
'''
for index, row in merged_df.iterrows():
    
    if row['is_equalized'] == False:
        print(f"Inizio equalizzazione...\n")    
        
        for index, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
            equalizza_immagini(row['path'])
        
        print(f"\nFine equalizzazione.\n")
        row['is_equalized'] = True
        
    else:
        print("Immagini già equalizzate")
        break
'''

#################################################### 'merged_df' #################################################### 

# VOGLIO CALCOLARE PER OGNI RIGA DI 'merged_df' IL NUMERO DI PIXEL BIANCHI E NERI DI OGNI SLICE

bar_format = "{l_bar}\x1b[33m{bar}\x1b[0m{r_bar}]"

    # Rosso: \x1b[31m
    # Verde: \x1b[32m
    # Giallo: \x1b[33m
    # Blu: \x1b[34m
    # Magenta: \x1b[35m
    # Ciano: \x1b[36m
    # Bianco: \x1b[37m

for index, row in tqdm(merged_df.iterrows(), total=len(merged_df), bar_format=bar_format):
    
    # Lettura dell'immagine
    slice = cv.imread(row['path'])
    slice_g_s = cv.cvtColor(slice, cv.COLOR_BGR2GRAY)

    # Conteggio dei pixel bianchi e neri
    num_bianchi = cv.countNonZero(slice_g_s)
    num_neri = np.count_nonzero(np.all(slice == 0, axis=2))
    
    # Rapporto bianchi/neri 

    # Aggiunta dei risultati come colonne del dataframe
    merged_df.at[index, 'white_pixels'] = num_bianchi
    merged_df.at[index, 'black_pixels'] = num_neri
    merged_df.at[index, 'white_px/black_px'] = num_bianchi/num_neri   

merged_df['white_pixels'] = merged_df['white_pixels']
merged_df['black_pixels'] = merged_df['black_pixels']
merged_df['white_px/black_px'] = merged_df['white_px/black_px'].round(6)

merged_df.to_csv('merged_df.csv', index=False)
