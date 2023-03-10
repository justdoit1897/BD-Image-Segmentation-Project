import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import mask_ops as mop

from itertools import zip_longest 

BASE_DIR = "../BD-Image-Segmentation-Comp/" 

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

########################################### PARTE 1 ###########################################

train_df = pd.read_csv(TRAIN_CSV)

print(train_df)
#train_df.head()

# Calcolo del numero di istanze per ogni classe
class_counts = train_df['class'].value_counts()

# Creazione del grafico a barre con colori personalizzati per ogni classe
colors = ['red', 'green', 'blue']
plt.bar(class_counts.index, class_counts.values, color=[colors[i] for i in range(len(class_counts))])

# Aggiunta delle etichette dell'asse x e y e del titolo del grafico
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Number of Instances per Class')

# Mostrare il grafico
#plt.show()

# Voglio mostrare solo le istanze del DataFrame che hanno una segmentazione non nulla

# Eliminazione delle righe con valori mancanti nella colonna "segmentation"
df_filtered = train_df.dropna(subset=['segmentation'])

# Calcolo del numero di istanze per ogni classe nella colonna "class" del DataFrame filtrato
class_counts = df_filtered['class'].value_counts()

# Creazione del grafico a barre con colori personalizzati per ogni classe
colors = ['red', 'green', 'blue']
plt.bar(class_counts.index, class_counts.values, color=[colors[i] for i in range(len(class_counts))])

# Aggiunta delle etichette dell'asse x e y e del titolo del grafico
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Number of Instances per Class (with non-null segmentation)')

# Mostrare il grafico
#plt.show()

'''

Abbiamo creato un DataFrame di esempio con alcune righe che hanno valori mancanti nella colonna 
"segmentation" e altre righe con una segmentazione definita. Abbiamo quindi utilizzato il metodo 
dropna() del DataFrame per eliminare le righe con valori mancanti nella colonna "segmentation" e 
creare un nuovo DataFrame "df_filtered" solo con le righe rimanenti.

Successivamente, abbiamo utilizzato il DataFrame filtrato "df_filtered" per calcolare il numero 
di istanze per ogni classe nella colonna "class", utilizzando la funzione value_counts() come nel 
precedente esempio. Infine, abbiamo creato un grafico a barre con i risultati e abbiamo aggiunto 
le etichette dell'asse x, dell'asse y e del titolo del grafico.

'''

# Sostituisco le classi con valori numerici per una miglior leggibilità
class_mapping = {'large_bowel': 0, 'small_bowel': 1, 'stomach': 2}

train_df['class'] = train_df['class'].replace(class_mapping)

splits = train_df['id'].str.split("_", n = 4, expand = True)
print(splits)

train_df['case_id'] = splits[0]
train_df['day_id'] = splits[1]
train_df['slice_id'] = splits[3].astype(int)

train_df['case_id'] = train_df['case_id'].str.replace('case', '').astype(int) 
train_df['day_id'] = train_df['day_id'].str.replace('day', '').astype(int)

# Elimino le colonne "class" e "segmentation" e ne salvo il contenuto in due liste quasi omonime
classe = train_df.pop('class')
segmentation = train_df.pop('segmentation')

# Raggruppo gli elementi delle liste in gruppi di tre
grouped_segmentation = list(zip_longest(*[iter(segmentation)]*3, fillvalue=None))
grouped_class = list(zip_longest(*[iter(classe)]*3, fillvalue=None))

# Elimino i duplicati sulla colonna "id"
train_df = train_df.drop_duplicates(subset=['id'])

# Reinserisco le colonne alla fine del dataframe
train_df.insert(len(train_df.columns), 'class', grouped_class)
train_df.insert(len(train_df.columns), 'segmentation', grouped_segmentation)

# ordino in ordine crescente in base a case_id, day_id e slice_id
train_df = train_df.sort_values(by=['case_id', 'day_id', 'slice_id'], ascending=True).reset_index(drop=True)

print(train_df)

########################################### PARTE 2 ###########################################

list_slices = glob.glob(TRAIN_DIR+'/*/*/scans/*.png')
#print(list_slices)

image_details = pd.DataFrame({'path':list_slices})

splits = image_details['path'].str.split("/", n = 7, expand = True)

image_details[['case_id', 'day_id']] = splits[4].str.split("_", expand = True)

image_details['case_id'] = image_details['case_id'].str.replace('case', '').astype(int) 
image_details['day_id'] = image_details['day_id'].str.replace('day', '').astype(int)

image_details['slice_name'] = splits[6]

slice_info = image_details['slice_name'].str.split(n=6, expand=True, pat="_")

image_details['slice_id'] = slice_info[1].astype(int)

image_details['width'] = slice_info[2].astype(int)
image_details['height'] = slice_info[3].astype(int)

image_details['width_px'] = slice_info[4].astype(float)
#.round(2).apply(lambda x: '{:.2f}'.format(x))
image_details['height_px'] = slice_info[5].str.replace('.png', '', regex=False).astype(float)

###########################
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


###########################

# ordino in ordine crescente in base a case_id, day_id e slice_id
image_details = image_details.sort_values(by=['case_id', 'day_id', 'slice_id'], ascending=True).reset_index(drop=True)

# aggiungo al df la riga contenente le maschere di segmentazione
image_details.insert(len(image_details.columns), 'segmentation', grouped_segmentation)
print("\n-------------------------------------------------------------------- image_details_merged_ --------------------------------------------------------------------\n")
#image_details.head()
print(image_details)

# Voglio vedere quante immagini hanno profondità dei pixel 1.50 mm e quante 1.63 mm
num_images_150 = image_details.loc[(image_details['width_px'] == 1.50) & (image_details['height_px'] == 1.50)].shape[0]
print("Il numero di immagini con width_px e height_px pari a 1.50 mm è:", num_images_150)

num_images_163 = image_details.loc[(image_details['width_px'] == 1.63) & (image_details['height_px'] == 1.63)].shape[0]
print("Il numero di immagini con width_px e height_px pari a 1.63 mm è:", num_images_163)

print(num_images_150+num_images_163)

# Dati per il grafico
labels = ['1.50 mm', '1.63 mm']
num_images = [num_images_150, num_images_163]

# Colori per le barre
colors = ['blue', 'orange']

# Creazione del grafico a barre
fig, ax = plt.subplots()
ax.bar(labels, num_images, color=colors)

# Aggiunta di titoli e label
ax.set_title('Numero di slice per larghezza e altezza dei pixel')
ax.set_xlabel('Larghezza e altezza pixel (in mm)')
ax.set_ylabel('Numero di slice')

# Visualizzazione del grafico
#plt.show()

# Voglio vedere se le immagini con una data profondità sono tutte delle stesse dimensioni

widths_px_1_50 = image_details[(image_details['width_px'] == 1.5) & (image_details['height_px'] == 1.5)]['width'].unique()
print(widths_px_1_50)

heights_px_1_50 = image_details[(image_details['width_px'] == 1.5) & (image_details['height_px'] == 1.5)]['height'].unique()
print(heights_px_1_50)

widths_px_63 = image_details[(image_details['width_px'] == 1.63) & (image_details['height_px'] == 1.63)]['width'].unique()
print(widths_px_63)

heights_px_1_63 = image_details[(image_details['width_px'] == 1.63) & (image_details['height_px'] == 1.63)]['height'].unique()
print(heights_px_1_63)

# si evince che le slice con profondità dei pixel pari a 1.50 mm hanno dimensione variabile, 
# mentre quelle a profondità pari a 1.63 mm hanno tutte la stessa dimensione

# Adesso voglio vedere se le immagini con larghezza pari a 266 hanno altezza variabile e viceversa

heights_266 = image_details[(image_details['width'] == 266)]['height'].unique()
print(heights_266)

widths_266 = image_details[(image_details['height'] == 266)]['width'].unique()
print(widths_266)

# Voglio vedere quante immagini hanno larghezza 266
num_images_266_width = image_details.loc[(image_details['width'] == 266)].shape[0]
print("Il numero di immagini con larghezza 266 px è:", num_images_266_width)

# Voglio vedere quante immagini hanno altezza 266
num_images_266_height = image_details.loc[(image_details['height'] == 266)].shape[0]
print("Il numero di immagini con altezza 266 px è:", num_images_266_height)

num_images_266x266 = image_details.loc[(image_details['width'] == 266) & (image_details['height'] == 266)].shape[0]
print("Il numero di immagini 266x266 è:", num_images_266x266)
# -> Ho dimostrato che le immagini con altezza/larghezza 266 sono quadrate

# Faccio lo stesso con le immagini larghe 234
heights_234 = image_details[(image_details['width'] == 234)]['height'].unique()
print(heights_234)

widths_234 = image_details[(image_details['height'] == 234)]['width'].unique()
print(widths_234)

# Voglio vedere quante immagini hanno larghezza 234
num_images_234_width = image_details.loc[(image_details['width'] == 234)].shape[0]
print("Il numero di immagini con larghezza 234 px è:", num_images_234_width)

# Voglio vedere quante immagini hanno altezza 234
num_images_234_height = image_details.loc[(image_details['height'] == 234)].shape[0]
print("Il numero di immagini con altezza 234 px è:", num_images_234_height)

num_images_234x234 = image_details.loc[(image_details['width'] == 234) & (image_details['height'] == 234)].shape[0]
print("Il numero di immagini 234x234 è:", num_images_234x234)

# -> Ho dimostrato che le immagini con altezza/larghezza 234 sono quadrate
        
# Mi torno l'altezza delle immagini larghe 360

heights_360 = image_details[(image_details['width'] == 360)]['height'].unique()
print(heights_360)

# Mi torno la larghezza delle immagini alte 310

widths_310 = image_details[(image_details['height'] == 310)]['width'].unique()
print(widths_310)

# Voglio vedere quante immagini hanno altezza 360
num_images_360_width = image_details.loc[(image_details['width'] == 360)].shape[0]
print("Il numero di immagini con larghezza 360 px è:", num_images_360_width)

# Voglio vedere quante immagini hanno altezza 310
num_images_310_height = image_details.loc[(image_details['height'] == 310)].shape[0]
print("Il numero di immagini con altezza 310 px è:", num_images_310_height)

num_images_360x310 = image_details.loc[(image_details['width'] == 360) & (image_details['height'] == 310)].shape[0]
print("Il numero di immagini 360x310 è:", num_images_360x310)

# Ho dimostrato che le tutte immagini larghe 360 hanno altezza 310

tot_slice = num_images_266x266 + num_images_234x234 + num_images_360x310

print("totale slice profondità pixel 1.50 mm: ", tot_slice)

# I valori combaciano :)
# Creo un grafico dal quale si evince quanto fatto in maniera qualitativa

# Dati per il grafico
labels = ['234x234', '266x266', '360x310']
num_slices = [num_images_234x234, num_images_266x266, num_images_360x310]

# Colori per le barre
colors = ['blue', 'orange', 'grey']

# Creazione del grafico a barre
plt.bar(labels, num_slices, color=colors)

# Aggiunta di titoli e label
plt.title('Numero di slice per larghezza e altezza dei pixel con profondità 1.50 mm')
plt.xlabel('Larghezza e altezza pixel (in px)')
plt.ylabel('Numero di slice')

# Visualizzazione del grafico
#plt.show()

# Dal dataframe "image_details" mi estraggo le righe le cui slice hanno dimensione 234x234
image_details_234x234 = image_details[(image_details['width'] == 234) & (image_details['height'] == 234)].copy().reset_index()
print("\n-------------------------------------------------------------------- image_details_234x234 --------------------------------------------------------------------\n")
print(image_details_234x234)
print(image_details_234x234['path'][0])

# contare il numero di righe con colonna "segmentation" diversa da "(nan, nan, nan)"
count_234 = 0
for i in range(len(image_details_234x234)):
    is_nan = True
    for j in range(len(image_details_234x234['segmentation'][i])):
        if not pd.isna(image_details_234x234['segmentation'][i][j]):
            is_nan = False
            break
    if is_nan:
        count_234 += 1

'''
Questo ciclo for scorre ogni elemento della colonna "segmentation", controllando se tutti i valori sono NaN. 
Se tutti i valori sono NaN, il contatore delle righe viene incrementato. 
'''

print(f"\nIl numero di righe con colonna 'segmentation' diversa da (nan, nan, nan) è {image_details_234x234.shape[0] - count_234}.")
print(f"Il numero delle righe totali è {image_details_234x234.shape[0]}.")
print(f"\nCi sono esattamente {count_234} slice di dimensione 234x234 con maschere totalmente vuote.")
print(f"\nLa percentuale di slice di dimensione 234x234 con maschere totalmente vuote è pari al {count_234/image_details_234x234.shape[0]}/100")

#lista prima riga contenente le 3 codifiche relative a classe 0, 1 e 2 
#print(image_details_234x234['segmentation'][0])

# primo elemento lista prima riga
#print(image_details_234x234['segmentation'][0][0])


#print("maschere di slice di dimensioni 234x234 non vuote: ", count)
 

# Dal dataframe "image_details" mi estraggo le righe le cui slice hanno dimensione 266x266
image_details_266x266 = image_details[(image_details['width'] == 266) & (image_details['height'] == 266)].copy().reset_index()
print("\n-------------------------------------------------------------------- image_details_266x266 --------------------------------------------------------------------\n")
print(image_details_266x266)

# contare il numero di righe con colonna "segmentation" diversa da "(nan, nan, nan)"
count_266 = 0
for i in range(len(image_details_266x266)):
    is_nan = True
    for j in range(len(image_details_266x266['segmentation'][i])):
        if not pd.isna(image_details_266x266['segmentation'][i][j]):
            is_nan = False
            break
    if is_nan:
        count_266 += 1

'''
Questo ciclo for scorre ogni elemento della colonna "segmentation", controllando se tutti i valori sono NaN. 
Se tutti i valori sono NaN, il contatore delle righe viene incrementato. 
'''

print(f"\nIl numero di righe con colonna 'segmentation' diversa da (nan, nan, nan) è {image_details_266x266.shape[0] - count_266}.")
print(f"Il numero delle righe totali è {image_details_266x266.shape[0]}.")
print(f"\nCi sono esattamente {count_266} slice di dimensione 266x266 con maschere totalmente vuote.")
print(f"\nLa percentuale di slice di dimensione 266x266 con maschere totalmente vuote è pari al {count_266/image_details_266x266.shape[0]}/100")

#print(image_details_266x266['segmentation'][0])
#print(image_details_266x266['segmentation'][0][0])

# Dal dataframe "image_details" mi estraggo le righe le cui slice hanno dimensione 276x276
image_details_276x276 = image_details[(image_details['width'] == 276) & (image_details['height'] == 276)].copy().reset_index()
print("\n-------------------------------------------------------------------- image_details_276x276 --------------------------------------------------------------------\n")
print(image_details_276x276)

# contare il numero di righe con colonna "segmentation" diversa da "(nan, nan, nan)"
count_276 = 0
for i in range(len(image_details_276x276)):
    is_nan = True
    for j in range(len(image_details_276x276['segmentation'][i])):
        if not pd.isna(image_details_276x276['segmentation'][i][j]):
            is_nan = False
            break
    if is_nan:
        count_276 += 1

'''
Questo ciclo for scorre ogni elemento della colonna "segmentation", controllando se tutti i valori sono NaN. 
Se tutti i valori sono NaN, il contatore delle righe viene incrementato. 
'''

print(f"\nIl numero di righe con colonna 'segmentation' diversa da (nan, nan, nan) è {image_details_276x276.shape[0] - count_276}.")
print(f"Il numero delle righe totali è {image_details_276x276.shape[0]}.")
print(f"\nCi sono esattamente {count_276} slice di dimensione 276x276 con maschere totalmente vuote.")
print(f"\nLa percentuale di slice di dimensione 276x276 con maschere totalmente vuote è pari al {count_276/image_details_276x276.shape[0]}/100")

# Dal dataframe "image_details" mi estraggo le righe le cui slice hanno dimensione 360x310
image_details_360x310 = image_details[(image_details['width'] == 360) & (image_details['height'] == 310)].copy().reset_index()
print("\n-------------------------------------------------------------------- image_details_360x310 --------------------------------------------------------------------\n")
print(image_details_360x310)

# contare il numero di righe con colonna "segmentation" diversa da "(nan, nan, nan)"
count = 0
for i in range(len(image_details_360x310)):
    is_nan = True
    for j in range(len(image_details_360x310['segmentation'][i])):
        if not pd.isna(image_details_360x310['segmentation'][i][j]):
            is_nan = False
            break
    if is_nan:
        count += 1

'''
Questo ciclo for scorre ogni elemento della colonna "segmentation", controllando se tutti i valori sono NaN. 
Se tutti i valori sono NaN, il contatore delle righe viene incrementato. 
'''

print(f"\nIl numero di righe con colonna 'segmentation' diversa da (nan, nan, nan) è {image_details_360x310.shape[0] - count}.")
print(f"Il numero delle righe totali è {image_details_360x310.shape[0]}.")
print(f"\nCi sono esattamente {count} slice di dimensione 360x310 con maschere totalmente vuote.")
print(f"\nLa percentuale di slice di dimensione 360x310 con maschere totalmente vuote è pari al {count/image_details_360x310.shape[0]}/100")

########################################### PARTE 3 ###########################################
'''
# Dati per il grafico
labels = ['234x234', '266x266', '276x276', '360x310']
counters = [count_234, count_266, count_276, count]

# Colori per le barre
colors = ['red', 'green', 'blue', 'orange']

# Creazione del grafico a barre
fig, ax = plt.subplots()
ax.bar(labels, counters, color=colors)

# Aggiunta di titoli e label
ax.set_title('Numero di maschere vuote per dimensione slice')
ax.set_xlabel('Larghezza e altezza slice (in px)')
ax.set_ylabel('Numero di slice')

# Visualizzazione del grafico
plt.show()
'''
########################################### PARTE 4 ###########################################

# RAGGRUPPARE PER SLICE_ID IN ORDINE CRESCENTE
# CONFRONTARE SE LE IMMAGINI CHE HANNO LO STESSO SLICE_ID E GIORNO DIVERSO SONO LE STESSE
 
########################################### PARTE 5 ###########################################

os.environ["QT_QPA_PLATFORM"] = "xcb"

import math

def split_rle(rle_strings):
    
    all_pixels = []
    all_lengths = []
    
    for rle_string in rle_strings:
        
        if isinstance(rle_string, float) and math.isnan(rle_string):
            
            # Se rle_string è NaN, aggiungo liste vuote a all_pixels e all_lengths
            all_pixels.append([])
            all_lengths.append([])
        else:
            pixels = []
            lengths = []

            # Converto la RLE in una lista di interi
            rle = [int(x) for x in rle_string.split(' ')]

            for i, c in enumerate(rle):
                if i % 2 == 0:
                    pixels.append(int(c))
                else:
                    lengths.append(int(c))

            all_pixels.append(pixels)
            all_lengths.append(lengths)

    return all_pixels, all_lengths

def posizioni_pixel(pixels, larghezza):
    col = [px % larghezza for px in pixels]
    row = [px // larghezza for px in pixels]
    return row, col

# def crea_maschera_vuota(path, width, height):

#     # Creo una maschera vuota con le dimensioni dei pixel calcolate
#     mask = np.zeros((height, width, 3), dtype=np.uint8)

#     cv.imwrite(path, mask)
    
#     print(f"\nMaschera vuota di dimensione {width}x{height} creata in: {path}")

'''

def colora_maschera(mask_path, rows, cols, colors, lengths):
    
    mask = cv.imread(mask_path)
    
    for color in colors:
        
        color = np.array(color).reshape((1, 1, -1))
        
        for x, y, l in zip(cols, rows, lengths):
            mask[y:y+l, x:x+l] = color
        
    cv.imwrite(mask_path, mask)

'''
    
# def colora_maschera(mask_path, rows, cols, color, lengths):
#     # print(f"Rows is: {rows}\nCols is: {cols}\n")
    
#     for index, row in enumerate(rows):
#         col = cols[index]
        
#         if not row or not col:
#             continue
        
#         mask = cv.imread(mask_path)
        
#         color = np.array(color).reshape((1, 1, -1))
          
#         for x, y, l in zip(col, row, lengths):
#             mask[y:y+l, x:x+l] = color
#         cv.imwrite(mask_path, mask)        
    
    # color = np.array(color).reshape((1, 1, -1))
    
# def colora_maschera(mask_path, rows, cols, lengths):
    
#     maschera_vuota = cv.imread(mask_path)
    
#     for idx, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        
#         color = np.array(color).reshape((1, 1, -1))
        
#         for x, y, l in zip(cols[idx], rows[idx], lengths[idx]):
#             maschera_vuota[y:y+l, x:x+l] = color
    
#     cv.imwrite(mask_path, maschera_vuota)
#     print(f"\nSto colorando la maschera vuota in: {mask_path}")


pixels = []
lengths = []

for element in image_details['segmentation']:
    
    pixels_data, lengths_data = split_rle(element)
    
    pixels.append(pixels_data)
    lengths.append(lengths_data)

rows_img_all = []
cols_img_all = []

for pixel in pixels:
    
    rows = []
    cols = []
    
    for element in pixel:    
        row, col = posizioni_pixel(element, 266)
        rows.append(row)
        cols.append(col)
    
    rows_img_all.append(rows)
    cols_img_all.append(cols)
    
# print(rows_img_all[100])
# print('\n')
# print(cols_img_all[100])    

# Devo creare tutte le maschere vuote di dimensione 266x266

num_righe_image_details = image_details.shape[0]

#print(pixels[100])
print('\n')
#print(lengths[100])

#pixels, lenghts = split_rle_0(image_details['segmentation'])

image_details['is_created_mask'] = [False] * image_details.shape[0]
# image_details['is_created_mask'] = [True] * image_details.shape[0]

# print(image_details)    

#print(pixels[100][0])
#print('\n')

# print(lengths[100])
# print('\n')

# print(rows_img_all[100])
# print('\n')

# print(cols_img_all[100])
# print('\n')

colori = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])]

# for row in image_details.iterrows():

#     if row[1]['is_created_mask'] == False:
#         crea_maschera_vuota(row[1]['mask_path'], 266, 266)
#         row[1]['is_created_mask'] = True
#     else:
#         print("maschere vuote già create")
    
#     for i, color in enumerate(colori):
#         colora_maschera(row[1]['mask_path'], rows_img_all[row[0]],
#         cols_img_all[row[0]], color, lengths[row[0]][i])
    
    
    # for index, i in enumerate(rows_img_all):
    #     # print(i, cols_img_all[index])
    #     cols = cols_img_all[index]
    #     length = lengths[index]
        
    #     for j, element in enumerate(i):
    #         for color in colori:
    #             colora_maschera(row[1]['mask_path'], element, cols[j], color, length[j])
        
    
    # colora_maschera(row[1]['mask_path'], )
    
# mask_path, rows, cols, lengths

# for index, row in image_details.iterrows():
    
#     mask_path = row['mask_path']
#     print(f"\nWorking on {mask_path}\n")
    
#     if row['is_created_mask'] == False:
#             mop.crea_maschera_vuota(mask_path, 266, 266)
#             row['is_created_mask'] = True
#     else:
#         print("Maschere vuote già create")
    
#     mop.colora_maschera(mask_path, rows_img_all[index], cols_img_all[index], lengths[index])

data = {
    'Righe': rows_img_all,
    'Colonne': cols_img_all,
    'Lunghezze Run': lengths
}

row_cols_len_df = pd.DataFrame(data)

print(f"\n--- DATAFRAME RIGHE, COLONNE E LUNGHEZZE RUN ---\n{row_cols_len_df.head(5)}")

image_details_full_df = image_details.merge(row_cols_len_df, left_index=True, right_index=True)

print(f"\n--- DATAFRAME COMPLETO ---\n{image_details.head(5)}")

mop.genera_tutte_maschere(image_details_full_df)

# mop.genera_tutte_maschere(image_details, rows_img_all, cols_img_all, lengths)