# Imports

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import modules.mask_ops as mop

from tqdm import tqdm
from itertools import zip_longest 

# Consts

BASE_DIR = "../BD-Image-Segmentation-Comp/" 

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

########################################### PARTE 1 - OPERAZIONI PRELIMINARI ###########################################

# Definizione di un dataframe a partire da train.csv e presentazione del contenuto
train_df = pd.read_csv(TRAIN_CSV)
print(train_df.head())
# CODICE PER NOTEBOOK
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

# Presentazione del grafico
#plt.show()

# 1.2 - Calcolo istanze con segmentazione non nulla: l'obiettivo qui è evidenziare quante e quali
# righe del dataframe contengono informazioni circa la segmentazione

# Si filtra il dataframe eliminando le righe con valori mancanti nella colonna "segmentation" (i.e. 'nan')
df_filtered = train_df.dropna(subset=['segmentation'])

# Viene effettuato il conteggio delle istanze per ogni classe, essendo certi che il DataFrame filtrato non
# contiene valori nulli
class_counts = df_filtered['class'].value_counts()

# Creazione del grafico a barre con colori personalizzati per ogni classe
colors = ['red', 'green', 'blue']
plt.bar(class_counts.index, class_counts.values, color=[colors[i] for i in range(len(class_counts))])

# Aggiunta delle etichette dell'asse x e y e del titolo del grafico
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Number of Instances per Class (with non-null segmentation)')

# Presentazione del grafico
#plt.show()

# ######################################## COMMENTO ########################################
# Abbiamo creato un DataFrame di esempio con alcune righe che hanno valori mancanti nella colonna 
# "segmentation" e altre righe con una segmentazione definita. Abbiamo quindi utilizzato il metodo 
# dropna() del DataFrame per eliminare le righe con valori mancanti nella colonna "segmentation" e 
# creare un nuovo DataFrame "df_filtered" solo con le righe rimanenti.

# Successivamente, abbiamo utilizzato il DataFrame filtrato "df_filtered" per calcolare il numero 
# di istanze per ogni classe nella colonna "class", utilizzando la funzione value_counts() come nel 
# precedente esempio. Infine, abbiamo creato un grafico a barre con i risultati e abbiamo aggiunto 
# le etichette dell'asse x, dell'asse y e del titolo del grafico.
# ##########################################################################################

# 1.3 - Espansione e ordinamento del dataframe: l'obiettivo è quello di ricavare informazioni aggiuntive
# dal dataframe di partenza, oltre che di effettuare un migliore ordinamento delle righe dello stesso

# Sostituzione delle classi con valori numerici per una migliore leggibilità
class_mapping = {'large_bowel': 0, 'small_bowel': 1, 'stomach': 2}
train_df['class'] = train_df['class'].replace(class_mapping)

# A partire dall'id di ogni riga, è possibile estrarre informazioni sull'identificativo del caso clinico,
# del giorno di osservazione e dello specifico frame della scansione
splits = train_df['id'].str.split("_", n = 4, expand = True)
print(splits)

train_df['case_id'] = splits[0]
train_df['day_id'] = splits[1]
train_df['slice_id'] = splits[3].astype(int)

train_df['case_id'] = train_df['case_id'].str.replace('case', '').astype(int) 
train_df['day_id'] = train_df['day_id'].str.replace('day', '').astype(int)

# Ciò che si vuole ottenere, adesso, è un dataframe compattato, in cui ogni riga associata a un caso/giorno/slice,
# contenga le informazioni relative alle tre etichette del problema in un'unica struttura dati.

# Si eliminano le colonne "class" e "segmentation" e se ne salva il contenuto in due liste
classe = train_df.pop('class')
segmentation = train_df.pop('segmentation')

# Si raggruppano gli elementi delle liste a tre a tre
grouped_segmentation = list(zip_longest(*[iter(segmentation)]*3, fillvalue=None))
grouped_class = list(zip_longest(*[iter(classe)]*3, fillvalue=None))

# Si eliminano i duplicati per la colonna "id"
train_df = train_df.drop_duplicates(subset=['id'])

# Le colonne processate in precedenza possono essere, quindi, appese al dataframe (in posizione finale), ottenendo
# un dataframe compattato nel numero di righe
train_df.insert(len(train_df.columns), 'class', grouped_class)
train_df.insert(len(train_df.columns), 'segmentation', grouped_segmentation)

# Viene dato un ordinamento per valori crescenti in di case_id, day_id e slice_id
train_df = train_df.sort_values(by=['case_id', 'day_id', 'slice_id'], ascending=True).reset_index(drop=True)

print(train_df.head())
# CODICE NOTEBOOK
# train_df.head()

########################################### PARTE 2 - STATISTICHE IMMAGINI ###########################################
# 2.1 - Individuazione percorsi e aumento dimensioni dataframe: l'obiettivo qui è associare ogni riga del dataframe ad
# un'immagine del dataset, da cui estrarre, in un secondo momento, delle informazioni utili per eventuali valutazioni
# statistiche.

list_slices = glob.glob(TRAIN_DIR+'/*/*/scans/*.png')
#print(list_slices)

# Si inizializza un dataframe sulla base dei percorsi individuati dalla glob, per poi procedere all'estrazione di
# informazioni associate a caso clinico, giorno, scansione (e tanti altri) direttamente dai file .png del dataset
image_details = pd.DataFrame({'path':list_slices})

splits = image_details['path'].str.split("/", n = 7, expand = True)

# case_id e day_id
image_details[['case_id', 'day_id']] = splits[4].str.split("_", expand = True)
image_details['case_id'] = image_details['case_id'].str.replace('case', '').astype(int) 
image_details['day_id'] = image_details['day_id'].str.replace('day', '').astype(int)

# slice id
image_details['slice_name'] = splits[6]
slice_info = image_details['slice_name'].str.split(n=6, expand=True, pat="_")
image_details['slice_id'] = slice_info[1].astype(int)

# dimensioni
image_details['width'] = slice_info[2].astype(int)
image_details['height'] = slice_info[3].astype(int)

# dimensioni dei pixel
image_details['width_px'] = slice_info[4].astype(float)
#.round(2).apply(lambda x: '{:.2f}'.format(x))
image_details['height_px'] = slice_info[5].str.replace('.png', '', regex=False).astype(float)

# Per funzioni implementate in seguito, si rende necessario definire i percorsi in cui si desidera
# salvare le maschere associate a un caso

# Vengono creati i percorsi che conterranno le maschere (sullo stesso livello della cartella scans)

splits[5] = splits[5].str.replace('scans', 'masks')
splits[6] = splits[6].str.replace('slice', 'mask_slice')

percorsi_cartelle = splits.drop(columns=[6])
percorsi_cartelle = percorsi_cartelle.apply(lambda x: '/'.join(x.astype(str)), axis=1)

percorsi_maschere = splits.apply(lambda x: '/'.join(x.astype(str)), axis=1)

# Viene inserita nel dataframe una dimensione contenente tutti i percorsi appena individuati
image_details.insert(1, 'mask_path', percorsi_maschere)

for path in percorsi_cartelle:
    # Creo la nuova cartella se non esiste già
    if not os.path.exists(path):
        os.mkdir(path) 
        print("\nCartella creata con successo!\n")

# Si procede all'ordinamento crescente dei valori di case_id, day_id e slice_id del dataframe
image_details = image_details.sort_values(by=['case_id', 'day_id', 'slice_id'], ascending=True).reset_index(drop=True)

# aggiungo al df la riga contenente le maschere di segmentazione
image_details.insert(len(image_details.columns), 'segmentation', grouped_segmentation)
print("\n-------------------------------------------------------------------- image_details_merged_ --------------------------------------------------------------------\n")
print(image_details.head())
# Codice NOTEBOOK
# image_details.head()

# Per evitare il ricalcolo ad ogni esecuzione del codice, si salva il dataframe in un opportuno .csv
image_details.to_csv('image_details_merged.csv', index=False)

# image_details = pd.read_csv('image_details_merged.csv')

# 2.2 - Analisi statistica delle immagini: a seguito di ispezione visiva di alcuni campioni, si è cercato di indagare
# su alcune caratteristiche salienti di ogni immagine. L'obiettivo qui è di individuare correlazioni tra la dimensione
# dei pixel e la qualità delle immagini, ecc.

# Si cerca di capire, innanzitutto, quante immagini hanno dimensione del pixel [1.50 mm x 1.50 mm] e quante [1.63 mm x 1.63 mm]
num_images_150 = image_details.loc[(image_details['width_px'] == 1.50) & (image_details['height_px'] == 1.50)].shape[0]
print("Il numero di immagini con width_px e height_px pari a 1.50 mm è:", num_images_150)

num_images_163 = image_details.loc[(image_details['width_px'] == 1.63) & (image_details['height_px'] == 1.63)].shape[0]
print("Il numero di immagini con width_px e height_px pari a 1.63 mm è:", num_images_163)

# Si prova che non vi siano immagini caratterizzate da ulteriori dimensioni dei pixel
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
# plt.show()

# Si cerca di capire, adesso, se le immagini con una data profondità sono tutte delle stesse dimensioni

widths_px_1_50 = image_details[(image_details['width_px'] == 1.5) & (image_details['height_px'] == 1.5)]['width'].unique()
print(widths_px_1_50)

heights_px_1_50 = image_details[(image_details['width_px'] == 1.5) & (image_details['height_px'] == 1.5)]['height'].unique()
print(heights_px_1_50)

widths_px_1_63 = image_details[(image_details['width_px'] == 1.63) & (image_details['height_px'] == 1.63)]['width'].unique()
print(widths_px_1_63)

heights_px_1_63 = image_details[(image_details['width_px'] == 1.63) & (image_details['height_px'] == 1.63)]['height'].unique()
print(heights_px_1_63)

# Eseguendo il codice sovrastante, si evince come solo le slice con profondità dei pixel pari a 1.63 mm hanno tutte la stessa dimensione

# VALUTAZIONE DIMENSIONI DELLE IMMAGINI
# Adesso si vuole vedere se le immagini con larghezza pari a 266 px hanno altezza variabile e viceversa

heights_266 = image_details[(image_details['width'] == 266)]['height'].unique()
print(heights_266)

widths_266 = image_details[(image_details['height'] == 266)]['width'].unique()
print(widths_266)

# Viene calcolato il numero di immagini con:
# - Larghezza 266 px
num_images_266_width = image_details.loc[(image_details['width'] == 266)].shape[0]
print("Il numero di immagini con larghezza 266 px è:", num_images_266_width)

# - Altezza 266 px
num_images_266_height = image_details.loc[(image_details['height'] == 266)].shape[0]
print("Il numero di immagini con altezza 266 px è:", num_images_266_height)

# Combinando i risultati, si evince che una dimensione a 266 px è associata a immagini quadrate
num_images_266x266 = image_details.loc[(image_details['width'] == 266) & (image_details['height'] == 266)].shape[0]
print("Il numero di immagini 266x266 è:", num_images_266x266)

# Si ripete il processo con le immagini larghe 234 px
heights_234 = image_details[(image_details['width'] == 234)]['height'].unique()
print(heights_234)

widths_234 = image_details[(image_details['height'] == 234)]['width'].unique()
print(widths_234)

# Viene calcolato il numero di immagini con:
# - Larghezza 234 px
num_images_234_width = image_details.loc[(image_details['width'] == 234)].shape[0]
print("Il numero di immagini con larghezza 234 px è:", num_images_234_width)

# - Altezza 234 px
num_images_234_height = image_details.loc[(image_details['height'] == 234)].shape[0]
print("Il numero di immagini con altezza 234 px è:", num_images_234_height)

# Combinando i risultati, si evince che una dimensione a 234 px è associata a immagini quadrate
num_images_234x234 = image_details.loc[(image_details['width'] == 234) & (image_details['height'] == 234)].shape[0]
print("Il numero di immagini 234x234 è:", num_images_234x234)

# Si completa adesso l'analisi, individuando i valori associati ad altezze di 310 px o larghezze di 360 px

heights_360 = image_details[(image_details['width'] == 360)]['height'].unique()
print(heights_360)


widths_310 = image_details[(image_details['height'] == 310)]['width'].unique()
print(widths_310)

# Viene calcolato il numero di immagini con:
# - Larghezza 360 px
num_images_360_width = image_details.loc[(image_details['width'] == 360)].shape[0]
print("Il numero di immagini con larghezza 360 px è:", num_images_360_width)

# - Altezza 310 px
num_images_310_height = image_details.loc[(image_details['height'] == 310)].shape[0]
print("Il numero di immagini con altezza 310 px è:", num_images_310_height)

# Combinando i risultati, si evince che le immagini larghe 360 px hanno tutte la stessa altezza (310 px)
num_images_360x310 = image_details.loc[(image_details['width'] == 360) & (image_details['height'] == 310)].shape[0]
print("Il numero di immagini 360x310 è:", num_images_360x310)

tot_slice = num_images_266x266 + num_images_234x234 + num_images_360x310

print("totale slice profondità pixel 1.50 mm: ", tot_slice)

# Viene realizzato un grafico dal quale si evince, in modo intuitivo, quanto detto in maniera qualitativa

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

# STATISTICA SEGMENTAZIONE
# Dal dataframe "image_details" vengono estratte le righe le cui slice hanno dimensione 234x234
image_details_234x234 = image_details[(image_details['width'] == 234) & (image_details['height'] == 234)].reset_index().drop(columns='index')
print("\n-------------------------------------------------------------------- image_details_234x234 --------------------------------------------------------------------\n")
print(image_details_234x234.head())

# CODICE NOTEBOOK
# image_details_234x234.head()

# Per evitare di eseguire ogni volta il codice, il dataframe filtrato viene salvato in locale
image_details_234x234.to_csv('image_details_234x234.csv', index=False)

# image_details_234x234 = pd.read_csv('image_details_234x234.csv')

# L'obiettivo adesso è di contare il numero di righe con colonna "segmentation" diversa da "(nan, nan, nan)"
# Si inizia individuando il numero di righe in cui 'segmentation' registra un valore di '(nan, nan, nan)'
image_details_234x234_w_seg = image_details_234x234.merge(train_df, on=['case_id', 'day_id', 'slice_id']).drop_duplicates()

count_234 = 0
for i in range(len(image_details_234x234_w_seg)):
    is_nan = True
    for j in range(len(image_details_234x234_w_seg['segmentation'][i])):
        if not pd.isna(image_details_234x234_w_seg['segmentation'][i][j]):
            is_nan = False
            break
    if is_nan:
        count_234 += 1

# Questo ciclo for scorre ogni elemento della colonna "segmentation", controllando se tutti i valori sono NaN. 
# Se tutti i valori sono NaN, il contatore delle righe viene incrementato. 

print(f"\nIl numero di righe con colonna 'segmentation' diversa da (nan, nan, nan) è {image_details_234x234_w_seg.shape[0] - count_234}.")
print(f"\nIl numero delle righe totali è {image_details_234x234.shape[0]}.")
print(f"\nCi sono esattamente {count_234} slice di dimensione 234x234 con maschere totalmente vuote.")
print(f"\nLa percentuale di slice di dimensione 234x234 con maschere totalmente vuote è pari al {round(count_234/image_details_234x234_w_seg.shape[0]*100, 4)}%")
 
# Viene reiterato il procedeimento per le righe del dataframe 'image_details' le cui slice hanno dimensione 266 x 266
image_details_266x266 = image_details[(image_details['width'] == 266) & (image_details['height'] == 266)].reset_index().drop(columns='index')
print("\n-------------------------------------------------------------------- image_details_266x266 --------------------------------------------------------------------\n")
print(image_details_266x266.head())

# CODICE NOTEBOOK
# image_details_266x266.head()

# Per evitare di eseguire ogni volta il codice, il dataframe filtrato viene salvato in locale
image_details_266x266.to_csv('image_details_266x266.csv', index=False)

# image_details_266x266 = pd.read_csv('image_details_266x266.csv')

image_details_266x266_w_seg = image_details_266x266.merge(train_df, on=['case_id', 'day_id', 'slice_id']).drop_duplicates()

count_266 = 0
for i in range(len(image_details_266x266_w_seg)):
    is_nan = True
    for j in range(len(image_details_266x266_w_seg['segmentation'][i])):
        if not pd.isna(image_details_266x266_w_seg['segmentation'][i][j]):
            is_nan = False
            break
    if is_nan:
        count_266 += 1

print(f"\nIl numero di righe con colonna 'segmentation' diversa da (nan, nan, nan) è {image_details_266x266_w_seg.shape[0] - count_266}.")
print(f"\nIl numero delle righe totali è {image_details_266x266_w_seg.shape[0]}.")
print(f"\nCi sono esattamente {count_266} slice di dimensione 266x266 con maschere totalmente vuote.")
print(f"\nLa percentuale di slice di dimensione 266x266 con maschere totalmente vuote è pari al {round(count_266/image_details_266x266_w_seg.shape[0]*100, 4)}%")

# Viene reiterato il procedeimento per le righe del dataframe 'image_details' le cui slice hanno dimensione 276 x 276
image_details_276x276 = image_details[(image_details['width'] == 276) & (image_details['height'] == 276)].reset_index().drop(columns='index')
print("\n-------------------------------------------------------------------- image_details_276x276 --------------------------------------------------------------------\n")
print(image_details_276x276.head())

# CODICE NOTEBOOK
# image_details_276x276.head()

# Per evitare di eseguire ogni volta il codice, il dataframe filtrato viene salvato in locale
image_details_276x276.to_csv('image_details_276x276.csv', index=False)

# image_details_276x276 = pd.read_csv('image_details_276x276.csv')

image_details_276x276_w_seg = image_details_276x276.merge(train_df, on=['case_id', 'day_id', 'slice_id']).drop_duplicates()

count_276 = 0
for i in range(len(image_details_276x276_w_seg)):
    is_nan = True
    for j in range(len(image_details_276x276_w_seg['segmentation'][i])):
        if not pd.isna(image_details_276x276_w_seg['segmentation'][i][j]):
            is_nan = False
            break
    if is_nan:
        count_276 += 1

print(f"\nIl numero di righe con colonna 'segmentation' diversa da (nan, nan, nan) è {image_details_276x276_w_seg.shape[0] - count_276}.")
print(f"\nIl numero delle righe totali è {image_details_276x276_w_seg.shape[0]}.")
print(f"\nCi sono esattamente {count_276} slice di dimensione 276x276 con maschere totalmente vuote.")
print(f"\nLa percentuale di slice di dimensione 276x276 con maschere totalmente vuote è pari al {round(count_276/image_details_276x276_w_seg.shape[0]*100)}%")

# Viene reiterato il procedeimento per le righe del dataframe 'image_details' le cui slice hanno dimensione 360 x 310
image_details_360x310 = image_details[(image_details['width'] == 360) & (image_details['height'] == 310)].reset_index().drop(columns='index')
print("\n-------------------------------------------------------------------- image_details_360x310 --------------------------------------------------------------------\n")
print(image_details_360x310.head())

# CODICE NOTEBOOK
# image_details_360x310.head()

# Per evitare di eseguire ogni volta il codice, il dataframe filtrato viene salvato in locale
image_details_360x310.to_csv('image_details_360x310.csv', index=False)

# image_details_360x310 = pd.read_csv('image_details_360x310.csv')

image_details_360x310_w_seg = image_details_360x310.merge(train_df, on=['case_id', 'day_id', 'slice_id']).drop_duplicates()

count_360 = 0
for i in range(len(image_details_360x310_w_seg)):
    is_nan = True
    for j in range(len(image_details_360x310_w_seg['segmentation'][i])):
        if not pd.isna(image_details_360x310_w_seg['segmentation'][i][j]):
            is_nan = False
            break
    if is_nan:
        count_360 += 1

print(f"\nIl numero di righe con colonna 'segmentation' diversa da (nan, nan, nan) è {image_details_360x310_w_seg.shape[0] - count_360}.")
print(f"\nIl numero delle righe totali è {image_details_360x310_w_seg.shape[0]}.")
print(f"\nCi sono esattamente {count_360} slice di dimensione 360x310 con maschere totalmente vuote.")
print(f"\nLa percentuale di slice di dimensione 360x310 con maschere totalmente vuote è pari al {round(count_360/image_details_360x310_w_seg.shape[0]*100)}%")

########################################### PARTE 3 ###########################################
'''
# Dati per il grafico
labels = ['234x234', '266x266', '276x276', '360x310']
counters1 = [count_234, count_266, count_276, count_360]
counters2 = [segmented_234, segmented_266, segmented_276, segmented_360]
counters3 = [(count_234 + segmented_234), (count_266 + segmented_266), (count_276 + segmented_276), (count_360 + segmented_360)]
lower_bound = 0
upper_bound = count_266 + segmented_266 + 1000

# Colori per le barre
colors = ['red', 'green', 'blue', 'orange']

# Creazione del grafico a barre
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.bar(labels, counters1, color=colors)
ax2.bar(labels, counters2, color=colors)
ax3.bar(labels, counters3, color=colors)

# Aggiunta di titoli, label e scala
ax1.set_title('N° di maschere vuote per dimensione slice')
ax1.set_ylabel('Numero di slice')
ax1.set_ybound(lower_bound, upper_bound)
ax2.set_title('N° di segmentazioni per dimensione slice')
ax2.set_xlabel('Larghezza e altezza slice (in px)')
ax2.set_ybound(lower_bound, upper_bound)
ax3.set_title('Totale')
ax3.set_ybound(lower_bound, upper_bound)

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


# def colora_maschera(mask_path, rows, cols, colors, lengths):
    
#     mask = cv.imread(mask_path)
    
#     for color in colors:
        
#         color = np.array(color).reshape((1, 1, -1))
        
#         for x, y, l in zip(cols, rows, lengths):
#             mask[y:y+l, x:x+l] = color
        
#     cv.imwrite(mask_path, mask)

 
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

# print(pixels[100])
# print('\n')
# print(lengths[100])

#pixels, lenghts = split_rle_0(image_details['segmentation'])

data = {
    'Righe': rows_img_all,
    'Colonne': cols_img_all,
    'Lunghezze Run': lengths
}

row_cols_len_df = pd.DataFrame(data)

print(f"\n--- DATAFRAME RIGHE, COLONNE E LUNGHEZZE RUN ---\n{row_cols_len_df.head(5)}")

image_details_full_df = image_details.merge(row_cols_len_df, left_index=True, right_index=True)

print(f"\n--- DATAFRAME COMPLETO ---\n{image_details.head(5)}")

#print(image_details_360x310['segmentation'][101])

#mop.genera_tutte_maschere(image_details_full_df)

# mop.genera_tutte_maschere(image_details, rows_img_all, cols_img_all, lengths)

###################################### PARTE 6 ######################################

# Voglio equalizzare le immagini e sovrascriverle nei percorsi di origine

'''

La funzione equalizza_immagini legge un'immagine in scala di grigi dal percorso specificato, 
ne equalizza l'istogramma e sovrascrive l'immagine originale con quella equalizzata. 
Infine, viene stampato un messaggio che indica che l'immagine equalizzata è stata salvata 
al percorso specificato. 

'''

def equalizza_immagini(path: str):
    
    """Funzione usata per effettuare l'equalizzazione dell'istogramma (con conseguente
    sovrascrittura) dell'immagine in un dato percorso

    Args:
        path (str): percorso in cui salvare l'immagine
    """
    
    # Leggo l'immagine situata in 'path' in scala di grigi
    
    img = cv.imread(path, 0) 

    equ = cv.equalizeHist(img)
    
    cv.imwrite(path, equ)
    
    #print(f"\nEqualizzazione immagine in: {path}")

image_details['is_equalized'] = [True] * image_details.shape[0]
# image_details['is_equalized'] = [False] * image_details.shape[0]    

for index, row in image_details.iterrows():
    
    if row['is_equalized'] == False:
        print(f"Inizio equalizzazione...\n")    
        
        for index, row in tqdm(image_details.iterrows(), total=len(image_details)):
            equalizza_immagini(row['path'])
        
        print(f"\nFine equalizzazione.\n")
        row['is_equalized'] = True
        
    else:
        print("Immagini già equalizzate")
        break


###################################### PARTE 7 ######################################

# def colora_maschera(mask_path, rows, cols, colors, lengths):
    
#     mask = cv.imread(mask_path)
    
#     for color in colors:
        
#         color = np.array(color).reshape((1, 1, -1))
        
#         for x, y, l in zip(cols, rows, lengths):
#             mask[y:y+l, x:x+l] = color
        
#     cv.imwrite(mask_path, mask)


# def colora_maschera(rle_encodings: list[str], height: int, width: int) -> np.ndarray:
    
#     # segmentation, shape (altezza, larghezza)
    
#     # Generiamo un numpy array (inizialmente appiattito)
#     mask_array = np.zeros((height, width, 3), dtype=np.uint8)
    
#     color = 0
    
#     for index, encoding in enumerate(rle_encodings):
        
#         # Piccola conversione per le codifiche in nan
#         if encoding is np.nan:
#             continue
            
#         if index == 0:
#             color = np.array([255, 0, 0]) # R
#         elif index == 1:
#             color = np.array([0, 255, 0])   # G
#         else:
#             color = np.array([0, 0, 255])   # B
        
#         # Genero una lista di numeri per ogni elemento di rle_encodings
#         segm = np.asarray(encoding.split(), dtype=int)
        
#         # Get start point and length between points
#         start_point = segm[0::2] - 1
#         length_point = segm[1::2]
        
#         # Compute the location of each endpoint
#         end_point = start_point + length_point
        
#         for start, end in zip(start_point, end_point):
#             mask_array[start:end] = 1
        
#     mask_array = mask_array.reshape(height, width, 3)
    
#     return mask_array

# def colora_maschera(rle_encodings: list[str], height: int, width: int) -> np.ndarray:
    
#     # Generiamo un numpy array (inizialmente appiattito)
#     mask_array = np.zeros((height, width, 3), dtype=np.uint8)
    
#     for index, encoding in enumerate(rle_encodings):
        
#         # Piccola conversione per le codifiche in nan
#         if encoding is np.nan:
#             continue
            
#         # Genero una lista di numeri per ogni elemento di rle_encodings
#         segm = np.asarray(encoding.split(), dtype=int)
        
#         # Get start point and length between points
#         start_point = segm[0::2] - 1
#         length_point = segm[1::2]
        
#         # Compute the location of each endpoint
#         end_point = start_point + length_point
        
#         # Coloriamo la maschera
#         if index == 0:
#             color = (255, 0, 0)  # rosso
#         elif index == 1:
#             color = (0, 255, 0)  # verde
#         else:
#             color = (0, 0, 255)  # blu
        
#         print(color)
#         print(color[index])
        
#         for start, end in zip(start_point, end_point):
#             mask_array[start:end, :, index] = color[index]
            
#     return mask_array


image_details['is_created_mask'] = [True] * image_details.shape[0]
# image_details['is_created_mask'] = [False] * image_details.shape[0]

print(f"Inizio creazione maschere vuote...\n")    

for index, row in tqdm(image_details.iterrows(), total=len(image_details)):
    
    mask_path = row['mask_path']
    
    if row['is_created_mask'] == False:
        
        mop.crea_maschera_vuota(mask_path, row['height'], row['width'])
        
        row['is_created_mask'] = True
        
    else:
        #print("Maschere vuote già create")
        continue

print(f"\nFine creazione maschere vuote.\n")

print(f"Inizio colorazione maschere vuote...\n")   

for index, row in tqdm(image_details.iterrows(), total=len(image_details)):
    
    mask_path = row['mask_path']
    
    if row['is_created_mask'] == True:
        
        red_segment = mop.rle_to_image(row['segmentation'][0], row['height'], row['width'])
        green_segment = mop.rle_to_image(row['segmentation'][1], row['height'], row['width'])
        blue_segment = mop.rle_to_image(row['segmentation'][2], row['height'], row['width'])

        merged_mask = cv.merge([red_segment, green_segment, blue_segment])
        
        cv.imwrite(mask_path, merged_mask)  # Save merged mask to mask_path
        
    else:
        print("Maschera vuota inesistente.")
        continue
    
print(f"\nFine colorazione maschere vuote.\n")
  
    
####################################################################




