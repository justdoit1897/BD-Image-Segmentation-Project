import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import glob

BASE_DIR = "../BD-Image-Segmentation-Comp/" 
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

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
plt.show()

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
plt.show()

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

list_slices = glob.glob(TRAIN_DIR+'/*/*/scans/*.png')
#print(list_slices)

image_details = pd.DataFrame({'path':list_slices})

splits = image_details['path'].str.split("/", n = 7, expand = True)

image_details[['case_id', 'day_id']] = splits[4].str.split("_", expand = True)

image_details['case_id'] = image_details['case_id'].str.replace('case', '').astype(int) 
image_details['day_id'] = image_details['day_id'].str.replace('day', '').astype(int)

image_details['slice_name'] = splits[6]

slice_info = image_details['slice_name'].str.split(n=6, expand=True, pat="_")

image_details['slice_id'] = slice_info[1]

image_details['width'] = slice_info[2].astype(int)
image_details['height'] = slice_info[3].astype(int)

image_details['width_px'] = slice_info[4].astype(float)
#.round(2).apply(lambda x: '{:.2f}'.format(x))
image_details['height_px'] = slice_info[5].str.replace('.png', '', regex=False).astype(float)

# ordino in ordine crescente in base a case_id e day_id
image_details = image_details.sort_values(by=['case_id', 'day_id'], ascending=[True, True])

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
plt.show()

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
        
        ###############################################

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
        
        ###############################################
        
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
fig = plt.figure(figsize=(10, 5))
plt.bar(labels, num_slices, color=colors)

# Aggiunta di titoli e label
plt.title('Numero di slice per larghezza e altezza dei pixel con profondità 1.50 mm')
plt.xlabel('Larghezza e altezza pixel (in px)')
plt.ylabel('Numero di slice')

# Visualizzazione del grafico
plt.show()