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

train_df = train_df.groupby('id').agg({'class': ','.join, 'segmentation': ','.join})
print(train_df)