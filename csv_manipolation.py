import os
import pandas as pd

BASE_DIR = "../BD-Image-Segmentation-Comp/" 
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

def sort_list(lst):
    
    # Suddivido ogni elemento in una lista di parti basata sul carattere di separazione '_'
    lst_parts = [elem.split('_') for elem in lst]

    # Ordino la lista di parti in base ai criteri desiderati
    lst_parts.sort(key=lambda x: (int(x[0][4:]), int(x[1][3:]), int(x[3])))

    # Ricompongo ogni lista in una stringa usando la funzione join
    return ['_'.join(parts) for parts in lst_parts]

# Leggo il file CSV
df = pd.read_csv(os.path.join(TRAIN_CSV))

# Sostituisco le classi con valori numerici per una miglior leggibilità
class_mapping = {'large_bowel': 0, 'small_bowel': 1, 'stomach': 2}

df['class'] = df['class'].replace(class_mapping)

# Converto gli elementi della colonna "segmentation" in stringhe
df['segmentation'] = df['segmentation'].astype(str)

# Raggruppamento delle righe in base all'id e combinazione dei valori delle colonne "class" e "segmentation" in una lista
grouped = df.groupby('id').agg({'class': list, 'segmentation': list})

# Reset dell'indice per ottenere un nuovo DataFrame
compact_df = grouped.reset_index()

df['id'] = sort_list(df['id'].tolist())

compact_df['id'] = sort_list(compact_df['id'].tolist())

# Creo tre nuove colonne
df[['case_id', 'day', 'id_slice']] = df['id'].str.split('_', expand=True)[[0, 1, 3]]

compact_df[['case_id', 'day', 'id_slice']] = compact_df['id'].str.split('_', expand=True)[[0, 1, 3]]

# Ordino la colonna "case_id" in ordine crescente e la formatto in modo da avere solo l'id del caso
df['case_id'] = df['case_id'].str.replace('case', '')

compact_df['case_id'] = compact_df['case_id'].str.replace('case', '')

# Stessa cosa faccio per la colonna "day"
df['day'] = df['day'].str.replace('day', '') 

compact_df['day'] = compact_df['day'].str.replace('day', '')

# Rimuovo la colonna 'id' originale
df = df.drop('id', axis=1)

compact_df = compact_df.drop('id', axis=1)

# Sposto le colonne appena create all'inizio
df = df.reindex(columns=['case_id', 'day', 'id_slice'] + [col for col in df.columns if col not in ['case_id', 'day', 'id_slice']])

compact_df = compact_df.reindex(columns=['case_id', 'day', 'id_slice'] + [col for col in compact_df.columns if col not in ['case_id', 'day', 'id_slice']])

# Stampo il dataframe
print(df)
#df.to_csv('df_ordinato.csv', index=False)

# Stampo il dataframe compatto
print(compact_df)
#compact_df.to_csv('df_ordinato_compattato.csv', index=False)

compact_df['case_id'] = compact_df['case_id'].astype(int)
compact_df['day'] = compact_df['day'].astype(int)
compact_df['id_slice'] = compact_df['id_slice'].astype(int)

order_by_slice = compact_df.sort_values(by=['case_id', 'id_slice'])

order_by_slice = order_by_slice.sort_values(by=['case_id', 'id_slice'], ascending=[True, True]).reset_index()

order_by_slice = order_by_slice.drop('index', axis=1)

print(order_by_slice)

# raggruppa per case_id e conta quanti giorni ci sono per ciascun case_id
days_per_case = df.groupby('case_id')['day'].nunique().reset_index()

days_per_case['case_id'] = days_per_case['case_id'].astype(int)

days_per_case = days_per_case.sort_values(by='case_id', ascending=True)

grouped = days_per_case.groupby('day')['case_id'].apply(list).reset_index(name='cases_per_day')

print(days_per_case)
print(grouped)

#print(df_filtrato)

#lista = compact_df['segmentation'].tolist()

#from itertools import chain

#unique_segments = set(chain.from_iterable(compact_df['segmentation']))
#filtered_segments = list(filter(lambda x: x != ['nan', 'nan', 'nan'], unique_segments))


order_by_slice = order_by_slice.drop(order_by_slice[order_by_slice['segmentation'].apply(lambda x: x== ['nan', 'nan', 'nan'])].index)

print(order_by_slice)
