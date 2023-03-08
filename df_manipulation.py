import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# BASE_DIR = "../BD-Image-Segmentation-Comp/" 
BASE_DIR = "BD-Image-Seg-Dataset/" 

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

cases = os.listdir(TRAIN_DIR)
#print(cases)

cases_id = [x.replace('case', '') for x in cases]
#print(cases)

cases_path = []
case_days = []

for case in cases:
    
    case_path = os.path.join(TRAIN_DIR, case)
    case_days = os.listdir(case_path)
    
    # ordino i giorni in ordine crescente
    case_days.sort()
    
    case_dict = {case_path: [os.path.join(case_path, day) for day in case_days]}
    cases_path.append(case_dict)

percorsi = []

for case_dict in cases_path:
    for case_path, day_paths in case_dict.items():
        case_name = os.path.basename(case_path)
        for day_path in day_paths:
            day_name = os.path.basename(day_path)
            percorsi.append([case_path, day_path+"/scans"])

headers = ["case_path", "scans_day_path"]

df = pd.DataFrame(percorsi, columns=headers)

df['case_id'] = df['case_path'].str.extract('case(\d+)', expand=False).astype(int)

# inserisco la colonna all'inizio
df.insert(0, 'case_id', df.pop('case_id'))

# estraggo il numero del giorno e creo una nuova colonna
df['day'] = df['scans_day_path'].str.extract('day(\d+)').astype(int)

# ordino in ordine crescente in base a case_id e day
df = df.sort_values(by=['case_id', 'day'], ascending=[True, True])

# inserisco la colonna day dopo case_id all'inizio
df.insert(1, 'day', df.pop('day'))

# distribuisco di nuovo gli indici del df
df = df.reset_index(drop=True)


# scans_per_day = []

# for _, row in df.iterrows():
#     case_path = row["case_path"]
#     scan_path = row["scans_day_path"]
    
#     scans = os.listdir(scan_path)
    
#     scans.sort()
    
#     scans_day_path = [os.path.join(scan_path, scan) for scan in scans]
    
#     scans_per_day.append(scans_day_path)
    
# df["scans"] = scans_per_day

# print(df)
# df.to_csv('aaa.csv', index=False)


# Loop attraverso ogni riga del DataFrame
for i, row in df.iterrows():
    
    # Ottiengo il percorso della directory
    scans_path = row['scans_day_path']
    
    # Ottiengo i nomi di tutti i file nella directory
    scans = os.listdir(scans_path)
    
    # Loop attraverso ogni nome di file e aggiungi una nuova riga al nuovo DataFrame
    for scan in scans:
        new_row = {'scans_name': scan, 'scans_day_path': scans_path}
        
# Visualizza il DataFrame aggiornato
print(df.head(10))
