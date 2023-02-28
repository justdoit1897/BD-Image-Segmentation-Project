# Sezione import
print("\n--- INIZIO DELLE IMPORT ---\n")

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize

print("\n--- FINE DELLE IMPORT ---\n")

print("\nDefinizione delle directories:")
BASE_DIR = 'fardella_tortorici/BD-Image-Seg-Dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
# Il file train.csv contiene i metadati utili alla segmentazione e alla classificazione
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

print(f"* Directory train.csv: {TRAIN_CSV}\n* Directory file di train: {TRAIN_DIR}\n")

# SEED impostato di default per garantire la riproducibilità della run
SEED = 42

# Come prima cosa, generiamo un dataframe dal file train.csv
train_df = pd.read_csv(TRAIN_CSV)

# DA USARE SE INSERITO NEL NOTEBOOK
# train_df.head(10)
print(f"\n--- Prime 10 righe di train.csv ---\n {train_df.head(10)}\n")

train_df.info()

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Funzione per modellare, ed estrarre, alcune informazioni preliminari
    dal dataframe ricavato dal training set. 
    
    Nello specifico, vengono estratte:
    
        - la classe per la segmentazione;
        
        - il numero del caso clinico;
        
        - il giorno di valutazione del caso;
        
        - un identificativo per il frame della scansione.

    Args:
        df (pd.DataFrame): il dataframe non processato

    Returns:
        pd.DataFrame: il datafrcase123_day20_slice_0001ame processato
    """
    
    df["segmentation"] = df["segmentation"].astype("str")
    df["case_id"] = df["id"].apply(lambda x: x.split("_")[0][4:])
    df["day_id"] = df["id"].apply(lambda x: x.split("_")[1][3:])
    df["slice_id"] = df["id"].apply(lambda x: x.split("_")[-1])
    
    return df
    
train_df = data_preprocessing(train_df)

# DA USARE SE INSERITO NEL NOTEBOOK
# train_df.head(10)
print(f"\n--- Prime 10 righe del dataframe processato ---\n {train_df.head(10)}\n")

def estrai_file_da_id(base_dir: str, case_id: str) -> str:
    """Funzione usata per ricavare il percorso del file di una scansione a partire
    da un id contenuto nel dataframe.

    Args:
        base_dir (str): percorso di partenza, da cui costruire la struttura delle cartelle
        case_id (str): identificativo del dataframe da cui ricavare il file

    Returns:
        str: percorso del file ricavato
    """
    
    # Ricaviamo la cartella del caso a partire dall'id salvato nel dataframe, formattato
    # sempre come "caseXYZ_dayDD_slice_SSSS"
    case_folder = case_id.split("_")[0]
    
    # In modo analogo, ricaviamo la cartella del giorno associato al caso appena estratto
    # e l'inizio del nome del file dello slice relativo all'id salvato nel dataframe
    day_folder = "_".join(case_id.split("_")[:2])
    file_starter = "_".join(case_id.split("_")[2:])
    
    # Generiamo, a partire dalle info estratte, il percorso delle scansioni 
    folder = os.path.join(base_dir, case_folder, day_folder, "scans")
    
    # Ricaviamo i file con un nome avente struttura simile (i.e. che iniziano nello stesso
    # modo)
    file = glob(f"{folder}/{file_starter}*")
    
    # Poiché glob genera una lista, siamo forzati a considerare il primo file in modo
    # esplicito, ma la glob restituirà sempre un solo file.
    file = file[0]
    
    return file

train_df["path"] = train_df["id"].apply(lambda x: estrai_file_da_id(TRAIN_DIR, x))

# DA USARE SE INSERITO NEL NOTEBOOK
# train_df.head(10)
print(f"\n--- Prime 10 righe del dataframe aumentato con percorso ---\n {train_df.head(10)}\n")

# A partire dai percorsi dei file estratti, possiamo accedere a nuove informazioni, come
# l'altezza e la larghezza delle immagini, che andiamo ad usare come ulteriori chiavi del
# dataframe.

train_df["height"] = train_df["path"].apply(lambda x: os.path.split(x)[-1].split("_")[2]).astype("int")
train_df["width"] = train_df["path"].apply(lambda x: os.path.split(x)[-1].split("_")[3]).astype("int")

# DA USARE SE INSERITO NEL NOTEBOOK
# train_df.head(10)
print(f"\n--- Prime 10 righe del dataframe aumentato con altezza e larghezza ---\n {train_df.head(10)}\n")

# Per la fase di classificazione, piuttosto che usare delle categorie, preferiamo passare
# ad etichette numeriche 
class_names = train_df["class"].unique()
print(f"\nClassi estratte dal problema:\n{class_names}")

# Andiamo a sostituire il valore della classe con l'indice corrispondente
# nella lista di etichette appena estratta
for index, label in enumerate(class_names):
    train_df["class"].replace(label, index, inplace = True)

# DA USARE SE INSERITO NEL NOTEBOOK
# train_df.head(10)
print(f"\n--- Prime 10 righe del dataframe dopo aver sostituito le classi ---\n {train_df.head(10)}\n")
