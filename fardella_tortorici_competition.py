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

# Dobbiamo sfruttare, adesso, la codifica RLE delle maschere, quindi
# implementiamo delle funzioni per:
#   - estrarre le informazioni di codifica RLE per una singola immagine
#   - definire i pixel sotto la maschera di una singola immagine
#   - preparare le maschere

def prepare_mask_data(string: str) -> tuple[list[int], list[int]]:
    """Funzione usata per estrarre i dati utili alla definizione di una maschera
    a partire da una stringa

    Args:
        string (str): la stringa da cui estrapolare la maschera

    Returns:
        tuple[list[int], list[int]]: liste dei pixel di inizio delle run e delle
        corrispettive lunghezze
    """
    
    # Data una stringa passata in input, sfruttiamo il fatto che gli elementi
    # sono separati da spazi 
    all_values = map(int, string.split(" "))
    
    # Definiamo le due liste contenenti i pixel di inizio della run
    # e le lunghezze delle run
    starterIndex, pixelCount = [], []
    
    for index, value in enumerate(all_values):
        
        # Sfruttiamo il fatto che la RLE presenta, come valore di indice
        # pari, la lunghezza di una run e, come valore di indice dispari,
        # il pixel da cui ha inizio la run.
        if index % 2:
            
            # i valori pari vanno in pixelCount
            pixelCount.append(value)
        else:
            
            # i valori dispari vanno in starterIndex
            starterIndex.append(value)
            
    return starterIndex, pixelCount
    
def indici_posizione_pixel(indexes: list[int], counts: list[int]) -> list:
    """Funzione per determinare,in modo globale, tutti i pixel coperti da una maschera

    Args:
        indexes (list[int]): lista di pixel da cui inizia la run di una RLE
        counts (list[int]): lista di lunghezze delle run di una RLE

    Returns:
        list: lista dei pixel coperti dalla maschera
    """
    
    # Definiamo una lista da riempire coi pixel dell'immagine che sono coperti da una maschera
    final_arr = []
    
    for index, counts in zip(indexes, counts):
        # Incrementiamo la lista con il numero specifico dei pixel coperti dalla maschera
        # (es. starterIndex[i] = 10, pixelCount[i] = 20 => verranno coperti i pixel in 10...30)
        final_arr += [index + i for i in range(counts)]
        
    return final_arr

def prepare_mask(string: str, height: int, width: int) -> np.ndarray:
    """Funzione usata per preparare la maschera associata a un'immagine.

    Args:
        string (str): stringa da cui decodificare le informazioni sulla maschera in RLE
        height (int): altezza dell'immagine
        width (int): larghezza dell'immagine

    Returns:
        np.ndarray: maschera associata all'immagine
    """
    
    # Generiamo gli array necessari a definire le maschere
    indexes, counts = prepare_mask_data(string)
    
    # Definiamo la lista degli indici dei pixel che sono coperti da maschera
    pos_pixel_indexes = indici_posizione_pixel(indexes, counts)
    
    # Generiamo un numpy array (inizialmente appiattito)
    mask_array = np.zeros(height * width)
    
    # Si sostituiscono i valori del suddetto array con degli 1, sulla base
    # dei pixel appartenenti alla maschera
    mask_array[pos_pixel_indexes] = 1
    
    # Viene restituita la maschera nella forma opportuna (w x h)
    return mask_array.reshape(width, height)

# Implementiamo, poi, una semplice funzione per il caricamento delle immagini

def carica_immagine(path: str) -> Image:
    """Funzione per il caricamento di un'immagine e la sua conversione
    in RGB

    Args:
        path (str): percorso da cui caricare l'immagine

    Returns:
        Image: oggetto PIL.Image rappresentativo dell'immagine in RGB
    """
    
    # Sfruttando Pillow, carichiamo l'immagine associata al percorso e la
    # convertiamo in RGB
    image = Image.open(path).convert('RGB')
    
    return image

# Per la data preparation, definiamo innanzitutto una classe UWDataset che estende
# la classe Dataset di pytorch
class UWDataset(Dataset):
    
    # Costruttore della classe
    def __init__(self, meta_df, h=256, w=256):
        super().__init__()
        self.meta_df = meta_df
        self.h = h
        self.w = w
        self.resize = Resize((h, w))
        
    # Override della proprietà __len__, intesa come lunghezza del dataframe
    # associato    
    def __len__(self):
        return len(self.meta_df)
    
    # Override della proprietà __getitem__, che restituisce la coppia
    # (immagine, maschera)
    def __getitem__(self, index):
        
        # Recuperiamo il percorso dell'immagine, contenuta nel campo path del dataframe
        path = self.meta_df.loc[index, "path"]
        
        # Carichiamo l'immagine dal percorso appena ottenuto
        image = carica_immagine(path)
        
        # Carichiamo l'altezza e la larghezza delle maschere originali
        mask_h, mask_w = self.meta_df.loc[index, "height"], self.meta_df.loc[index, "width"]
        
        # Estraiamo, quindi, la codifica della segmentazione e carichiamo la maschera
        mask_string = self.meta_df.loc[index, "segmentation"]
        main_mask_channel = self.carica_maschera(string=mask_string, h=mask_h, w=mask_w)
        
        # Per essere utilizzabili in pytorch, dobbiamo trasformare l'immagine e la maschera
        # in tensori, usando la funzione ToTensor
        # updating those in tensor format
        image = ToTensor()(self.resize(image))
        main_mask_channel = ToTensor()(self.resize(main_mask_channel))
        
        # Carichiamo, infine, la maschera originale e l'etichetta di classe
        mask = torch.zeros((3, self.h, self.w))
        class_label = self.meta_df.loc[index, "class"]
        mask[class_label, ...] = main_mask_channel
        
        return image, mask
    
    def carica_maschera(self, string: str, h: int, w: int) -> Image:
        """Funzione per caricare un'immagine rappresentativa della maschera.

        Args:
            string (str): stringa rappresentante la codifica RLE della maschera
            h (int): altezza della maschera
            w (int): larghezza della maschera

        Returns:
            Image: immagine rappresentativa della maschera
        """
        # Controlliamo se la codifica della maschera sia valida oppure un NaN
        if string != "nan":
            # Se la codifica è valida, restituiamo la maschera attraverso la funzione
            # prepare_mask definita in precedenza            
            return Image.fromarray(prepare_mask(string, h, w))
        # Altrimenti, la maschera sarà una matrice di 0
        return Image.fromarray(np.zeros((h, w)))

# Carichiamo il dataset    
ds = UWDataset(train_df)
print(f"\n--- Lunghezza del dataset : {len(ds)} ---")

# Per avere una prova del corretto funzionamento di quanto detto,
# prendiamo un elemento dal dataset, stampiamo le dimensioni di 
# immagine e maschera
image, mask = ds[194]
# DA USARE SOLO NEL NOTEBOOK
# image.shape, mask.shape
print(f"\nDimensione dell'immagine 194: {image.shape}\nDimensione della maschera per l'immagine 194: {mask.shape}\n")

combined_im_mask = torch.cat([image, mask], dim=2)

# Adesso, diamo una dimostrazione visiva di quanto ottenuto
def show_image(tensor_image: torch.Tensor, name: str):
    """Funzione per visualizzare un'immagine definita da un Tensor

    Args:
        tensor_image (torch.Tensor): tensore dell'immagine da mostrare
        name (str): titolo dell'immagine
    """
    plt.figure(figsize=(20, 20))
    plt.imshow(tensor_image.permute(1,2,0))
    plt.title(name, size=30)
    plt.show()

# show_image(combined_im_mask, "Immagine & Maschera")

# Dobbiamo fare una divisione del training set per ricavare un validation set: proviamo con percentuale 90-10

# Calcoliamo la dimensione del training set
train_size = int(len(ds)*0.9)

# Per sottrazione, ricaviamo la dimensione del validation set
val_size = len(ds) - train_size

# Sfruttando pytorch, implementiamo la divisione
train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
print(f"Length of the training dataset : {len(train_ds)}")
print(f"Length of the validation dataset : {len(val_ds)}")
