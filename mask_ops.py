"""Questo modulo serve per eseguire una serie di operazioni
relative alle maschere utili alla competizione.
"""

from pandas import DataFrame
import numpy as np
import cv2 as cv
import math

# def crea_maschera_vuota(path: str, width: int, height: int):
#     """Funzione usata per creare e salvare una maschera vuota 
#     delle dimensioni specificate

#     Args:
#         path (str): percorso in cui salvare la maschera
#         width (int): larghezza della maschera
#         height (int): altezza della maschera
#     """
#     # Creo una maschera vuota con le dimensioni dei pixel calcolate
#     mask = np.zeros((height, width, 3), dtype=np.uint8)

#     cv.imwrite(path, mask)
    
#     print(f"\nMaschera vuota di dimensione {width}x{height} creata in: {path}")


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

def prepare_mask(rle_encodings: list[str], height: int, width: int) -> np.ndarray:
    """Funzione usata per preparare la maschera associata a un'immagine.

    Args:
        rle_encodings (list[str]): lista di codifiche RLE da cui ricavare le informazioni
        height (int): altezza dell'immagine
        width (int): larghezza dell'immagine

    Returns:
        np.ndarray: maschera associata all'immagine
    """
    # Generiamo un numpy array (inizialmente appiattito)
    mask_array = np.zeros((height * width, 3))
    
    color = 0
    
    for index, encoding in enumerate(rle_encodings):
        
        # Piccola conversione per le codifiche in nan
        if encoding is np.nan:
            continue
            
        if index == 0:
            color = np.array([255, 0, 0]) # R
        elif index == 1:
            color = np.array([0, 255, 0])   # G
        else:
            color = np.array([0, 0, 255])   # B
        
        # Generiamo gli array necessari a definire le maschere
        indexes, counts = prepare_mask_data(encoding)
    
        # Definiamo la lista degli indici dei pixel che sono coperti da maschera
        pos_pixel_indexes = indici_posizione_pixel(indexes, counts)
        
        # Replichiamo l'array "color" lungo l'asse 0
        color = np.repeat(color[np.newaxis, :], len(pos_pixel_indexes), axis=0)
    
        # Si sostituiscono i valori del suddetto array con degli 1, sulla base
        # dei pixel appartenenti alla maschera
        mask_array[pos_pixel_indexes] = color
    
    # Viene restituita la maschera nella forma opportuna (w x h)
    return mask_array.reshape(width, height, 3)


# def crea_maschera_vuota(path: str, width: int, height: int, pixel_w: float, pixel_h: float):
#     """Funzione usata per creare e salvare una maschera vuota 
#     delle dimensioni specificate

#     Args:
#         path (str): percorso in cui salvare la maschera
#         width (int): larghezza della maschera
#         height (int): altezza della maschera
#     """
#     # Creo una maschera vuota con le dimensioni dei pixel calcolate
#     mask = np.zeros((math.ceil(height * pixel_h), math.ceil(width * pixel_w), 3), dtype=np.uint8)

#     cv.imwrite(path, mask)
    
#     print(f"\nMaschera vuota di dimensione {width}x{height} creata in: {path}")


# def colora_maschera(mask_path: str, rows_triple, cols_triple, run_len_triple):
#     print(f"\nSto colorando la maschera vuota in: {mask_path}")
    
#     maschera = cv.imread(mask_path)
    
#     for idx, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        
#         color = np.array(color).reshape((1, 1, -1))
        
#         for x, y, l in zip(cols_triple[idx], rows_triple[idx], run_len_triple[idx]):
#             maschera[y:y+l, x:x+l] = color

#         print(f"Colore: {color}\nRighe: {rows_triple[idx]}\nColonne: {cols_triple[idx]}\n")
    
#     cv.imwrite(mask_path, maschera)

    
def override_mask_on_img(mask_array: np.array, rgb_slice: cv.Mat) -> cv.Mat:
    """Funzione usata per sovrapporre una maschera ad un'immagine

    Args:
        mask_array (np.array): maschera da sovrapporre
        rgb_slice (Mat): immagine su cui applicare la maschera

    Returns:
        Mat: immagine con maschera sovrapposta
    """
    
    bool_index = mask_array != (0, 0, 0)

    rgb_slice[bool_index] = mask_array[bool_index]

    return rgb_slice
    
    
# def genera_tutte_maschere(image_details: DataFrame, righe_maschera_global: list, colonne_maschera_global: list, lunghezze_run_global: list):
#     for index, row in image_details.iterrows():
        
#         mask_path = row['mask_path']
#         # print(f"\nWorking on {mask_path}\n")
        
#         if row['is_created_mask'] == False:
#                 crea_maschera_vuota(mask_path, 266, 266)
#                 row['is_created_mask'] = True
#         # else:
#             # print("Maschere vuote già create")
        
#         colora_maschera(mask_path, righe_maschera_global[index], colonne_maschera_global[index], lunghezze_run_global[index])
        
def genera_tutte_maschere(image_details_df: DataFrame):
    
    for row in image_details_df.iterrows():
        # Raccolgo le variabili di interesse
        slice_path = row[1]['path']
        mask_path = row[1]['mask_path']
        rle_triple = row[1]['segmentation']
        altezza = row[1]['height']
        larghezza = row[1]['width']
        
        print(f"\nPercorso: {mask_path}\nDimensioni:{larghezza}x{altezza}\n")
        
        if row[1]['is_created_mask'] == False:
                mask_array = prepare_mask(rle_encodings=rle_triple, height=altezza, width=larghezza)

                rgb_slice = cv.imread(slice_path)

                overriden_w_mask = override_mask_on_img(mask_array=mask_array, rgb_slice=rgb_slice)
                
                cv.imwrite(mask_path, overriden_w_mask)
                
                row[1]['is_created_mask'] = True
        else:
            print("Maschere vuote già create")
            continue
            
        