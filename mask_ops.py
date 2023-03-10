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

def crea_maschera_vuota(path: str, width: int, height: int, pixel_w: float, pixel_h: float):
    """Funzione usata per creare e salvare una maschera vuota 
    delle dimensioni specificate

    Args:
        path (str): percorso in cui salvare la maschera
        width (int): larghezza della maschera
        height (int): altezza della maschera
    """
    # Creo una maschera vuota con le dimensioni dei pixel calcolate
    mask = np.zeros((math.ceil(height * pixel_h), math.ceil(width * pixel_w), 3), dtype=np.uint8)

    cv.imwrite(path, mask)
    
    print(f"\nMaschera vuota di dimensione {width}x{height} creata in: {path}")


def colora_maschera(mask_path: str, rows_triple, cols_triple, run_len_triple):
    print(f"\nSto colorando la maschera vuota in: {mask_path}")
    
    maschera = cv.imread(mask_path)
    
    for idx, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        
        color = np.array(color).reshape((1, 1, -1))
        
        for x, y, l in zip(cols_triple[idx], rows_triple[idx], run_len_triple[idx]):
            maschera[y:y+l, x:x+l] = color

        print(f"Colore: {color}\nRighe: {rows_triple[idx]}\nColonne: {cols_triple[idx]}\n")
    
    cv.imwrite(mask_path, maschera)
    
    
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
        
        # Raccolgo le variabili di interesse: percorso, righe per le tre maschere,
        # colonne per le tre maschere e lunghezza delle run per le tre maschere.
        mask_path = row[1]['mask_path']
        rows_triple = row[1]['Righe']
        cols_triple = row[1]['Colonne']
        run_len_triple = row[1]['Lunghezze Run']
        altezza_pixel = row[1]['height_px']
        larghezza_pixel = row[1]['width_px']
        
        if row[1]['is_created_mask'] == False:
                crea_maschera_vuota(mask_path, 266, 266, larghezza_pixel, altezza_pixel)
                row[1]['is_created_mask'] = True
        else:
            print("Maschere vuote già create")
        
        colora_maschera(mask_path, rows_triple, cols_triple, run_len_triple)