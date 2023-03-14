"""Questo modulo serve per eseguire una serie di operazioni
relative alle maschere utili alla competizione.
"""

# Imports

from pandas import DataFrame
import numpy as np
import cv2 as cv

# Consts

COLORS = [(1,0,0), (0,1,0), (0,0,1)]

# Functions

def crea_maschera_vuota(path, width, height):

    # Creo una maschera vuota con le dimensioni dei pixel calcolate
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    cv.imwrite(path, mask)

def mask_from_segmentation(segmentation: str, shape: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    """Funzione usata per creare una maschera per un segmento codificato RLE

    Args:
        segmentation (str): stringa rappresentativa del segmento in codifica RLE
        shape (tuple[int, int]): forma della maschera (altezza, larghezza)
        color (tuple[int, int, int]): tupla di colori RGB (normalizzata in [0, 1])

    Returns:
        np.ndarray: maschera associata a un segmento codificato RLE
    """

    # Estraiamo una lista globale di posizioni iniziali e di run length dalla stringa
    segm = np.asarray(segmentation.split(), dtype=int)

    # Separiamo la lista in modo da evidenziare posizioni iniziali e run lenghts associate
    start_point = segm[0::2] - 1
    length_point = segm[1::2]

    # Calcoliamo, per ogni punto iniziale estratto, il corrispondente punto finale della run
    end_point = start_point + length_point

    # Creiamo la maschera (inizialmente vuota)
    case_mask = np.zeros((shape[0]*shape[1], 3), dtype=np.uint8)

    # Coloriamo tutti i pixel compresi tra inizio e fine della run del colore passato in input
    for start, end in zip(start_point, end_point):
        case_mask[start:end] = color

    # Riorganizziamo la maschera come matrice (altezza x larghezza x 3)
    case_mask = case_mask.reshape((shape[0], shape[1], 3))
    
    return case_mask


def prepare_mask(rle_encodings: list[str], height: int, width: int) -> np.ndarray:
    """Funzione usata per preparare la maschera globale (a partire da codifica RLE) associata a una scansione.

    Args:
        rle_encodings (list[str]): lista di codifiche RLE da cui ricavare le informazioni
        height (int): altezza dell'immagine
        width (int): larghezza dell'immagine

    Returns:
        np.ndarray: maschera associata all'immagine
    """
    
    
    mask_glob = np.zeros((height, width, 3), dtype=np.uint8)
    
    for index, encoding in enumerate(rle_encodings):
        
        if encoding is np.nan:
            continue
        
        mask = mask_from_segmentation(encoding, (height, width), COLORS[index])
            
        mask_glob += mask

    return mask_glob

    
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
    
        
# def genera_tutte_maschere(image_details_df: DataFrame):
    
#     for row in image_details_df.iterrows():
#         # Raccolgo le variabili di interesse
#         slice_path = row[1]['path']
#         mask_path = row[1]['mask_path']
#         rle_triple = row[1]['segmentation']
#         altezza = row[1]['height']
#         larghezza = row[1]['width']
        
#         print(f"\nPercorso: {mask_path}\nDimensioni:{larghezza}x{altezza}\n")
        
#         if row[1]['is_created_mask'] == False:
#                 mask_array = prepare_mask(rle_encodings=rle_triple, height=altezza, width=larghezza)

#                 # rgb_slice = cv.imread(slice_path)

#                 # overriden_w_mask = override_mask_on_img(mask_array=mask_array, rgb_slice=rgb_slice)
                
#                 cv.imwrite(mask_path, 255*mask_array)
                
#                 row[1]['is_created_mask'] = True
#         else:
#             print("Maschere vuote già create")
#             continue

