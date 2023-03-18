"""Questo modulo serve per eseguire una serie di operazioni
relative alle maschere utili alla competizione.
"""

# Imports

from pandas import DataFrame
import math
import numpy as np
import cv2 as cv

# Consts

COLORS = [(1,0,0), (0,1,0), (0,0,1)]

# Functions

def crea_maschera_vuota(path: str, width: int, height: int):
    """Funzione usata per creare un segnaposto per una maschera
    in un dato percorso

    Args:
        path (str): percorso in cui salvare la maschera vuota
        width (int): larghezza desiderata per la maschera
        height (int): altezza desiderata per la maschera
    """
    # Creo una maschera vuota con le dimensioni dei pixel calcolate
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    cv.imwrite(path, mask)


def rle_encode(mask: np.ndarray) -> str:
    """Funzione usata per generare una codifica RLE a partire da una maschera binaria

    Args:
        mask (np.ndarray): maschera binaria da codificare in RLE

    Returns:
        str: codifica RLE della maschera
    """
    
    rle_string = ""
    
    mask = mask.flatten()

    # Aggiungi uno zero-padding alla maschera per evitare problemi di indicizzazione
    padded_mask = np.pad(mask, (1,), mode="constant")

    # Inizializza l'indice corrente a 0
    current_index = 0
    
    # Inizializza la lunghezza corrente a 0
    current_length = 0

    # Itera attraverso ciascun pixel della riga
    for pixel in padded_mask:
        # Se il pixel è 1 e il pixel precedente era 0, inizia una nuova sequenza
        if pixel == 1 and padded_mask[current_index - 1] == 0:
            rle_string += str(current_index) + " "
            current_length = 1
        # Se il pixel è 1 e il pixel precedente era 1, aumenta la lunghezza della sequenza corrente
        elif pixel == 1 and padded_mask[current_index - 1] == 1:
            current_length += 1
        # Se il pixel è 0 e il pixel precedente era 1, termina la sequenza corrente
        elif pixel == 0 and padded_mask[current_index - 1] == 1:
            rle_string += str(current_length) + " "
            current_length = 0
        
        # Incrementa l'indice corrente
        current_index += 1

    # Se c'è ancora una sequenza attiva alla fine della riga, termina la sequenza
    if current_length != 0:
        rle_string += str(current_length) + " "

    # Restituisci la stringa RLE
    return rle_string


def rle_to_image(rle_code: str, height: int, width: int) -> np.ndarray:
    """Funzione usata per convertire una maschera codificata RLE in 
    un'immagine di data altezza e larghezza

    Args:
        rle_code (str): stringa rappresentativa della segmentazione RLE
        height (int): altezza desiderata per l'immagine
        width (int): larghezza desiderata per l'immagine

    Returns:
        np.ndarray: immagine della maschera associata alla stringa RLE
    """
    # Verifica se la codifica è 'nan'
    
    if isinstance(rle_code, float) and math.isnan(rle_code):
        return np.zeros((height, width), dtype=np.uint8)
    
    # decodifica la codifica RLE
    rle_numbers = [int(i) for i in rle_code.split()]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)

    # crea un'immagine vuota
    img = np.zeros(height*width, dtype=np.uint8)

    # colora i pixel coperti dalla maschera
    for index, length in rle_pairs:
        index -= 1
        img[index:index+length] = 255

    # ridimensiona l'immagine e la restituisce
    return img.reshape((height, width))


# def mask_from_segmentation(segmentation: str, shape: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
#     """Funzione usata per creare una maschera per un segmento codificato RLE

#     Args:
#         segmentation (str): stringa rappresentativa del segmento in codifica RLE
#         shape (tuple[int, int]): forma della maschera (altezza, larghezza)
#         color (tuple[int, int, int]): tupla di colori RGB (normalizzata in [0, 1])

#     Returns:
#         np.ndarray: maschera associata a un segmento codificato RLE
#     """

#     # Estraiamo una lista globale di posizioni iniziali e di run length dalla stringa
#     segm = np.asarray(segmentation.split(), dtype=int)

#     # Separiamo la lista in modo da evidenziare posizioni iniziali e run lenghts associate
#     start_point = segm[0::2] - 1
#     length_point = segm[1::2]

#     # Calcoliamo, per ogni punto iniziale estratto, il corrispondente punto finale della run
#     end_point = start_point + length_point

#     # Creiamo la maschera (inizialmente vuota)
#     case_mask = np.zeros((shape[0]*shape[1], 3), dtype=np.uint8)

#     # Coloriamo tutti i pixel compresi tra inizio e fine della run del colore passato in input
#     for start, end in zip(start_point, end_point):
#         case_mask[start:end] = color

#     # Riorganizziamo la maschera come matrice (altezza x larghezza x 3)
#     case_mask = case_mask.reshape((shape[0], shape[1], 3))
    
#     return case_mask


# def prepare_mask(rle_encodings: list[str], height: int, width: int) -> np.ndarray:
#     """Funzione usata per preparare la maschera globale (a partire da codifica RLE) associata a una scansione.

#     Args:
#         rle_encodings (list[str]): lista di codifiche RLE da cui ricavare le informazioni
#         height (int): altezza dell'immagine
#         width (int): larghezza dell'immagine

#     Returns:
#         np.ndarray: maschera associata all'immagine
#     """
    
    
#     mask_glob = np.zeros((height, width, 3), dtype=np.uint8)
    
#     for index, encoding in enumerate(rle_encodings):
        
#         if encoding is np.nan:
#             continue
        
#         mask = mask_from_segmentation(encoding, (height, width), COLORS[index])
            
#         mask_glob += mask

#     return mask_glob

    
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


def interpolate_masks(masks:list, output_path:str) -> tuple[np.ndarray, list[str]]:
    """Funzione usata per generare una maschera per interpolazione dei dati
    di un batch di immagini (5) e rispettiva codifica RLE.

    Args:
        masks (list): lista di percorsi contenenti immagini
        output_path (str): percorso in cui salvare la maschera interpolata
        
    Returns:
        tuple[np.ndarray, list[str]]: maschera interpolata e codifica RLE dei tre canali
    """
    
    # Inizialmente, si leggono le maschere
    img1 = cv.imread(masks[0], cv.IMREAD_COLOR)
    img2 = cv.imread(masks[1], cv.IMREAD_COLOR)
    img3 = cv.imread(masks[2], cv.IMREAD_COLOR)
    img4 = cv.imread(masks[3], cv.IMREAD_COLOR)
    img5 = cv.imread(masks[4], cv.IMREAD_COLOR)

    # Si calcola la media delle intensità, pixel per pixel, delle maschere
    mean_mask = np.mean(np.array([img1, img2, img3, img4, img5]), axis=0)

    # Si traspone la scala dell'immagine (per avere valori compresi tra 0 e 255)
    mean_mask = (mean_mask - np.min(mean_mask)) / (np.max(mean_mask) - np.min(mean_mask)) * 255

    # Si converte l'immagine in formato uint8
    mean_mask = mean_mask.astype(np.uint8)
    
    # Bisogna saturare, adesso l'informazione del colore per i 3 canali
    # Si creano tre maschere su base condizionale (una per canale)
    mask_red = mean_mask[:, :, 0] >= 1
    mask_green = mean_mask[:, :, 1] >= 1
    mask_blue = mean_mask[:, :, 2] >= 1

    # Si sostituisce ai pixel di ciascuna maschera il valore di colore
    # totalmente saturo
    mean_mask[mask_red] = (255, 0, 0)
    mean_mask[mask_green] = (0, 255, 0)
    mean_mask[mask_blue] = (0, 0, 255)
    
    # Si converte l'immagine in RGB
    # NB: LA CONVERSIONE DI COLORE SI USA SOLO PER ELABORARE L'IMMAGINE IN PYTHON
    # IL SALVATAGGIO AVVIENE SECONDO QUELLA CHE PER OPENCV E' CODIFICA BGR
    mean_mask = cv.cvtColor(mean_mask, cv.COLOR_BGR2RGB)

    # Si salva l'immagine in output_path
    cv.imwrite(output_path, mean_mask)
    
    return (mean_mask, [rle_encode(mean_mask[:,:,0]/255), rle_encode(mean_mask[:,:,1]/255), rle_encode(mean_mask[:,:,2]/255)])