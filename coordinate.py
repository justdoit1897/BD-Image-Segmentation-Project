""" Questo modulo serve per creare 
"""


def posizione_a_coordinate(posizione: int, larghezza: int) -> tuple[int, int]:
    """Funzione per ricavare le coordinate di un pixel a partire
    dalla sua posizione e dalla larghezza dell'immagine

    Args:
        posizione (int): posizione del pixel nell'immagine in forma vettorizzata
        larghezza (int): larghezza dell'immagine

    Returns:
        tuple[int, int]: coordinate x e y del pixel
    """
    
    x = posizione % larghezza
    y = posizione // larghezza
    return x, y

###############################################################
# PER COLORARE LA MASCHERA CREO UN'IMMAGINE VUOTA, LARGA E ALTA QUANTO
# L'IMMAGINE ASSOCIATA DEL TRAINING SET.
# SUCCESSIVAMENTE, IN BASE ALLA CLASSE DELLA MASCHERA, OVVERO SE 
# E' DI CLASSE 0 (STOMACO), ECC., SETTO UN DETERMINATO COLORE 
# STOMACO -> ROSSO 
# CRASSO -> VERDE 
# TENUE -> BLU

from PIL import Image

def crea_base_maschera(larghezza: int, altezza: int) -> Image:
    """
    Crea un'immagine vuota RGB delle dimensioni specificate.
    
    Args:
        larghezza (int): la larghezza dell'immagine.
        altezza (int): l'altezza dell'immagine.
        
    Returns:
        Image: un'immagine PIL vuota delle dimensioni specificate.
    """
    
    return Image.new("RGB", (larghezza, altezza))


# Crea un'immagine vuota di larghezza 100 e altezza 100

immagine = Image.new('RGB', (100, 100))

# Seleziona un pixel e coloralo di rosso
x = 50
y = 50
colore = (255, 0, 0)  # rosso in formato RGB
immagine.putpixel((x, y), colore)

# Salva l'immagine su disco
immagine.save('immagine.png')

#######################################################################################

import numpy as np
from PIL import Image

def disegna_maschera(rle: list[int], width: int, height: int) -> np.ndarray:
    """Funzione usata per generare un'immagine contenente una maschera
    specificata attraverso una codifica RLE

    Args:
        rle (list[int]): codifica rle da cui disegnare la maschera
        width (int): larghezza dell'immagine contenente la maschera
        height (int): altezza dell'immagine contenente la maschera

    Raises:
        ValueError: la codifica dell'immagine è incompleta

    Returns:
        np.ndarray: l'array dei pixel che fanno parte della maschera
    """
    
    # Creo un array numpy vuoto con le dimensioni dell'immagine
    img_data = np.zeros((height, width, 3), dtype=np.uint8)
    # print(f"\nGenerato {type(img_data)} di dimensione ({height}, {width})\n")
    
    # Itero sulla sequenza RLE invertita
    color = None
    x, y = 0, 0
    for i, val in enumerate(rle):
        if i % 2 == 0:
            # Siamo in una posizione pari, quindi abbiamo la posizione del pixel
            # x_pix, y_pix = posizione_a_coordinate(val, width)
            
            # x colonna, y riga
            
            color = val
            # print(f"\nPosizione nell'array pari: color = {val}")
            
        else:
            # Siamo in una posizione dispari, quindi abbiamo la lunghezza della sequenza
            # print(f"\nPosizione nell'array dispari: sequenza lunga {val} pixel")
            for j in range(val):
                print(f"(x,y) = ({y},{x})")
                img_data[y, x, :] = color
                
                x += 1
                if x == width:
                    # Siamo arrivati all'ultima colonna, quindi passiamo alla riga successiva
                    x = 0
                    y += 1
                    if y == height:
                        # Siamo arrivati all'ultima riga, quindi abbiamo decodificato tutta l'immagine
                        return img_data
    
    # Se siamo arrivati qui, allora abbiamo decodificato solo una parte dell'immagine
    raise ValueError("RLE incompleta")

# Esempi di utilizzo
# RLE COMPLETA: LA FUNZIONE RIESCE A GENERARE UN'IMMAGINE
# 
# rle = [255, 5, 0, 2, 255, 3, 0, 1, 255, 4, 0, 1, 255, 8]
# RLE INCOMPLETA: LA FUNZIONE NON RIESCE A GENERARE UN'IMMAGINE,
# CHIAMANDO UN ERRORE

rle = [255, 5, 0, 2, 255, 3, 0, 1, 255, 4]

width, height = 8, 3
img_data = disegna_maschera(rle, width, height)
img = Image.fromarray(img_data, 'RGB')
img.show()
