
def posizione_a_coordinate(posizione, larghezza):
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

def crea_base_maschera(larghezza, altezza):
    
    """
    Crea un'immagine vuota RGB delle dimensioni specificate.
    :param larghezza: la larghezza dell'immagine.
    :param altezza: l'altezza dell'immagine.
    :return: un'immagine PIL vuota delle dimensioni specificate.
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

def disegna_maschera(rle, width, height):
    # Creo un array numpy vuoto con le dimensioni dell'immagine
    img_data = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Itero sulla sequenza RLE invertita
    color = None
    x, y = 0, 0
    for i, val in enumerate(rle):
        if i % 2 == 0:
            # Siamo in una posizione pari, quindi abbiamo la posizione del pixel
            
            #x_pix, y_pix = posizione_a_coordinate(val, width)
            
            # x colonna, y riga
            
            color = val
            
        else:
            
            # Siamo in una posizione dispari, quindi abbiamo la lunghezza della sequenza
            for j in range(val):
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

# Esempio di utilizzo
rle = [255, 5, 0, 2, 255, 3, 0, 1, 255, 4]
width, height = 8, 3
img_data = disegna_maschera(rle, width, height)
img = Image.fromarray(img_data, 'RGB')
img.show()
