import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

"""
Questa funzione prende in input la larghezza e l'altezza dell'immagine in pixel 
e la dimensione fisica dei pixel come una tupla di valori in millimetri. 
Calcola quindi le dimensioni dei pixel in pixel arrotondando al valore intero più vicino. 
Infine, crea una maschera vuota di dimensioni (height*pixel_height, width*pixel_width) 
utilizzando la funzione numpy.zeros() e restituisce la maschera creata.
"""

# Creo un'immagine a partire da una codifica RLE invertita
#rle = '15323 4 15587 8 15852 10 16117 11 16383 12 16649 12 16915 12 17181 12 17447 12 17713 12 17979 12 18245 12 18511 12 18777 12 19043 12 19309 12 19575 12 19841 12 20107 12 20373 12 20639 12 20905 12 21171 12 21437 12 21703 12 21969 12 22235 12 22501 12 22767 12 23033 12 23299 12 23565 12 23831 12 24097 12 24363 12 24629 12 24895 12 25161 13 25427 13 25693 14 25959 14 26224 15 26489 16 26755 17 27020 19 27286 20 27552 21 27818 21 28084 21 28350 22 28616 22 28882 22 29147 23 29413 23 29678 24 29944 24 30210 25 30476 25 30742 25 31008 25 31274 25 31540 25 31806 25 32072 24 32338 24 32604 24 32871 22 33137 22 33403 21 33669 21 33936 19 34203 17 34469 14 34736 12 35003 11 35271 8 35539 3'
rle = '12927 6 13186 15 13450 19 13715 21 13980 24 14245 26 14510 27 14776 28 15042 29 15307 31 15574 30 15840 31 16106 31 16372 31 16638 31 16904 31 17170 31 17437 30 17703 30 17969 30 18235 30 18502 29 18768 29 19034 29 19300 29 19567 29 19833 29 20099 29 20365 29 20631 28 20897 28 21163 28 21429 28 21695 28 21961 28 22227 29 22493 29 22759 29 23025 29 23292 28 23558 28 23824 28 24090 28 24357 27 24623 27 24889 27 25155 27 25421 27 25687 27 25953 27 26219 28 26485 28 26750 29 27016 29 27282 29 27548 29 27814 30 28080 30 28345 31 28611 31 28877 31 29143 31 29409 31 29675 32 29941 32 30208 31 30475 30 30741 29 31008 28 31275 26 31541 25 31808 24 32074 23 32340 22 32606 22 32872 22 33139 20 33405 20 33672 18 33940 15 34207 12 34482 1'

##########################################################################
#                                   OKK

def split_rle(rle_string):
    
    pixels = []
    lengths = []
    
    # Converto la RLE in una lista di interi
    rle = [int(x) for x in rle_string.split(' ')]
    
    for i, c in enumerate(rle):
        if i % 2 == 0:
            pixels.append(int(c))
        else:
            lengths.append(int(c))
    return pixels, lengths

def coordinate_pixel(posizioni, larghezza):
    x_coord = [pos % larghezza for pos in posizioni]
    y_coord = [pos // larghezza for pos in posizioni]
    return x_coord, y_coord

"""
Questa funzione prende in input la larghezza e l'altezza dell'immagine in pixel 
e la dimensione fisica dei pixel come una tupla di valori in millimetri. 
Calcola quindi le dimensioni dei pixel in pixel arrotondando al valore intero più vicino. 
Infine, crea una maschera vuota di dimensioni (height*pixel_height, width*pixel_width) 
utilizzando la funzione numpy.zeros() e restituisce la maschera creata.

    QUESTO PERCHE' IL NOME DEI FILE DELLA CARTELLA "train" include 4 numeri (es. 276_276_1.63_1.63.png), 
    rispettivamente altezza/larghezza della porzione (pixel in interi) e altezza/larghezza della spaziatura 
    (mm in numeri in virgola mobile). I primi due numeri definiscono la risoluzione della slide, mentre le 
    altre due la dimensione fisica del pixel.
"""

def crea_maschera_vuota(width, height, pixel_size):
    
    # Calcola le dimensioni dei pixel in pixel
    pixel_width = round(pixel_size[0] * width)
    pixel_height = round(pixel_size[1] * height)

    # Crea una maschera vuota con le dimensioni dei pixel calcolate
    mask = np.zeros((height*pixel_height, width*pixel_width), dtype=np.uint8)

    return mask

def crea_maschera(width, height):

    # Crea una maschera vuota con le dimensioni dei pixel calcolate
    mask = np.zeros((height, width), dtype=np.uint8)

    return mask

"""
def riempi_maschera(mask, x_coords, y_coords, color):
    
    # Riempie una maschera con i punti dati dalle coordinate con un colore specifico.
    for x, y in zip(x_coords, y_coords):
        mask[y, x] = color
    return mask
"""

def riempi_maschera(mask, x_coords, y_coords, color, lengths=None):
    
    """Riempie una maschera con i punti dati dalle coordinate con un colore specifico e una lunghezza specifica."""
    if lengths is None:
        lengths = [1] * len(x_coords)

    for x, y, l in zip(x_coords, y_coords, lengths):
        mask[y:y+l, x:x+l] = color

    return mask

res = split_rle(rle)

print(res)

# accede alla lista delle posizioni
posizioni = res[0]
print(f"\n--- posizioni : {posizioni} ---")

# accede alla lista delle ripetizioni
ripetizioni = res[1]
print(f"\n--- ripetizioni : {ripetizioni} ---")

# accede alle coordinate (lista di lista)
coordinate = coordinate_pixel(posizioni, 276)
print(f"\n--- coordinate : {coordinate} ---")

# accede alla lista delle colonne
colonne = coordinate[0]
print(f"\n--- colonne : {colonne} ---")

# accede alla lista delle righe
righe = coordinate[1]
print(f"\n--- righe : {righe} ---")

pixel_size = (1.5, 1.5)

#mask = crea_maschera_vuota(266, 266, pixel_size)
mask = crea_maschera(266, 266)
 
riempi_maschera(mask, colonne, righe, 255, ripetizioni)
#out = riempi_maschera(mask, righe, colonne, 255, ripetizioni)

os.environ["QT_QPA_PLATFORM"] = "xcb"

plt.imshow(np.array(mask))
plt.show()

'''
# Visualizzo la maschera
cv2.imshow("Maschera", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''