from cv2 import COLOR_GRAY2BGR, COLOR_GRAY2RGB, Mat
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv

os.environ["QT_QPA_PLATFORM"] = "xcb"

# def split_rle(rle_string):
    
#     pixels = []
#     lengths = []
    
#     # Converto la RLE in una lista di interi
#     rle = [int(x) for x in rle_string.split(' ')]
    
#     for i, c in enumerate(rle):
#         if i % 2 == 0:
#             pixels.append(int(c))
#         else:
#             lengths.append(int(c))
#     return pixels, lengths

# def split_rle_0(rle_strings):
    
#     all_pixels = []
#     all_lengths = []
    
#     for rle_string in rle_strings:
#         pixels = []
#         lengths = []

#         # Converto la RLE in una lista di interi
#         rle = [int(x) for x in rle_string.split(' ')]

#         for i, c in enumerate(rle):
#             if i % 2 == 0:
#                 pixels.append(int(c))
#             else:
#                 lengths.append(int(c))

#         all_pixels.append(pixels)
#         all_lengths.append(lengths)

#     return all_pixels, all_lengths

# def posizioni_pixel(pixels, larghezza):
#     col = [px % larghezza for px in pixels]
#     row = [px // larghezza for px in pixels]
#     return row, col

# def crea_maschera_vuota(width, height):

#     # Crea una maschera vuota con le dimensioni dei pixel calcolate
#     mask = np.zeros((height, width, 3), dtype=np.uint8)

#     return mask


# '''

# def crea_maschera_vuota(width, height):

#     # Crea una maschera vuota con le dimensioni dei pixel calcolate
#     mask = np.zeros((height, width), dtype=np.uint8)

#     return mask

# '''

# def colora_maschera(maschera_vuota, rows, cols, color, lengths):
#     color = np.array(color).reshape((1, 1, -1))
#     for x, y, l in zip(cols, rows, lengths):
#         maschera_vuota[y:y+l, x:x+l] = color
#     return maschera_vuota


# '''

# def colora_maschera(maschera_vuota, rows, cols, lengths):
    
#     for x, y, l in zip(cols, rows, lengths):
#         maschera_vuota[y:y+l, x:x+l] = 255
    
#     return maschera_vuota


# '''

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
        if encoding == 'nan':
            encoding = '0 0'
            
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


# rle = "12927 6 13186 15 13450 19 13715 21 13980 24 14245 26 14510 27 14776 28 15042 29 15307 31 15574 30 15840 31 16106 31 16372 31 16638 31 16904 31 17170 31 17437 30 17703 30 17969 30 18235 30 18502 29 18768 29 19034 29 19300 29 19567 29 19833 29 20099 29 20365 29 20631 28 20897 28 21163 28 21429 28 21695 28 21961 28 22227 29 22493 29 22759 29 23025 29 23292 28 23558 28 23824 28 24090 28 24357 27 24623 27 24889 27 25155 27 25421 27 25687 27 25953 27 26219 28 26485 28 26750 29 27016 29 27282 29 27548 29 27814 30 28080 30 28345 31 28611 31 28877 31 29143 31 29409 31 29675 32 29941 32 30208 31 30475 30 30741 29 31008 28 31275 26 31541 25 31808 24 32074 23 32340 22 32606 22 32872 22 33139 20 33405 20 33672 18 33940 15 34207 12 34482 1"

# PROVA RLE IMMAGINE 276x276
# Se la codifica contiene nan, si può considerare come una coppia 0 0
encodings = [
    '18636 6 18910 19 19185 23 19460 25 19736 27 20012 29 20288 31 20564 32 20840 34 21116 35 21393 7 21401 27 21669 6 21679 26 21947 3 21957 25 22235 25 22513 24 22790 25 23068 25 23345 25 23622 25 23900 23 24178 23 24455 23 24733 22 '+
    '25011 20 25289 19 25566 18 25843 17 26119 17 26395 17 26671 17 26948 15 27225 13 27502 11 27780 8 28058 4', # large_bowel
    'nan', # small_bowel
    '23324 13 23597 18 23871 22 24146 24 24421 26 24696 27 24972 28 25248 29 25524 30 25799 33 26075 34 26352 34 26628 34 26904 35 27181 35 27458 34 27735 34 28012 34 28289 34 28567 33 28845 32 29121 34 29398 35 29677 34 29955 34 30234 35 30512 34 30788 35 31064 36 '+
    '31340 37 31616 37 31891 39 32168 38 32446 37 32724 35 33001 35 33278 34 33555 33 33832 32 34108 32 34385 31 34661 31 34938 30 35214 31 35490 31 35762 35 36035 38 36309 41 36584 42 36859 43 37134 44 37409 45 37686 43 37963 42 38240 41 38517 40 38794 39 39071 37 '+
    '39348 36 39625 35 39902 33 40179 32 40460 27 40737 26 41014 25 41290 24 41566 23 41842 22 42119 20 42395 13 42673 6'    # stomach
]

# PROVA RLE IMMAGINE 234x234
# Se la codifica contiene nan, si può considerare come una coppia 0 0
# encodings = [
#     'nan', # large_bowel
#     '22320 4 22553 6 22786 7 23020 8 23254 8 23488 7 23722 6 23957 4', # small_bowel
#     '22381 6 22614 8 22847 10 23080 12 23313 14 23546 15 23780 15 24013 17 24246 18 24479 19 24713 19 24947' +
#     ' 19 25181 19 25415 19 25649 19 25883 19 26117 18 26351 18 26585 18 26820 17 27054 16 27289 15 27524 13 27760 10'    # stomach
# ]

# PROVA RLE DI DEFAULT (MARIO)
# large_bowel = '21716 5 21725 5 21981 17 22246 19 22512 21 22778 23 23044 24 23310 26 23576 27 23842 29 24108 30 24374 32 24640 33 24907 32 25173 33 25441 31 25709 30 25978 28 26246 27 26513 27 26780 26 27046 27 27313 26 27580 25 27847 25 28114 24 28383 21 28652 19 28918 19 29184 19 29450 19 29716 20 29983 19 30249 21 30516 23 30782 25 31048 26 31315 26 31581 26 31848 26 32114 26 32381 25 32647 24 32914 23 33181 21 33447 20 33714 18 33981 16 34248 14 34515 12 34782 11 35049 9 35316 7 36128 5 36392 8 36658 9 36923 11 37189 11 37455 11 37721 12 37987 12 38253 14 38519 15 38785 16 39051 17 39317 17 39583 18 39849 17 40115 17 40381 16 40647 16 40913 14 41179 13 41445 13 41711 13 41978 11 42244 11 42511 9 42779 5'
# small_bowel = '26234 4 26498 8 26764 10 27029 13 27295 16 27561 19 27825 22 28090 23 28355 24 28621 24 28887 24 29153 24 29419 25 29685 25 29951 26 30218 25 30485 26 30751 27 31018 27 31285 27 31551 27 31818 26 32085 25 32353 23 32622 21 32889 20 33155 21 33422 20 33688 20 33955 19 34223 17 34493 13 34759 13 35026 12 35293 10 35560 8 35828 4'
# stomach = '22749 8 23013 11 23277 14 23541 17 23806 19 24071 21 24335 23 24599 25 24863 27 25127 29 25392 30 25657 31 25922 32 26188 31 26454 30 26720 29 26986 27 27252 24 27518 21 27785 17 28051 15 28319 10'

#pixels, lenghts = split_rle_0(df['segmentation'])
#pixels, lenghts = split_rle_0([large_bowel, small_bowel, stomach])

#print(pixels)
#print(lenghts)

#print(rows)
#print(cols)

def override_mask_on_img(mask_array: np.array, rgb_slice: Mat) -> Mat:
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

height = 276
width = 276

mask_array = prepare_mask(rle_encodings=encodings, width=width, height=height)

# rgb_slice = cv.imread("train/case139/case139_day0/scans/slice_0075_234_234_1.50_1.50.png")
rgb_slice = cv.imread("train/case118/case118_day0/scans/slice_0031_276_276_1.63_1.63.png")

overriden_w_mask = override_mask_on_img(mask_array=mask_array, rgb_slice=rgb_slice)

plt.imshow(np.array(overriden_w_mask))
plt.show()

# r, g, b = cv.split(rgb_slice)

# equalised_slice = cv.merge((cv.equalizeHist(r), cv.equalizeHist(g), cv.equalizeHist(b)))

# cv.imwrite('EQUALISED_RGB_slice_0031_276_276_1.63_1.63.png', equalised_slice)

# bool_index = mask_array != (0, 0, 0)

# rgb_slice[bool_index] = mask_array[bool_index]

# cv.imwrite('OVERRIDDEN_MASK_slice_0031_276_276_1.63_1.63.png', rgb_slice.astype(np.uint8))

# plt.imshow(np.array(rgb_slice))
# plt.show()


# maschera_0 = colora_maschera(mask, rows_0, cols_0, np.array([255, 0, 0]), lenghts_0)
# maschera_1 = colora_maschera(mask, rows_1, cols_1, np.array([0, 255, 0]), lenghts_1)
# maschera_2 = colora_maschera(mask, rows_2, cols_2, np.array([0, 0, 255]), lenghts_2)

# '''
# maschera_0 = colora_maschera(mask, rows_0, cols_0, lenghts_0)
# maschera_1 = colora_maschera(mask, rows_1, cols_1, lenghts_1)
# maschera_2 = colora_maschera(mask, rows_2, cols_2, lenghts_2)
# '''

# plt.imshow(np.array(maschera_2))

# plt.show()

# DECOMMENTARE DA QUI

# img = cv.imread("slice_0001_234_234_1.50_1.50.png")

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# equ = cv.equalizeHist(gray)

# #res = np.hstack(equ) #stacking images side-by-side
# #cv.imwrite('slice_0001_234_234_1.50_1.50_.png',res)

# plt.imshow(np.array(equ))

# plt.show()

# # Imposta la nuova dimensione
# new_size = (266, 266)

# # Ridimensiona l'immagine
# resized_img = cv.resize(equ, new_size)

# maschera_2 = cv.cvtColor(maschera_2, cv.COLOR_BGR2GRAY)
# alpha = 0.5  # definisci il peso della maschera
# output = cv.addWeighted(resized_img, 1-alpha, maschera_2, alpha, 0)  # sovrappone l'immagine alla maschera

# plt.imshow(np.array(output))

# plt.show()