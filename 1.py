import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv

os.environ["QT_QPA_PLATFORM"] = "xcb"

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

def split_rle_0(rle_strings):
    
    all_pixels = []
    all_lengths = []
    
    for rle_string in rle_strings:
        pixels = []
        lengths = []

        # Converto la RLE in una lista di interi
        rle = [int(x) for x in rle_string.split(' ')]

        for i, c in enumerate(rle):
            if i % 2 == 0:
                pixels.append(int(c))
            else:
                lengths.append(int(c))

        all_pixels.append(pixels)
        all_lengths.append(lengths)

    return all_pixels, all_lengths


def posizioni_pixel(pixels, larghezza):
    col = [px % larghezza for px in pixels]
    row = [px // larghezza for px in pixels]
    return row, col

def crea_maschera_vuota(width, height):

    # Crea una maschera vuota con le dimensioni dei pixel calcolate
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    return mask

'''

def crea_maschera_vuota(width, height):

    # Crea una maschera vuota con le dimensioni dei pixel calcolate
    mask = np.zeros((height, width), dtype=np.uint8)

    return mask

'''

def colora_maschera(maschera_vuota, rows, cols, color, lengths):
    color = np.array(color).reshape((1, 1, -1))
    for x, y, l in zip(cols, rows, lengths):
        maschera_vuota[y:y+l, x:x+l] = color
    return maschera_vuota


'''

def colora_maschera(maschera_vuota, rows, cols, lengths):
    
    for x, y, l in zip(cols, rows, lengths):
        maschera_vuota[y:y+l, x:x+l] = 255
    
    return maschera_vuota


'''

rle = "12927 6 13186 15 13450 19 13715 21 13980 24 14245 26 14510 27 14776 28 15042 29 15307 31 15574 30 15840 31 16106 31 16372 31 16638 31 16904 31 17170 31 17437 30 17703 30 17969 30 18235 30 18502 29 18768 29 19034 29 19300 29 19567 29 19833 29 20099 29 20365 29 20631 28 20897 28 21163 28 21429 28 21695 28 21961 28 22227 29 22493 29 22759 29 23025 29 23292 28 23558 28 23824 28 24090 28 24357 27 24623 27 24889 27 25155 27 25421 27 25687 27 25953 27 26219 28 26485 28 26750 29 27016 29 27282 29 27548 29 27814 30 28080 30 28345 31 28611 31 28877 31 29143 31 29409 31 29675 32 29941 32 30208 31 30475 30 30741 29 31008 28 31275 26 31541 25 31808 24 32074 23 32340 22 32606 22 32872 22 33139 20 33405 20 33672 18 33940 15 34207 12 34482 1"

large_bowel = '21716 5 21725 5 21981 17 22246 19 22512 21 22778 23 23044 24 23310 26 23576 27 23842 29 24108 30 24374 32 24640 33 24907 32 25173 33 25441 31 25709 30 25978 28 26246 27 26513 27 26780 26 27046 27 27313 26 27580 25 27847 25 28114 24 28383 21 28652 19 28918 19 29184 19 29450 19 29716 20 29983 19 30249 21 30516 23 30782 25 31048 26 31315 26 31581 26 31848 26 32114 26 32381 25 32647 24 32914 23 33181 21 33447 20 33714 18 33981 16 34248 14 34515 12 34782 11 35049 9 35316 7 36128 5 36392 8 36658 9 36923 11 37189 11 37455 11 37721 12 37987 12 38253 14 38519 15 38785 16 39051 17 39317 17 39583 18 39849 17 40115 17 40381 16 40647 16 40913 14 41179 13 41445 13 41711 13 41978 11 42244 11 42511 9 42779 5'
small_bowel = '26234 4 26498 8 26764 10 27029 13 27295 16 27561 19 27825 22 28090 23 28355 24 28621 24 28887 24 29153 24 29419 25 29685 25 29951 26 30218 25 30485 26 30751 27 31018 27 31285 27 31551 27 31818 26 32085 25 32353 23 32622 21 32889 20 33155 21 33422 20 33688 20 33955 19 34223 17 34493 13 34759 13 35026 12 35293 10 35560 8 35828 4'
stomach = '22749 8 23013 11 23277 14 23541 17 23806 19 24071 21 24335 23 24599 25 24863 27 25127 29 25392 30 25657 31 25922 32 26188 31 26454 30 26720 29 26986 27 27252 24 27518 21 27785 17 28051 15 28319 10'

pixels_0, lenghts_0 = split_rle(large_bowel)
pixels_1, lenghts_1 = split_rle(small_bowel)
pixels_2, lenghts_2 = split_rle(stomach)

#pixels, lenghts = split_rle_0(df['segmentation'])
#pixels, lenghts = split_rle_0([large_bowel, small_bowel, stomach])

#print(pixels)
#print(lenghts)

rows_0, cols_0 = posizioni_pixel(pixels_0, 266)
rows_1, cols_1 = posizioni_pixel(pixels_1, 266)
rows_2, cols_2 = posizioni_pixel(pixels_2, 266)

#print(rows)
#print(cols)

mask = crea_maschera_vuota(266, 266)

#plt.imshow(np.array(mask))
#plt.show()

maschera_0 = colora_maschera(mask, rows_0, cols_0, np.array([255, 0, 0]), lenghts_0)
maschera_1 = colora_maschera(mask, rows_1, cols_1, np.array([0, 255, 0]), lenghts_1)
maschera_2 = colora_maschera(mask, rows_2, cols_2, np.array([0, 0, 255]), lenghts_2)

'''
maschera_0 = colora_maschera(mask, rows_0, cols_0, lenghts_0)
maschera_1 = colora_maschera(mask, rows_1, cols_1, lenghts_1)
maschera_2 = colora_maschera(mask, rows_2, cols_2, lenghts_2)
'''

plt.imshow(np.array(maschera_2))

plt.show()

img = cv.imread("slice_0001_234_234_1.50_1.50.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
equ = cv.equalizeHist(gray)

#res = np.hstack(equ) #stacking images side-by-side
#cv.imwrite('slice_0001_234_234_1.50_1.50_.png',res)

plt.imshow(np.array(equ))

plt.show()

# Imposta la nuova dimensione
new_size = (266, 266)

# Ridimensiona l'immagine
resized_img = cv.resize(equ, new_size)

maschera_2 = cv.cvtColor(maschera_2, cv.COLOR_BGR2GRAY)
alpha = 0.5  # definisci il peso della maschera
output = cv.addWeighted(resized_img, 1-alpha, maschera_2, alpha, 0)  # sovrappone l'immagine alla maschera

plt.imshow(np.array(output))

plt.show()