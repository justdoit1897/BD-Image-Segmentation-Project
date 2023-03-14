import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def rle_to_image(rle_code, height, width):
    # Verifica se la codifica è 'nan'
    if rle_code == 'nan':
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

# definisci le dimensioni dell'immagine
height = 266    
width = 266

# esempio di 3 maschere RLE
# rle_code_0 = '42008 5 42366 8 42725 10 43085 11 43445 11 43805 11 44164 12 44524 12 44884 12 45243 13 45603 13 45963 13 46322 13 46681 14 47041 13 47400 14 47760 14 48120 13 48480 13 48840 12 49201 10 49562 8 49923 5'
# rle_code_1 = 'nan'
# rle_code_2 = '31877 10 32235 13 32594 15 32953 17 33312 19 33671 20 34031 21 34391 21 34750 23 35110 24 35470 26 35830 27 36190 27 36550 28 36910 29 37270 30 37630 30 37990 31 38350 32 38710 33 39071 33 39431 33 39791 34 40152 34 40513 34 40873 34 41234 34 41595 33 41955 34 42316 34 42676 34 43037 34 43398 33 43758 34 44119 33 44479 33 44839 34 45199 34 45559 34 45920 33 46280 33 46640 33 47001 32 47361 32 47721 33 48082 32 48442 32 48802 32 49163 31 49523 31 49883 32 50243 32 50603 32 50962 33 51322 34 51682 34 52041 35 52401 36 52754 43 53112 45 53470 48 53829 49 54189 49 54548 50 54908 50 55267 51 55627 51 55987 50 56348 49 56708 49 57068 49 57428 49 57789 47 58149 47 58510 45 58872 42 59234 39 59595 37 59956 35 60318 32 60679 30 61041 27 61402 25 61763 23 62124 20 62486 17 62849 12'

rle_code_0 = 'nan'
rle_code_1 = 'nan'
rle_code_2 = '38223 4 38483 12 38747 15 39012 17 39277 19 39542 20 39807 22 40071 24 40335 26 40600 27 40864 29 41129 30 41394 31 41659 32 41925 31 42191 30 42457 30 42723 29 42989 28 43255 27 43522 25 43788 24 44055 22 44322 20 44589 18 44856 15 45128 7'


# converte le maschere in immagini
img_1 = rle_to_image(rle_code_0, height, width)
img_2 = rle_to_image(rle_code_1, height, width)
img_3 = rle_to_image(rle_code_2, height, width)

# sovrappone le 3 immagini colorate

# Merge the three masks into a three-channel mask
merged_mask = cv.merge([img_1, img_2, img_3])

# mostra l'immagine risultante
plt.imshow(merged_mask)
plt.show()