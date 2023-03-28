'''

In OpenCV, il cropping di un'immagine viene eseguito rispetto all'origine (0, 0) 
dell'immagine stessa, ovvero l'angolo in alto a sinistra dell'immagine. 
Questo significa che se si specificano le coordinate di un'area rettangolare da ritagliare, 
si fa riferimento all'angolo in alto a sinistra dell'area rettangolare rispetto all'origine 
dell'immagine.

'''

'''

La funzione prende in input un'immagine e restituisce le coordinate dei pixel bianchi 
più esterni, ovvero i pixel che si trovano più a sinistra, più a destra, più in alto e 
più in basso rispetto all'oggetto bianco nell'immagine. La funzione utilizza la libreria 
OpenCV per trovare i contorni degli oggetti bianchi nell'immagine e quindi determinare i 
pixel bianchi più esterni.

'''

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def find_outermost_white_pixels(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    x, y, w, h = cv2.boundingRect(cnt)
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_right = (x + w, y + h)
    bottom_left = (x, y + h)

    return top_left, top_right, bottom_right, bottom_left


# Carica un'immagine in scala di grigi
img = cv2.imread("slice_0081_266_266_1.50_1.50.png")

# Trova i bordi dell'immagine
top_left, top_right, bottom_right, bottom_left = find_outermost_white_pixels(img)

# Stampa le coordinate dei bordi
print(top_left, top_right, bottom_right, bottom_left)

# image = cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 1)

cropped = img[top_left[1] : bottom_left[1], top_left[0] : top_right[0]]

plt.imshow(cropped)
plt.show()

# merged_df = pd.read_csv('merged_df.csv')

h, w = cropped.shape[:2]
print(cropped.shape[:2])

border_w = (300 - w) // 2
border_h = (300 - h) // 2

immagine_ridimensionata = cv2.copyMakeBorder(cropped, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

plt.imshow(immagine_ridimensionata)
plt.show()

print(immagine_ridimensionata.shape[:2])

normale = cv2.resize(img, [300, 300])

plt.imshow(normale)
plt.show()
print(normale.shape[:2])

'''
import cv2
import numpy as np

# # leggi l'immagine
img = cv2.imread("image.jpg")

# converte in scala di grigi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# applica una soglia binaria
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# trova i contorni
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# calcola il centro di massa
M = cv2.moments(contours[0])
cx = int(M["m10"] / M["m00"])
cy = int(M["m01"] / M["m00"])

# confronta con la metà della larghezza dell'immagine
if cx > img.shape[0] / 2:
    print("L'informazione è più spostata a destra")
else:
    print("L'informazione è più spostata a sinistra")
'''