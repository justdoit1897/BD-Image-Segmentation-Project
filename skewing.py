import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

img = cv.imread("slice_0081_266_266_1.50_1.50.png")
rows, cols = img.shape[:2]
angle = -10  # angolazione in gradi

def do_skewing(slice, mask):
    
    rows, cols = slice.shape[:2]
    
    # # angolazione in gradi
    # angle = random.randint(-10, 10)
    # while angle == 0:
    #     angle = random.randint(-10, 10)
    
    angle = 180
    
    
    
    # definisco la matrice di trasformazione affine
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    M1 = np.float32([[-1, 0, cols-1], [0, 1, 0]])
    M2 = np.float32([[1, 0, 0], [0, -1, rows-1]])
    
    img_mirror = cv.warpAffine(img, M, (cols, rows))

    # applico lo skewing alla slice
    skewed_slice = cv.warpAffine(slice, M, (cols, rows))
    
    # applico lo skewing alla maschera corrispondente
    skewed_mask = cv.warpAffine(mask, M2, (cols, rows))
    
    return skewed_slice, skewed_mask

imgg, skewed_img = do_skewing(img, img)


# Crea una figura e due assi
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Mostra le immagini sugli assi
ax1.imshow(img)
ax2.imshow(skewed_img)

# Imposta i titoli degli assi
ax1.set_title("Slice originale")
ax2.set_title("Slice con skewing")

# Mostra la figura
plt.show()
