import os
import cv2 as cv
import matplotlib.pyplot as plt
os.environ["QT_QPA_PLATFORM"] = "xcb"
import numpy as np


# Carica le due immagini
slice = cv.imread('./train/case80/case80_day0/scans/slice_0081_266_266_1.50_1.50.png')
mask = cv.imread('./train/case80/case80_day0/masks/mask_slice_0081_266_266_1.50_1.50.png')

# Specifica il peso delle due immagini
alpha = 0.2  # peso per l'immagine A
beta = 1 - alpha  # peso per l'immagine B

# Sovrappone le due immagini
result = cv.addWeighted(slice, alpha, mask, beta, 0)

# Definisco i nomi delle classi e i rispettivi colori
class_names = ['Large bowel', 'Small bowel', 'Stomach']
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

# Creo una figura vuota
fig, ax = plt.subplots()

# Aggiungo una legenda con i nomi delle classi e i rispettivi colori
legend_handles = [plt.Rectangle((0,0),1,1, color=c, ec="k") for c in colors]
ax.legend(legend_handles, class_names)

# Mostra il risultato
plt.imshow(result)
plt.show()

# Mostra l'immagine
# plt.imshow(slice, cmap='gray')
# plt.show()

# Mostra l'istogramma
plt.hist(slice.ravel(), bins=256, range=(0, 255))
plt.show()

slice_g_s = cv.cvtColor(slice, cv.COLOR_BGR2GRAY)

# Conta il numero di pixel bianchi
white_pixels = cv.countNonZero(slice_g_s)
black_pixels = np.count_nonzero(np.all(slice == 0, axis=2))

# Mostra il numero di pixel bianchi e neri
print("Il numero di pixel bianchi nell'immagine è:", white_pixels)
print("Il numero di pixel neri nell'immagine è:", black_pixels)

print(f"\nRapporto pixel bianchi/neri: {white_pixels/black_pixels}")