# import numpy as np
# from scipy.interpolate import interp2d
# from skimage import io
# import cv2 as cv

# Carica le immagini e le maschere di segmentazione
# img1 = cv.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/scans/slice_0049_266_266_1.50_1.50.png')
# mask1 = cv.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0049_266_266_1.50_1.50.png')      # maschera di cui si vuole fare l'interpolazione

# img2 = cv.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/scans/slice_0050_266_266_1.50_1.50.png')
# mask2 = cv.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0050_266_266_1.50_1.50.png')

# img3 = cv.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/scans/slice_0051_266_266_1.50_1.50.png')
# mask3 = cv.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0051_266_266_1.50_1.50.png')
# # ...

# # Definisci il numero di immagini vicine da usare per l'interpolazione
# n = 3

# # Inizializza l'array dei valori interpolati
# interp_mask = np.zeros_like(mask1)

# # Ciclo sui pixel della maschera di segmentazione
# for i in range(mask1.shape[0]):
#     for j in range(mask1.shape[1]):
#         # Verifica se il pixel è nero in tutti i canali RGB
#         if np.all(mask1[i,j] == [0, 0, 0]):
#             # Se il pixel è nero, cerca i n pixel vicini nella maschera di segmentazione
#             neighbors = []
#             for k in range(1, n+1):
#                 if i-k >= 0 and np.any(mask1[i-k,j] != [0, 0, 0]):
#                     neighbors.append((i-k, j, mask1[i-k,j]))
#                 if i+k < mask1.shape[0] and np.any(mask1[i+k,j] != [0, 0, 0]):
#                     neighbors.append((i+k, j, mask1[i+k,j]))
#                 if j-k >= 0 and np.any(mask1[i,j-k] != [0, 0, 0]):
#                     neighbors.append((i, j-k, mask1[i,j-k]))
#                 if j+k < mask1.shape[1] and np.any(mask1[i,j+k] != [0, 0, 0]):
#                     neighbors.append((i, j+k, mask1[i,j+k]))
#             # Se ci sono almeno n pixel vicini con un valore noto, usa la loro media per stimare il valore del pixel corrente
#             if len(neighbors) >= n:
#                 val = sum([np.mean(p[2]) for p in neighbors]) / n
#                 interp_mask[i,j] = [val, val, val]
#             else:
#                 # Altrimenti cerca gli n pixel vicini
#                 img_data = []
#                 for img, mask in [(img2, mask2), (img3, mask3)]:
#                     img_data.extend([(img[p[0],p[1]], p[2]) for p in neighbors if mask[p[0],p[1]] != 0])
#                 # Se ci sono almeno n pixel vicini con un valore noto tra le immagini vicine, usa la loro media per stimare il valore del pixel corrente
#                 if len(img_data) >= n:
#                     val = sum([p[0]*p[1] for p in img_data]) / sum([p[1] for p in img_data])
#                     interp_mask[i,j] = val

# # Salva la maschera di segmentazione interpolata

# cv.imwrite('interp_mask.png', interp_mask)
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caricamento delle immagini delle maschere
img1 = cv2.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0076_266_266_1.50_1.50.png')
img2 = cv2.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0077_266_266_1.50_1.50.png')
img3 = cv2.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0078_266_266_1.50_1.50.png')

# Conversione delle immagini in scala di grigi
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Applicazione di un filtro di media per rimuovere eventuali rumori
kernel = np.ones((5, 5), np.float32)/25
gray1 = cv2.filter2D(gray1, -1, kernel)
gray2 = cv2.filter2D(gray2, -1, kernel)
gray3 = cv2.filter2D(gray3, -1, kernel)

# Interpolazione delle tre immagini per ottenere la maschera finale
final_mask = cv2.addWeighted(gray1, 1/3, gray2, 1/3, 0)
final_mask = cv2.addWeighted(final_mask, 1, gray3, 1/3, 0)

# Salvataggio della maschera finale
cv2.imwrite('final_mask.png', final_mask)

'''
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

os.environ["QT_QPA_PLATFORM"] = "xcb"

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

# Esempio di utilizzo
masks = [
        '../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0074_266_266_1.50_1.50.png',
        '../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0075_266_266_1.50_1.50.png',
        '../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0076_266_266_1.50_1.50.png', 
        '../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0077_266_266_1.50_1.50.png', 
        '../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0078_266_266_1.50_1.50.png'
        ]
output_path = 'interpolated_mask.png'
interpolated_mask, encoded_channels = interpolate_masks(masks, output_path)

plt.imshow(interpolated_mask)
plt.show()

# Adesso si vuole confrontare l'utilizzo di una maschera predefinita del dataset con una maschera ottenuta
# per interpolazione (in teoria dovrebbe essere più completa poiché frutto dell'evoluzione delle acquisizioni)

# Caricamento dell'immagine e della maschera (da dataset)
img = cv.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/scans/slice_0074_266_266_1.50_1.50.png')

mask_before = cv.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0074_266_266_1.50_1.50.png')

# Conversione dell'immagine in RGB
mask_before = cv.cvtColor(mask_before, cv.COLOR_BGR2RGB)

mask_after = cv.imread('interpolated_mask.png')

# Sovrapposizione della maschera all'immagine
before = cv.addWeighted(img, 1, mask_before, 1, 0)

bool_index = img <= (200, 200, 200)

img[bool_index] = mask_after[bool_index]
# after = cv.addWeighted(img, 0.3, mask_after, 0.7, 0)

# VISUALIZZAZIONE DELLE IMMAGINI

# Crea una figura e due assi
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Mostra le immagini sugli assi
ax1.imshow(before)
ax2.imshow(img)

# Imposta i titoli degli assi
ax1.set_title("Mask w/o interpol.")
ax2.set_title("Mask w. interpol.")

# Mostra la figura
plt.show()

# from scipy import interpolate
# import numpy as np
# import cv2

# # Lettura delle immagini vicine
# img1 = cv2.imread("../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0050_266_266_1.50_1.50.png")
# img2 = cv2.imread("../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0051_266_266_1.50_1.50.png")
# img3 = cv2.imread("../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0052_266_266_1.50_1.50.png")
# img4 = cv2.imread("../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0053_266_266_1.50_1.50.png")

# # Lettura della maschera di segmentazione
# mask = cv2.imread("../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/mask_slice_0049_266_266_1.50_1.50.png")

# # Estrazione dei canali RGB delle immagini vicine
# r1, g1, b1 = img1[:,:,0], img1[:,:,1], img1[:,:,2]
# r2, g2, b2 = img2[:,:,0], img2[:,:,1], img2[:,:,2]
# r3, g3, b3 = img3[:,:,0], img3[:,:,1], img3[:,:,2]
# r4, g4, b4 = img4[:,:,0], img4[:,:,1], img4[:,:,2]

# # Creazione della griglia di punti
# x = np.arange(mask.shape[1])
# y = np.arange(mask.shape[0])
# xx, yy = np.meshgrid(x, y)

# # Creazione della griglia di valori
# values = np.dstack((r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4))

# # Creazione della funzione interpolante
# f = interpolate.interp2d(xx, yy, values, kind='cubic')

# # Applicazione della funzione interpolante alla maschera di segmentazione
# mask_interpolated = f(x, y)

# # Salvataggio della maschera interpolata
# cv2.imwrite("mask_interpolated.png", mask_interpolated)

