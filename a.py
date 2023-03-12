import cv2
import numpy as np
import os

BASE_DIR = "../BD-Image-Segmentation-Comp/" 
TRAIN_DIR = os.path.join(BASE_DIR, 'train')

# Carica le immagini di input
img1 = cv2.imread(os.path.join(TRAIN_DIR, "case146", "case146_day0", "scans", "slice_0003_266_266_1.50_1.50.png"), 0)
img2 = cv2.imread(os.path.join(TRAIN_DIR, "case146", "case146_day0", "scans", "slice_0004_266_266_1.50_1.50.png"), 0)

# Inizializza l'algoritmo SIFT
sift = cv2.SIFT_create()

# Trova i punti chiave e i descrittori per le due immagini
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Inizializza il matcher di features di tipo FLANN
matcher = cv2.FlannBasedMatcher()

# Trova le corrispondenze tra i descrittori delle due immagini
matches = matcher.knnMatch(des1, des2, k=2)

# Conserva solo le corrispondenze che superano una soglia di distanza
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Estrae i punti chiave corrispondenti nelle due immagini
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Stima la trasformazione che allinea le due immagini
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Applica la trasformazione all'immagine di input
aligned_img = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

# Salva l'immagine allineata
cv2.imwrite('aligned_image.png', aligned_img)