import numpy as np
import io

SCANS_PATH = "./training/scans.tif"
MASKS_PATH = "./training/masks.tif"

scans = io.imread(SCANS_PATH)[:1024]
masks = io.imread(MASKS_PATH)[:1024]

print(f"SCANS SHAPE: {scans.shape}")
print(f"MASKS SHAPE: {masks.shape}")


# y_true e y_pred sono i tensori delle maschere di segmentazione

# calcola la distanza di Hausdorff per ogni coppia di maschere di segmentazione
dH = []
for i in range(len(y_true)):
    dH.append(hausdorff_distance(y_true[i], y_pred[i]))

# converte dH in un array numpy per calcolare il valore minimo e massimo
dH = np.array(dH)

# calcola il valore minimo e massimo della distanza di Hausdorff per ogni maschera
dH_min = np.min(dH, axis=(1,2,3))
dH_max = np.max(dH, axis=(1,2,3))

scaled_dH = (dH - dH_min.reshape(-1, 1, 1, 1)) / (dH_max.reshape(-1, 1, 1, 1) - dH_min.reshape(-1, 1, 1, 1))
