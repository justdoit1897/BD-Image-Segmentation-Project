import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from PIL import Image
from torchvision.transforms import ToTensor, Resize
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from glob import glob

BASE_DIR = "../BD-Image-Segmentation-Comp/" 
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

#apro una immagine della cartella "train"


img = cv.imread(os.path.join(TRAIN_DIR, "case146", "case146_day0", "scans", "slice_0003_266_266_1.50_1.50.png"), 0) 
#img_buona = cv.imread(os.path.join(TRAIN_DIR, "case35", "case35_day15", "scans", "slice_0048_276_276_1.63_1.63.png"), 0) 

#img_ritagliata = img_buona[4:270, 4:270]

equ = cv.equalizeHist(img)

res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('res.png',res)

##################################################################################################

# import packages
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
  
# reading main image
img1 = cv.imread(os.path.join(TRAIN_DIR, "case146", "case146_day0", "scans", "slice_0003_266_266_1.50_1.50.png"), 0) 
  
# checking the number of channels
print('No of Channel is: ' + str(img1.ndim))
  
# reading reference image
img2 = cv.imread(os.path.join(TRAIN_DIR, "case35", "case35_day15", "scans", "slice_0048_276_276_1.63_1.63.png"), 0) 
  
# checking the number of channels
print('No of Channel is: ' + str(img2.ndim))
  
image = img1
reference = img2
  
matched = match_histograms(image, reference, channel_axis=None)
  
  
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, 
                                    figsize=(8, 3),
                                    sharex=True, sharey=True)
  
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()
  
ax1.imshow(image)
ax1.set_title('Source')
ax2.imshow(reference)
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')
  
plt.tight_layout()
plt.show()
  
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
  
for i, img in enumerate((image, reference, matched)):
    for c, c_color in enumerate(('red', 'green', 'blue')):
        img_hist, bins = exposure.histogram(img[..., c], 
                                            source_range='dtype')
        axes[c, i].plot(bins, img_hist / img_hist.max())
        img_cdf, bins = exposure.cumulative_distribution(img[..., c])
        axes[c, i].plot(bins, img_cdf)
        axes[c, 0].set_ylabel(c_color)
  
axes[0, 0].set_title('Source')
axes[0, 1].set_title('Reference')
axes[0, 2].set_title('Matched')
  
plt.tight_layout()
plt.show()

