import numpy as np # linear algebra
import pandas as pd

from PIL import Image

def rle_decode(mask_rle: str = '', shape: tuple = (234, 234)):
    '''
    Decode rle encoded mask.
    
    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str='img.jpg', shape: tuple = (234, 234)):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask
            
    return masks

########################################################################

def inverseRleToMask(inverseRle, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)
    rleNumbers = [int(x) for x in inverseRle.split()]
    index = 0
    for i in range(0, len(rleNumbers), 2):
        start = index + rleNumbers[i]
        end = start + rleNumbers[i+1]
        mask[start:end] = 1
        index = end
    mask = mask.reshape((height, width)).T
    return mask


large_bowel = '21716 5 21725 5 21981 17 22246 19 22512 21 22778 23 23044 24 23310 26 23576 27 23842 29 24108 30 24374 32 24640 33 24907 32 25173 33 25441 31 25709 30 25978 28 26246 27 26513 27 26780 26 27046 27 27313 26 27580 25 27847 25 28114 24 28383 21 28652 19 28918 19 29184 19 29450 19 29716 20 29983 19 30249 21 30516 23 30782 25 31048 26 31315 26 31581 26 31848 26 32114 26 32381 25 32647 24 32914 23 33181 21 33447 20 33714 18 33981 16 34248 14 34515 12 34782 11 35049 9 35316 7 36128 5 36392 8 36658 9 36923 11 37189 11 37455 11 37721 12 37987 12 38253 14 38519 15 38785 16 39051 17 39317 17 39583 18 39849 17 40115 17 40381 16 40647 16 40913 14 41179 13 41445 13 41711 13 41978 11 42244 11 42511 9 42779 5'
small_bowel = '26234 4 26498 8 26764 10 27029 13 27295 16 27561 19 27825 22 28090 23 28355 24 28621 24 28887 24 29153 24 29419 25 29685 25 29951 26 30218 25 30485 26 30751 27 31018 27 31285 27 31551 27 31818 26 32085 25 32353 23 32622 21 32889 20 33155 21 33422 20 33688 20 33955 19 34223 17 34493 13 34759 13 35026 12 35293 10 35560 8 35828 4'
stomach = '22749 8 23013 11 23277 14 23541 17 23806 19 24071 21 24335 23 24599 25 24863 27 25127 29 25392 30 25657 31 25922 32 26188 31 26454 30 26720 29 26986 27 27252 24 27518 21 27785 17 28051 15 28319 10'

'''
img_0 = inverseRleToMask(large_bowel, 234, 234)
img_1 = inverseRleToMask(small_bowel, 234, 234)
img_2 = inverseRleToMask(stomach, 234, 234)

import matplotlib.pyplot as plt

plt.imshow(img_0)
plt.show()

plt.imshow(img_1)
plt.show()

plt.imshow(img_2)
plt.show()
'''

import numpy as np

# Stringa di codifica RLE inversa

rle_string = large_bowel

altezza = 234
larghezza = 234

# Creazione dell'array numpy vuoto
mask = np.zeros(altezza * larghezza, dtype=np.uint8)

# Decodifica della stringa di codifica RLE inversa
rle_numbers = [int(x) for x in rle_string.split()]
starts = rle_numbers[0::2]
lengths = rle_numbers[1::2]

# Creazione della maschera
current_position = 0
for start, length in zip(starts, lengths):
    current_position += start
    mask[current_position : current_position + length] = 255
    current_position += length

# Ridimensionamento della maschera
mask = mask.reshape(altezza, larghezza)

# Visualizzazione della maschera
import matplotlib.pyplot as plt
plt.imshow(mask, cmap='gray')
plt.show()

