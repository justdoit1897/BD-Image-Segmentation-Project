import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "../BD-Image-Segmentation-Comp/" 
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

cases = os.listdir(TRAIN_DIR)
print(cases)

# ordino in ordine crescente la lista
cases.sort(key=lambda x: int(x[4:]))
print(cases)

cases_path = []
case_days = []

'''

for case in cases:
    
    case_path = os.path.join(TRAIN_DIR, case)
    
    case_days = os.listdir(case_path)
    
    cases_path.append({case_path: case_days})
    
# Lettura percorsi: [giorni]    
for case_path_dict in cases_path:
    
    case_path = list(case_path_dict.keys())[0]
    
    case_days = case_path_dict[case_path]
    
    print(f"I giorni del caso {case_path} sono: {case_days}")
    
'''

for case in cases:
    
    case_path = os.path.join(TRAIN_DIR, case)
    case_days = os.listdir(case_path)
    
    # ordino i giorni in ordine crescente
    case_days.sort()
    
    case_dict = {case_path: [os.path.join(case_path, day) for day in case_days]}
    cases_path.append(case_dict)

"""

for case_path_dict in cases_path:
    
    case_path = list(case_path_dict.keys())[0]
    
    case_days = case_path_dict[case_path]
    
    print(f"I percorsi dei giorni del caso {case_path} sono: {case_days}")

"""
    
for case_dict in cases_path:
    for case_path, day_paths in case_dict.items():
        print(f"\tCase path: {case_path}\n")
        for day_path in day_paths:
            print(f"Day path: {day_path}")
            for scan_path in os.listdir(day_path):
                print(f"Scan path: {os.path.join(day_path, scan_path)}\n")

'''
La lista scans_per_day contiene le scansioni suddivise per caso e giorno, in cui ogni elemento
rappresenta un caso, ogni caso è una lista di giorni e ogni giorno è una lista di scansioni 
in ordine crescente.
'''

scans_per_day = []

for case in cases:
    case_path = os.path.join(TRAIN_DIR, case)
    case_days = os.listdir(case_path)

    scans_by_case = []

    for day in case_days:
        day_path = os.path.join(case_path, day)
        scan_path = os.path.join(day_path, "scans")
        scans = os.listdir(scan_path)
        scans.sort()
        scans_by_case.append(scans)

    scans_per_day.append(scans_by_case)

from tabulate import tabulate

for case, scans in zip(cases, scans_per_day):
    print(f"Case {case[4:]}")
    headers = ["Day", "Scans"]
    data = []
    for day in scans:
        data.append([f"Day {scans.index(day)}", "\n".join(day)])
    print(tabulate(data, headers=headers, tablefmt="grid"))





# Carica l'immagine
img = cv2.imread("slice_0021_266_266_1.50_1.50.png")

# Imposta la scala in mm
scale_x = 1.50
scale_y = 1.50

# Calcola le dimensioni dell'immagine scalata
height, width, _ = img.shape

print(height, width)

new_height = int(scale_x * height)
new_width = int(scale_y * width)

print(new_height, new_width)

# Ridimensiona l'immagine
img_scaled = cv2.resize(img, (new_width, new_height))

# Imposta l'extent dell'immagine in mm
extent = [0, new_width*scale_x, 0, new_height*scale_y]

# Rappresenta l'immagine con scala in mm
plt.imshow(img_scaled, extent=extent)

# Mostra l'immagine
plt.show()
