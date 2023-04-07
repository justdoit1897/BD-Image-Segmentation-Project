import os
import tifffile

# percorso alla cartella train
TRAIN_PATH = "../BD-Image-Segmentation-Comp/train/"

for casex_folder in os.listdir(TRAIN_PATH):
    
    # print(casex_folder)
    
    casex_folder_path = os.path.join(TRAIN_PATH, casex_folder)
    
    for day in os.listdir(casex_folder_path):
        
        day_path = os.path.join(TRAIN_PATH, casex_folder, day_path)
        
        for element in os.listdir(day_path):
            print(element)
            print('\n')
        

    
    # if os.path.isdir(casex_folder_path):
    #     # lista di tutti i file nella cartella scans
    #     scan_files = sorted(os.listdir(os.path.join(casex_folder_path, 'scans')))
    #     # numero di slice
    #     num_slices = len(scan_files)
    #     # carica ogni immagine nella lista "data"
    #     data = []
    #     for scan_file in scan_files:
    #         scan_path = os.path.join(casex_folder_path, 'scans', scan_file)
    #         data.append(tifffile.imread(scan_path))
    #     # crea un file TIFF 3D utilizzando la lista "data"
    #     output_path = os.path.join(casex_folder_path, f'{casex_folder}.tif')
    #     tifffile.imwrite(output_path, data, metadata={'axes': 'ZYX', 'z': num_slices})
