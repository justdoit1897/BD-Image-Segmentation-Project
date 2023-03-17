'''
Per stimare la somiglianza tra due immagini, puoi utilizzare una metrica di distanza 
come la SSIM (Structural Similarity Index). La SSIM è una metrica di qualità delle 
immagini che misura la somiglianza strutturale tra due immagini. Essa tiene conto di 
fattori come la luminosità, il contrasto e la struttura dell'immagine, ed è stata 
dimostrata essere una metrica più accurata rispetto ad altre metriche di distanza come 
la Mean Squared Error (MSE) o la Peak Signal-to-Noise Ratio (PSNR). 
'''

# from skimage import io, metrics

# # Carica le due immagini
# img1 = io.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/scans/slice_0001_266_266_1.50_1.50.png', as_gray=True)
# img2 = io.imread('../BD-Image-Segmentation-Comp/train/case2/case2_day1/scans/slice_0002_266_266_1.50_1.50.png', as_gray=True)

# # Calcola la similarità strutturale
# ssim = metrics.structural_similarity(img1, img2)

# # Stampa il valore di similarità
# print(f"Similarità strutturale: {ssim}")

# from skimage import io, metrics
# import pandas as pd

# # Lettura del dataframe
# df = pd.read_csv('merged_df.csv')

# print(df)

# # Raggruppa le slice per giorno e caso
# grouped = df.groupby(['case_id', 'day_id'])

# print(grouped)

# # Ciclo sui gruppi di slice
# for name, group in grouped:
    
#     count = 0
    
#     # Ciclo sulle slice
#     for i in range(len(group)):
#         # Confronta la slice con tutte le altre slice dello stesso giorno
#         for j in range(i + 1, len(group)):
#             # Carica le due immagini
#             img1 = io.imread(group.iloc[i]['path'], as_gray=True)
#             img2 = io.imread(group.iloc[j]['path'], as_gray=True)
#             # Calcola la similarità strutturale
#             ssim = metrics.structural_similarity(img1, img2)
#             # Stampa il valore di similarità
#             print(f"Caso: {name[0]}, Giorno: {name[1]}, Slice {i+1} vs Slice {j+1}: Similarità strutturale = {ssim}")
#             count = count + 1
#     print(f"\ncount = {count}\n")
            
            
'''

import numpy as np
from skimage import io, metrics

# Lista dei giorni disponibili
days = df['day'].unique()

# Inizializza una matrice vuota
ssim_matrix = np.zeros((len(days), len(days)))

# Calcola la similarità tra le immagini di ogni giorno e salva i risultati nella matrice
for i, day_i in enumerate(days):
    img_i = io.imread(df.loc[df['day'] == day_i, 'image_path'].iloc[0], as_gray=True)
    for j, day_j in enumerate(days):
        img_j = io.imread(df.loc[df['day'] == day_j, 'image_path'].iloc[0], as_gray=True)
        ssim_matrix[i, j] = metrics.structural_similarity(img_i, img_j)

print(ssim_matrix)

'''

# from skimage import io, metrics
# import numpy as np
# from tqdm import tqdm
# import pandas as pd

# # Leggi il dataframe
# df = pd.read_csv('merged_df.csv')

# # Inizializza la matrice delle similarità
# n_slices = len(df)
# n_days = len(df['day_id'].unique())
# ssim_matrix = np.zeros((n_slices, n_slices))

# # Confronta le slice dello stesso giorno
# for day in tqdm(df['day_id'].unique(), desc="Days processed", position=0):
#     day_slices = df[df['day_id']==day]
#     n_slices_day = len(day_slices)
#     for i in tqdm(range(n_slices_day), desc=f"Day {day}", position=1):
#         for j in range(i+1, n_slices_day):
#             # Carica le due immagini
#             img1_path = day_slices.iloc[i]['path']
#             img2_path = day_slices.iloc[j]['path']
#             img1 = io.imread(img1_path, as_gray=True)
#             img2 = io.imread(img2_path, as_gray=True)
#             # Calcola la similarità strutturale
#             ssim = metrics.structural_similarity(img1, img2)
#             # Salva il valore di similarità nella matrice
#             ssim_matrix[i,j] = ssim
#             ssim_matrix[j,i] = ssim

# # Salva la matrice in un file
# np.savetxt('ssim_matrix.csv', ssim_matrix, delimiter=',')


# from skimage import io, metrics
# import pandas as pd
# from tqdm import tqdm

# # Lettura del dataframe
# df = pd.read_csv('merged_df.csv')

# # Raggruppa le slice per giorno e caso
# grouped = df.groupby(['case_id', 'day_id'])

# # Ciclo sui gruppi di slice
# for name, group in grouped:
#     # Inizializza la progress bar
#     pbar = tqdm(total=len(group)*(len(group)-1)//2, desc=f"Caso: {name[0]}, Giorno: {name[1]}")
#     # Matrice delle similarità
#     similarity_matrix = []
#     # Ciclo sulle slice
#     for i in range(len(group)):
#         # Confronta la slice con tutte le altre slice dello stesso giorno
#         for j in range(i + 1, len(group)):
#             # Carica le due immagini
#             img1 = io.imread(group.iloc[i]['path'], as_gray=True)
#             img2 = io.imread(group.iloc[j]['path'], as_gray=True)
#             # Calcola la similarità strutturale
#             ssim = metrics.structural_similarity(img1, img2)
#             # Aggiungi la similarità alla matrice
#             similarity_matrix.append(ssim)
#             # Aggiorna la progress bar
#             pbar.update(1)
#     # Stampa la matrice delle similarità
#     print(similarity_matrix)


from skimage import io, metrics
import pandas as pd
from tqdm import tqdm

# Lettura del dataframe
df = pd.read_csv('merged_df.csv')

# Raggruppa le slice per giorno e caso
grouped = df.groupby(['case_id', 'day_id'])

# Inizializza il dataframe per salvare le informazioni di similarità
similar_images_df = pd.DataFrame(columns=['case_id', 'day_id', 'path_1', 'path_2', 'similarity'])

# Ciclo sui gruppi di slice
for name, group in grouped:
    # Ciclo sulle slice
    for i in tqdm(range(len(group)), desc=f"Caso: {name[0]}, Giorno: {name[1]}"):
        # Confronta la slice con tutte le altre slice dello stesso giorno
        for j in range(i + 1, len(group)):
            # Carica le due immagini
            img1 = io.imread(group.iloc[i]['path'], as_gray=True)
            img2 = io.imread(group.iloc[j]['path'], as_gray=True)
            # Calcola la similarità strutturale
            ssim = metrics.structural_similarity(img1, img2)
            # Se il coefficiente di similarità è maggiore o uguale a 0.95, salva le informazioni
            if ssim >= 0.95:
                row = {
                    'case_id': name[0],
                    'day_id': name[1],
                    'path1': group.iloc[i]['path'],
                    'path2': group.iloc[j]['path'],
                    'ssim': ssim
                }
                similar_images_df = pd.concat([similar_images_df, pd.DataFrame(row, index=[0])], ignore_index=True)
            # Stampa il valore di similarità
            # print(f"Caso: {name[0]}, Giorno: {name[1]}, Slice {i+1} vs Slice {j+1}: Similarità strutturale = {ssim}")

# Salva il dataframe in un file CSV
similar_images_df.to_csv('similar_images.csv', index=False)

# # Stampa i percorsi delle immagini simili
# print("Immagini simili:")
# for path1, path2 in similar_images:
#     print(f"{path1}, {path2}")


# from skimage import io, metrics
# import pandas as pd
# from tqdm import tqdm

# # Lettura del dataframe
# df = pd.read_csv('merged_df.csv')

# # Raggruppa le slice per giorno e caso
# grouped = df.groupby(['case_id', 'day_id'])

# # Inizializza il dataframe per salvare le informazioni di similarità
# similar_images_df = pd.DataFrame(columns=['case_id', 'day_id', 'path_1', 'path_2', 'similarity'])

# # Ciclo sui gruppi di slice
# for name, group in tqdm(grouped):
#     # Ciclo sulle slice
#     for i in range(len(group)):
#         # Confronta la slice con tutte le altre slice dello stesso giorno
#         for j in range(i + 1, len(group)):
#             # Carica le due immagini
#             img1 = io.imread(group.iloc[i]['path'], as_gray=True)
#             img2 = io.imread(group.iloc[j]['path'], as_gray=True)
#             # Calcola la similarità strutturale
#             ssim = metrics.structural_similarity(img1, img2)
#             # Se il coefficiente di similarità è maggiore o uguale a 0.95, salva le informazioni
#             if ssim >= 0.95:
#                 row = {
#                     'case_id': name[0],
#                     'day_id': name[1],
#                     'path1': group.iloc[i]['path'],
#                     'path2': group.iloc[j]['path'],
#                     'ssim': ssim
#                 }
#                 similar_images_df = pd.concat([similar_images_df, pd.DataFrame(row, index=[0])], ignore_index=True)
                
# # Salva il dataframe in un file CSV
# similar_images_df.to_csv('similar_images.csv', index=False)