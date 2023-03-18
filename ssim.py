# Per stimare la somiglianza tra due immagini, una metrica di distanza molto utilizzata è
# la SSIM (Structural Similarity Index). Questa metrica tiene conto di fattori come 
# la luminosità, il contrasto e la struttura dell'immagine, ed è stato dimostrato che questa
# è una metrica più accurata rispetto ad altre metriche di distanza come la Mean Squared 
# Error (MSE) o la Peak Signal-to-Noise Ratio (PSNR). 

# Imports

from skimage import io, metrics
import pandas as pd
from tqdm import tqdm

# Lettura del dataframe
df = pd.read_csv('merged_df.csv')

# Raggruppa le slice per giorno e caso
grouped = df.groupby(['case_id', 'day_id'])

# Inizializza il dataframe per salvare le informazioni di similarità
similar_images_df = pd.DataFrame(columns=['case_id', 'day_id', 'path_1', 'path_2', 'ssim'])

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
                    'path_1': group.iloc[i]['path'],
                    'path_2': group.iloc[j]['path'],
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