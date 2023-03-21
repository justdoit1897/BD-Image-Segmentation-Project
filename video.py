import cv2 as cv
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Specifica i percorsi delle cartelle delle immagini
path1 = "../BD-Image-Segmentation-Comp/train/case2/case2_day1/masks/"
path2 = "../BD-Image-Segmentation-Comp/train/case2/case2_day1/scans/"
# Crea una lista contenente tutte le immagini nella prima cartella
img_list1 = os.listdir(path1)

# Ordina la lista in ordine crescente di nome file
img_list1.sort()

# Crea una lista contenente tutte le immagini nella seconda cartella
img_list2 = os.listdir(path2)

# Ordina la lista in ordine crescente di nome file
img_list2.sort()

# Specifica il nome del file video di output
video_name = "video_output.avi"

# Specifica la larghezza e l'altezza del video
width = 266
height = 266

# Inizializza l'oggetto VideoWriter
fourcc = cv.VideoWriter_fourcc(*"MJPG")
out = cv.VideoWriter(video_name, fourcc, 10.0, (width, height))

# Loop attraverso tutte le immagini
for i in range(min(len(img_list1), len(img_list2))):
    # Leggi le immagini dalle due cartelle
    img_path1 = os.path.join(path1, img_list1[i])
    img1 = cv.imread(img_path1)

    img_path2 = os.path.join(path2, img_list2[i])
    img2 = cv.imread(img_path2)

    # Ridimensiona le immagini alla dimensione desiderata
    img1 = cv.resize(img1, (width, height))
    img2 = cv.resize(img2, (width, height))

    # Sovrapponi le due immagini
    alpha = 0.5
    beta = 1 - alpha
    blended = cv.addWeighted(img1, alpha, img2, beta, 0)

    # Aggiungi l'immagine sovrapposta al video
    out.write(blended)

# Chiudi l'oggetto VideoWriter
out.release()

# Visualizza il video
cap = cv.VideoCapture(video_name)

while(cap.isOpened()):
    # Leggi un frame dal video
    ret, frame = cap.read()

    if ret:
        # Visualizza il frame
        cv.imshow('Video', frame)

        # Aspetta l'input da tastiera per 25 millisecondi
        # Se il tasto 'esc' viene premuto, esci dal loop
        # Se il tasto 'barra spaziatrice' viene premuto, metti in pausa il video
        key = cv.waitKey(25)
        if key == 27:       # 27 corrisponde al codice ASCII per il tasto 'esc'
            break
        elif key == 32:     # 32 corrisponde al codice ASCII per la barra spaziatrice
            cv.waitKey(-1)  # Aspetta l'input da tastiera indefinitivamente

    else:
        break

# Rilascia tutte le risorse
cap.release()
cv.destroyAllWindows()

