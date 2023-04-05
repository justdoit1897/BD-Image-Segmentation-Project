import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Crea un oggetto ImageDataGenerator con alcune trasformazioni di data augmentation
data_generator = ImageDataGenerator(
    rotation_range=15,  # ruota l'immagine in un range di 15 gradi
    width_shift_range=0.1,  # sposta l'immagine orizzontalmente in un range del 10% della larghezza
    height_shift_range=0.1,  # sposta l'immagine verticalmente in un range del 10% dell'altezza
    zoom_range=0.1,  # zoom in avanti o indietro in un range del 10%
    horizontal_flip=True,  # specchia l'immagine orizzontalmente
    vertical_flip=True,  # specchia l'immagine verticalmente
)

# Utilizza l'oggetto ImageDataGenerator per applicare le trasformazioni alle immagini
data_generator.fit(images)

# Addestra il modello utilizzando il data generator con le immagini aumentate
model.fit(data_generator.flow(images, labels, batch_size=32), epochs=10)
