import numpy as np
import tensorflow as tf
from keras import layers

def get_unet_3d(input_shape):
    

    # Definizione della 3D U-Net
    inputs = layers.Input(input_shape)

    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv1)

    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv2)

    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv3)

    pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = layers.Conv3D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv3D(256, 3, activation='relu', padding='same')(conv4)

    drop4 = layers.Dropout(0.5)(conv4)

    pool4 = layers.MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = layers.Conv3D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv3D(512, 3, activation='relu', padding='same')(conv5)

    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv3D(256, 2, activation='relu', padding='same')(layers.UpSampling3D(size=(2, 2, 2))(drop5))

    merge6 = layers.concatenate([drop4, up6], axis=-1)

    conv6 = layers.Conv3D(256, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv3D(256, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv3D(128, 2, activation='relu', padding='same')(layers.UpSampling3D(size=(2, 2, 2))(conv6))

    merge7 = layers.concatenate([conv3, up7], axis=-1)

    conv7 = layers.Conv3D(128, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv3D(64, 2, activation='relu', padding='same')(layers.UpSampling3D(size=(2, 2, 2))(conv7))

    merge8 = layers.concatenate([conv2, up8], axis=-1)

    conv8 = layers.Conv3D(64, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv3D(32, 2, activation='relu', padding='same')(layers.UpSampling3D(size=(2, 2, 2))(conv8))

    merge9 = layers.concatenate([conv1, up9], axis=-1)

    conv9 = layers.Conv3D(32, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv9)
    conv9 = layers.Conv3D(2, 3, activation='relu', padding='same')(conv9)

    conv10 = layers.Conv3D(1, 1, activation='softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    # # Compilazione della 3D U-Net
    # model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient, 'binary_crossentropy', HausdorffDistance()])
    # model.summary()
    
    return model

'''

Il codice definisce una rete neurale U-Net 3D per la segmentazione di immagini volumetriche in cui si utilizza la convoluzione tridimensionale. 
La U-Net 3D è una rete neurale convoluzionale in grado di segmentare volumi di immagini tridimensionali. La rete è composta da un'encoder e un decoder 
connessi da una serie di ponti di connessione, che consentono di preservare le informazioni spaziali e di migliorare la precisione della segmentazione.

Il codice definisce la struttura della rete utilizzando il modulo layers di Keras. L'input della rete è un tensore con dimensioni (None, None, None, 1) 
che rappresenta un volume di immagini tridimensionale con canale di profondità 1. La rete ha un'architettura a U e prevede l'utilizzo di convoluzioni 
tridimensionali, max pooling tridimensionali, convoluzioni tridimensionali trasposte (up-sampling) e connessioni skip.

La rete è composta da 9 blocchi convoluzionali, ognuno dei quali comprende due convoluzioni tridimensionali seguite da una funzione di attivazione ReLU. 
I primi quattro blocchi sono seguiti da una funzione di dropout con probabilità di dropout pari a 0,5. La rete termina con un blocco di convoluzione 
tridimensionale seguito da una funzione di attivazione softmax per produrre l'output finale della rete, che rappresenta la maschera segmentata 
dell'immagine volumetrica di input.

'''
