import tensorflow as tf

# definizione delle costanti
BATCH_SIZE = 32
NUM_IMAGES = 17000
IMAGE_SIZE = (200, 200)
NUM_CHANNELS = 3

# definizione della funzione per caricare le immagini e le maschere
def load_image_and_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=NUM_CHANNELS)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMAGE_SIZE)
    mask = tf.cast(mask, tf.float32) / 255.0
    
    return image, mask

# definizione del percorso delle immagini e delle maschere
images_path = "path_to_images_directory/"
masks_path = "path_to_masks_directory/"

# creazione della lista dei percorsi delle immagini e delle maschere
image_paths = [f"{images_path}{i}.png" for i in range(NUM_IMAGES)]
mask_paths = [f"{masks_path}{i}.png" for i in range(NUM_IMAGES)]

# creazione del Dataset per caricare le immagini e le maschere
dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
dataset = dataset.shuffle(buffer_size=NUM_IMAGES)
dataset = dataset.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
