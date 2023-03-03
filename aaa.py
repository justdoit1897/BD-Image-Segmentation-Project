from PIL import Image

def create_image_from_rle(width, height, rle_data):
    # Creiamo un'immagine vuota di dimensioni width * height
    img = Image.new('RGB', (width, height), color='white')
    pixels = img.load()

    # Inizializziamo le variabili per tenere traccia della posizione corrente e del colore corrente
    x, y = 0, 0
    current_color = None

    # Analizziamo la codifica RLE e coloriamo i pixel corrispondenti
    for i in range(0, len(rle_data), 2):
        pixel = rle_data[i]
        length = rle_data[i+1]

        # Se non abbiamo un colore corrente, lo inizializziamo
        if current_color is None:
            current_color = pixel

        # Coloriamo i pixel corrispondenti
        for j in range(length):
            pixels[x, y] = current_color
            x += 1

            # Se abbiamo raggiunto la fine della riga, passiamo alla riga successiva
            if x == width:
                x = 0
                y += 1

        # Cambiamo colore solo se necessario
        if i+2 < len(rle_data) and rle_data[i+2] != current_color:
            current_color = rle_data[i+2]

    return img

img = create_image_from_rle(5, 3, [255, 2, 0, 3, 255, 2, 0, 1])
img.show()
