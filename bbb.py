from PIL import Image

def create_image_from_rle(width, height, rle):
    # Create a new image with the specified width and height
    img = Image.new('RGB', (width, height), color='red')
    
    # Convert the RLE to a list of integers
    rle_list = [int(x) for x in rle.split(' ')]
    
    # Loop over the RLE and set the corresponding pixels in the image
    i = 0
    x = 0
    y = 0
    while i < len(rle_list):
        pixel = rle_list[i]
        length = rle_list[i+1]
        for j in range(length):
            img.putpixel((x, y), (pixel, pixel, pixel))
            x += 1
            if x == width:
                x = 0
                y += 1
        i += 2
    
    return img

width = 256
height = 256
rle = '28094 3 28358 7 28623 9 28889 9 29155 9 29421 9 29687 9 29953 9 30219 9 30484 10 30750 10 31016 10 31282 10 31548 10 31814 10 32081 9 32347 8 32614 6'

img = create_image_from_rle(width, height, rle)
img.show()
