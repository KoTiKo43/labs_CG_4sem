import numpy as np
from PIL import Image

image_mtrx = np.zeros((600, 800, 3), dtype=np.uint8)

for x in range(800):
    for y in range(600):
        image_mtrx[y, x] = ((x - y) * (x + y)) % 256

image = Image.fromarray(image_mtrx, mode='RGB')
image.save("3.png")
image.show()
