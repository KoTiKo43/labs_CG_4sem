import numpy as np
from PIL import Image

model_matrix = []

with open('model_1.obj') as file:
    for s in file:
        s_splt = s.split()
        
        if s_splt[0] == 'v':
            model_matrix.append(list(map(lambda x: float(x), s_splt[1:])))
            
H, W = 1000, 1000

image_mtrx = np.zeros((H, W), dtype=np.uint8)
for vertex in model_matrix:
    x = round(vertex[0] * 5000 + 500)
    y = round(vertex[1] * 5000 + 500)
    image_mtrx[-y, -x] = 255

image = Image.fromarray(image_mtrx, mode='L')
image.save('4.png')
image.show()
