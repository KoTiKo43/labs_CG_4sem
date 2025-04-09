import numpy as np
from PIL import Image, ImageOps
from math import ceil, floor, sqrt, cos, sin, pi

# Конфигурационные параметры
H, W = 2000, 2000
HT, WT = 1024, 1024
U0, V0 = W/2, H/2
AX, AY = 1000, 1000
LIGHT_DIR = np.array([0, 0, 1])

# Инициализация буферов
img_mat = np.zeros((H, W, 3), dtype=np.uint8)
img_text = np.zeros((H, W, 3), dtype=np.uint8)
z_buffer = np.full((H, W), np.inf, dtype=float)

def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    denominator = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    lambda0 = ((x - x2)*(y1 - y2) - (x1 - x2)*(y - y2)) / denominator
    lambda1 = ((x0 - x2)*(y - y2) - (x - x2)*(y0 - y2)) / denominator
    return (lambda0, lambda1, 1.0 - lambda0 - lambda1)

def compute_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    v1 = np.array([x1-x0, y1-y0, z1-z0])
    v2 = np.array([x2-x0, y2-y0, z2-z0])
    return np.cross(v1, v2)

def project_vertex(x, y, z):
    return AX*x/z + U0, AY*y/z + V0

def norm_scalar(normal):
    norm = np.linalg.norm(normal)
    return np.dot(LIGHT_DIR, normal) / norm

def draw_triangle(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, i0, i1, i2):
    # Проекция вершин
    proj0 = project_vertex(x0, y0, z0)
    proj1 = project_vertex(x1, y1, z1)
    proj2 = project_vertex(x2, y2, z2)
    
    # Ограничивающий прямоугольник
    x_min = max(0, int(min(x0, x1, x2)))
    x_max = min(W-1, int(max(x0, x1, x2)) + 1)
    y_min = max(0, int(min(y0, y1, y2)))
    y_max = min(H-1, int(max(y0, y1, y2)) + 1)
    
    # Проверка ориентации
    normal = compute_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    if norm_scalar(normal) > 0: return

    # Растеризация
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            lambdas = barycentric_coordinates(i, j, *proj0, *proj1, *proj2)
            if all(l >= 0 for l in lambdas):
                z = lambdas[0]*z0 + lambdas[1]*z1 + lambdas[2]*z2
                if z < z_buffer[j][i]:
                    intensity = sum(l*c for l, c in zip(lambdas, [i0, i1, i2]))
                    img_mat[j][i][0], img_mat[j][i][1], img_mat[j][i][2] = -255*intensity, -100*intensity, 0
                    z_buffer[j][i] = z

def apply_texture(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, uv0, uv1, uv2, texture, i0, i1, i2):
    # Наложение текстуры с учетом освещения
    proj0 = project_vertex(x0, y0, z0)
    proj1 = project_vertex(x1, y1, z1)
    proj2 = project_vertex(x2, y2, z2)
    
    xmin = floor(min(proj0[0], proj1[0], proj2[0]))
    xmax = ceil(max(proj0[0], proj1[0], proj2[0]))
    ymin = floor(min(proj0[1], proj1[1], proj2[1]))
    ymax = ceil(max(proj0[1], proj1[1], proj2[1]))
    
    if (xmin < 0):
        xmin = 0
    if (xmax > W):
        xmax = W
    if (ymin < 0):
        ymin = 0
    if (ymax > H):
        ymax = H
    
    normal = compute_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    if norm_scalar(normal) > 0: return

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambdas = barycentric_coordinates(i, j, *proj0, *proj1, *proj2)
            if all(l >= 0 for l in lambdas):
                z = lambdas[0]*z0 + lambdas[1]*z1 + lambdas[2]*z2
                if z < z_buffer[j][i]:
                    textur_c = [round(HT*(lambdas[0]*uv0[0]+lambdas[1]*uv1[0]+lambdas[2]*uv2[0])),
                                round(WT*(lambdas[0]*uv0[1]+lambdas[1]*uv1[1]+lambdas[2]*uv2[1]))]
                                                                     
                    color = texture.getpixel((textur_c[0],textur_c[1]))

                    img_mat[j][i] = (#-color[0]*(i0*lambdas[0]+i1*lambdas[1]+i2*lambdas[2]),
                                    #-color[1]*(i0*lambdas[0]+i1*lambdas[1]+i2*lambdas[2]),
                                    #-color[2]*(i0*lambdas[0]+i1*lambdas[1]+i2*lambdas[2]))
                                    color)
                    z_buffer[j][i] = z

def rotate_vertex(x, y, z, angles, translate):
    # Поворот и трансляция вершины
    rx = np.array([[1, 0, 0], 
                 [0, cos(angles[0]), sin(angles[0])], 
                 [0, -sin(angles[0]), cos(angles[0])]])
    
    ry = np.array([[cos(angles[1]), 0, sin(angles[1])], 
                 [0, 1, 0], 
                 [-sin(angles[1]), 0, cos(angles[1])]])
    
    rz = np.array([[cos(angles[2]), sin(angles[2]), 0], 
                 [-sin(angles[2]), cos(angles[2]), 0], 
                 [0, 0, 1]])
    
    rotated = np.dot(rz, np.dot(ry, np.dot(rx, np.array([x, y, z]))))
    return rotated + np.array(translate)

# Загрузка данных
model = []
texture_coords = []
faces = []

with open(r'C:\\Users\akimn\Documents\Лабы КГ\labs_CG_4sem\models\zayac.obj') as f:
    for line in f:
        parts = line.split()
        if not parts: continue
        if parts[0] == 'v': model.append(list(map(float, parts[1:4])))
        elif parts[0] == 'vt': texture_coords.append(list(map(float, parts[1:3])))
        elif parts[0] == 'f': faces.append([tuple(map(int, v.split('/'))) for v in parts[1:4]])

# Применение трансформаций
for i in range(len(model)):
    model[i] = rotate_vertex(*model[i], # x, y, z
                             (0, pi/2, 0), # Углы
                             (0, -0.03, 0.2) # tx, ty, tz
                             )

# Вычисление вершинных нормалей
vertex_normals = np.zeros((len(model), 3))
for face in faces:
    v0, v1, v2 = [model[idx[0]-1] for idx in face]
    normal = compute_normal(*v0, *v1, *v2)
    for idx in face:
        vertex_normals[idx[0]-1] += normal

# Нормализация нормалей
vertex_normals = [n/np.linalg.norm(n) if np.linalg.norm(n)!=0 else n for n in vertex_normals]

# Основной цикл рендеринга
texture = ImageOps.flip(Image.open(r'C:\\Users\akimn\Documents\Лабы КГ\labs_CG_4sem\textures\zayac.jpg'))

for face in faces:
    # Получение данных для треугольника
    v_indices = [idx[0]-1 for idx in face]
    vt_indices = [idx[1]-1 for idx in face]
    
    # Геометрические данные
    v0, v1, v2 = [model[i] for i in v_indices]
    intensities = [norm_scalar(vertex_normals[i]) for i in v_indices] # I0, I1, I2
    
    # Данные текстуры
    uv0 = texture_coords[vt_indices[0]]
    uv1 = texture_coords[vt_indices[1]]
    uv2 = texture_coords[vt_indices[2]]
    
    # Отрисовка
#    draw_triangle(img_mat, *v0, *v1, *v2, *intensities)
    apply_texture(img_text, *v0, *v1, *v2, uv0, uv1, uv2, texture, *intensities)

image = Image.fromarray(img_mat, mode='RGB')
image2 = Image.fromarray(img_text, mode='RGB')
image = ImageOps.flip(image)
image2 = ImageOps.flip(image2)
image.show()
image2.show()
#image.save(r'C:\\Users\akimn\Documents\Лабы КГ\labs_CG_4sem\output_images\Крол.png')
image2.save(r'C:\\Users\akimn\Documents\Лабы КГ\labs_CG_4sem\output_images\КролТекст.png')
