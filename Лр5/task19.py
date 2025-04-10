import numpy as np
from PIL import Image, ImageOps
from math import ceil, floor, cos, sin, pi
import quaternion as qu

# Конфигурационные параметры
H, W = 3000, 3000
FOV_X, FOV_Y = 1000, 1000
LIGHT_DIR = np.array([0, 0, 1])

# Инициализация буферов
output_image = np.zeros((H, W, 3), dtype=np.uint8)
z_buffer = np.full((H, W), np.inf, dtype=float)

def parse_obj(filepath):
    vertices = []
    texture_coords = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                vertices.append(np.array(list(map(float, parts[1:4]))))
            elif parts[0] == 'vt':
                texture_coords.append(list(map(float, parts[1:3])))
            elif parts[0] == 'f':
                face = []
                for part in parts[1:]:
                    indices = part.split('/')
                    v = int(indices[0])-1 if indices[0] else -1
                    vt = int(indices[1])-1 if len(indices)>1 and indices[1] else -1
                    face.append((v, vt))
                # Триангуляция полигона
                for i in range(1, len(face)-1):
                    faces.append([face[0], face[i], face[i+1]])
    
    return vertices, texture_coords, faces

def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    x0x2 = x0 - x2
    x1x2 = x1 - x2
    y1y2 = y1 - y2
    y0y2 = y0 - y2
    
    denominator = x0x2 * y1y2 - x1x2 * y0y2
    
    lambda0 = ((x - x2) * y1y2 - x1x2 * (y - y2)) / denominator
    lambda1 = (x0x2 * (y - y2) - (x - x2) * y0y2) / denominator
    return (lambda0, lambda1, 1.0 - lambda0 - lambda1)

def project_point(x, y, z):
    if z <= 1e-6:  # Точка позади камеры
        return (-1, -1)
    return (
        FOV_X * x / z + W/2,
        FOV_Y * y / z + H/2
    )

def compute_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    v1 = np.array([x1-x0, y1-y0, z1-z0])
    v2 = np.array([x2-x0, y2-y0, z2-z0])
    return np.cross(v1, v2)

def norm_scalar(normal):
    norm = np.linalg.norm(normal)
    return np.dot(LIGHT_DIR, normal) / norm

def euler_rotation_matrix(angles):
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
    
    rotation_matrix = np.dot(rz, np.dot(ry, rx))
    return rotation_matrix

def quat_rotation_matrix(q):
    a, b, c, d =  q.w, q.x, q.y, q.z
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    return np.array([
        [aa + bb - cc - dd, 2.0 * (b*c - a*d), 2.0 * (b*d + a*c)],
        [2.0 * (b*c - a*d), aa - bb + cc - dd, 2.0 * (c*d - a*b)],
        [2.0 * (b*d - a*c), 2.0 * (c*d + a*b), aa - bb - cc + dd]
    ])

def compute_vertex_normal(vertices, faces):
    # Вычисление вершинных нормалей
    vertex_normals = np.zeros((len(vertices), 3))
    
    for face in faces:
        v0, v1, v2 = [vertices[idx[0]] for idx in face]
        normal = compute_normal(*v0, *v1, *v2)
        norm = np.linalg.norm(normal)
        # Нормализация нормалей
        if norm > 0:
            normal /= norm
        for idx in face:
            vertex_normals[idx[0]] += normal

    return vertex_normals

def transform_vertex(v, scale, angles, translate, rotation_mode, quat):
    # Масштабирование
    scaled = v * np.array(scale)
    
    # Применение трансформаций
    if rotation_mode == 'euler':
        rot_mat = euler_rotation_matrix(angles)
    elif rotation_mode == 'quat':
        rot_mat = quat_rotation_matrix(quat)
    else:
        rot_mat = np.eye(3)
    
    rotated = np.dot(rot_mat, scaled)
        
    # Трансляция
    return rotated + np.array(translate)

def rasterize_triangle(proj_points, world_points, uv_coords, intencities, texture=None):
    (px0, py0), (px1, py1), (px2, py2) = proj_points
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = world_points
    
    # Проверка ориентации
    normal = compute_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    if norm_scalar(normal) > 1e-6: 
        return
    
    # Ограничивающий прямоугольник
    xmin = max(0, floor(min(px0, px1, px2)))
    xmax = min(W-1, ceil(max(px0, px1, px2)))
    ymin = max(0, floor(min(py0, py1, py2)))
    ymax = min(H-1, ceil(max(py0, py1, py2)))

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambda0, lambda1, lambda2 = barycentric_coordinates(i, j, px0, py0, px1, py1, px2, py2)
            
            if lambda0 < 0 or lambda1 < 0 or lambda2 < 0:
                continue
            
            z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
            if z < z_buffer[j][i]:
                if texture:
                    texture_c = [int(texture.height * (lambda0 * uv_coords[0][0] + lambda1 * uv_coords[1][0] + lambda2 * uv_coords[2][0])),
                                int(texture.width * (lambda0 * uv_coords[0][1] + lambda1 * uv_coords[1][1] + lambda2 * uv_coords[2][1]))]
                                                                     
                    color = texture.getpixel((texture_c[0], texture_c[1]))
                else:
                    color = (255, 255, 255)
                
                # Тонировка Гуро
                intencity = intencities[0] * lambda0 + intencities[1] * lambda1 + intencities[2] * lambda2
                intencity = min(0, intencity)

                final_color = (-color[0] * intencity,
                                -color[1] * intencity,
                                -color[2] * intencity)

                output_image[j][i] = final_color
                z_buffer[j][i] = z

def render_obj(obj_path, texture_path = None, scale = (1, 1, 1), angles = (0, 0, 0), translate = (0, 0, 0), rotation_mode='euler', quat=None):
    vertices, tex_coords, faces = parse_obj(obj_path)
    texture = ImageOps.flip(Image.open(texture_path)) if texture_path else None
    
    # Трансформация величин
    transformed = [transform_vertex(v, scale, angles, translate, rotation_mode, quat) for v in vertices]
    
    # Вычисление нормалей
    normals = compute_vertex_normal(transformed, faces)
    
    # Отрисовка треугольников
    for face in faces:
        # Получение данных для треугольника
        v_indices = [idx[0] for idx in face]
        vt_indices = [idx[1] for idx in face]
        
        world_points = [transformed[i] for i in v_indices]
        proj_points = [project_point(*p) for p in world_points]
        
        # Пропуск треугольников с невидимыми вершинами
        if any(x < 0 or y < 0 for x, y in proj_points):
            continue
    
        # Геометрические данные
        intensities = [norm_scalar(normals[i]) for i in v_indices]
    
        # Данные текстуры
        uv_coords = [tex_coords[i] for i in vt_indices] if tex_coords else None

        # Отрисовка
        rasterize_triangle(proj_points, world_points, uv_coords, intensities, texture)


angle = np.radians(45)  # Угол в радианах
axis = np.array([0, 1, 0])  # Ось Y
q = qu.from_rotation_vector(axis * angle)
render_obj(r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\models\man.obj',
           texture_path=r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\textures\man.bmp',
           scale=(3, 3, 3), angles=(0, pi+0.001, 0), translate=(0, 0, 2.5))
#render_obj(r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\models\cat.obj',
#           texture_path=r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\textures\cat.jpg',
#           scale=(0.05, 0.05, 0.05), angles=(pi/2, pi+0.001, 0), translate=(0, -0.9, 3))
#render_obj(r'C:\\Users\Nikita\Documents\КГ Лабы\labs_CG_4sem\models\zayac.obj',
#           texture_path=None,
#           scale=(1, 1, 1), angles=(0, pi/2, 0), translate=(0, -0.03, 0.2), rotation_mode='quat', quat=q)

image = Image.fromarray(output_image, mode='RGB')
image = ImageOps.flip(image)
image.show()
image.save(r'C:\\Users\akimn\Documents\Лабы репозитории\КГ Лабы\labs_CG_4sem\output_images\result_man.png')
