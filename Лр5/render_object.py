import numpy as np
from PIL import Image, ImageOps
from math import ceil, floor, sqrt, cos, sin, pi, radians
import quaternion

# ================= КОНФИГУРАЦИЯ РЕНДЕРА =================
IMG_WIDTH = 2000        # Ширина выходного изображения
IMG_HEIGHT = 2000       # Высота выходного изображения
FOV_X = 1000            # Коэффициент перспективы по X
FOV_Y = 1000            # Коэффициент перспективы по Y
LIGHT_DIR = np.array([0, 0, 1])  # Направление освещения

# ================= ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ =================
z_buffer = np.full((IMG_HEIGHT, IMG_WIDTH), np.inf)         # Z-буфер глубины
output_image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)     # Выходное изображение

# ================= ОСНОВНЫЕ ФУНКЦИИ =====================

def parse_obj(file_path):
    """Парсер OBJ-файлов с поддержкой полигонов и текстур"""
    vertices = []
    tex_coords = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                vertices.append(np.array(list(map(float, parts[1:4]))))
            elif parts[0] == 'vt':
                tex_coords.append(list(map(float, parts[1:3])))
            elif parts[0] == 'f':
                face = []
                for part in parts[1:]:
                    indices = part.split('/')
                    v = int(indices[0])-1 if indices[0] else -1
                    vt = int(indices[1])-1 if len(indices)>1 and indices[1] else -1
                    face.append((v, vt))
                # Триангуляция полигона
                for i in range(1, len(face)-1):
                    faces.append([face[0], face[i+1], face[i]])
    
    return vertices, tex_coords, faces

def euler_rotation_matrix(angles):
    """Создает матрицу поворота из углов Эйлера (в радианах)"""
    rx = np.array([[1, 0, 0],
                 [0, cos(angles[0]), -sin(angles[0])],
                 [0, sin(angles[0]), cos(angles[0])]])
    
    ry = np.array([[cos(angles[1]), 0, sin(angles[1])],
                 [0, 1, 0],
                 [-sin(angles[1]), 0, cos(angles[1])]])
    
    rz = np.array([[cos(angles[2]), -sin(angles[2]), 0],
                 [sin(angles[2]), cos(angles[2]), 0],
                 [0, 0, 1]])
    
    return rz @ ry @ rx

def quat_rotation_matrix(q):
    """Создает матрицу поворота из кватерниона"""
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w   ],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w   ],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y ]
    ])

def transform_vertex(v, scale, angles, translate, rotation_mode='euler', quat=None):
    """Применяет трансформации к вершине"""
    # Масштабирование
    scaled = v * np.array(scale)
    
    # Поворот
    if rotation_mode == 'euler':
        rot_mat = euler_rotation_matrix(angles)
    elif rotation_mode == 'quat':
        rot_mat = quat_rotation_matrix(quat)
    rotated = rot_mat @ scaled
    
    # Трансляция
    return rotated + np.array(translate)

def compute_vertex_normals(vertices, faces):
    """Вычисляет нормали вершин усреднением нормалей полигонов"""
    normals = np.zeros((len(vertices), 3))
    for face in faces:
        v0, v1, v2 = [vertices[idx[0]] for idx in face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal /= norm
        for idx in face:
            normals[idx[0]] += normal
    
    # Нормализация и защита от нуля
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1.0
    return normals / norms[:, np.newaxis]

def project_point(x, y, z):
    """Проецирует 3D точку на 2D плоскость с проверкой видимости"""
    if z <= 1e-6:  # Точка позади камеры
        return (-1, -1)
    return (
        int(FOV_X * x / z + IMG_WIDTH/2),
        int(FOV_Y * y / z + IMG_HEIGHT/2)
    )

def rasterize_triangle(proj_points, world_points, intensities, tex_coords=None, texture=None):
    """Растеризация треугольника с перспективно-корректной интерполяцией"""
    (x0, y0), (x1, y1), (x2, y2) = proj_points
    (wx0, wy0, wz0), (wx1, wy1, wz1), (wx2, wy2, wz2) = world_points
    
    # Проверка ориентации (backface culling)
    normal = np.cross([wx1-wx0, wy1-wy0, wz1-wz0], [wx2-wx0, wy2-wy0, wz2-wz0])
    if np.dot(normal, LIGHT_DIR) > 0:
        return
    
    # Определение границ растеризации
    x_min = max(0, min(x0, x1, x2))
    x_max = min(IMG_WIDTH-1, max(x0, x1, x2))
    y_min = max(0, min(y0, y1, y2))
    y_max = min(IMG_HEIGHT-1, max(y0, y1, y2))
    
    # Вычисление матрицы для барицентрических координат
    det = (y0 - y2)*(x1 - x2) + (x2 - x0)*(y1 - y2)
    if abs(det) < 1e-6:
        return
    
    inv_det = 1.0 / det
    y2y0 = y2 - y0
    x0x2 = x0 - x2
    y1y2 = y1 - y2
    x2x1 = x2 - x1
    
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            # Барицентрические координаты
            lambda0 = ((y - y2)*x2x1 + (x - x2)*y1y2) * inv_det
            lambda1 = ((y - y0)*x0x2 + (x - x0)*y2y0) * inv_det
            lambda2 = 1.0 - lambda0 - lambda1
            
            if lambda0 < 0 or lambda1 < 0 or lambda2 < 0:
                continue
            
            # Глубина с перспективной коррекцией
            z = 1.0 / (lambda0/wz0 + lambda1/wz1 + lambda2/wz2)
            if z >= z_buffer[y][x]:
                continue
            
            # Обновление буферов
            z_buffer[y][x] = z
            
            # Интерполяция интенсивности (Гуро)
            intensity = lambda0*intensities[0] + lambda1*intensities[1] + lambda2*intensities[2]
            intensity = max(0, min(1, intensity))
            
            # Текстурные координаты с коррекцией
            if texture and tex_coords:
            # Перспективно-корректная интерполяция
                inv_z0 = 1.0 / wz0
                inv_z1 = 1.0 / wz1
                inv_z2 = 1.0 / wz2

                # Интерполяция с учетом перспективы
                u_over_z = (lambda0 * tex_coords[0][0] * inv_z0 +
                    lambda1 * tex_coords[1][0] * inv_z1 +
                    lambda2 * tex_coords[2][0] * inv_z2)
        
                v_over_z = (lambda0 * tex_coords[0][1] * inv_z0 +
                    lambda1 * tex_coords[1][1] * inv_z1 +
                    lambda2 * tex_coords[2][1] * inv_z2)
        
                z = 1.0 / (lambda0*inv_z0 + lambda1*inv_z1 + lambda2*inv_z2)
                u = u_over_z * z
                v = v_over_z * z

                # Обработка выхода за границы [0,1]
                u = u - floor(u)  # Повторение текстуры
                v = v - floor(v)

                # Конвертация в пиксели текстуры (с учетом вертикального flip)
                tex_x = round(u * (texture.width-1))
                tex_y = round((1 - v) * (texture.height-1))  # Инвертируем V-координату

                color = texture.getpixel((tex_x, tex_y))
            else:
                color = (255, 255, 255)
            
            # Применение освещения
            final_color = (
                int(color[0] * intensity),
                int(color[1] * intensity),
                int(color[2] * intensity)
            )
            
            output_image[y][x] = np.clip(final_color, 0, 255)

def render_model(obj_path, texture_path=None, 
                scale=(1,1,1), angles=(0,0,0), translate=(0,0,0),
                rotation_mode='euler', quat=None):
    """Рендерит модель с заданными параметрами"""
    vertices, tex_coords, faces = parse_obj(obj_path)
    texture = ImageOps.flip(Image.open(texture_path)) if texture_path else None
    
    # Трансформация вершин
    transformed = [transform_vertex(v, scale, angles, translate, rotation_mode, quat)
                  for v in vertices]
    
    # Вычисление нормалей
    normals = compute_vertex_normals(transformed, faces)
    
    # Отрисовка треугольников
    for face in faces:
        v_indices = [idx[0] for idx in face]
        vt_indices = [idx[1] for idx in face]
        
        world_points = [transformed[i] for i in v_indices]
        proj_points = [project_point(*p) for p in world_points]
        
        # Пропуск треугольников с невидимыми вершинами
        if any(x < 0 or y < 0 for x, y in proj_points):
            continue
        
        intensities = [np.dot(normals[i], LIGHT_DIR) for i in v_indices]
        uv_coords = [tex_coords[i] for i in vt_indices] if tex_coords else None
        
        rasterize_triangle(
            proj_points,
            world_points,
            intensities,
            uv_coords,
            texture
        )

# Рендер первой модели (с текстурой)
render_model(
    r'C:\\Users\akimn\Documents\Лабы репозитории\Лабы КГ\labs_CG_4sem\models\zayac.obj',
    texture_path=r'C:\\Users\akimn\Documents\Лабы репозитории\Лабы КГ\labs_CG_4sem\textures\zayac.jpg',
    scale=(1, 1, 1),
    angles=(radians(30), radians(15), 0),
    translate=(0.2, -0.1, 1.5)
)
    
angle = np.radians(45)  # Угол в радианах
axis = np.array([0, 1, 0])  # Ось Y
q = quaternion.from_rotation_vector(axis * angle)
    
# Рендер второй модели (без текстуры)
render_model(
    r'C:\\Users\akimn\Documents\Лабы репозитории\Лабы КГ\labs_CG_4sem\models\man.obj',
    scale=(1.2, 1.2, 1.2),
    translate=(-0.3, 0.2, 2.0),
    rotation_mode='quat',
    quat=q
)
    
# Сохранение результата
Image.fromarray(output_image).save(r'C:\\Users\akimn\Documents\Лабы репозитории\Лабы КГ\labs_CG_4sem\output_images\result.png')
Image.fromarray(output_image).show()
