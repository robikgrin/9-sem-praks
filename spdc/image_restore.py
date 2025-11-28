import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from PIL import Image
import os

# ==========================================
# 1. КОНСТАНТЫ И УРАВНЕНИЯ (из PDF)
# ==========================================

# Длины волн (в мкм)
LAMBDA_PUMP = 0.405
LAMBDA_DEGEN = 0.810

# Уравнения Зельмейера для BBO (Eq 1.15, 1.16)
def get_no_sq(lam): # lam in um
    return 2.7405 + 0.0184 / (lam**2 - 0.0179) - 0.0155 * lam**2

def get_ne_sq(lam): # lam in um
    return 2.3730 + 0.0128 / (lam**2 - 0.0156) - 0.0044 * lam**2

def get_no(lam):
    return np.sqrt(get_no_sq(lam))

def get_ne_pure(lam):
    return np.sqrt(get_ne_sq(lam))

# Показатель преломления необыкновенной волны от угла alpha (Eq 1.17)
def get_ne_alpha(lam, alpha_rad):
    no = get_no(lam)
    ne = get_ne_pure(lam)
    # Формула 1.17: ne(alpha) = no * sqrt((tan^2(a)+1) / ((tan(a)*no/ne)^2 + 1))
    # Упрощенная форма для эллипсоида показателей преломления:
    # 1/n^2 = cos^2(a)/no^2 + sin^2(a)/ne^2
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    val = (cos_a / no)**2 + (sin_a / ne)**2
    return 1.0 / np.sqrt(val)

# ==========================================
# 2. ДАННЫЕ КАЛИБРОВКИ (от Пользователя)
# ==========================================

# Оптический путь
L0_optical = 60.0  # см

# --- Вертикальная калибровка (Y -> Угол) ---
# Линейка направлена вверх (чем больше см, тем выше точка, тем меньше Y пиксель)
# Данные: 0.5 см -> y=1080; 8.8 см -> y=284
y1, l1 = 295, 9
y2, l2 = 579, 5.5
cm_per_pixel_y = (l2 - l1) / (y2 - y1) # Отрицательное число, т.к. Y растет вниз
l_offset = l1 - cm_per_pixel_y * y1    # l = slope * y + offset

# Положение линейки для коллинеарного вырожденного (центр, theta=0)
l_coll_deg = 6.2  # см
# Найдем Y-координату центра (оптической оси) на камере
y_center = (l_coll_deg - l_offset) / cm_per_pixel_y
print(f"Калибровка Y: {cm_per_pixel_y:.4f} см/пикс. Центр пучка (theta=0) на Y={y_center:.1f}")

# --- Горизонтальная калибровка (X -> Длина волны) ---
# Данные: Красный (650 нм) -> x=1189.
# Накачка/Синий (810 нм в 1-м порядке) -> x_mean = (193+252)/2 = 222.5
x_red = 1189
lam_red = 0.650
x_blue = 222.5 # Среднее из x_min и x_max при движении кристалла
lam_blue = 0.810

# Линейная зависимость: lam = k * x + b
# Ожидаем, что 650 справа, 810 слева -> k будет отрицательным
k_x = (lam_red - lam_blue) / (x_red - x_blue)
b_x = lam_blue - k_x * x_blue

def x_to_lambda(x):
    return k_x * x + b_x

def lambda_to_x(lam):
    return (lam - b_x) / k_x

print(f"Калибровка X: {k_x:.6f} мкм/пикс. 810нм на X={lambda_to_x(0.810):.1f}")

# ==========================================
# 3. РАСЧЕТ УГЛА КРИСТАЛЛА alpha
# ==========================================

# Находим alpha0 (угол среза для коллинеарного вырожденного синхронизма)
# Условие: ne(wp, alpha0) = no(ws), где ws = wp/2
def condition_alpha0(alpha):
    return get_ne_alpha(LAMBDA_PUMP, alpha) - get_no(LAMBDA_DEGEN)

alpha0_guess = np.radians(29.0) # Ожидаем около 29 градусов для BBO
alpha0 = fsolve(condition_alpha0, alpha0_guess)[0]
print(f"Угол среза alpha0: {np.degrees(alpha0):.2f} град")

# Функция пересчета смещения линейки в изменение угла alpha
# Формула из раздела 4: (l - l0)/L0 = -2 * (alpha - alpha0) * ne
def get_alpha_for_ruler(l_ruler_val):
    delta_l = l_ruler_val - l_coll_deg
    # ne берем приблизительно в точке alpha0
    ne_val = get_ne_alpha(LAMBDA_PUMP, alpha0)
    
    # delta_l / L0 = -2 * delta_alpha * ne
    # delta_alpha = - delta_l / (2 * L0 * ne)
    # НО! Нужно учесть преломление на выходе из кристалла. 
    # Формула в методичке (l - l0)/L0 = -2(alpha - alpha0)ne уже учитывает
    # переход углов (угол внутри * ne ~ угол снаружи).
    
    delta_alpha = - delta_l / (2 * L0_optical * ne_val)
    return alpha0 + delta_alpha

# ==========================================
# 4. РАСЧЕТ ПЕРЕСТРОЕЧНОЙ КРИВОЙ
# ==========================================

def calculate_tuning_curve(alpha_crystal):
    # Диапазон длин волн для построения (сигнальная волна)
    # От 600 нм до 1100 нм (ограничиваем, чтобы не уйти в сингулярности)
    lambdas_s = np.linspace(0.600, 1.100, 500)
    
    thetas_ext = []
    valid_lambdas = []
    
    kp = 2 * np.pi * get_ne_alpha(LAMBDA_PUMP, alpha_crystal) / LAMBDA_PUMP
    
    for ls in lambdas_s:
        # Холостая волна по закону сохранения энергии
        li = 1.0 / (1.0/LAMBDA_PUMP - 1.0/ls)
        
        ks = 2 * np.pi * get_no(ls) / ls
        ki = 2 * np.pi * get_no(li) / li
        
        # Закон косинусов для треугольника волновых векторов:
        # ki^2 = kp^2 + ks^2 - 2*kp*ks*cos(theta_s)
        # cos(theta_s) = (kp^2 + ks^2 - ki^2) / (2*kp*ks)
        
        cos_theta = (kp**2 + ks**2 - ki**2) / (2 * kp * ks)
        
        if abs(cos_theta) <= 1:
            theta_int = np.arccos(cos_theta)
            
            # Закон Снеллиуса для выхода из кристалла: sin(theta_ext) = no * sin(theta_int)
            # (сигнальная волна обыкновенная)
            sin_ext = get_no(ls) * np.sin(theta_int)
            
            if abs(sin_ext) <= 1:
                theta_ext = np.arcsin(sin_ext)
                valid_lambdas.append(ls)
                thetas_ext.append(theta_ext)
                
                # Симметричная ветвь (отрицательный угол)
                valid_lambdas.append(ls)
                thetas_ext.append(-theta_ext)
    
    return np.array(valid_lambdas), np.array(thetas_ext)

# ==========================================
# 5. ОБРАБОТКА КОНКРЕТНОГО СЛУЧАЯ
# ==========================================

# Выбираем случай: Неколлинеарный вырожденный
target_ruler = 5.1 # см
current_alpha = get_alpha_for_ruler(target_ruler)
print(f"Анализ для линейки {target_ruler} см. Alpha = {np.degrees(current_alpha):.2f} град")

# Расчет кривой
lams, thetas = calculate_tuning_curve(current_alpha)

# Перевод физических величин в пиксели
# X: lambda -> pixel
x_pixels = lambda_to_x(lams)

# Y: theta -> pixel
# theta_ext ~ tan(theta) = y_dist / L0
# y_dist = theta_ext * L0 (в приближении малых углов или тангенса)
# y_pixel = y_center + (y_dist_cm / cm_per_pixel_y)
# ВАЖНО: Знак угла. Обычно верхняя ветвь соответствует одному знаку, нижняя другому.
y_pixels = y_center + (np.tan(thetas) * L0_optical) / cm_per_pixel_y

# ==========================================
# 6. ВИЗУАЛИЗАЦИЯ
# ==========================================

# Создаем "фейковое" изображение, так как реального файла нет
# В реальности вы сделаете: img = plt.imread('path/to/photo.bmp')
# img_width, img_height = 1920, 1200
# img = np.zeros((img_height, img_width, 3), dtype=np.uint8)


# Рисуем "шум" и примерный экспериментальный сигнал для наглядности (симуляция)
# (В реальном скрипте этот блок не нужен, просто загрузите фото)
# for i in range(len(x_pixels)):
#     ix, iy = int(x_pixels[i]), int(y_pixels[i])
#     if 0 <= ix < img_width and 0 <= iy < img_height:
#         # Рисуем толстую линию "эксперимента"
#         img[iy-2:iy+3, ix-2:ix+3, 1] = 200 # Green channel

# plt.figure(figsize=(10, 6))
# Если есть реальное фото, раскомментируйте следующую строку:
img = plt.imread('СПР/картинка 1 положение 5.1 см 2.png')



plt.imshow(img, aspect='auto') # aspect auto чтобы не искажать пропорции пикселей
plt.plot(x_pixels, y_pixels, 'r-', linewidth=1, label='Theory Fit')

plt.xlim(0, 1920)
plt.ylim(1200, 0) # Начало координат в левом верхнем углу
plt.xlabel('X pixel (Wavelength)')
plt.ylabel('Y pixel (Angle)')
plt.title(f'SPDC Tuning Curve Overlay (Ruler = {target_ruler} cm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Готово. Красная линия - теория, зеленая (на фоне) - симуляция сигнала.")