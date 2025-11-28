import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import matplotlib.patches as patches
from PIL import Image
import os

def get_spr_image(IMAGE_PATH, l_current, lam_red = 0.810, lam_blue = 0.610, x_red = 198, x_blue = 1256,  
                  L0 = 60, l1 = 9.0, l2 = 5.5, y1 = 295, y2 = 579, l_ref_collinear = 6.6):

    l_min = min(l1, l2)
    l_max = max(l1, l2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    gamma_y = (l_max - l_min) / ((y_max - y_min) * L0)

    print(f"Угловое расстояние на пиксель: {gamma_y:.5f} рад/пиксель")
    
    y_center = l_ref_collinear/ (gamma_y * L0)
    print(f"Центр картины (theta=0) находится на Y = {y_center:.1f}")

    k_x = (lam_red - lam_blue) / (x_blue - x_red)

    def x_to_lambda(x):
        return lam_red - k_x * (x - x_red)

    def lambda_to_x(lam):
        return (lam_red - lam)/ k_x + x_red

    print(f"Калибровка X: 650нм на {lambda_to_x(0.650):.0f} px, 810нм на {lambda_to_x(0.810):.0f} px")

    LAMBDA_PUMP = lam_red/2

    # Уравнения Зельмейера
    def get_no_sq(lam): 
        return 2.7405 + 0.0184 / (lam**2 - 0.0179) - 0.0155 * lam**2

    def get_ne_sq(lam): 
        return 2.3730 + 0.0128 / (lam**2 - 0.0156) - 0.0044 * lam**2

    def get_no(lam): return np.sqrt(get_no_sq(lam))
    def get_ne_pure(lam): return np.sqrt(get_ne_sq(lam))

    def get_ne_alpha(lam, alpha_rad):
        no = get_no(lam)
        ne = get_ne_pure(lam)
        
        return no * np.sqrt((np.tan(alpha_rad)**2 + 1)/((np.tan(alpha_rad) * no/ne)**2 + 1))

    def condition_alpha0(alpha, lam_pump):
        return np.abs(get_ne_alpha(lam_pump, alpha) - get_no(lam_pump*2))

    alpha0 = fsolve(condition_alpha0, x0 = np.radians(29.0), args = (LAMBDA_PUMP,) )[0]
    print(f"Угол наклона кристалла при кол выр режиме: {np.degrees(alpha0):.2f}°")

    # ==========================================
    # 3. РАСЧЕТ ДЛЯ ТЕКУЩЕГО ИЗОБРАЖЕНИЯ
    # ==========================================

    def find_alpha(alpha, l, l0, alpha0):
        return np.abs((alpha - alpha0) * get_ne_alpha(LAMBDA_PUMP, alpha) + (l - l0)/L0)

    alpha = fsolve(find_alpha, x0 = alpha0, args = (l_current, l_ref_collinear, alpha0,))[0]
    print(f"Угол наклона кристалла alpha: {np.degrees(alpha):.2f}°")

    print(f"Линейка: {l_current} см. Расчетный угол кристалла: {np.degrees(alpha):.2f}°")

    # 2. Строим перестроечную кривую
    def calculate_curve(alpha_cryst):
        lams = np.linspace(0.5, 0.9, 10000) # Диапазон длин волн
        valid_lams = []
        thetas_ext = []
        
        kp = 2 * np.pi * get_ne_alpha(LAMBDA_PUMP, alpha_cryst) / LAMBDA_PUMP
        
        for ls in lams:
            li = 1.0 / (1.0/LAMBDA_PUMP - 1.0/ls)
            if li < 0: 
                continue
            
            ks = 2 * np.pi * get_no(ls) / ls
            ki = 2 * np.pi * get_no(li) / li
            
            # Теорема косинусов: ki^2 = kp^2 + ks^2 - 2 kp ks cos(theta)
            cos_theta = (kp**2 + ks**2 - ki**2) / (2 * kp * ks)
            if abs(cos_theta) <= 1:
                theta_in = np.arccos(cos_theta)
                sin_out = get_no(ls) * np.sin(theta_in)
                
                if abs(sin_out) <= 1:
                    theta_out = np.arcsin(sin_out)
                    valid_lams.append(ls)
                    thetas_ext.append(theta_out)
                    
                    # Нижняя ветвь (отрицательный угол)
                    valid_lams.append(ls)
                    thetas_ext.append(-theta_out)
                    
        return np.array(valid_lams), np.array(thetas_ext)

    lams_fit, thetas_fit = calculate_curve(alpha)

    x_fit = lambda_to_x(lams_fit)
    y_fit = thetas_fit/gamma_y + 1.15*y_max

    # ==========================================
    # 4. ОТРИСОВКА
    # ==========================================
    plt.figure(figsize=(10, 8), dpi = 400)
    img_data = Image.open(IMAGE_PATH)
    plt.imshow(img_data, aspect='auto')
    sc = plt.scatter(x_fit, y_fit, c=np.flip(lams_fit), cmap='turbo', s=2, label=f'Расчет для {l_current}см на линейке', alpha=0.7)

    cbar = plt.colorbar(sc)
    cbar.set_label(r'Длины волн, мкм')
    plt.xlim(0, 1920)
    plt.ylim(0, 1200)
    plt.xlabel(r"Пиксели по оси X (длины волн $\lambda$)")
    plt.ylabel(r"Пиксели по оси Y (углы $\theta$)")
    if alpha < alpha0:
        plt.title(f"Коллинеарный невырожденный синхронизм.\n" + r'$\alpha$ = ' + f"{np.degrees(alpha):.2f}°, l = {l_current} см")
    elif alpha > alpha0:
        plt.title(f"Неколлинеарный вырожденный синхронизм.\n" + r'$\alpha$ = ' + f"{np.degrees(alpha):.2f}°, l = {l_current} см")
    else:
        plt.title(f"Коллинеарный вырожденный синхронизм.\n" + r'$\alpha$ = ' + f"{np.degrees(alpha):.2f}°, l = {l_current} см")
    plt.legend()
    plt.show()


def get_spr_rings(IMAGE_PATH, l_current, 
                  # Параметры калибровки (те же самые)
                  lam_red = 0.810, 
                  L0 = 60, l1 = 9.0, l2 = 5.5, y1 = 295, y2 = 579, 
                  l_ref_collinear = 6.6,
                  x_center_manual = 198*2 + 10, 
                  y_center_correction = 0):

    # 1. КАЛИБРОВКА (как в вашем коде)
    l_min = min(l1, l2)
    l_max = max(l1, l2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    # Угловой масштаб (радиан на пиксель)
    gamma_y = (l_max - l_min) / ((y_max - y_min) * L0)
    print(f"Угловое расстояние на пиксель: {gamma_y:.5f} рад/пиксель")
    
    y_center_calc =  l_ref_collinear/ (gamma_y * L0)
    
    y_center = y_center_calc + y_center_correction
    x_center = x_center_manual
    
    print(f"Центр кольца: X={x_center}, Y={y_center:.1f}")

    LAMBDA_PUMP = lam_red / 2

    def get_no_sq(lam): 
        return 2.7405 + 0.0184 / (lam**2 - 0.0179) - 0.0155 * lam**2
    def get_ne_sq(lam): 
        return 2.3730 + 0.0128 / (lam**2 - 0.0156) - 0.0044 * lam**2
    
    def get_no(lam): return np.sqrt(get_no_sq(lam))
    def get_ne_pure(lam): return np.sqrt(get_ne_sq(lam))
    
    def get_ne_alpha(lam, alpha_rad):
        no = get_no(lam)
        ne = get_ne_pure(lam)
        return no * np.sqrt((np.tan(alpha_rad)**2 + 1)/((np.tan(alpha_rad) * no/ne)**2 + 1))

    # Находим alpha0
    def condition_alpha0(alpha, lam_pump):
        return np.abs(get_ne_alpha(lam_pump, alpha) - get_no(lam_pump*2))
    
    alpha0 = fsolve(condition_alpha0, x0=np.radians(29.0), args=(LAMBDA_PUMP,))[0]
    
    # Находим текущий alpha
    def find_alpha(alpha, l, l0, alpha0):
        return np.abs((alpha - alpha0) * get_ne_alpha(LAMBDA_PUMP, alpha) + (l - l0)/L0)

    alpha = fsolve(find_alpha, x0=alpha0, args=(l_current, l_ref_collinear, alpha0,))[0]
    print(f"Линейка: {l_current} см. Alpha: {np.degrees(alpha):.2f}°")

    # 3. РАСЧЕТ РАДИУСОВ КОЛЕЦ
    # Цвета, которые мы хотим нарисовать (длины волн)
    target_lambdas = [0.455, 0.532, 0.650]
    plot_colors = ['blue', 'green', 'red']
    
    radii_pixels = []

    kp = 2 * np.pi * get_ne_alpha(LAMBDA_PUMP, alpha) / LAMBDA_PUMP

    for ls in target_lambdas:
        li = 1.0 / (1.0/LAMBDA_PUMP - 1.0/ls)
        if li < 0:
            radii_pixels.append(None)
            continue
            
        ks = 2 * np.pi * get_no(ls) / ls
        ki = 2 * np.pi * get_no(li) / li
        
        # Теорема косинусов для угла theta_in
        cos_theta = (kp**2 + ks**2 - ki**2) / (2 * kp * ks)
        
        if abs(cos_theta) <= 1:
            theta_in = np.arccos(cos_theta)
            sin_out = get_no(ls) * np.sin(theta_in) # Снеллиус
            
            if abs(sin_out) <= 1:
                theta_out = np.arcsin(sin_out)
                
                r_pix =  np.tan(theta_out) / gamma_y
                radii_pixels.append(r_pix)
            else:
                radii_pixels.append(None)
        else:
            radii_pixels.append(None)

    # 4. ОТРИСОВКА
    plt.figure(figsize=(10, 8), dpi = 400)
    img_data = Image.open(IMAGE_PATH)
    plt.imshow(img_data)
    ax = plt.gca()
    
    for r, col, lam in zip(radii_pixels, plot_colors, target_lambdas):
        if r is not None:
            circle = patches.Circle((x_center, y_center), radius=r, 
                                    edgecolor=col, facecolor='none', 
                                    linewidth=2, label=f'{int(lam*1000)} нм' + f', r={r:.1f} px')
            ax.add_patch(circle)

    plt.xlim(0, 1920)
    plt.ylim(1200, 0)
    
    plt.xlabel("X, пиксели)")
    plt.ylabel("Y, пиксели)")
    if alpha > alpha0:
        plt.title(f"Наблюдение колец в неколлинеарном вырожденном режиме\n" + 
                  r'$\alpha$ = ' + f"{np.degrees(alpha):.2f}°, l = {l_current} см")
    elif alpha < alpha0:
        plt.title(f"Наблюдение колец в неколлинеарном невырожденном режиме\n" + 
                  r'$\alpha$ = ' + f"{np.degrees(alpha):.2f}°, l = {l_current} см")
    else:
        plt.title(f"Наблюдение колец в коллинеарном вырожденном режиме\n" +
                  r'$\alpha$ = ' + f"{np.degrees(alpha):.2f}°, l = {l_current} см - показания линейнки")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    IMAGE_PATH = "./СПР/картинка 1 положение 5.1 см 2.png" 
    l_current = 5.1
    get_spr_rings(IMAGE_PATH, l_current)