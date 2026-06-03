import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4_system(f1, f2, a, b, y0, z0, h):
    n = int((b - a) / h)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    
    x[0] = a
    y[0] = y0
    z[0] = z0
    
    for i in range(n):
        K1_y = h * z[i]
        K1_z = h * f2(x[i], y[i], z[i])
        
        K2_y = h * (z[i] + K1_z / 2)
        K2_z = h * f2(x[i] + h/2, y[i] + K1_y/2, z[i] + K1_z/2)
        
        K3_y = h * (z[i] + K2_z / 2)
        K3_z = h * f2(x[i] + h/2, y[i] + K2_y/2, z[i] + K2_z/2)
        
        K4_y = h * (z[i] + K3_z)
        K4_z = h * f2(x[i] + h, y[i] + K3_y, z[i] + K3_z)
        
        y[i+1] = y[i] + (K1_y + 2*K2_y + 2*K3_y + K4_y) / 6
        z[i+1] = z[i] + (K1_z + 2*K2_z + 2*K3_z + K4_z) / 6
        x[i+1] = x[i] + h
    
    return x, y, z

def runge_kutta_4_first_order(f, a, b, y0, h):
    n = int((b - a) / h)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    
    x[0] = a
    y[0] = y0
    
    for i in range(n):
        K1 = h * f(x[i], y[i])
        K2 = h * f(x[i] + h/2, y[i] + K1/2)
        K3 = h * f(x[i] + h/2, y[i] + K2/2)
        K4 = h * f(x[i] + h, y[i] + K3)
        
        y[i+1] = y[i] + (K1 + 2*K2 + 2*K3 + K4) / 6
        x[i+1] = x[i] + h
    
    return x, y

def runge_kutta_adaptive(f1, f2, a, b, y0, z0, h0, tol=0.0001):
    x = [a]
    y = [y0]
    z = [z0]
    
    h = h0
    xi = a
    yi = y0
    zi = z0
    
    while xi < b:
        if xi + h > b:
            h = b - xi
        
        y1, z1 = rk4_step(f2, xi, yi, zi, h)
        y2_half, z2_half = rk4_step(f2, xi, yi, zi, h/2)
        y2, z2 = rk4_step(f2, xi + h/2, y2_half, z2_half, h/2)
        
        err = max(abs(y1 - y2), abs(z1 - z2))
        
        if err < tol:
            xi = xi + h
            yi = y1
            zi = z1
            x.append(xi)
            y.append(yi)
            z.append(zi)
        
        if err > 0:
            h = 0.9 * h * (tol / err) ** 0.2
        h = min(h, b - xi)
    
    return np.array(x), np.array(y), np.array(z)

def rk4_step(f, x, y, z, h):
    K1_y = h * z
    K1_z = h * f(x, y, z)
    
    K2_y = h * (z + K1_z / 2)
    K2_z = h * f(x + h/2, y + K1_y/2, z + K1_z/2)
    
    K3_y = h * (z + K2_z / 2)
    K3_z = h * f(x + h/2, y + K2_y/2, z + K2_z/2)
    
    K4_y = h * (z + K3_z)
    K4_z = h * f(x + h, y + K3_y, z + K3_z)
    
    y_new = y + (K1_y + 2*K2_y + 2*K3_y + K4_y) / 6
    z_new = z + (K1_z + 2*K2_z + 2*K3_z + K4_z) / 6
    
    return y_new, z_new

def runge_kutta_adaptive_first_order(f, a, b, y0, h0, tol=0.0001):
    x = [a]
    y = [y0]
    
    h = h0
    xi = a
    yi = y0
    
    while xi < b:
        if xi + h > b:
            h = b - xi
        
        y1 = rk4_step_first(f, xi, yi, h)
        y_half = rk4_step_first(f, xi, yi, h/2)
        y2 = rk4_step_first(f, xi + h/2, y_half, h/2)
        
        err = abs(y1 - y2)
        
        if err < tol:
            xi = xi + h
            yi = y1
            x.append(xi)
            y.append(yi)
        
        if err > 0:
            h = 0.9 * h * (tol / err) ** 0.2
        h = min(h, b - xi)
    
    return np.array(x), np.array(y)

def rk4_step_first(f, x, y, h):
    K1 = h * f(x, y)
    K2 = h * f(x + h/2, y + K1/2)
    K3 = h * f(x + h/2, y + K2/2)
    K4 = h * f(x + h, y + K3)
    
    y_new = y + (K1 + 2*K2 + 2*K3 + K4) / 6
    return y_new

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА")
print("Метод Рунге-Кутты для задачи Коши")
print("=" * 70)

print("\n" + "=" * 70)
print("ЗАДАНИЕ 1: Уравнение второго порядка")
print("=" * 70)

print("\n1.1 УСЛОВИЕ ЗАДАЧИ:")
print("Решить задачу Коши для дифференциального уравнения:")
print("y'' - 2y' + y = 5x·eˣ")
print("Начальные условия: y(0) = 1, y'(0) = 2")
print("Отрезок: [0; 1]")
print("Шаги: h = 0.1, 0.01, 0.001")

print("\n1.2 ПРИВЕДЕНИЕ К СИСТЕМЕ ПЕРВОГО ПОРЯДКА:")
print("Замена: z = y'")
print("Получаем систему:")
print("y' = z")
print("z' = 2z - y + 5x·eˣ")
print("Начальные условия: y(0) = 1, z(0) = 2")

def f1_sys(x, y, z):
    return z

def f2_sys(x, y, z):
    return 2*z - y + 5*x*np.exp(x)

a, b = 0, 1
y0, z0 = 1, 2

print("\n1.3 РЕШЕНИЕ С ПОСТОЯННЫМ ШАГОМ:")
steps = [0.1, 0.01, 0.001]
results = {}

for h in steps:
    x, y, z = runge_kutta_4_system(f1_sys, f2_sys, a, b, y0, z0, h)
    results[h] = (x, y, z)
    print(f"\nШаг h = {h}:")
    print(f"Количество точек: {len(x)}")
    print(f"{'x':>8} {'y':>12} {'z=y\'':>12}")
    step_print = max(1, len(x)//10)
    for i in range(0, len(x), step_print):
        print(f"{x[i]:8.{max(1, int(-np.log10(h)))}f} {y[i]:12.6f} {z[i]:12.6f}")

print("\n1.4 ГРАФИКИ РЕШЕНИЙ ДЛЯ РАЗНЫХ ШАГОВ:")
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
colors = ['blue', 'red', 'green']
for idx, h in enumerate(steps):
    x, y, z = results[h]
    if h == 0.1:
        plt.plot(x, y, 'o-', color=colors[idx], markersize=6, linewidth=2.5, 
                markevery=max(1, len(x)//11), label=f'h = {h}')
    elif h == 0.01:
        plt.plot(x, y, 's-', color=colors[idx], markersize=4, linewidth=1.5, 
                markevery=max(1, len(x)//11), label=f'h = {h}')
    else:
        plt.plot(x, y, '-', color=colors[idx], linewidth=2, label=f'h = {h}')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Решение y(x)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for idx, h in enumerate(steps):
    x, y, z = results[h]
    if h == 0.1:
        plt.plot(x, z, 'o-', color=colors[idx], markersize=6, linewidth=2.5, 
                markevery=max(1, len(x)//11), label=f'h = {h}')
    elif h == 0.01:
        plt.plot(x, z, 's-', color=colors[idx], markersize=4, linewidth=1.5, 
                markevery=max(1, len(x)//11), label=f'h = {h}')
    else:
        plt.plot(x, z, '-', color=colors[idx], linewidth=2, label=f'h = {h}')
plt.xlabel('x')
plt.ylabel("y'(x)")
plt.title("Производная y'(x)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

print("\n1.5 РЕШЕНИЕ С АДАПТИВНЫМ ШАГОМ:")
x_adapt, y_adapt, z_adapt = runge_kutta_adaptive(f1_sys, f2_sys, a, b, y0, z0, 0.1, 0.0001)
print(f"Количество точек с адаптивным шагом: {len(x_adapt)}")
print(f"Минимальный шаг: {np.min(np.diff(x_adapt)):.6f}")
print(f"Максимальный шаг: {np.max(np.diff(x_adapt)):.6f}")

plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(x_adapt, y_adapt, 'b.-', markersize=8, linewidth=2)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Решение с адаптивным шагом - y(x)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(x_adapt, z_adapt, 'r.-', markersize=8, linewidth=2)
plt.xlabel('x')
plt.ylabel("y'(x)")
plt.title("Решение с адаптивным шагом - y'(x)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nСравнение решений в точке x = 1:")
for h in steps:
    x, y, z = results[h]
    print(f"h = {h}: y(1) = {y[-1]:.6f}, y'(1) = {z[-1]:.6f}")
print(f"Адаптивный: y(1) = {y_adapt[-1]:.6f}, y'(1) = {z_adapt[-1]:.6f}")

print("\n" + "=" * 70)
print("ЗАДАНИЕ 2: Уравнение первого порядка")
print("=" * 70)

print("\n1.1 УСЛОВИЕ ЗАДАЧИ:")
print("Решить задачу Коши для дифференциального уравнения:")
print("y' = y + x·y²")
print("Начальное условие: y(0) = 1")
print("Отрезок: [0; 0.5]")

def f_first_order(x, y):
    return y + x * y**2

a2, b2 = 0, 0.5
y0_2 = 1

print("\n1.2 РЕШЕНИЕ С ПОСТОЯННЫМ ШАГОМ:")
steps2 = [0.1, 0.01, 0.001]
results2 = {}

for h in steps2:
    x, y = runge_kutta_4_first_order(f_first_order, a2, b2, y0_2, h)
    results2[h] = (x, y)
    print(f"\nШаг h = {h}:")
    print(f"Количество точек: {len(x)}")
    print(f"{'x':>8} {'y':>12}")
    step_print = max(1, len(x)//10)
    for i in range(0, len(x), step_print):
        print(f"{x[i]:8.{max(1, int(-np.log10(h)))}f} {y[i]:12.6f}")

print("\n1.3 ГРАФИКИ РЕШЕНИЙ:")
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
for idx, h in enumerate(steps2):
    x, y = results2[h]
    if h == 0.1:
        plt.plot(x, y, 'o-', color=colors[idx], markersize=6, linewidth=2.5,
                markevery=max(1, len(x)//6), label=f'h = {h}')
    elif h == 0.01:
        plt.plot(x, y, 's-', color=colors[idx], markersize=4, linewidth=1.5,
                markevery=max(1, len(x)//6), label=f'h = {h}')
    else:
        plt.plot(x, y, '-', color=colors[idx], linewidth=2, label=f'h = {h}')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title("Решение y' = y + xy², y(0) = 1")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
x_adapt2, y_adapt2 = runge_kutta_adaptive_first_order(f_first_order, a2, b2, y0_2, 0.1, 0.0001)
plt.plot(x_adapt2, y_adapt2, 'b.-', markersize=8, linewidth=2)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Решение с адаптивным шагом')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n1.4 РЕШЕНИЕ С АДАПТИВНЫМ ШАГОМ:")
print(f"Количество точек с адаптивным шагом: {len(x_adapt2)}")
print(f"Минимальный шаг: {np.min(np.diff(x_adapt2)):.6f}")
print(f"Максимальный шаг: {np.max(np.diff(x_adapt2)):.6f}")

plt.figure(figsize=(10, 6))
plt.plot(x_adapt2, y_adapt2, 'b.-', markersize=8, linewidth=2.5)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title("Адаптивный шаг: y' = y + xy², y(0) = 1")
plt.grid(True, alpha=0.3)
plt.show()

print("\nСравнение решений в точке x = 0.5:")
for h in steps2:
    x, y = results2[h]
    print(f"h = {h}: y(0.5) = {y[-1]:.6f}")
print(f"Адаптивный: y(0.5) = {y_adapt2[-1]:.6f}")

print("\n" + "=" * 70)
print("ВЫВОДЫ:")
print("=" * 70)
print("1. Метод Рунге-Кутты 4-го порядка обеспечивает высокую точность")
print("   даже при относительно крупном шаге.")
print("2. Уменьшение шага приводит к увеличению точности решения.")
print("3. Адаптивный шаг позволяет оптимизировать вычисления,")
print("   уменьшая шаг на сложных участках и увеличивая на простых.")
print("4. Результаты для разных шагов сходятся к точному решению.")