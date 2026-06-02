import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

def equation1(x):
    return 10 * np.cos(x) - 0.1 * x**2

def equation2(x):
    return x**3 + 3*x**2 - 7

def phi1_interval1(x):
    return np.arccos(0.01 * x**2)

def phi1_interval2(x):
    return np.sqrt(100 * np.cos(x))

def phi1_interval3(x):
    return -np.sqrt(100 * np.cos(x))

def phi2_interval1(x):
    return np.sqrt(7/(x+3))

def phi2_interval2(x):
    return (7 - x**3) / (3*x)

def phi2_interval3(x):
    return np.cbrt(7 - 3*x**2)

def check_convergence(phi, x0, label):
    try:
        h = 1e-6
        deriv = (phi(x0 + h) - phi(x0 - h)) / (2*h)
        q = abs(deriv)
        print(f"{label}: |φ'(x0)| = {q:.6f}")
        if q < 1:
            print(f"{label}: Условие сходимости выполнено (|φ'| < 1)")
            return True
        else:
            print(f"{label}: Условие сходимости НЕ выполнено (|φ'| >= 1)")
            return False
    except:
        print(f"{label}: Не удалось проверить сходимость в точке x0")
        return False

def simple_iteration(phi, x0, epsilon=1e-6, max_iter=1000):
    x_prev = x0
    iterations = 0
    
    for i in range(max_iter):
        try:
            x_next = phi(x_prev)
            iterations += 1
            
            if abs(x_next - x_prev) < epsilon:
                return x_next, iterations, True
            
            x_prev = x_next
        except:
            return None, iterations, False
    
    return None, max_iter, False

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА: МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
print("=" * 60)

print("\n" + "=" * 60)
print("ЗАДАЧА 1: 10cos(x) - 0.1x² = 0")
print("=" * 60)

print("\n1.1 УСЛОВИЕ ЗАДАЧИ:")
print("Решить уравнение 10cos(x) - 0.1x² = 0 методом простой итерации.")

print("\n1.2 ГРАФИК ФУНКЦИИ f(x) = 10cos(x) - 0.1x²")
x_plot = np.linspace(-10, 10, 1000)
y_plot = equation1(x_plot)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = 10cos(x) - 0.1x²')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 10cos(x) - 0.1x²')
plt.legend()

print("\n1.3 ПРЕОБРАЗОВАНИЕ К ВИДУ x = φ(x):")
print("Исходное уравнение: 10cos(x) - 0.1x² = 0")
print("\nВариант 1: x = arccos(0.01x²)")
print("10cos(x) = 0.1x² → cos(x) = 0.01x² → x = arccos(0.01x²)")
print("Область определения: |0.01x²| ≤ 1, то есть |x| ≤ 10")

print("\nВариант 2: x = √(100cos(x)) для x ≥ 0")
print("10cos(x) = 0.1x² → x² = 100cos(x) → x = √(100cos(x))")
print("Область определения: cos(x) ≥ 0, то есть x ∈ [-π/2+2πk, π/2+2πk]")

print("\nВариант 3: x = -√(100cos(x)) для x ≤ 0")
print("x² = 100cos(x) → x = -√(100cos(x))")

print("\n1.4 ПРОВЕРКА СХОДИМОСТИ ДЛЯ РАЗЛИЧНЫХ ИНТЕРВАЛОВ:")
print("\nАнализ графика показывает корни приблизительно в точках:")
print("x₁ ≈ -4, x₂ ≈ 2, x₃ ≈ 6, x₄ ≈ -7")

roots_to_find = [
    ("Корень x₁ ≈ -7 (интервал [-8, -6])", -7, phi1_interval3, "φ(x) = -√(100cos(x))"),
    ("Корень x₂ ≈ -4 (интервал [-5, -3])", -4, phi1_interval1, "φ(x) = arccos(0.01x²)"),
    ("Корень x₃ ≈ 2 (интервал [1, 3])", 2, phi1_interval2, "φ(x) = √(100cos(x))"),
    ("Корень x₄ ≈ 6 (интервал [5, 7])", 6, phi1_interval1, "φ(x) = arccos(0.01x²)"),
]

for desc, x0, phi, phi_desc in roots_to_find:
    print(f"\n{desc}:")
    print(f"Используем {phi_desc}")
    check_convergence(phi, x0, f"x₀ = {x0}")
    
    root, iters, converged = simple_iteration(phi, x0)
    if converged:
        print(f"Найден корень: x = {root:.10f}")
        print(f"Количество итераций: {iters}")
        print(f"Проверка: f({root:.6f}) = {equation1(root):.2e}")
    else:
        print("Метод не сошелся")

print("\n" + "=" * 60)
print("ЗАДАЧА 2: x³ + 3x² - 7 = 0")
print("=" * 60)

print("\n1.1 УСЛОВИЕ ЗАДАЧИ:")
print("Решить уравнение x³ + 3x² - 7 = 0 методом простой итерации.")

print("\n1.2 ГРАФИК ФУНКЦИИ f(x) = x³ + 3x² - 7")
x_plot2 = np.linspace(-5, 5, 1000)
y_plot2 = equation2(x_plot2)

plt.subplot(1, 2, 2)
plt.plot(x_plot2, y_plot2, 'r-', linewidth=2, label='f(x) = x³ + 3x² - 7')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = x³ + 3x² - 7')
plt.legend()
plt.tight_layout()
plt.show()

print("\n1.3 ПРЕОБРАЗОВАНИЕ К ВИДУ x = φ(x):")
print("Исходное уравнение: x³ + 3x² - 7 = 0")
print("\nВариант 1: x = √(7/(x+3))")
print("x³ + 3x² = 7 → x²(x+3) = 7 → x = √(7/(x+3))")
print("Область определения: 7/(x+3) > 0 и x+3 ≠ 0, то есть x > -3")

print("\nВариант 2: x = (7 - x³)/(3x)")
print("3x² = 7 - x³ → x = (7 - x³)/(3x)")
print("Область определения: x ≠ 0")

print("\nВариант 3: x = ∛(7 - 3x²)")
print("x³ = 7 - 3x² → x = ∛(7 - 3x²)")

print("\n1.4 ПРОВЕРКА СХОДИМОСТИ ДЛЯ РАЗЛИЧНЫХ ИНТЕРВАЛОВ:")
print("\nАнализ графика показывает корни приблизительно в точках:")
print("x₁ ≈ -3.5 (отрицательный корень)")
print("x₂ ≈ 1.3 (положительный корень)")
print("x₃ ≈ -1.5 (еще один отрицательный корень)")

roots_to_find2 = [
    ("Корень x₁ ≈ 1.3 (интервал [1, 2])", 1.3, phi2_interval1, "φ(x) = √(7/(x+3))"),
    ("Корень x₁ ≈ 1.3 (интервал [1, 2])", 1.3, phi2_interval3, "φ(x) = ∛(7-3x²)"),
    ("Корень x₂ ≈ -1.5 (интервал [-2, -1])", -1.5, phi2_interval3, "φ(x) = ∛(7-3x²)"),
    ("Корень x₃ ≈ -3.5 (интервал [-4, -3])", -3.5, phi2_interval2, "φ(x) = (7-x³)/(3x)"),
]

for desc, x0, phi, phi_desc in roots_to_find2:
    print(f"\n{desc}:")
    print(f"Используем {phi_desc}")
    check_convergence(phi, x0, f"x₀ = {x0}")
    
    root, iters, converged = simple_iteration(phi, x0)
    if converged:
        print(f"Найден корень: x = {root:.10f}")
        print(f"Количество итераций: {iters}")
        print(f"Проверка: f({root:.6f}) = {equation2(root):.2e}")
    else:
        print("Метод не сошелся")

print("\n" + "=" * 60)
