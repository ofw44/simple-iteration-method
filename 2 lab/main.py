import numpy as np
import matplotlib.pyplot as plt

def check_diagonal_dominance(A):
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        summ = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= summ:
            return False
    return True

def rearrange_for_dominance(A, b):
    n = len(A)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    for i in range(n):
        max_idx = i
        max_val = abs(A[i][i])
        for k in range(i+1, n):
            if abs(A[k][i]) > max_val:
                max_val = abs(A[k][i])
                max_idx = k
        if max_idx != i:
            A[[i, max_idx]] = A[[max_idx, i]]
            b[[i, max_idx]] = b[[max_idx, i]]
    
    return A.tolist(), b.tolist()

def simple_iteration_system(A, b, eps=0.0001, max_iter=1000):
    n = len(A)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    alpha = np.zeros((n, n))
    beta = np.zeros(n)
    
    for i in range(n):
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = -A[i][j] / A[i][i]
            else:
                alpha[i][j] = 0
    
    x = beta.copy()
    
    for k in range(max_iter):
        x_new = beta.copy()
        for i in range(n):
            summ = 0
            for j in range(n):
                summ += alpha[i][j] * x[j]
            x_new[i] += summ
        
        if max(abs(x_new[i] - x[i]) for i in range(n)) < eps:
            return x_new, k+1
        
        x = x_new
    
    return x, max_iter

def newton_system(x, y):
    f1 = np.cos(y + 0.5) - x - 2
    f2 = np.sin(x) - 2*y - 1
    return np.array([f1, f2])

def jacobian(x, y):
    j11 = -1
    j12 = -np.sin(y + 0.5)
    j21 = np.cos(x)
    j22 = -2
    return np.array([[j11, j12], [j21, j22]])

def newton_method(x0, y0, eps=0.0001, max_iter=100):
    x, y = x0, y0
    
    for k in range(max_iter):
        F = newton_system(x, y)
        J = jacobian(x, y)
        
        try:
            delta = np.linalg.solve(J, -F)
        except:
            return None, None, 0
        
        x_new = x + delta[0]
        y_new = y + delta[1]
        
        if max(abs(x_new - x), abs(y_new - y)) < eps:
            return x_new, y_new, k+1
        
        x, y = x_new, y_new
    
    return x, y, max_iter

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №2")
print("Метод итераций. Метод Ньютона")
print("=" * 70)

print("\n" + "=" * 70)
print("ЗАДАНИЕ 1: Метод итераций для СЛАУ")
print("=" * 70)

print("\n1.1 УСЛОВИЕ ЗАДАЧИ:")
print("Решить систему линейных уравнений методом итерации с точностью ε = 0.0001")
print("0.16x₁ + 0.13x₂ - 0.32x₃ - 0.16x₄ = 0.64")
print("0.34x₁ - 0.08x₂ + 0.15x₃ - 0.18x₄ = -1.42")
print("0.15x₁ + 0.32x₂ + 0.13x₃ - 0.25x₄ = 2.06")
print("0.11x₁ - 0.26x₂ - 0.08x₃ + 0.24x₄ = -0.83")

A = [
    [0.16, 0.13, -0.32, -0.16],
    [0.34, -0.08, 0.15, -0.18],
    [0.15, 0.32, 0.13, -0.25],
    [0.11, -0.26, -0.08, 0.24]
]
b = [0.64, -1.42, 2.06, -0.83]

print("\n1.2 ПРЕОБРАЗОВАННАЯ СИСТЕМА:")
print("Проверка диагонального преобладания исходной матрицы:")

if check_diagonal_dominance(A):
    print("Диагональное преобладание выполняется")
else:
    print("Диагональное преобладание НЕ выполняется")
    print("Выполняем перестановку строк для достижения преобладания...")
    A_new, b_new = rearrange_for_dominance(A, b)
    print("\nПреобразованная система:")
    for i in range(len(A_new)):
        eq = ""
        for j in range(len(A_new[i])):
            sign = " + " if A_new[i][j] >= 0 and j > 0 else " "
            eq += f"{sign}{A_new[i][j]:.2f}x{j+1}"
        print(f"{eq} = {b_new[i]}")
    
    if check_diagonal_dominance(A_new):
        print("\nПосле перестановки диагональное преобладание выполняется")
        A = A_new
        b = b_new
    else:
        print("\nДиагональное преобладание всё ещё не выполняется")
        print("Приводим систему к виду x = αx + β")
        
        A = [
            [0.34, -0.08, 0.15, -0.18],
            [0.15, 0.32, 0.13, -0.25],
            [0.11, -0.26, -0.08, 0.24],
            [0.16, 0.13, -0.32, -0.16]
        ]
        b = [-1.42, 2.06, -0.83, 0.64]

print("\nСистема в виде x = αx + β:")
n = len(A)
alpha = np.zeros((n, n))
beta = np.zeros(n)

for i in range(n):
    beta[i] = b[i] / A[i][i]
    for j in range(n):
        if i != j:
            alpha[i][j] = -A[i][j] / A[i][i]

for i in range(n):
    eq = f"x{i+1} = {beta[i]:.6f}"
    for j in range(n):
        if alpha[i][j] != 0:
            sign = " + " if alpha[i][j] >= 0 else " - "
            eq += f"{sign}{abs(alpha[i][j]):.6f}·x{j+1}"
    print(eq)

print("\n1.5 ПРОВЕРКА СХОДИМОСТИ:")
norms = []
for i in range(n):
    norm = sum(abs(alpha[i][j]) for j in range(n))
    norms.append(norm)
    print(f"||α{i+1}|| = {norm:.6f}")

max_norm = max(norms)
if max_norm < 1:
    print(f"\nmax||αᵢ|| = {max_norm:.6f} < 1")
    print("Условие сходимости выполнено!")
else:
    print(f"\nmax||αᵢ|| = {max_norm:.6f} >= 1")
    print("Условие сходимости НЕ выполнено")

print("\n1.6 ПРОГРАММНЫЕ СРЕДСТВА:")
print("Язык программирования: Python 3")
print("Библиотеки: NumPy (работа с массивами), Matplotlib (графики)")

print("\n1.7 РЕШЕНИЕ:")
x, iterations = simple_iteration_system(A, b, 0.0001)
print(f"Метод итераций (ε = 0.0001):")
for i in range(len(x)):
    print(f"x{i+1} = {x[i]:.6f}")
print(f"Количество итераций: {iterations}")

print("\nПроверка решения (подстановка в исходную систему):")
original_A = [
    [0.16, 0.13, -0.32, -0.16],
    [0.34, -0.08, 0.15, -0.18],
    [0.15, 0.32, 0.13, -0.25],
    [0.11, -0.26, -0.08, 0.24]
]
original_b = [0.64, -1.42, 2.06, -0.83]

for i in range(len(original_A)):
    summ = sum(original_A[i][j] * x[j] for j in range(len(x)))
    print(f"Уравнение {i+1}: левая часть = {summ:.6f}, правая часть = {original_b[i]:.6f}, разница = {abs(summ - original_b[i]):.2e}")

print("\n" + "=" * 70)
print("ЗАДАНИЕ 2: Метод Ньютона для нелинейной системы")
print("=" * 70)

print("\n1.1 УСЛОВИЕ ЗАДАЧИ:")
print("Решить систему нелинейных уравнений методом Ньютона с точностью ε = 0.0001")
print("cos(y + 0.5) - x = 2")
print("sin(x) - 2y = 1")

print("\n1.3 ГРАФИК ФУНКЦИЙ:")

x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

Z1 = np.cos(Y + 0.5) - X - 2
Z2 = np.sin(X) - 2*Y - 1

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Линии уровня (пересечение - решение)')
plt.legend(['cos(y+0.5)-x=2', 'sin(x)-2y=1'])

plt.subplot(1, 2, 2)
x_range = np.linspace(-3, 3, 100)
y1 = np.cos(x_range + 0.5) - 2
y2 = (np.sin(x_range) - 1) / 2
plt.plot(x_range, y1, 'b-', linewidth=2, label='y = cos(x+0.5)-2')
plt.plot(x_range, y2, 'r-', linewidth=2, label='y = (sin(x)-1)/2')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Функции, разрешённые относительно y')
plt.legend()

plt.tight_layout()
plt.show()

print("\n1.4 МАТРИЦА ЯКОБИ:")
print("Система уравнений:")
print("F₁(x,y) = cos(y + 0.5) - x - 2 = 0")
print("F₂(x,y) = sin(x) - 2y - 1 = 0")
print("\nМатрица Якоби J(x,y):")
print("         [∂F₁/∂x  ∂F₁/∂y]   [  -1     -sin(y+0.5) ]")
print("J(x,y) = [                ] = [                    ]")
print("         [∂F₂/∂x  ∂F₂/∂y]   [ cos(x)      -2      ]")

print("\n1.5 ПРОВЕРКА СХОДИМОСТИ:")
x0, y0 = -2, -1
print(f"Начальное приближение: x₀ = {x0}, y₀ = {y0}")
J0 = jacobian(x0, y0)
det_J0 = np.linalg.det(J0)
print(f"Определитель матрицы Якоби в начальной точке: det(J) = {det_J0:.6f}")
if abs(det_J0) > 1e-10:
    print("Матрица Якоби невырождена - условие сходимости выполняется")
else:
    print("Матрица Якоби вырождена - необходимо другое начальное приближение")

print("\n1.6 ПРОГРАММНЫЕ СРЕДСТВА:")
print("Язык программирования: Python 3")
print("Библиотеки: NumPy (решение СЛАУ, работа с матрицами)")
print("Matplotlib (визуализация графиков)")

print("\n1.7 РЕШЕНИЕ:")
x_sol, y_sol, iters = newton_method(-2, -1, 0.0001)

if x_sol is not None:
    print(f"Метод Ньютона (ε = 0.0001):")
    print(f"x = {x_sol:.6f}")
    print(f"y = {y_sol:.6f}")
    print(f"Количество итераций: {iters}")
    
    print("\nПроверка решения:")
    f1_check = np.cos(y_sol + 0.5) - x_sol - 2
    f2_check = np.sin(x_sol) - 2*y_sol - 1
    print(f"F₁(x,y) = cos({y_sol:.6f} + 0.5) - {x_sol:.6f} - 2 = {f1_check:.2e}")
    print(f"F₂(x,y) = sin({x_sol:.6f}) - 2·{y_sol:.6f} - 1 = {f2_check:.2e}")
else:
    print("Метод Ньютона не сошелся")
    print("Пробуем другое начальное приближение...")
    x_sol, y_sol, iters = newton_method(-1, -0.5, 0.0001)
    if x_sol is not None:
        print(f"x = {x_sol:.6f}")
        print(f"y = {y_sol:.6f}")
        print(f"Количество итераций: {iters}")