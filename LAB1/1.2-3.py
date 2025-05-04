import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from tabulate import tabulate

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_idx, test_idx = indices[:train_size], indices[train_size:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

reg_sklearn = LinearRegression()
reg_sklearn.fit(X_train, y_train)
y_pred_sklearn = reg_sklearn.predict(X_test)

X_train_mean = np.mean(X_train)
y_train_mean = np.mean(y_train)

numerator = sum((X_train[i] - X_train_mean) * (y_train[i] - y_train_mean) for i in range(len(X_train)))
denominator = sum((X_train[i] - X_train_mean) ** 2 for i in range(len(X_train)))

b1 = numerator / denominator
b0 = y_train_mean - b1 * X_train_mean

y_pred_custom = b0 + b1 * X_test

mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)
mape_sklearn = MAPE(y_test, y_pred_sklearn)

mae_custom = mean_absolute_error(y_test, y_pred_custom)
r2_custom = r2_score(y_test, y_pred_custom)
mape_custom = MAPE(y_test, y_pred_custom)

plt.figure(figsize=(12, 5))
plt.scatter(X_test, y_test, color='blue', label='Тестовые данные')
plt.plot(X_test, y_pred_sklearn, color='red', linewidth=2, label='Scikit-Learn')
plt.plot(X_test, y_pred_custom, color='green', linestyle='dashed', linewidth=2, label='Собственная реализация')
plt.title('Сравнение регрессионных моделей')
plt.xlabel('Индекс массы тела')
plt.ylabel('Прогрессия заболевания')
plt.legend()
plt.grid(True)
plt.show()

print("\nКоэффициенты регрессии:")
print(f"Scikit-Learn: intercept = {reg_sklearn.intercept_:}, coef = {reg_sklearn.coef_[0]:}")
print(f"Собственный алгоритм: b0 = {b0}, b1 = {b1}")

print("\nМетрики качества моделей:")
metrics_data = [
    ['MAE', mae_sklearn, mae_custom],
    ['R^2', r2_sklearn, r2_custom],
    ['MAPE (%)', mape_sklearn, mape_custom]
]
print(tabulate(metrics_data, headers=['Метрика', 'Scikit-Learn', 'Собственная реализация'], floatfmt=".4f"))

results = []

print("\nАнализ качества моделей:")
print("1. MAE (Mean Absolute Error) - средняя абсолютная ошибка. Чем меньше, тем лучше.")
print(f"   Scikit-Learn: {mae_sklearn:.2f}, Собственная: {mae_custom:.2f}")

print("\n2. R² (Коэффициент детерминации) - показывает долю объясненной дисперсии. Ближе к 1 - лучше.")
print(f"   Scikit-Learn: {r2_sklearn:.4f}, Собственная: {r2_custom:.4f}")

print("\n3. MAPE (Mean Absolute Percentage Error) - средняя абсолютная процентная ошибка. Чем меньше, тем лучше.")
print(f"   Scikit-Learn: {mape_sklearn:.2f}%, Собственная: {mape_custom:.2f}%")

print("\nВывод: Обе реализации показывают схожие результаты, что подтверждает правильность")
print("собственной реализации алгоритма линейной регрессии. Небольшие различия в метриках")
print("могут быть вызваны особенностями реализации алгоритмов.")