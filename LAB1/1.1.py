import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = list(reader)
    return headers, data


def get_statistics(data, col_idx):
    column = [float(row[col_idx]) for row in data]
    stats = {
        'count': len(column),
        'min': min(column),
        'max': max(column),
        'mean': np.mean(column)
    }
    return stats


def plot_data(x, y, x_label, y_label):
    plt.figure(figsize=(15, 5))

    # 1. Исходные точки
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, color='blue')
    plt.title('Исходные данные')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    # 2. Регрессионная прямая
    plt.subplot(1, 3, 2)
    plt.scatter(x, y, color='blue')

    # Вычисление параметров прямой
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

    b1 = numerator / denominator
    b0 = y_mean - b1 * x_mean

    # Построение прямой
    x_line = np.array([min(x), max(x)])
    y_line = b0 + b1 * x_line
    plt.plot(x_line, y_line, color='red')
    plt.title('Регрессионная прямая')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    # 3. Квадраты ошибок
    plt.subplot(1, 3, 3)
    plt.scatter(x, y, color='blue')
    plt.plot(x_line, y_line, color='red')

    # Добавление квадратов ошибок
    for xi, yi in zip(x, y):
        y_pred = b0 + b1 * xi
        error = yi - y_pred
        if error != 0:
            rect = Rectangle((xi, y_pred), abs(error), abs(error),
                             linewidth=1, edgecolor='green', facecolor='green', alpha=0.1)
            plt.gca().add_patch(rect)

    plt.title('Квадраты ошибок')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return b0, b1


def main():
    filename = input("Введите имя CSV файла: ")
    headers, data = read_data(filename)

    print("\nДоступные столбцы:")
    for i, header in enumerate(headers):
        print(f"{i}: {header}")

    x_col = int(input("Выберите столбец для X: "))
    y_col = int(input("Выберите столбец для Y: "))

    x = [float(row[x_col]) for row in data]
    y = [float(row[y_col]) for row in data]

    print("\nСтатистика по X:")
    x_stats = get_statistics(data, x_col)
    print(f"Количество: {x_stats['count']}")
    print(f"Минимум: {x_stats['min']:.2f}")
    print(f"Максимум: {x_stats['max']:.2f}")
    print(f"Среднее: {x_stats['mean']:.2f}")

    print("\nСтатистика по Y:")
    y_stats = get_statistics(data, y_col)
    print(f"Количество: {y_stats['count']}")
    print(f"Минимум: {y_stats['min']:.2f}")
    print(f"Максимум: {y_stats['max']:.2f}")
    print(f"Среднее: {y_stats['mean']:.2f}")

    b0, b1 = plot_data(x, y, headers[x_col], headers[y_col])
    print(f"\nУравнение регрессионной прямой: y = {b0:.2f} + {b1:.2f}x")


if __name__ == "__main__":
    main()