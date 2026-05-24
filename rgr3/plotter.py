import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def _base_axes(title):
        plt.figure(figsize=(10, 6))
        plt.xlabel("Число ядер, cores")
        plt.ylabel("TDP, Вт")
        plt.title(title)
        plt.grid(True, alpha=0.3)

    @staticmethod
    def scatter_plot(x, y):
        Plotter._base_axes("Диаграмма рассеяния: cores и TDP")
        plt.scatter(x, y, color="steelblue", edgecolor="black", alpha=0.8)
        plt.tight_layout()
        plt.savefig("scatter.png", dpi=200)
        plt.close()

    @staticmethod
    def models_plot(x, y, linear_model, quad_model, power_model):
        x_grid = np.linspace(x.min(), x.max(), 300)

        linear_grid = (
            linear_model.coefficients["a"] + linear_model.coefficients["b"] * x_grid
        )
        quad_grid = (
            quad_model.coefficients["a"]
            + quad_model.coefficients["b"] * x_grid
            + quad_model.coefficients["c"] * x_grid ** 2
        )
        power_grid = (
            power_model.coefficients["a"] * x_grid ** power_model.coefficients["b"]
        )

        Plotter._base_axes("Сравнение регрессионных моделей")
        plt.scatter(x, y, color="black", alpha=0.7, label="Исходные данные")
        plt.plot(x_grid, linear_grid, color="red", linewidth=2, label="Линейная")
        plt.plot(x_grid, quad_grid, color="green", linewidth=2, label="Квадратичная")
        plt.plot(x_grid, power_grid, color="blue", linewidth=2, label="Степенная")
        plt.legend()
        plt.tight_layout()
        plt.savefig("models_comparison.png", dpi=200)
        plt.close()

    @staticmethod
    def model_plot(x, y, x_grid, y_grid, filename, title, line_color, label):
        Plotter._base_axes(title)
        plt.scatter(x, y, color="black", alpha=0.7, label="Исходные данные")
        plt.plot(x_grid, y_grid, color=line_color, linewidth=2, label=label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()

    @staticmethod
    def individual_models_plot(x, y, linear_model, quad_model, power_model):
        x_grid = np.linspace(x.min(), x.max(), 300)

        linear_grid = (
            linear_model.coefficients["a"] + linear_model.coefficients["b"] * x_grid
        )
        quad_grid = (
            quad_model.coefficients["a"]
            + quad_model.coefficients["b"] * x_grid
            + quad_model.coefficients["c"] * x_grid ** 2
        )
        power_grid = (
            power_model.coefficients["a"] * x_grid ** power_model.coefficients["b"]
        )

        Plotter.model_plot(
            x,
            y,
            x_grid,
            linear_grid,
            "linear_model.png",
            "Линейная регрессия на фоне исходных точек",
            "red",
            "Линейная модель",
        )
        Plotter.model_plot(
            x,
            y,
            x_grid,
            quad_grid,
            "quadratic_model.png",
            "Квадратичная регрессия на фоне исходных точек",
            "green",
            "Квадратичная модель",
        )
        Plotter.model_plot(
            x,
            y,
            x_grid,
            power_grid,
            "power_model.png",
            "Степенная регрессия на фоне исходных точек",
            "blue",
            "Степенная модель",
        )

    @staticmethod
    def residual_plot(x, residuals, filename, title):
        plt.figure(figsize=(9, 6))
        plt.scatter(x, residuals, color="darkorange", edgecolor="black", alpha=0.8)
        plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
        plt.xlabel("Число ядер, cores")
        plt.ylabel("Остатки e_i")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
