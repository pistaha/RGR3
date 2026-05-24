import math

import numpy as np
from scipy.stats import t

from rgr3.models import ModelResult
from rgr3.utils import fmt


class RegressionAnalysis:
    def __init__(self, x, y):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.n = len(self.x)
        self.y_mean = self.y.mean()
        self.tss = np.sum((self.y - self.y_mean) ** 2)

    def _calc_metrics(self, y_hat):
        residuals = self.y - y_hat
        rss = np.sum(residuals ** 2)
        r2 = 1 - rss / self.tss
        approx_error = 100 / self.n * np.sum(np.abs((self.y - y_hat) / self.y))
        return residuals, rss, r2, approx_error

    def linear_model(self):
        x_mean = self.x.mean()
        y_mean = self.y.mean()
        sxx = np.sum((self.x - x_mean) ** 2)
        sxy = np.sum((self.x - x_mean) * (self.y - y_mean))
        b = sxy / sxx
        a = y_mean - b * x_mean
        y_hat = a + b * self.x
        residuals, rss, r2, approx_error = self._calc_metrics(y_hat)

        print("\nЛИНЕЙНАЯ МОДЕЛЬ: y = a + b*x")
        print(f"x_mean = {fmt(x_mean)}")
        print(f"y_mean = {fmt(y_mean)}")
        print(f"Sxx = {fmt(sxx)}")
        print(f"Sxy = {fmt(sxy)}")
        print(f"a = {fmt(a)}")
        print(f"b = {fmt(b)}")
        print(f"Уравнение: y_hat = {fmt(a)} + {fmt(b)}*x")
        print(f"RSS_linear = {fmt(rss)}")
        print(f"TSS = {fmt(self.tss)}")
        print(f"R2_linear = {fmt(r2)}")
        print(f"A_linear = {fmt(approx_error)} %")

        return ModelResult(
            name="Линейная",
            equation=f"y = {a:.4f} + {b:.4f}*x",
            y_hat=y_hat,
            residuals=residuals,
            rss=rss,
            r2=r2,
            approx_error=approx_error,
            coefficients={"a": a, "b": b, "x_mean": x_mean, "y_mean": y_mean, "Sxx": sxx},
        )

    @staticmethod
    def gaussian_elimination_3x3(A, B):
        aug = np.array(
            [
                [A[0][0], A[0][1], A[0][2], B[0]],
                [A[1][0], A[1][1], A[1][2], B[1]],
                [A[2][0], A[2][1], A[2][2], B[2]],
            ],
            dtype=float,
        )

        print("\nРасширенная матрица системы:")
        print(np.round(aug, 4))

        m21 = aug[1, 0] / aug[0, 0]
        print(f"\nR2 <- R2 - {fmt(m21)}*R1")
        aug[1] = aug[1] - m21 * aug[0]
        print(np.round(aug, 4))

        m31 = aug[2, 0] / aug[0, 0]
        print(f"\nR3 <- R3 - {fmt(m31)}*R1")
        aug[2] = aug[2] - m31 * aug[0]
        print(np.round(aug, 4))

        m32 = aug[2, 1] / aug[1, 1]
        print(f"\nR3 <- R3 - {fmt(m32)}*R2")
        aug[2] = aug[2] - m32 * aug[1]
        print(np.round(aug, 4))

        print("\nВерхнетреугольная система:")
        print(np.round(aug, 4))

        c = aug[2, 3] / aug[2, 2]
        b = (aug[1, 3] - aug[1, 2] * c) / aug[1, 1]
        a = (aug[0, 3] - aug[0, 2] * c - aug[0, 1] * b) / aug[0, 0]

        print("\nОбратная подстановка:")
        print(f"c = {fmt(c)}")
        print(f"b = {fmt(b)}")
        print(f"a = {fmt(a)}")

        return a, b, c

    def quadratic_model(self):
        sum_x = np.sum(self.x)
        sum_x2 = np.sum(self.x ** 2)
        sum_x3 = np.sum(self.x ** 3)
        sum_x4 = np.sum(self.x ** 4)
        sum_y = np.sum(self.y)
        sum_xy = np.sum(self.x * self.y)
        sum_x2y = np.sum((self.x ** 2) * self.y)

        print("\nКВАДРАТИЧНАЯ МОДЕЛЬ: y = a + b*x + c*x^2")
        print(f"n = {self.n}")
        print(f"sum_x = {fmt(sum_x)}")
        print(f"sum_x2 = {fmt(sum_x2)}")
        print(f"sum_x3 = {fmt(sum_x3)}")
        print(f"sum_x4 = {fmt(sum_x4)}")
        print(f"sum_y = {fmt(sum_y)}")
        print(f"sum_xy = {fmt(sum_xy)}")
        print(f"sum_x2y = {fmt(sum_x2y)}")

        A = np.array(
            [
                [self.n, sum_x, sum_x2],
                [sum_x, sum_x2, sum_x3],
                [sum_x2, sum_x3, sum_x4],
            ],
            dtype=float,
        )
        B = np.array([sum_y, sum_xy, sum_x2y], dtype=float)

        print("\nСистема нормальных уравнений в численном виде:")
        print(
            f"{fmt(A[0,0])}*a + {fmt(A[0,1])}*b + {fmt(A[0,2])}*c = {fmt(B[0])}"
        )
        print(
            f"{fmt(A[1,0])}*a + {fmt(A[1,1])}*b + {fmt(A[1,2])}*c = {fmt(B[1])}"
        )
        print(
            f"{fmt(A[2,0])}*a + {fmt(A[2,1])}*b + {fmt(A[2,2])}*c = {fmt(B[2])}"
        )

        a_gauss, b_gauss, c_gauss = self.gaussian_elimination_3x3(A, B)
        numpy_solution = np.linalg.solve(A, B)

        print("\nПроверка через numpy.linalg.solve:")
        print(
            f"a = {fmt(numpy_solution[0])}, b = {fmt(numpy_solution[1])}, c = {fmt(numpy_solution[2])}"
        )

        y_hat = a_gauss + b_gauss * self.x + c_gauss * self.x ** 2
        residuals, rss, r2, approx_error = self._calc_metrics(y_hat)

        print(f"Уравнение: y_hat = {fmt(a_gauss)} + {fmt(b_gauss)}*x + {fmt(c_gauss)}*x^2")
        print(f"RSS_quad = {fmt(rss)}")
        print(f"R2_quad = {fmt(r2)}")
        print(f"A_quad = {fmt(approx_error)} %")

        return ModelResult(
            name="Квадратичная",
            equation=f"y = {a_gauss:.4f} + {b_gauss:.4f}*x + {c_gauss:.4f}*x^2",
            y_hat=y_hat,
            residuals=residuals,
            rss=rss,
            r2=r2,
            approx_error=approx_error,
            coefficients={"a": a_gauss, "b": b_gauss, "c": c_gauss},
        )

    def power_model(self):
        X = np.log(self.x)
        Y = np.log(self.y)
        X_mean = X.mean()
        Y_mean = Y.mean()
        sxx = np.sum((X - X_mean) ** 2)
        sxy = np.sum((X - X_mean) * (Y - Y_mean))
        b = sxy / sxx
        A = Y_mean - b * X_mean
        a = math.exp(A)
        y_hat = a * self.x ** b
        residuals, rss, r2, approx_error = self._calc_metrics(y_hat)

        print("\nСТЕПЕННАЯ МОДЕЛЬ: y = a*x^b")
        print(f"X_mean = {fmt(X_mean)}")
        print(f"Y_mean = {fmt(Y_mean)}")
        print(f"SXX = {fmt(sxx)}")
        print(f"SXY = {fmt(sxy)}")
        print(f"A = ln(a) = {fmt(A)}")
        print(f"a = {fmt(a)}")
        print(f"b = {fmt(b)}")
        print(f"Уравнение: y_hat = {fmt(a)}*x^{fmt(b)}")
        print(f"RSS_power = {fmt(rss)}")
        print(f"R2_power = {fmt(r2)}")
        print(f"A_power = {fmt(approx_error)} %")

        return ModelResult(
            name="Степенная",
            equation=f"y = {a:.4f}*x^{b:.4f}",
            y_hat=y_hat,
            residuals=residuals,
            rss=rss,
            r2=r2,
            approx_error=approx_error,
            coefficients={"a": a, "b": b, "A": A},
        )

    def linear_statistics(self, linear_result):
        a = linear_result.coefficients["a"]
        b = linear_result.coefficients["b"]
        x_mean = linear_result.coefficients["x_mean"]
        sxx = linear_result.coefficients["Sxx"]
        rss = linear_result.rss

        s2 = rss / (self.n - 2)
        s = math.sqrt(s2)
        se_b = s / math.sqrt(sxx)
        se_a = s * math.sqrt(1 / self.n + x_mean ** 2 / sxx)

        alpha = 0.05
        t_crit = t.ppf(1 - alpha / 2, df=self.n - 2)

        a_left = a - t_crit * se_a
        a_right = a + t_crit * se_a
        b_left = b - t_crit * se_b
        b_right = b + t_crit * se_b

        t_observed = b / se_b
        p_value = 2 * (1 - t.cdf(abs(t_observed), df=self.n - 2))
        significant = abs(t_observed) > t_crit

        print("\nПОДРОБНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ ЛИНЕЙНОЙ МОДЕЛИ")
        print(f"RSS_linear = {fmt(rss)}")
        print(f"s^2 = {fmt(s2)}")
        print(f"s = {fmt(s)}")
        print(f"SE_a = {fmt(se_a)}")
        print(f"SE_b = {fmt(se_b)}")
        print(f"t_crit = {fmt(t_crit)}")
        print(f"Доверительный интервал для a: [{fmt(a_left)}; {fmt(a_right)}]")
        print(f"Доверительный интервал для b: [{fmt(b_left)}; {fmt(b_right)}]")
        print(f"t_observed = {fmt(t_observed)}")
        print(f"p_value = {fmt(p_value)}")
        if significant:
            print("Вывод: H0 отвергается, коэффициент b статистически значим.")
        else:
            print("Вывод: H0 не отвергается, коэффициент b статистически незначим.")

        return {
            "rss": rss,
            "s2": s2,
            "s": s,
            "se_a": se_a,
            "se_b": se_b,
            "t_crit": t_crit,
            "a_interval": (a_left, a_right),
            "b_interval": (b_left, b_right),
            "t_observed": t_observed,
            "p_value": p_value,
            "significant": significant,
        }
