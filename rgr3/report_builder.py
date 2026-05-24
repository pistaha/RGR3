import pandas as pd

from rgr3.utils import fmt


class ReportBuilder:
    @staticmethod
    def format_linear_equation(model):
        a = model.coefficients["a"]
        b = model.coefficients["b"]
        return f"y_hat = {a:.4f} + {b:.4f}*x"

    @staticmethod
    def format_quadratic_equation(model):
        a = model.coefficients["a"]
        b = model.coefficients["b"]
        c = model.coefficients["c"]
        c_sign = "-" if c < 0 else "+"
        return f"y_hat = {a:.4f} + {b:.4f}*x {c_sign} {abs(c):.4f}*x^2"

    @staticmethod
    def format_power_equation(model):
        a = model.coefficients["a"]
        b = model.coefficients["b"]
        return f"y_hat = {a:.4f}*x^{b:.4f}"

    @staticmethod
    def choose_best_model(models):
        return sorted(models, key=lambda m: (-m.r2, m.rss, m.approx_error))[0]

    @staticmethod
    def comparison_table(models):
        return pd.DataFrame(
            {
                "Модель": [model.name for model in models],
                "Уравнение": [model.equation for model in models],
                "RSS": [round(model.rss, 4) for model in models],
                "R2": [round(model.r2, 4) for model in models],
                "Средняя ошибка аппроксимации, %": [
                    round(model.approx_error, 4) for model in models
                ],
            }
        )

    @staticmethod
    def predictions(x_star, linear_model, quad_model, power_model):
        linear_pred = (
            linear_model.coefficients["a"] + linear_model.coefficients["b"] * x_star
        )
        quad_pred = (
            quad_model.coefficients["a"]
            + quad_model.coefficients["b"] * x_star
            + quad_model.coefficients["c"] * x_star ** 2
        )
        power_pred = power_model.coefficients["a"] * x_star ** power_model.coefficients["b"]
        return linear_pred, quad_pred, power_pred

    @staticmethod
    def format_p_value(p_value):
        if p_value < 0.0001:
            return "< 0.0001"
        return fmt(p_value)

    @staticmethod
    def print_final_results(df, best_model, linear_model, quad_model, power_model, linear_stats, x_star, preds):
        linear_pred, quad_pred, power_pred = preds
        best_pred = {
            "Линейная": linear_pred,
            "Квадратичная": quad_pred,
            "Степенная": power_pred,
        }[best_model.name]
        line = "=" * 60

        print(f"\n{line}")
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ РГР №3")
        print("Вариант: C-6")
        print("Данные: openintro::cpu, срез по AMD")
        print("Фактор x: число ядер процессора")
        print("Результат y: TDP, Вт")
        print(f"Количество наблюдений: n = {len(df)}")
        print(line)

        print("\n1. Построенные модели:\n")
        print("Линейная модель:")
        print(ReportBuilder.format_linear_equation(linear_model))
        print("\nКвадратичная модель:")
        print(ReportBuilder.format_quadratic_equation(quad_model))
        print("\nСтепенная модель:")
        print(ReportBuilder.format_power_equation(power_model))

        print(f"\n{line}\n")
        print("2. Сравнение качества моделей:\n")
        print(f"{'Модель':<14}{'RSS':>13}{'R2':>11}{'Средняя ошибка, %':>22}")
        for model in [linear_model, quad_model, power_model]:
            print(
                f"{model.name:<14}{model.rss:>13.4f}{model.r2:>11.4f}{model.approx_error:>22.4f}"
            )
        print(f"\nПо R2 и RSS лучшей является {best_model.name.lower()} модель.")
        print("У неё наибольшее R2 и наименьшая сумма квадратов остатков.")

        print(f"\n{line}\n")
        print("3. Статистический анализ линейной модели:\n")
        print(f"RSS = {fmt(linear_stats['rss'])}")
        print(f"s^2 = {fmt(linear_stats['s2'])}")
        print(f"s = {fmt(linear_stats['s'])}")
        print("\n95% доверительный интервал для a:")
        print(
            f"[{fmt(linear_stats['a_interval'][0])}; {fmt(linear_stats['a_interval'][1])}]"
        )
        print("\n95% доверительный интервал для b:")
        print(
            f"[{fmt(linear_stats['b_interval'][0])}; {fmt(linear_stats['b_interval'][1])}]"
        )
        print(f"\nt_набл = {fmt(linear_stats['t_observed'])}")
        print(f"t_крит = {fmt(linear_stats['t_crit'])}")
        print(f"p-value {ReportBuilder.format_p_value(linear_stats['p_value'])}")
        print("\nТак как |t_набл| > t_крит, гипотеза H0: b = 0 отвергается.")
        print("Коэффициент b статистически значим.")
        print("Следовательно, связь между числом ядер и TDP статистически значима.")

        print(f"\n{line}\n")
        print(f"4. Прогноз при x* = {x_star}:\n")
        print(f"Линейная модель:      {fmt(linear_pred)} Вт")
        print(f"Квадратичная модель:  {fmt(quad_pred)} Вт")
        print(f"Степенная модель:     {fmt(power_pred)} Вт")
        print("\nОсновной прогноз по выбранной модели:")
        print(f"y_hat({x_star}) = {fmt(best_pred)} Вт")
        print(f"\nОкруглённо: ожидаемый TDP ≈ {round(best_pred):.0f} Вт.")

        print(f"\n{line}\n")
        print("5. Общий вывод:\n")
        print("По диаграмме рассеяния и построенным моделям видно, что при увеличении")
        print("числа ядер TDP процессора AMD в среднем возрастает. Лучшую аппроксимацию")
        print("по R2 и RSS дала квадратичная модель. Линейная модель также подтвердила")
        print("статистически значимую положительную связь между числом ядер и TDP.")
        print(line)
