from rgr3 import CSV_PATH
from rgr3.analysis import RegressionAnalysis
from rgr3.data_loader import DataLoader
from rgr3.plotter import Plotter
from rgr3.report_builder import ReportBuilder
from rgr3.utils import fmt


def main():
    print("РГР №3 по математической статистике")
    print("Тема: Регрессионные модели и их интерпретация")

    loader = DataLoader(CSV_PATH)
    df = loader.load()
    loader.print_basic_info(df)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    Plotter.scatter_plot(x, y)
    print("\nДиаграмма рассеяния сохранена в файл scatter.png")

    analysis = RegressionAnalysis(x, y)

    linear_model = analysis.linear_model()
    quad_model = analysis.quadratic_model()
    power_model = analysis.power_model()

    Plotter.models_plot(x, y, linear_model, quad_model, power_model)
    print("\nОбщий график моделей сохранён в файл models_comparison.png")

    Plotter.residual_plot(x, linear_model.residuals, "residuals_linear.png", "Остатки линейной модели")
    Plotter.residual_plot(x, quad_model.residuals, "residuals_quad.png", "Остатки квадратичной модели")
    Plotter.residual_plot(x, power_model.residuals, "residuals_power.png", "Остатки степенной модели")
    print("Графики остатков сохранены в файлы residuals_linear.png, residuals_quad.png, residuals_power.png")

    models = [linear_model, quad_model, power_model]
    comparison_df = ReportBuilder.comparison_table(models)

    comparison_df.to_csv("model_comparison.csv", index=False, encoding="utf-8-sig")
    print("\nТаблица сравнения сохранена в файл model_comparison.csv")

    best_model = ReportBuilder.choose_best_model(models)

    linear_stats = analysis.linear_statistics(linear_model)

    x_star = 20
    preds = ReportBuilder.predictions(x_star, linear_model, quad_model, power_model)
    linear_pred, quad_pred, power_pred = preds

    ReportBuilder.print_final_results(
        df=df,
        best_model=best_model,
        linear_model=linear_model,
        quad_model=quad_model,
        power_model=power_model,
        linear_stats=linear_stats,
        x_star=x_star,
        preds=preds,
    )


if __name__ == "__main__":
    main()
