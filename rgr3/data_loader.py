import pandas as pd

from rgr3.utils import fmt


class DataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):
        df = pd.read_csv(self.csv_path, encoding="utf-8-sig")

        if "x" not in df.columns or "y" not in df.columns:
            raise ValueError("В CSV-файле должны быть столбцы x и y.")

        df = df[["x", "y"]].copy()
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna()

        if (df["x"] <= 0).any() or (df["y"] <= 0).any():
            raise ValueError("Все значения x и y должны быть больше 0.")

        return df

    @staticmethod
    def print_basic_info(df):
        x = df["x"]
        y = df["y"]

        print("\nПервые строки таблицы:")
        print(df.head().round(4).to_string(index=False))

        print("\nБазовые характеристики:")
        print(f"n = {len(df)}")
        print(f"sum(x) = {fmt(x.sum())}")
        print(f"sum(y) = {fmt(y.sum())}")
        print(f"mean(x) = {fmt(x.mean())}")
        print(f"mean(y) = {fmt(y.mean())}")
        print(f"min(x) = {fmt(x.min())}")
        print(f"max(x) = {fmt(x.max())}")
        print(f"min(y) = {fmt(y.min())}")
        print(f"max(y) = {fmt(y.max())}")
