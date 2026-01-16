import os
import numpy as np
import pandas as pd

# === НАСТРОЙКИ ===
DATA_PATH = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\data\clean\Data_29_10_full.csv"
OUT_DIR   = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\outputs"
Z_THRESH  = 3.0   # порог для |z| по гауссовому фильтру

# Возможные названия колонки возраста
AGE_CANDIDATES = ["Age", "age"]

def detect_age_column(df: pd.DataFrame) -> str:
    for col in AGE_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"Не удалось найти колонку возраста среди: {AGE_CANDIDATES}. "
        f"Текущие колонки: {list(df.columns)}"
    )

def build_age_groups(df: pd.DataFrame, age_col: str) -> pd.Series:
    """
    Создаём возрастные диапазоны (группы) для внутригрупповой z-нормализации.
    Если хочешь, потом можно будет подправить границы.
    """
    age = df[age_col]
    # На всякий случай игнорируем NaN при расчёте min/max
    min_age = int(np.nanmin(age))
    max_age = int(np.nanmax(age))

    # Пример: шаг 5 лет (0–5, 5–10, ..., до max_age+5)
    upper = max(35, max_age + 5)
    bins = list(range(0, upper + 1, 5))  # 0,5,10,...,upper
    if min_age < 0:
        bins.insert(0, min_age - 1)

    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]
    age_groups = pd.cut(age, bins=bins, labels=labels, include_lowest=True)

    return age_groups

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Загружаю данные из:\n{DATA_PATH}\n")
    df = pd.read_csv(DATA_PATH)

    print(f"Размеры исходного набора: {df.shape[0]} строк, {df.shape[1]} столбцов")

    # === 1. Определяем колонку возраста и возрастные группы ===
    age_col = detect_age_column(df)
    print(f"\nКолонка возраста определена как: '{age_col}'")

    df["AgeGroup"] = build_age_groups(df, age_col)

    print("\nПример распределения по возрастным группам:")
    print(df["AgeGroup"].value_counts(dropna=False))

    # === 2. Числовые колонки, по которым будем фильтровать ===
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Исключаем возраст и возможный ID (если есть)
    exclude_cols = {age_col, "ID"}
    num_cols = [c for c in num_cols if c not in exclude_cols]

    print(f"\nЧисловых колонок для фильтрации: {len(num_cols)}")

    # === 3. Внутригрупповая z-нормализация + гауссов фильтр ===
    print("\nНачинаю внутригрупповую z-нормализацию и фильтрацию выбросов...")

    # Группируем по возрастным группам (явно укажем observed=False, чтобы убрать предупреждение)
    group = df.groupby("AgeGroup", observed=False)

    # Средние и стандартные отклонения по группам
    means = group[num_cols].transform("mean")
    stds  = group[num_cols].transform("std")

    # z-скоры
    z_scores = (df[num_cols] - means) / stds

    # std > 0, чтобы не делить на ноль
    std_positive = stds > 0

    # булева маска выбросов: там, где |z| > порога и std > 0
    outliers = (z_scores.abs() > Z_THRESH) & std_positive

    # Маска для подсчёта — заменим NaN на False
    removed_mask = outliers.fillna(False)

    # === ВАЖНО: применяем маску ТОЛЬКО к числовым колонкам через .mask(), а не через df.loc[...]
    # в outliers те же индексы и колонки, что и в df[num_cols]
    df[num_cols] = df[num_cols].mask(outliers, np.nan)

    # === 4. Подсчёт статистики по удалённым значениям ===
    total_numeric_cells = df[num_cols].size
    total_removed = int(removed_mask.values.sum())

    print("\n=== РЕЗУЛЬТАТЫ ФИЛЬТРАЦИИ ===")
    print(f"Всего числовых ячеек (после возможных NaN до фильтрации): {total_numeric_cells}")
    print(f"Удалено (заменено на NaN) значений по критерию |z| > {Z_THRESH}: {total_removed}")
    if total_numeric_cells > 0:
        print(f"Процент удалённых значений: {total_removed / total_numeric_cells * 100:.3f}%")

    # 4.1. Сводка по колонкам
    removed_by_col = removed_mask.sum().sort_values(ascending=False)
    print("\nТоп-20 колонок по числу удалённых значений:")
    print(removed_by_col.head(20))

    # 4.2. Сводка по возрастным группам
    removed_per_row = removed_mask.sum(axis=1)
    removed_by_group = removed_per_row.groupby(df["AgeGroup"]).sum().sort_values(ascending=False)

    print("\nУдалённых значений по возрастным группам:")
    print(removed_by_group)

    # === 5. Сохранение очищенного датасета ===
    out_path = os.path.join(OUT_DIR, "Data_29_10_full_zfiltered.csv")
    df = df.drop(columns=["AgeGroup"])

    df.to_csv(out_path, index=False)
    print(f"\nОчищенный файл сохранён в:\n{out_path}")

if __name__ == "__main__":
    main()
