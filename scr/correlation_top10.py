import pandas as pd
from scipy import stats
import os

# --- 1. Настройка путей ---
DATA_PATH = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\data\clean\Data_29_10_new_full_clean.csv"
OUTPUT_DIR = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\outputs\correlations"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "top10_vo2max_correlations.xlsx")
TARGET_VARIABLE = "VO2max_per_kg"

def calculate_correlations(df, target_col):
    """
    Рассчитывает корреляции Пирсона, Спирмена и N (количество)
    для целевой переменной со всеми числовыми столбцами.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    results = []

    if target_col not in numeric_cols:
        print(f"Ошибка: Целевая переменная '{target_col}' не найдена или не является числовой.")
        return None

    for col in numeric_cols:
        if col == target_col:
            continue
        
        # 2. Попарное удаление NaN
        # Это важно, чтобы 'n' было корректным для каждой пары
        temp_df = df[[target_col, col]].dropna()
        
        # 3. Расчет 'n'
        n = len(temp_df)
        
        # Для расчета корреляции нужно хотя бы 2 значения
        if n < 2:
            continue
            
        # 4. Расчет коэффициентов
        pearson_corr, _ = stats.pearsonr(temp_df[target_col], temp_df[col])
        spearman_corr, _ = stats.spearmanr(temp_df[target_col], temp_df[col])
        
        results.append({
            "Название переменной": col,
            "Коэффициент Пирсона": pearson_corr,
            "Коэффици Kорреляции Спирмена": spearman_corr, # Исправлено название
            "n": n
        })
        
    return pd.DataFrame(results)

def main():
    # --- 2. Загрузка данных ---
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Данные успешно загружены из {DATA_PATH}")
    except FileNotFoundError:
        print(f"Ошибка: Файл данных не найден по пути {DATA_PATH}")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    # --- 3. Расчет корреляций ---
    correlation_results = calculate_correlations(df, TARGET_VARIABLE)
    
    if correlation_results is None or correlation_results.empty:
        print("Не удалось рассчитать корреляции.")
        return

    # --- 4. Поиск Топ-10 ---
    # Создаем столбец с абсолютным значением для сортировки
    correlation_results["Abs_Pearson"] = correlation_results["Коэффициент Пирсона"].abs()
    
    # Сортируем и берем топ-10
    top_10_df = correlation_results.sort_values(by="Abs_Pearson", ascending=False).head(10)
    
    # Выбираем итоговые столбцы в нужном порядке
    final_columns = ["Название переменной", "Коэффициент Пирсона", "Коэффици Kорреляции Спирмена", "n"]
    final_df = top_10_df[final_columns]

    # --- 5. Сохранение в Excel ---
    try:
        # Убедимся, что директория существует
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        final_df.to_excel(OUTPUT_FILE, index=False)
        
        print("\n--- Топ 10 корреляций ---")
        print(final_df.to_string(index=False)) # Выводим в консоль для проверки
        print(f"\nРезультат успешно сохранен в: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Ошибка при сохранении файла Excel: {e}")

# --- Запуск скрипта ---
if __name__ == "__main__":
    main()