import pandas as pd
import numpy as np
import os

# --- 1. Пути к файлам ---
# Исходный файл (сейчас это Excel)
load_path = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\data\clean\Data_29.10.xlsx"

# Куда сохраняем очищенный CSV
save_path = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\data\clean\clean_data_29_10.csv"

try:
    # --- 2. ЗАГРУЗКА ДАННЫХ В ЗАВИСИМОСТИ ОТ РАСШИРЕНИЯ ---
    ext = os.path.splitext(load_path)[1].lower()
    print(f"Загружаю файл: {load_path}")
    print(f"Обнаружено расширение: {ext}")

    if ext in [".xlsx", ".xls"]:
        # Excel-файл
        df = pd.read_excel(
            load_path,
            na_values='-'   # "-" считаем пропуском
        )
    elif ext == ".csv":
        # CSV-файл с ; как разделителем (как у тебя было изначально)
        df = pd.read_csv(
            load_path,
            sep=';',
            na_values='-'
        )
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")

    print("\n--- Файл успешно загружен ---")
    print(f"Размеры: {df.shape[0]} строк, {df.shape[1]} столбцов")

    # --- 3. УДАЛЕНИЕ ЛИШНИХ/ПУСТЫХ СТОЛБЦОВ ---

    # 3.1. Явно удаляем проблемный столбец с датой
    col_to_drop = "HRV_LV Diameter Systole cm"
    if col_to_drop in df.columns:
        df = df.drop(columns=[col_to_drop])
        print(f"\nУдалён столбец: {col_to_drop}")
    else:
        print(f"\nСтолбец '{col_to_drop}' не найден (возможно, уже удалён).")

    # 3.2. Удаляем столбцы, где 100% пропусков, кроме тех, что хотим сохранить
    keep_empty_cols = ["Max Time Arms s"]  # даже если там всё NaN — оставляем

    null_percent = df.isna().mean() * 100
    empty_cols_all = null_percent[null_percent == 100].index.tolist()
    empty_cols_to_drop = [c for c in empty_cols_all if c not in keep_empty_cols]

    if empty_cols_to_drop:
        df = df.drop(columns=empty_cols_to_drop)
        print("\nУдалены столбцы с 100% пропусков (кроме явно оставленных):")
        for c in empty_cols_to_drop:
            print("  -", c)
    else:
        print("\nНет столбцов с 100% пропусков (кроме сохранённых).")

    print(f"\nРазмеры после удаления пустых столбцов: {df.shape[0]} строк, {df.shape[1]} столбцов")

    # --- 4. ОЧИСТКА ТИПОВ ДАННЫХ ---
    print("\n--- Очистка типов данных в текстовых столбцах ---")

    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Найдено object-столбцов: {len(object_cols)}")
    if "Name" in object_cols:
        print("Столбец 'Name' исключён из преобразования типов (оставляем как текстовый).")

    for col in object_cols:
        if col == 'Name':
            continue
        # заменяем запятые на точки (если они есть в строковом представлении)
        df[col] = df[col].str.replace(',', '.', regex=False)
        # пробуем привести к числовому типу
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("--- Очистка типов завершена. ---")

    # --- 5. Информация о типах ---
    print("\n" + "="*40 + "\n")
    print("--- Итоговая информация о типах ---")
    df.info()

    # --- 6. Проверка Sport и Nosology ---
    print("\n" + "="*40 + "\n")
    print("--- Проверка типов Sport и Nosology ---")
    for col in ["Sport", "Nosology"]:
        if col in df.columns:
            print(f"Столбец '{col}': тип {df[col].dtype}, пропусков: {df[col].isna().sum()}")
        else:
            print(f"ВНИМАНИЕ: столбец '{col}' не найден в данных.")

    # --- 7. УДАЛЕНИЕ ДУБЛИКАТОВ (по всем столбцам, кроме 'Name') ---
    print("\n" + "="*40 + "\n")
    print("--- Очистка дубликатов (по всем данным, КРОМЕ 'Name') ---")

    rows_before = df.shape[0]
    all_cols = df.columns.tolist()

    if 'Name' in all_cols:
        all_cols_without_name = [c for c in all_cols if c != 'Name']
    else:
        all_cols_without_name = all_cols
        print("ВНИМАНИЕ: столбец 'Name' не найден, дубликаты ищутся по всем столбцам.")

    df.drop_duplicates(subset=all_cols_without_name, keep='first', inplace=True)
    rows_after = df.shape[0]

    print(f"Было строк:  {rows_before}")
    print(f"Стало строк: {rows_after}")
    print(f"Удалено дубликатов: {rows_before - rows_after}")

    # --- 8. СОХРАНЕНИЕ РЕЗУЛЬТАТА ---
    print("\n" + "="*40 + "\n")
    print(f"--- Сохранение очищенного файла в: {save_path} ---")

    # создаём директорию при необходимости
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # сохраняем в CSV с запятой-разделителем и точкой в качестве decimal
    df.to_csv(save_path, sep=',', decimal='.', index=False, encoding='utf-8-sig')

    print("\n🎉 Успешно сохранено! 🎉")
    print("Файл 'clean_data_29_10.csv' готов к дальнейшей работе.")

# --- Обработка ошибок ---
except pd.errors.ParserError:
    print(f"ОШИБКА: Не удалось корректно прочитать табличный файл (ParserError).")
except FileNotFoundError:
    print(f"ОШИБКА: Файл не найден по пути: {load_path}")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")
