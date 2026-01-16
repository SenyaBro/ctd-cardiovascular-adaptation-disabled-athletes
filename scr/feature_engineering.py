import pandas as pd
import numpy as np
import os

# --- 1. Определение путей ---
load_path = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\data\clean\clean_data_29_10.csv"
save_path = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\data\clean\Data_29_10_full.csv"

try:
    # --- 2. Загрузка данных ---
    print(f"--- Загрузка файла: {load_path} ---")
    
    #  Явно указываем разделитель
    df = pd.read_csv(load_path, sep=',')
    
    # === Сначала вычисляем значения ===
    
    vo2_per_kg = None
    imt = None
    

    # b) Расчет ИМТ (Индекс Массы Тела)
    if 'Height' in df.columns and 'Weight' in df.columns:
        height_in_meters = df['Height'] / 100
        imt = df['Weight'] / (height_in_meters ** 2)
    else:
        print("- ВНИМАНИЕ: 'Height' или 'Weight' не найдены.")

    # === Теперь вставляем их на нужные позиции ===
    

        
    if imt is not None:
        df.insert(7, 'IMT', imt)
        print("- Столбец 'IMT' успешно вставлен на 8-ю позицию (индекс 7).")


    # --- 3. Сохранение итогового файла ---
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df.to_csv(save_path, sep=',', decimal='.', index=False)
    
    print("\n" + "="*40)
    print(f"🎉 Успешно сохранен новый файл: {save_path}")
    print(f"Новые столбцы вставлены на 5-ю и 8-ю позиции.")


except FileNotFoundError:
    print(f"ОШИБКА: Не удалось найти исходный файл: {load_path}")
    print("Убедитесь, что вы сначала запустили 'data_preprocessing.py'")
except KeyError as e:
    print(f"ОШИБКА: Не найден столбец {e}, необходимый для расчетов.")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")
    