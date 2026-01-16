# -*- coding: utf-8 -*-
import os
import re
import pandas as pd

# === ПУТИ ===
INPUT_XLSX  = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\data\clean\clean_main_data.xlsx"
OUTPUT_DIR = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\outputs"
XLSX_RAW   = os.path.join(OUTPUT_DIR, "selected_metrics2.xlsx")
XLSX_NUM   = os.path.join(OUTPUT_DIR, "selected_metrics_numeric2.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === СПИСОК ТРЕБУЕМЫХ ПЕРЕМЕННЫХ (в желаемом порядке) ===
requested = [
    "HR Resting", "HR max", "HR 3min.1",
    "Lying_HST",
    "HR1min_HST_Recovery", "HR2min_HST_Recovery", "HR3min_HST_Recovery",
    "HSTI",
    "RR", "PQ", "QTc", "QRS",
    "VC_L", "VC Percent",
    "FVC L",
    "FEV1 L", "FEV1 L.1", "FEV1 Percent",
    "Gensler Index",
    "Tiffeneau Index Ratio", "Tiffeneau Index Percent",
    "PEF Ls", "PEF Percent",
    "HR Lying", "HR Standing", "HR 2 Standing",
    "ΔHR1 (Lying-Standing)", "HR1 (Lying-Lying)",
    "SBP Lying", "DBP Lying", "SBP Standing", "DBP Standing",
    "SV Lying", "SV Standing",
    "SI Lying", "SI Standing",
    "EDV Lying", "EDV Standing",
    "EDI Lying", "EDI Standing",
    "EF Lying", "EF Standing",
    "Respiratory Rate",
    "Blood Volume",
    "Inotropy",
    "Vascular_Tone",
    "CO Lying", "CO Standing",
    "CI Lying", "CI Standing",
    "TPR Lying",
    "Chronotropy",
]

# === ХЕЛПЕРЫ ДЛЯ МЭППИНГА КОЛОНОК ===
def norm(s: str) -> str:
    """Упрощённая нормализация имени: нижний регистр, убрать пробелы/подчёркивания/точки/кавычки."""
    s = str(s).replace("\u00A0", " ").strip().lower()
    s = re.sub(r"[\"'%]+", "", s)
    s = re.sub(r"[\s_\.\/\\-]+", "", s)
    return s

def normalize_requested(name: str) -> str:
    return norm(name)

# === ЗАГРУЗКА ДАННЫХ ===
df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

# мягкая нормализация заголовков
df.columns = (
    pd.Index(df.columns)
      .str.replace("\u00A0", " ", regex=False)
      .str.replace(r'^[\'"]+|[\'"]+$', "", regex=True)
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
)

# карта "нормализованное имя -> реальное имя"
norm_to_real = {}
for c in df.columns:
    key = norm(c)
    norm_to_real.setdefault(key, c)

# подбираем реальные имена из df под запрошенные
found_cols, missing = [], []
for req in requested:
    key = normalize_requested(req)
    real = norm_to_real.get(key)
    if real is None:
        # fallback: поиск по вхождению нормализованной подстроки
        candidates = [real_name for nk, real_name in norm_to_real.items() if key == nk or key in nk]
        real = candidates[0] if candidates else None
    if real is None:
        missing.append(req)
    elif real not in found_cols:
        found_cols.append(real)

if missing:
    print("⚠️ Не найдены в данных (проверь названия/источник):")
    for m in missing:
        print("  -", m)
if not found_cols:
    raise SystemExit("Не удалось подобрать ни одной колонки. Останавливаюсь.")

# === ВЫБОРКА НУЖНЫХ КОЛОНОК ===
df_sel = df.loc[:, found_cols].copy()

# === КОРРЕКЦИЯ PQ/QRS: умножить на 1000, если значение < 1 ===
for col in ["PQ", "QRS"]:
    if col in df_sel.columns:
        df_sel[col] = pd.to_numeric(df_sel[col], errors="coerce")
        mask = df_sel[col] < 1
        n_changed = int(mask.sum())
        if n_changed > 0:
            df_sel.loc[mask, col] = df_sel.loc[mask, col] * 1000
            print(f"✅ Умножено на 1000 в '{col}': {n_changed} значений (<1)")

# === СОХРАНЕНИЕ: «как есть» ===
with pd.ExcelWriter(XLSX_RAW, engine="openpyxl") as xw:
    df_sel.to_excel(xw, sheet_name="data", index=False)
print(f"✅ Сохранил: {XLSX_RAW}")

# === ЧИСЛОВАЯ ВЕРСИЯ + СТАТИСТИКА ===
df_num = df_sel.apply(pd.to_numeric, errors="coerce")

# список колонок для статистики — берём только реально найденные из requested
req_norm_set = {norm(r) for r in requested}
stats_cols = [c for c in df_num.columns if norm(c) in req_norm_set]

# считаем min, max, mean, std
stats_df = pd.DataFrame({
    "min":  df_num[stats_cols].min(skipna=True),
    "max":  df_num[stats_cols].max(skipna=True),
    "mean": df_num[stats_cols].mean(skipna=True),
    "std":  df_num[stats_cols].std(skipna=True, ddof=1),
}).rename_axis("variable").reset_index()

# доля пропусков для справки
na_share = (df_num.isna().sum() / len(df_num)).rename("Na_share").to_frame()

with pd.ExcelWriter(XLSX_NUM, engine="openpyxl") as xw:
    df_num.to_excel(xw, sheet_name="numeric", index=False)
    na_share.to_excel(xw, sheet_name="na_share")
    stats_df.to_excel(xw, sheet_name="stats", index=False)

print(f"✅ Сохранил: {XLSX_NUM}")
print("\n— Итог —")
print("Найдено колонок:", len(found_cols))
print("Выгружено строк:", len(df_sel))
if missing:
    print("Отсутствуют:", ", ".join(missing))
