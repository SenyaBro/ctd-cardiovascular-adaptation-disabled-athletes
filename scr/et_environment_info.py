import sys
import os
import pandas as pd
import numpy as np
import scipy
import statsmodels
import matplotlib
import seaborn as sns

# Путь для сохранения файла
out_dir = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\outputs"
os.makedirs(out_dir, exist_ok=True)

save_path = os.path.join(out_dir, "environment_info.txt")

# Сбор версий ключевых библиотек
info = {
    "Python version": sys.version.replace("\n", " "),
    "pandas version": pd.__version__,
    "numpy version": np.__version__,
    "scipy version": scipy.__version__,
    "statsmodels version": statsmodels.__version__,
    "matplotlib version": matplotlib.__version__,
    "seaborn version": sns.__version__,
}

# Сохранение в файл
with open(save_path, "w", encoding="utf-8") as f:
    f.write("Environment information for reproducibility\n")
    f.write("==========================================\n\n")
    for lib, ver in info.items():
        f.write(f"{lib}: {ver}\n")

print(f"Файл успешно сохранён: {save_path}")
