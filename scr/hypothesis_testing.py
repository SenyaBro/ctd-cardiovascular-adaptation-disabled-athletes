import os
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# === PATHS ===
DATA_PATH = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\data\clean\Data_29_10_full_zfiltered.csv"
OUT_DIR   = r"C:\Users\Ars\projects\university\Data_Lab_Urfu_2025\outputs"

# === SETTINGS ===
ALPHA = 0.05
GROUP_COLS = ["Sport", "Nosology"]
METRICS    = ["VO2max", "QTc"]   # target variables

# === FONT SETTINGS ===
AXIS_LABEL_FONTSIZE = 14   # подписи осей
TICK_LABEL_FONTSIZE = 12   # подписи делений (числа на осях)

# Axis labels
GROUP_LABELS = {
    "Sport": "Sports discipline",
    "Nosology": "Nosology",
}

METRIC_LABELS = {
    "VO2max": "VO₂max (ml/kg/min)",
    "QTc": "QTc (ms)",
}

# Effect size names (for printing / Excel)
EFFECT_TYPE_NAME = {
    "eta_squared": "η² (eta squared)",
    "epsilon_squared": "ε² (epsilon squared)",
}

# === Nosology mapping ===
# ВАЖНО: отредактируй под свои реальные коды нозологий.
# Если код не найден в словаре, на графике останется исходное значение.
NOSOLOGY_MAP = {
    1: "HI",
    2: "IDD",
    3: "VI",
    4: "MPI",
}

# === Sport mapping (codes → 2–3 letter abbreviations) ===
# 1 – para-swimming
# 2 – cross-country skiing
# 3 – para-equestrian
# 4 – para-hockey
# 5 – para-football
# 7 – wheel-chair dancing
SPORT_MAP = {
    1: "PS",   # para-swimming
    2: "CCS",   # cross-country skiing
    3: "PEQ",   # para-equestrian
    4: "PH",   # para-hockey
    5: "PFB",   # para-football
    7: "WCD",   # wheel-chair dancing
}


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
#  Assumption checks and group tests
# ----------------------------------------------------------------------
def check_assumptions(series_by_group: Dict[str, np.ndarray]) -> Tuple[bool, bool]:
    """
    normal: Shapiro–Wilk in each group (n>=3)
    equal_var: Levene test across groups.
    Returns (normal, equal_var).
    """
    normal = True

    for g, arr in series_by_group.items():
        arr = arr[~np.isnan(arr)]
        if len(arr) >= 3:
            stat, p = stats.shapiro(arr)
            if p < ALPHA:
                normal = False

    # Levene test (>=2 groups with data)
    valid_groups = [v[~np.isnan(v)] for v in series_by_group.values()
                    if np.isfinite(v).sum() > 1]
    equal_var = True
    if len(valid_groups) >= 2:
        _, p_lev = stats.levene(*valid_groups, center="median")
        equal_var = (p_lev >= ALPHA)

    return normal, equal_var


def run_oneway_test(series_by_group: Dict[str, np.ndarray]) -> Tuple[str, float, float, Dict]:
    """
    Choose One-way ANOVA or Kruskal–Wallis depending on assumptions.
    Returns:
      test_name, p_value, statistic, extra_info_dict

    extra_info:
      - k (number of groups)
      - n_total
      - assumptions: {normal, equal_var}
      - effect_size (eta² or epsilon²)
      - effect_type: 'eta_squared' or 'epsilon_squared'
    """
    normal, equal_var = check_assumptions(series_by_group)
    groups = [v[~np.isnan(v)] for v in series_by_group.values()]
    labels = list(series_by_group.keys())

    # filter out tiny groups
    good = []
    good_labels = []
    for lab, arr in zip(labels, groups):
        if len(arr) >= 2:
            good.append(arr)
            good_labels.append(lab)

    groups = good
    labels = good_labels

    if len(groups) < 2:
        raise ValueError("Not enough groups with data for comparison.")

    n_total = int(sum(len(g) for g in groups))
    k = len(groups)

    extra = {
        "assumptions": {"normal": normal, "equal_var": equal_var},
        "k": k,
        "n_total": n_total,
    }

    if normal and equal_var:
        # --- One-way ANOVA ---
        test_name = "One-way ANOVA"
        stat, p = stats.f_oneway(*groups)

        # eta² via F, k, N:
        # eta² = (F * (k - 1)) / (F * (k - 1) + (N - k))
        eta_sq = (stat * (k - 1)) / (stat * (k - 1) + (n_total - k)) if n_total > k else np.nan

        extra["effect_size"] = float(eta_sq)
        extra["effect_type"] = "eta_squared"

    else:
        # --- Kruskal–Wallis ---
        test_name = "Kruskal–Wallis test"
        stat, p = stats.kruskal(*groups)

        # ε² = (H - k + 1) / (n - k)
        if n_total > k:
            eps_sq = (stat - k + 1) / (n_total - k)
            eps_sq = max(0.0, float(eps_sq))  # small negatives due to rounding
        else:
            eps_sq = np.nan

        extra["effect_size"] = float(eps_sq)
        extra["effect_type"] = "epsilon_squared"

    extra["statistic"] = float(stat)
    return test_name, float(p), float(stat), extra


# ----------------------------------------------------------------------
#  KMO & Bartlett
# ----------------------------------------------------------------------
def compute_kmo_bartlett(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> Optional[Tuple[float, np.ndarray, float, float, int, int, int, List[str]]]:
    """
    Compute overall KMO, per-variable KMO, Bartlett's chi² and p-value.

    Returns:
      (kmo_overall, kmo_per_variable, chi2, p_value, df_bartlett,
       n_obs, n_variables, used_columns)
    or None if computation is impossible.
    """
    if exclude_cols is None:
        exclude_cols = []

    num = df.select_dtypes(include=[np.number]).copy()

    # exclude grouping etc. from factor analysis
    for col in exclude_cols:
        if col in num.columns:
            num = num.drop(columns=col)

    # drop constant or almost empty columns
    drop_cols = []
    for c in num.columns:
        if num[c].count() < 5 or num[c].std(skipna=True) == 0:
            drop_cols.append(c)
    if drop_cols:
        num = num.drop(columns=drop_cols)

    num = num.dropna()
    n_obs, p = num.shape
    if p < 2 or n_obs < 10:
        warnings.warn("Not enough data for KMO/Bartlett.")
        return None

    cols_used = list(num.columns)

    # correlation matrix
    corr = np.corrcoef(num.values, rowvar=False)

    # inverse (use pseudo-inverse for numerical stability)
    corr_inv = np.linalg.pinv(corr)

    # partial correlation matrix
    d = np.sqrt(np.diag(corr_inv))
    denom = np.outer(d, d)
    partial = -corr_inv / denom
    np.fill_diagonal(partial, 0.0)

    corr_sq = corr ** 2
    partial_sq = partial ** 2
    np.fill_diagonal(corr_sq, 0.0)
    np.fill_diagonal(partial_sq, 0.0)

    # overall KMO
    kmo_num = np.sum(corr_sq)
    kmo_den = kmo_num + np.sum(partial_sq)
    kmo_overall = kmo_num / kmo_den

    # per-variable KMO
    kmo_per_var = np.sum(corr_sq, axis=0) / (np.sum(corr_sq, axis=0) +
                                             np.sum(partial_sq, axis=0))

    # Bartlett's test of sphericity
    det_corr = np.linalg.det(corr)
    if det_corr <= 0:
        warnings.warn("Correlation matrix determinant <= 0, Bartlett test not valid.")
        chi2 = np.nan
        p_value = np.nan
    else:
        chi2 = -(n_obs - 1 - (2 * p + 5) / 6) * np.log(det_corr)
        df_bartlett = int(p * (p - 1) / 2)
        p_value = 1 - stats.chi2.cdf(chi2, df_bartlett)

    df_bartlett = int(p * (p - 1) / 2)

    return float(kmo_overall), kmo_per_var, float(chi2), float(p_value), df_bartlett, n_obs, p, cols_used


# ----------------------------------------------------------------------
#  Top correlations for VO2max and QTc
# ----------------------------------------------------------------------
def top_correlations(df: pd.DataFrame, target: str, top_n: int = 10) -> pd.DataFrame:
    """
    Spearman correlations between target and all other numeric variables.
    Returns top_n by |rho|.
    """
    if target not in df.columns:
        return pd.DataFrame()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in num_cols:
        return pd.DataFrame()

    num_cols.remove(target)
    rows = []

    for col in num_cols:
        sub = df[[target, col]].dropna()
        if len(sub) < 5:
            continue
        rho, p = stats.spearmanr(sub[target], sub[col])
        if np.isnan(rho):
            continue
        rows.append({
            "target": target,
            "variable": col,
            "rho": float(rho),
            "p_value": float(p),
            "n": int(len(sub)),
        })

    if not rows:
        return pd.DataFrame()

    corr_df = pd.DataFrame(rows)
    corr_df["abs_rho"] = corr_df["rho"].abs()
    corr_df = corr_df.sort_values("abs_rho", ascending=False).head(top_n)
    corr_df = corr_df.drop(columns=["abs_rho"])

    return corr_df


# ----------------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------------
def make_boxplot(df: pd.DataFrame, group_col: str, value_col: str, outdir: str):
    """
    Boxplot for value_col by group_col.
    - only left and bottom axes
    - no title
    - X: 'Sports discipline' / 'Nosology'
    - Y: metric with units
    - Nosology: use English abbreviations (NOSOLOGY_MAP)
    - Sport: use abbreviations (SPORT_MAP)
    """
    sub = df[[group_col, value_col]].dropna()
    if sub[group_col].nunique() < 2:
        return

    if group_col == "Nosology":
        # map nosology codes → HI/IDD/VI/MPI
        codes = pd.to_numeric(sub[group_col], errors="coerce")
        mapped = codes.map(NOSOLOGY_MAP)
        sub = sub.assign(_group=mapped.fillna(sub[group_col].astype(str)))

    elif group_col == "Sport":
        # map sport codes → PS/XC/EQ/HK/FB/WD
        codes = pd.to_numeric(sub[group_col], errors="coerce")
        mapped = codes.map(SPORT_MAP)
        sub = sub.assign(_group=mapped.fillna(sub[group_col].astype(str)))

    else:
        sub = sub.assign(_group=sub[group_col].astype(str))

    data = [
        sub.loc[sub["_group"] == g, value_col].values
        for g in sorted(sub["_group"].unique())
    ]
    labels = sorted(sub["_group"].unique())

    fig, ax = plt.subplots()
    ax.boxplot(data, labels=labels, showmeans=True)

    # only left & bottom axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(GROUP_LABELS.get(group_col, group_col), fontsize=AXIS_LABEL_FONTSIZE)
    ylabel = METRIC_LABELS.get(value_col, value_col)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)

    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    fname = os.path.join(outdir, f"boxplot_{group_col}_{value_col}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)

# ----------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------
def main():
    ensure_outdir(OUT_DIR)

    print(f"Loading data from:\n{DATA_PATH}\n")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    results_rows: List[dict] = []

    # 1) Group comparisons (Sport / Nosology × VO2max / QTc)
    for group_col in GROUP_COLS:
        if group_col not in df.columns:
            warnings.warn(f"Column '{group_col}' is missing — skipping.")
            continue

        for value_col in METRICS:
            if value_col not in df.columns:
                warnings.warn(f"Column '{value_col}' is missing — skipping.")
                continue

            print(f"\n=== Analysis: {GROUP_LABELS.get(group_col, group_col)} → "
                  f"{METRIC_LABELS.get(value_col, value_col)} ===")

            sub = df[[group_col, value_col]].dropna()
            if sub[group_col].nunique() < 2:
                print("Less than 2 groups with data — test not performed.")
                continue

            # for tests use original group codes as strings
            groups_labels = sub[group_col].astype("Int64", errors="ignore").astype(str)

            series_by_group: Dict[str, np.ndarray] = {}
            for g in sorted(groups_labels.unique()):
                mask = (groups_labels == g)
                arr = sub.loc[mask, value_col].to_numpy(dtype=float)
                series_by_group[str(g)] = arr

            non_empty_groups = [v for v in series_by_group.values() if len(v) >= 2]
            if len(non_empty_groups) < 2:
                print("Not enough groups with ≥2 observations — skipping.")
                continue

            try:
                test_name, p_value, stat, extra = run_oneway_test(series_by_group)
            except ValueError as e:
                print(f"Test not performed: {e}")
                continue

            row = {
                "grouping": group_col,
                "metric": value_col,
                "test": test_name,
                "p_value": p_value,
                "statistic": stat,
                "k_groups": extra.get("k"),
                "n_total": extra.get("n_total"),
                "assumption_normal": extra["assumptions"]["normal"],
                "assumption_equal_var": extra["assumptions"]["equal_var"],
                "effect_type": extra.get("effect_type"),
                "effect_size": extra.get("effect_size"),
            }
            results_rows.append(row)

            eff_name = EFFECT_TYPE_NAME.get(row["effect_type"], row["effect_type"])
            print(f"Test: {test_name}")
            print(f"p-value: {p_value:.5f}")
            print(f"Statistic: {stat:.4f}")
            print(f"Effect size ({eff_name}): {row['effect_size']:.4f}")

            # boxplot with updated labels
            make_boxplot(
                sub.assign(**{group_col: groups_labels}),
                group_col=group_col,
                value_col=value_col,
                outdir=OUT_DIR,
            )

    res_df = pd.DataFrame(results_rows) if results_rows else pd.DataFrame()

    # 2) Effect size barplot
    if not res_df.empty:
        eff = res_df.dropna(subset=["effect_size"]).reset_index(drop=True)
    if not eff.empty:
        fig, ax = plt.subplots()

        indices = np.arange(1, len(eff) + 1)
        ax.bar(indices, eff["effect_size"].values)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(indices)
        ax.set_xticklabels(indices)

        # axis labels – bold & larger
        ax.set_xlabel(
            "Effect index",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        ax.set_ylabel(
            "Effect size (η² / ε², dimensionless)",
            fontsize=AXIS_LABEL_FONTSIZE,
        )


        mapping_lines = []
        for i, row in eff.iterrows():
            idx = i + 1
            group_label = GROUP_LABELS.get(row["grouping"], row["grouping"])
            metric_label = METRIC_LABELS.get(row["metric"], row["metric"])
            effect_symbol = "η²" if row["effect_type"] == "eta_squared" else "ε²"
            mapping_lines.append(
                f"{idx}: {group_label} → {metric_label}, {effect_symbol}"
            )

        fig.subplots_adjust(bottom=0.3)
        fig.text(
            0.01,
            0.02,
            "\n".join(mapping_lines),
            ha="left",
            va="bottom",
            fontsize=8,
        )

        eff_plot_path = os.path.join(OUT_DIR, "effect_sizes_barplot.png")
        fig.savefig(eff_plot_path, dpi=150)
        plt.close(fig)
        print(f"\n📊 Effect-size plot saved to:\n{eff_plot_path}")


    # 3) KMO and Bartlett
    kmo_result = compute_kmo_bartlett(df, exclude_cols=GROUP_COLS)
    if kmo_result is not None:
        (
            kmo_overall,
            kmo_per_var,
            chi2,
            p_bartlett,
            df_bartlett,
            n_obs,
            p_vars,
            cols_used,
        ) = kmo_result

        print("\nKMO and Bartlett:")
        print(f"Overall KMO: {kmo_overall:.3f}")
        print(f"Bartlett chi²({df_bartlett}) = {chi2:.2f}, p = {p_bartlett:.3e}")

        kmo_summary_df = pd.DataFrame([{
            "KMO_overall": kmo_overall,
            "Bartlett_chi2": chi2,
            "Bartlett_df": df_bartlett,
            "Bartlett_p": p_bartlett,
            "n_observations": n_obs,
            "n_variables": p_vars,
        }])

        kmo_per_var_df = pd.DataFrame({
            "variable": cols_used,
            "KMO_partial": kmo_per_var,
        })
    else:
        kmo_summary_df = pd.DataFrame()
        kmo_per_var_df = pd.DataFrame()

    # 4) Top correlations for VO2max and QTc
    top_vo2 = top_correlations(df, "VO2max", top_n=10)
    top_qtc = top_correlations(df, "QTc", top_n=10)

    if not top_vo2.empty:
        print("\nTop correlations for VO2max (Spearman):")
        print(top_vo2)
    if not top_qtc.empty:
        print("\nTop correlations for QTc (Spearman):")
        print(top_qtc)

    # 5) Save everything to a single Excel file
    excel_path = os.path.join(OUT_DIR, "hypothesis_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        if not res_df.empty:
            res_df.to_excel(writer, sheet_name="group_tests", index=False)
        if not kmo_summary_df.empty:
            kmo_summary_df.to_excel(writer, sheet_name="KMO_Bartlett_summary", index=False)
        if not kmo_per_var_df.empty:
            kmo_per_var_df.to_excel(writer, sheet_name="KMO_per_variable", index=False)
        if not top_vo2.empty:
            top_vo2.to_excel(writer, sheet_name="TopCorr_VO2max", index=False)
        if not top_qtc.empty:
            top_qtc.to_excel(writer, sheet_name="TopCorr_QTc", index=False)

    print(f"\n📄 Excel file with results saved to:\n{excel_path}")


if __name__ == "__main__":
    main()
