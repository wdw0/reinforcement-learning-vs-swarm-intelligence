"""
graph.py — Statistical comparison and boxplot for all agents.

Computes Welch t-test and Mann-Whitney U for selected pairs,
prints a  p-value table, and saves the boxplot figure.

Usage:
    python graph.py

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Score vectors ──────────────────────────────────────────────────────────────

# Baselines (provided by the assignment)
rule_based = np.array([
    12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22, 9.95,
    19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 15.43, 15.13, 22.50,
    25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 13.80, 15.13, 12.35, 16.19,
])

human = np.array([
    27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21, 18.92, 25.71, 20.05,
    31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 28.45, 12.59, 33.01,
    21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 22.96, 9.41, 35.22,
])

neural_ag = np.array([
    38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30, 39.76, 48.17,
    44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 67.17, 53.54, 33.59, 49.24,
    52.65, 16.35, 44.05, 56.59, 63.23, 43.96, 43.82, 19.19, 28.36, 18.65,
])

# Agents implemented in this work
dqn = np.array([
    37.62, 44.52, 16.41, 121.17, 23.91, 16.83, 1.44, 26.65, 1.23, 20.25,
    19.14, 10.73, 106.68, 16.39, 18.40, 26.71, 128.10, 2.25, 19.25, 14.14,
    115.32, 21.59, 16.65, 1.64, 23.15, 27.82, 52.20, 111.18, 37.52, 74.19,
])

abc = np.array([
    83.06, 101.02, 109.47, 93.84, 77.94, 122.61, 32.60, 52.02, 37.34, 19.41,
    20.10, 58.47, 19.50, 96.62, 106.86, 105.33, 118.74, 53.81, 20.88, 49.24,
    58.83, 36.63, 20.94, 89.70, 24.08, 118.56, 98.87, 41.63, 35.02, 100.66,
])

bat = np.array([
    11.43, 85.48, 15.85, 19.14, 3.64, 37.25, 26.00, 23.01, 14.72, 20.07,
    59.37, 48.62, 36.18, 17.27, 22.61, 35.28, 21.63, 11.46, 16.21, 27.96,
    90.42, 25.20, 36.72, 4.05, 45.84, 26.62, 17.01, 27.88, 15.99, 19.05,
])

gwo = np.array([
    51.24, 17.10, 12.02, 106.50, 8.38, 16.03, 106.41, 16.69, 114.96, 17.94,
    112.53, 108.66, 29.57, 19.76, 19.10, 18.96, 108.48, 135.12, 17.10, 32.12,
    107.40, 109.11, 108.21, 108.21, 2.84, 17.13, 10.84, 15.92, 117.75, 108.12,
])

firefly = np.array([
    161.76, 89.88, 80.99, 24.78, 130.26, 45.51, 107.67, 118.92, 107.40, 110.10,
    102.73, 118.74, 17.42, 121.71, 113.07, 108.66, 89.79, 17.72, 23.19, 125.31,
    110.10, 31.98, 113.16, 117.48, 8.64, 112.35, 107.49, 58.88, 116.40, 35.20,
])

fss = np.array([
    99.49, 121.80, 101.38, 32.27, 20.65, 22.02, 95.45, 99.31, 25.68, 7.02,
    22.97, 23.68, 101.38, 101.02, 22.61, 17.49, 94.82, 24.50, 23.24, 95.27,
    15.94, 15.89, 20.70, 43.69, 16.30, 21.99, 95.81, 23.86, 20.76, 31.80,
])

q_linear = np.array([
    40.83, 15.04, 32.16, 15.81, 21.99, 21.32, 18.34, 32.87, 34.48, 26.71,
    23.59, 22.97, 21.37, 23.28, 2.05, 6.85, 28.94, 27.60, 44.77, 28.05,
    32.87, 17.50, 2.25, 27.69, 22.62, 23.86, 38.95, 17.98, 17.18, 28.76,
])

q_tile = np.array([
    19.14, 17.36, 30.37, 13.65, 32.78, 15.76, 21.37, 15.58, 22.17, 3.48,
    23.68, 23.68, 27.52, 18.52, 22.26, 23.24, 27.16, 19.14, 18.87, 25.38,
    25.20, 13.65, 23.86, 26.71, 19.14, 18.52, 19.63, 31.62, 16.65, 18.16,
])

# ── Agent registry ─────────────────────────────────────────────────────────────
# Each entry: (label_for_plot, array, is_baseline)
agents = [
    ("Baseado em regras",    rule_based,   True),
    ("Humano",               human,        True),
    ("Rede neural (AG)",     neural_ag,    True),
    ("Bat Algorithm",        bat,          False),
    ("DQN",                  dqn,          False),
    ("Fish School Search",   fss,          False),
    ("Grey Wolf Optimizer",  gwo,          False),
    ("ABC / iABC",           abc,          False),
    ("Firefly Algorithm",    firefly,      False),
    ("Q-Learning linear",  q_linear,     False),
    ("Q-Learning tile",    q_tile,       False),
]

# ── Descriptive statistics ─────────────────────────────────────────────────────
print("=" * 70)
print(f"{'Agente':<25} {'Média':>8} {'DP':>8} {'Mediana':>8} {'Máx':>8} {'CV':>6}")
print("=" * 70)
for label, arr, _ in sorted(agents, key=lambda x: np.mean(x[1])):
    m, s, med, mx = np.mean(arr), np.std(arr, ddof=1), np.median(arr), np.max(arr)
    print(f"{label:<25} {m:>8.2f} {s:>8.2f} {med:>8.2f} {mx:>8.2f} {s/m:>6.2f}")

# ── Statistical tests ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TESTES DE HIPÓTESE  (Welch t-test + Mann-Whitney U, bilateral, α=0.05)")
print("=" * 70)

# Pairs of interest: each implemented agent vs. key references + DQN
implemented = [(l, a) for l, a, b in agents if not b]
references = {
    "Regras":       rule_based,
    "Humano":       human,
    "Neural (AG)":  neural_ag,
}

header = f"{'Par':<38} {'Welch p':>10} {'MW-U p':>10} {'Sig.':>5}"
print(header)
print("-" * len(header))

rows = []
for lbl, arr in implemented:
    for ref_name, ref_arr in references.items():
        _, p_welch = stats.ttest_ind(arr, ref_arr, equal_var=False)
        _, p_mwu   = stats.mannwhitneyu(arr, ref_arr, alternative="two-sided")
        sig = "**" if p_welch < 0.05 and p_mwu < 0.05 else ("*" if p_welch < 0.05 or p_mwu < 0.05 else "")
        pair = f"{lbl} vs {ref_name}"
        print(f"{pair:<38} {p_welch:>10.4f} {p_mwu:>10.4f} {sig:>5}")
        rows.append((pair, p_welch, p_mwu))

# Cross-comparisons among implemented agents
print()
print("-- Entre agentes implementados --")
dqn_row = ("DQN", dqn)
for lbl, arr in implemented:
    if lbl == "DQN":
        continue
    _, p_welch = stats.ttest_ind(arr, dqn, equal_var=False)
    _, p_mwu   = stats.mannwhitneyu(arr, dqn, alternative="two-sided")
    sig = "**" if p_welch < 0.05 and p_mwu < 0.05 else ("*" if p_welch < 0.05 or p_mwu < 0.05 else "")
    pair = f"{lbl} vs DQN"
    print(f"{pair:<38} {p_welch:>10.4f} {p_mwu:>10.4f} {sig:>5}")

# ── LaTeX table snippet ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SNIPPET LaTeX (tab-estatistica) — copiar para ch4-avaliacao.tex")
print("=" * 70)
print(r"""
\begin{table}[htb]
\centering
\caption{Valores-$p$ dos testes de Welch e Mann-Whitney U para pares
         selecionados. Negrito: ambos os testes com $p < 0{,}05$.}
\label{tab-estatistica}
\begin{tabular}{lcc}
\toprule
Par comparado & Welch $p$ & Mann-Whitney $p$ \\
\midrule""")

ref_short = {"Baseado em regras": "Regras", "Humano": "Humano",
             "Rede neural (AG)": "Neural (AG)", "DQN": "DQN"}

all_pairs = []
for lbl, arr in implemented:
    for ref_name, ref_arr in {**references, "DQN": dqn}.items():
        if lbl == "DQN" and ref_name == "DQN":
            continue
        _, p_welch = stats.ttest_ind(arr, ref_arr, equal_var=False)
        _, p_mwu   = stats.mannwhitneyu(arr, ref_arr, alternative="two-sided")
        all_pairs.append((lbl, ref_name, p_welch, p_mwu))

for lbl, ref_name, p_w, p_m in all_pairs:
    bold_open  = r"\textbf{" if (p_w < 0.05 and p_m < 0.05) else ""
    bold_close = "}"         if (p_w < 0.05 and p_m < 0.05) else ""
    pw_str = f"{bold_open}{p_w:.4f}{bold_close}"
    pm_str = f"{bold_open}{p_m:.4f}{bold_close}"
    print(f"{lbl} vs {ref_name} & {pw_str} & {pm_str} \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}""")

# ── Boxplot ────────────────────────────────────────────────────────────────────
# Order by mean (ascending)
plot_agents = sorted(agents, key=lambda x: np.mean(x[1]))
labels = [l for l, _, _ in plot_agents]
data   = [a for _, a, _ in plot_agents]
colors = ["#d9e8f5" if b else "#fde9cc" for _, _, b in plot_agents]

fig, ax = plt.subplots(figsize=(14, 6))

bp = ax.boxplot(data, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=1.0),
                capprops=dict(linewidth=1.0),
                flierprops=dict(marker="o", markersize=3, alpha=0.5))

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Pontuação", fontsize=10)
ax.set_title("Distribuição das pontuações nos 30 episódios de avaliação\n"
             "(azul = baseline, laranja = agentes implementados)", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#d9e8f5", label="Baseline"),
    Patch(facecolor="#fde9cc", label="Implementado neste trabalho"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

plt.tight_layout()
outfile = "boxplot_comparativo.png"
plt.savefig(outfile, dpi=150)
print(f"\nBoxplot salvo em: {outfile}")
