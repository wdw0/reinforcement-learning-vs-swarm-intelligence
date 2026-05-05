import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, wilcoxon

# Seus arrays de resultados:
methods = {
    'Voo dos Morcegos'      : np.array([134.04, 108.3, 65.29, 119.1, 94.55, 17.26, 21.14, 16.12, 121.98, 112.17,
                                    24.88, 21.58, 19.41, 18.83, 111.81, 23.96, 23.68, 48.71, 104.26, 123.96,
                                    61.34, 111., 18.25, 89.97, 19.27, 80.9, 22.7, 116.31, 84.77, 221.07,]),

    'Rule-Based'            : np.array([12.47,  22.43,  15.91,  17.27,  20.65,   8.02,  26.09,  15.76,   4.84,  19.59,
                                      8.00,  30.19,   6.08,  18.79,  22.70,  16.84,  29.21,  17.98,   4.60,  16.30,
                                     21.99,   8.36,  18.52,   2.13,  16.75,  15.86,  15.43,  12.77,  20.32,  14.98]),

    'GA Neural Implementado': np.array([ 37.25, 116.67, 120.45, 111.54, 108.57, 116.04,  17.72, 105.15, 113.52, 109.02,
                                    110.64, 110.73,  19.76, 105.42, 118.2,  225.57, 110.73, 117.75, 111.54, 112.71,
                                    106.23, 109.47,  21.19, 110.01, 130.53, 112.26, 113.52, 115.95, 113.43, 132.15,]),

    'GA Neural'             : np.array([38.32,  54.53,  61.16,  27.55,  16.08,  26.00,  25.33,  18.30,  39.76,  48.17,
                                     44.77,  47.54,  75.43,  23.68,  16.83,  15.81,  67.17,  53.54,  33.59,  49.24,
                                     52.65,  16.35,  44.05,  56.59,  63.23,  43.96,  43.82,  19.19,  28.36,  18.65]),

    'Human'                 : np.array([27.34,  17.63,  39.33,  17.44,   1.16,  24.04,  29.21,  18.92,  25.71,  20.05,
                                     31.88,  15.39,  22.50,  19.27,  26.33,  23.67,  16.82,  28.45,  12.59,  33.01,
                                     21.74,  14.23,  27.90,  24.80,  11.35,  30.12,  17.08,  22.96,   9.41,  35.22])

}

# 1) Cria DataFrame com os 30 scores
df_raw = pd.DataFrame(methods)

# 2) Calcula média e desvio padrão
df_stats = df_raw.agg(['mean','std']).T.rename(columns={'mean':'Mean','std':'StdDev'})

# 3) Junta raw + stats
df_combined = pd.concat([df_raw.T, df_stats], axis=1)

# 4) Imprime como tabela Markdown (ou simplesmente print)
print(df_stats.to_markdown(floatfmt=".5f"))

# 5) Testes estatísticos
pairs = []
names = list(methods.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = names[i], names[j]
        x, y = methods[a], methods[b]
        t_stat, p_t = ttest_ind(x, y)
        try:
            w_stat, p_w = wilcoxon(x, y)
        except ValueError:
            w_stat, p_w = None, None
        pairs.append({
            'A': a, 'B': b,
            't-stat': t_stat, 'p(t)': p_t,
            'W-stat': w_stat, 'p(w)': p_w,
            'sig_t': p_t < 0.05,
            'sig_w': (p_w is not None and p_w < 0.05)
        })
df_tests = pd.DataFrame(pairs)
print(df_tests.to_markdown(floatfmt=".5f", index=False))

# 6) Boxplot
plt.figure(figsize=(10,6))
plt.boxplot([methods[n] for n in names], labels=names, showmeans=True)
plt.xticks(rotation=45)
plt.ylabel('Score')
plt.title('Comparação de Agentes')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
