from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import shapiro

from binary_stats import apriori_exclusions


if __name__ == '__main__':
    root = Path('methodology study')

    df_exclusions = pd.read_excel(root / 'Hilde_overview.xlsx')
    df_exclusions['max. height loss (mm) Geomagic original'] = df_exclusions['Difference.Height.mm']
    df_exclusions['Tooth'] = df_exclusions['Element']
    # dfs_3dwa = extract_heights(df_exclusions)


    dfs_3dwa = pd.read_excel(root / '3DWA pairs' / '3dwa_heights_double.xlsx', sheet_name=None)

    dfs_3dwa = apriori_exclusions(df_exclusions, dfs_3dwa)

    labels1, labels2 = [], []
    for patient, df in dfs_3dwa.items():
        df = df[~pd.isna(df).any(axis=1)]
        labels1.extend(df['0-5'] if '0-5' in df else df['0-6'])
        labels2.extend(df['0-5_operator2' if '0-5_operator2' in df else '0-6_operator2'])

    gts = -np.array(labels1)
    preds = -np.array(labels2)

    print(shapiro(gts).pvalue)
    print(shapiro(preds).pvalue)

    print(np.quantile(gts - preds, 0.025))
    print(np.quantile(gts - preds, 0.975))
    print(np.mean(gts - preds))
    print(scipy.stats.wilcoxon(gts, preds))

    _, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[1].scatter(np.mean([gts, preds], axis=0), gts - preds, alpha=0.8)
    axs[1].axhline(0, c='k')
    axs[1].axhline(np.quantile(gts - preds, 0.025), c='k', linestyle='--')
    axs[1].axhline(np.quantile(gts - preds, 0.975), c='k', linestyle='--')
    axs[1].set_xlabel('Mean profile loss (mm)')
    axs[1].set_ylabel('Difference in profile loss (operator 1 - operator 2)')
    axs[1].set_xlim(0, axs[1].get_xlim()[1])
    axs[1].text(.01, axs[1].get_ylim()[1] - 0.015, 'n=75')

    slope, intercept, r_value, _, _ = scipy.stats.linregress(gts, preds)
    axs[0].scatter(gts, preds, alpha=0.8)
    axs[0].plot([0, 1], [intercept, intercept + slope], color='k', linewidth=2, label=f'Pearson\'s $r$ = {r_value:.3f}')
    axs[0].set_xlabel('Operator 1 profile loss (mm)')
    axs[0].set_ylabel('Operator 2 profile loss (mm)')
    axs[0].legend()

    plt.savefig('stats_double.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()



