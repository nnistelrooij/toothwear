from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


if __name__ == '__main__':
    root = Path('methodology study')

    dfs_ours = pd.read_excel(root / 'ours_heights_intra.xlsx', sheet_name=None)

    values = np.concatenate([df.iloc[:, 1:].to_numpy() for df in dfs_ours.values()])
    values = values[~np.any(np.isnan(values), axis=1)]

    gts = -values[:, 1]
    preds = -values[:, 0]


    print(scipy.stats.shapiro(gts))
    print(scipy.stats.shapiro(preds))
    print(scipy.stats.kstest(gts, preds))
    print(np.quantile(gts - preds, 0.025))
    print(np.quantile(gts - preds, 0.975))
    print(np.mean(gts - preds))
    print(np.std(gts - preds) / np.sqrt(2))
    print(scipy.stats.wilcoxon(gts, preds))

    _, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[1].scatter(np.mean([gts, preds], axis=0), gts - preds, alpha=0.8)
    axs[1].axhline(0, c='k')
    axs[1].axhline(np.quantile(gts - preds, 0.025), c='k', linestyle='--')
    axs[1].axhline(np.quantile(gts - preds, 0.975), c='k', linestyle='--')
    axs[1].set_xlabel('Mean profile loss (mm)')
    axs[1].set_ylabel('Difference in profile loss (Automated 1 - Automated 2)')
    axs[1].text(.13, 0.17, f'n={gts.shape[0]}')

    slope, intercept, r_value, _, _ = scipy.stats.linregress(preds, gts)
    axs[0].scatter(preds, gts, alpha=0.8)
    axs[0].plot([0, 1.3], [intercept, intercept + slope * 1.3], color='k', linewidth=2, label=f'Pearson\'s $r$ = {r_value:.3f}')
    axs[0].set_xlabel('Automated profile loss 1 (mm)')
    axs[0].set_ylabel('Automated profile loss 2 (mm)')
    axs[0].legend()

    plt.savefig('figures/stats_intra.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()

    k = 3
