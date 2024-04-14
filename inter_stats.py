import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from binary_stats import apriori_exclusions


def flatten_list(df, dfs_ours):
    out_dfs = {}
    for patient in dfs_ours:
        df_patient = df[df['ESO.ID'] == patient.replace('-', '')]
        df_out = copy.deepcopy(dfs_ours[patient])
        for i, row in df_patient.iterrows():
            fdi = row['Tooth']
            interval = '-'.join(str(row['Interval']).split('.'))
            loss = row['max. height loss (mm) Geomagic original']
            if df_out.loc[df_out['FDI'] == fdi, interval].isna().any():
                continue
            df_out.loc[df_out['FDI'] == fdi, interval] = -loss

        out_dfs[patient] = df_out

    return out_dfs



if __name__ == '__main__':
    root = Path('methodology study')

    df_exclusions = pd.read_excel(root / 'overview.xlsx')
    # dfs_3dwa = extract_heights(df_exclusions)


    dfs_ours = pd.read_excel(root / 'AI pairs' / 'ours_max_heights.xlsx', sheet_name=None)
    dfs_ours = apriori_exclusions(df_exclusions, dfs_ours)
    
    dfs_3dwa = pd.read_excel(root / '3DWA pairs' / '3dwa_max_heights.xlsx', sheet_name=None)
    # dfs_3dwa = flatten_list(df_exclusions, dfs_ours)
    dfs_3dwa = apriori_exclusions(df_exclusions, dfs_3dwa)

    gts, preds = [], []
    for patient in dfs_3dwa:
        gt_heights = dfs_3dwa[patient].iloc[:, 1:].to_numpy()
        pred_heights = dfs_ours[patient].iloc[:, 1:].to_numpy()

        gts.extend(gt_heights[~np.isnan(gt_heights)])
        preds.extend(pred_heights[~np.isnan(pred_heights)])

    gts = -np.array(gts)
    preds = -np.array(preds)

    print(scipy.stats.kstest(gts, preds))
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
    axs[1].set_ylabel('Difference in profile loss (3DWA - Automated)')
    axs[1].text(.01, 0.39, 'n=516')

    slope, intercept, r_value, _, _ = scipy.stats.linregress(preds, gts)
    axs[0].scatter(preds, gts, alpha=0.8)
    axs[0].plot([0, 1.2], [intercept, intercept + slope * 1.2], color='k', linewidth=2, label=f'Pearson\'s $r$ = {r_value:.3f}')
    axs[0].set_xlabel('Automated profile loss (mm)')
    axs[0].set_ylabel('3DWA profile loss (mm)')
    axs[0].legend()

    plt.savefig('stats_inter.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()