import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

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

    gts, preds = [{k: [] for k in ['0-1', '0-3', '0-5']} for _ in range(2)]
    for patient in dfs_3dwa:
        for k in gts:
            gt_heights = dfs_3dwa[patient][k].to_numpy()
            pred_heights = dfs_ours[patient][k].to_numpy()

            gts[k].extend(gt_heights[~np.isnan(gt_heights)])
            preds[k].extend(pred_heights[~np.isnan(pred_heights)])

    gts = {k: -np.array(gt) for k, gt in gts.items()}
    preds = {k: -np.array(pred) for k, pred in preds.items()}

    gts_df = pd.DataFrame()
    gts_df['3DWA profile loss (mm)'] = np.concatenate(list(gts.values()))
    gts_df['Time interval'] = [f'{inter} years' for k, v in gts.items() for inter in [k]*v.shape[0]]

    preds_df = pd.DataFrame()
    preds_df['Automated profile loss (mm)'] = np.concatenate(list(preds.values()))
    preds_df['Time interval'] = [f'{inter} years' for k, v in preds.items() for inter in [k]*v.shape[0]]

    g = sns.catplot(
        data=gts_df, x='Time interval', y='3DWA profile loss (mm)', hue='Time interval',
        kind='violin', aspect=6/5,
    )
    g.set(ylim=(-0.1, 1.4))
    plt.savefig('stats_time_3dwa.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()

    g = sns.catplot(
        data=preds_df, x='Time interval', y='Automated profile loss (mm)', hue='Time interval',
        kind='violin', aspect=6/5,
    )
    g.set(ylim=(-0.1, 1.4))
    plt.savefig('stats_time_auto.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()