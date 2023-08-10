from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def apriori_exclusions(
    df_exclusions: pd.DataFrame,
    dfs: Dict[str, pd.DataFrame],
    patients: List[str]=['A-20', 'A-24', 'A-25', 'A-27', 'A-28', 'A-29', 'A-41', 'A-46'],
):
    heights = df_exclusions['max. height loss (mm) Geomagic original']
    is_excluded = (heights < 0).to_numpy()

    patient_exclusions = {}
    for idx in np.nonzero(is_excluded)[0]:
        patient = df_exclusions['ESO.ID'][idx]
        patient = f'{patient[0]}-{patient[1:]}' if '-' not in patient else patient
        fdi = df_exclusions['Tooth'][idx]

        patient_exclusions.setdefault(patient, set([18, 28, 38, 48])).add(fdi)

    dfs = {k: v for k, v in dfs.items() if k in patients}
    for patient, df in dfs.items():
        if patient not in patient_exclusions:
            continue
        
        for exclusion in patient_exclusions[patient]:
            df[df['FDI'] == exclusion] = np.nan

    return dfs


def measurement_exclusions(
    dfs1: Dict[str, pd.DataFrame],
    dfs2: Dict[str, pd.DataFrame],        
):
    for patient in dfs1:
        df1 = dfs1[patient]
        df2 = dfs2[patient]

        diffs1 = df1.iloc[:, 2:].to_numpy() - df1.iloc[:, 1:-1].to_numpy()
        diffs2= df2.iloc[:, 2:].to_numpy() - df2.iloc[:, 1:-1].to_numpy()
        is_outlier = (diffs1 >= 0.2).any(axis=1) | (diffs2 >= 0.3).any(axis=1)

        df1[is_outlier] = np.nan
        df2[is_outlier] = np.nan

    return dfs1, dfs2



def direct_wear(dfs_3dwa, dfs_ours):
    wears_3dwa = np.zeros(0)
    wears_ours = np.zeros(0)

    for patient in dfs_3dwa:
        for followup in dfs_3dwa[patient].columns[1:]:
            wears_3dwa = np.concatenate((wears_3dwa, dfs_3dwa[patient][followup]))
            wears_ours = np.concatenate((wears_ours, dfs_ours[patient][followup]))

    wears_3dwa = wears_3dwa[~np.isnan(wears_3dwa)]
    wears_ours = wears_ours[~np.isnan(wears_ours)]

    print(stats.shapiro(wears_3dwa))
    print(stats.shapiro(wears_ours))

    result = stats.wilcoxon(wears_3dwa, wears_ours)
    print('n=', wears_3dwa.shape[0])
    print(result)

    return (wears_3dwa, wears_ours), result.pvalue


def indirect_wear_progression(dfs_3dwa, dfs_ours):
    wears_3dwa = np.zeros(0)
    wears_ours = np.zeros(0)

    for patient in dfs_3dwa:
        for i, col1 in enumerate(dfs_3dwa[patient].columns[1:-1]):
            col2 = dfs_3dwa[patient].columns[i + 2]
            # for col2 in dfs_3dwa[patient].columns[i + 2:]:
            diffs_3dwa = dfs_3dwa[patient][col2] - dfs_3dwa[patient][col1]
            diffs_ours = dfs_ours[patient][col2] - dfs_ours[patient][col1]

            diffs_3dwa = diffs_3dwa[~np.isnan(diffs_3dwa)]
            diffs_ours = diffs_ours[~np.isnan(diffs_ours)]

            wears_3dwa = np.concatenate((wears_3dwa, diffs_3dwa))
            wears_ours = np.concatenate((wears_ours, diffs_ours))

    print(stats.shapiro(wears_3dwa))
    print(stats.shapiro(wears_ours))

    result = stats.wilcoxon(wears_3dwa, wears_ours)
    print('n=', wears_3dwa.shape[0])
    print(result)
    
    return (wears_3dwa, wears_ours), result.pvalue


def violin_plots(direct_wears, indirect_wears, axs):
    axs[0].violinplot(direct_wears[0], showmedians=True)
    axs[0].set_xticks([1, 2], labels=['3DWA', 'Ours'])
    axs[0].set_ylabel('Wear (mm)')

    axs[1].violinplot(indirect_wears[0], showmedians=True)
    axs[1].set_xticks([1, 2], labels=['3DWA', 'Ours'])
    axs[1].set_ylabel('Wear progression (mm)')


def mean_diff_plot(data, sd_limit=1.96, ax=None):
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))
    
    (m2, m1), pvalue = data

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    ax.scatter(means, diffs, s=15, alpha=0.66) # Plot the means against the diffs.
    ax.axhline(mean_diff, color='gray', linewidth=1, linestyle='-')  # draw mean line.

    # annotate sample size
    ax.annotate(
        f'n={m1.shape[0]}',
        xy=(0.01, 0.99),
        horizontalalignment='left',
        verticalalignment='top',
        xycoords='axes fraction',
        fontsize=10,
    )

    # Annotate p-value
    ax.annotate(
        f'p={pvalue:.3f}' if np.round(pvalue, 3) > 0 else 'p<0.001',
        xy=(0.99, 0.99),
        horizontalalignment='right',
        verticalalignment='top',
        xycoords='axes fraction',
        fontsize=10,
    )

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        print(mean_diff, lower, upper)
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, color='gray', linewidth=1, linestyle='--')
        # ax.annotate(f'-{sd_limit} SD: {lower:0.2g}',
        #             xy=(0.99, 0.07),
        #             horizontalalignment='right',
        #             verticalalignment='bottom',
        #             fontsize=14,
        #             xycoords='axes fraction')
        # ax.annotate(f'+{sd_limit} SD: {upper:0.2g}',
        #             xy=(0.99, 0.92),
        #             horizontalalignment='right',
        #             fontsize=14,
        #             xycoords='axes fraction')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

    ax.set_ylabel('Difference')
    ax.set_xlabel('Means')
    # ax.tick_params(labelsize=13)
    fig.tight_layout()
    return fig


def bland_altman_plots(direct_wears, indirect_wears, axs):
    mean_diff_plot(direct_wears, ax=axs[0])
    mean_diff_plot(indirect_wears, ax=axs[1])


if __name__ == '__main__':
    root = Path('methodology study')

    df_exclusions = pd.read_excel(root / 'overview.xlsx')
    dfs_3dwa = pd.read_excel(root / '3dwa_heights.xlsx', sheet_name=None)
    dfs_ours = pd.read_excel(root / 'ours_heights.xlsx', sheet_name=None)

    dfs_3dwa = apriori_exclusions(df_exclusions, dfs_3dwa)
    dfs_ours = apriori_exclusions(df_exclusions, dfs_ours)
    dfs_3dwa, dfs_ours = measurement_exclusions(dfs_3dwa, dfs_ours)

    direct_wears = direct_wear(dfs_3dwa, dfs_ours)
    indirect_wears = indirect_wear_progression(dfs_3dwa, dfs_ours)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6.5))
    violin_plots(direct_wears, indirect_wears, axs=axs[:, 0])
    bland_altman_plots(direct_wears, indirect_wears, axs=axs[:, 1])
    # plt.tight_layout()
    for ax, letter in zip(axs.flatten(), 'abcd'):
        ax.annotate(
            f'({letter})',
            xy=(0.01, 0.01),
            horizontalalignment='left',
            verticalalignment='bottom',
            xycoords='axes fraction',
            fontsize=12,
            weight='bold',
        )
    plt.savefig('stats.png', dpi=500, bbox_inches='tight', pad_inches=0.0)
    plt.show()
