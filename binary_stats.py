from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def draw_confusion_matrix(
    cm,
    labels,
    ax,
    title: str='',
    xaxis: bool=True,
    yaxis: bool=True,
    colorbar: bool=True,
):
    norm_cm = cm / cm.max()
    disp = ConfusionMatrixDisplay(norm_cm, display_labels=labels)
    disp.plot(cmap='magma', ax=ax, colorbar=colorbar)

    if not xaxis:
        disp.ax_.xaxis.set_visible(False)
    if not yaxis:
        disp.ax_.yaxis.set_visible(False)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            disp.text_[i, j].set_text(int(cm[i, j]))
    
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
    disp.ax_.images[0].set_norm(normalize)
    
    # draw y ticklabels vertically
    offset = matplotlib.transforms.ScaledTranslation(-0.1, 0, disp.figure_.dpi_scale_trans)
    for label in disp.ax_.get_yticklabels():
        label.set_rotation(90)
        label.set_transform(label.get_transform() + offset)
        label.set_ha('center')
        label.set_rotation_mode('anchor')

    ax.set_ylabel('3DWA protocol')
    ax.set_xlabel('Automated method')
    if title:
        ax.set_title(title)


def extract_heights(
    df: pd.DataFrame,
    patients: List[str]=['A-20', 'A-24', 'A-25', 'A-27', 'A-28', 'A-29', 'A-41', 'A-46'],
):
    ret = {}
    for patient in patients:
        patient_df = df[df['ESO.ID'] == f'A{patient[2:]}']

        ret_df = pd.DataFrame({
            'FDI': [10 * q + e for q in [1,2,3,4] for e in [1,2,3,4,5,6,7,8]],
            '0-1': [np.nan for _ in range(32)],
            '0-3': [np.nan for _ in range(32)],
            '0-5': [np.nan for _ in range(32)],
        })
        for i, row in patient_df.iterrows():
            fdi = row['Tooth']
            interval = str(row['Interval']).replace('.', '-')
            ret_df.loc[ret_df['FDI'] == fdi, interval] = row['max. height loss (mm) Geomagic original']
        
        ret[patient] = ret_df

    return ret


def apriori_exclusions(
    df_exclusions: pd.DataFrame,
    dfs: Dict[str, pd.DataFrame],
    patients: List[str]=['A-02', 'A-20', 'A-24', 'A-25', 'A-27', 'A-28', 'A-29', 'A-40', 'A-41', 'A-46', 'a-47'],
):
    heights = df_exclusions['max. height loss (mm) Geomagic original']
    is_excluded = (heights < 0).to_numpy()

    patient_exclusions = defaultdict(lambda: set([18, 28, 38, 48]))
    for idx in np.nonzero(is_excluded)[0]:
        patient = df_exclusions['ESO.ID'][idx]
        patient = f'{patient[0]}-{patient[1:]}' if '-' not in patient else patient
        fdi = df_exclusions['Tooth'][idx]

        patient_exclusions[patient].add(fdi)

    patient_inclusions = {
        patient: np.unique(df_exclusions[df_exclusions['ESO.ID'] == patient]['Tooth'])
        for patient in np.unique(df_exclusions['ESO.ID'])
    }
    patient_inclusions = {
        f'{patient[0]}-{patient[1:]}' if '-' not in patient else patient: v
        for patient, v in patient_inclusions.items()
    }
    patient_inclusions = {
        patient: set(patient_inclusions[patient]) - set(patient_exclusions[patient])
        for patient in patient_inclusions
    }

    dfs = {k: v for k, v in dfs.items() if k in patients}
    for patient, df in dfs.items():
        if patient not in patient_exclusions:
            continue
        
        for exclusion in patient_exclusions[patient]:
            mask = df['FDI'] == exclusion
            tmp = df.loc[mask, 'FDI']
            df[mask] = np.nan
            df.loc[mask, 'FDI'] = tmp

    return dfs


def determine_metrics(cm, labels):
    metrics = defaultdict(list)
    for idx, label in zip(range(cm.shape[0]), labels):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx].sum() - tp
        tn = cm.sum() - tp - fp - fn

        metrics[f'{label}_acc'] = (tp + tn) / (tp + fp + fn + tn)
        metrics[f'{label}_prec'] = tp / (tp + fp)
        metrics[f'{label}_sens'] = tp / (tp + fn)
        metrics[f'{label}_spec'] = tn / (tn + fp)
        # metrics[f'{label}_npv'] = tn / (tn + fn)
        metrics[f'{label}_f1'] = 2 * tp / (2 * tp + fp + fn)

    for key, value in list(metrics.items()):
        metrics[key.split('_')[1]] = metrics[key.split('_')[1]] + [value]

    metrics = {key: np.mean(value) for key, value in metrics.items()}

    return metrics


def direct_confusion_matrices(
    df_3dwa: Dict[str, pd.DataFrame],
    df_ours: Dict[str, pd.DataFrame],
    # thresholds: np.ndarray=np.array([0, 0.2, 1, 2, 8]),
    thresholds: np.ndarray=np.array([0, 0.2]),
    verbose: bool=False,
):
    out = np.zeros((32, 3, thresholds.shape[0], thresholds.shape[0]), dtype=int)
    gts, scores = np.zeros((2, 0))
    for patient in df_3dwa:
        for i, fdi in enumerate(df_3dwa[patient]['FDI']):
            for j, interval in enumerate(df_3dwa[patient].columns[1:]):
                gt_wear = df_3dwa[patient].loc[df_3dwa[patient]['FDI'] == fdi, interval]
                pred_wear = df_ours[patient].loc[df_ours[patient]['FDI'] == fdi, interval]

                if pd.isna(gt_wear).item() or pd.isna(pred_wear).item():
                    continue

                # gt_wear /= int(interval[-1])
                # pred_wear /= int(interval[-1])

                gt_label = thresholds.shape[0] - 1 - (thresholds[::-1] <= np.abs(gt_wear.item())).argmax()
                pred_label = thresholds.shape[0] - 1 - (thresholds[::-1] <= np.abs(pred_wear.item())).argmax()
                score = np.abs(pred_wear.item())

                out[i, j, gt_label, pred_label] += 1
                gts = np.concatenate((gts, [gt_label]))
                scores = np.concatenate((scores, [score]))

    RocCurveDisplay.from_predictions(gts, scores)
    plt.show()

    thresholds = np.concatenate(([0], scores))
    f1s = []
    for thr in thresholds:
        pred = scores >= thr
        tp = (gts * pred).sum()
        fp = ((1 - gts) * pred).sum()
        fn = (gts * ~pred).sum()
        tn = ((1 - gts) * ~pred).sum()
        f1 = 2 * tp / (2 * tp + fp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        f1s.append(acc)

    f1s = np.array(f1s)
    thr = thresholds[np.argmax(f1s)]
    print(thr)

    ConfusionMatrixDisplay.from_predictions(gts, scores >= thr)
    plt.show()


    for i, interval in enumerate(['0-1 years', '0-3 years', '0-5 years']):
        draw_confusion_matrix(
            cm=out.sum(0)[i],
            labels=['Without lesion', 'With lesion'],
            ax=plt.gca(),
            yaxis=i == 0,
            colorbar=False,
        )
        plt.savefig(f'figures/cm_{interval}', dpi=500, bbox_inches='tight', pad_inches=None)
        if verbose: plt.show()
        plt.close()
    
    draw_confusion_matrix(out.sum((0, 1)), ['Without lesion', 'With lesion'], plt.gca(), yaxis=False)
    plt.title('Total')
    plt.savefig(f'figures/cm_total', dpi=500, bbox_inches='tight', pad_inches=None)
    if verbose: plt.show()
    plt.close()

    return out


def indirect_confusion_matrices(
    df_3dwa: Dict[str, pd.DataFrame],
    df_ours: Dict[str, pd.DataFrame],
    # thresholds: np.ndarray=np.array([0, 0.2, 1, 2, 8]),
    thresholds: np.ndarray=np.array([-3, 0.07]),
):
    out = np.zeros((32, 3, thresholds.shape[0], thresholds.shape[0]), dtype=int)
    for patient in df_3dwa:
        for i, fdi in enumerate(df_3dwa[patient]['FDI']):
            for j, interval1 in enumerate(df_3dwa[patient].columns[1:-1]):
                for k, interval2 in enumerate(df_3dwa[patient].columns[2 + j:]):
                    gt_wear1 = df_3dwa[patient].loc[df_3dwa[patient]['FDI'] == fdi, interval1].item()
                    gt_wear2 = df_3dwa[patient].loc[df_3dwa[patient]['FDI'] == fdi, interval2].item()

                    pred_wear1 = df_ours[patient].loc[df_ours[patient]['FDI'] == fdi, interval1].item()
                    pred_wear2 = df_ours[patient].loc[df_ours[patient]['FDI'] == fdi, interval2].item()

                    if (
                        np.isnan(gt_wear1) or np.isnan(pred_wear1)
                        or np.isnan(gt_wear2) or np.isnan(pred_wear2)
                    ):
                        continue

                    gt_wear = np.abs(gt_wear2) - np.abs(gt_wear1)
                    pred_wear = np.abs(pred_wear2) - np.abs(pred_wear1)

                    gt_wear /= int(interval2[-1]) - int(interval1[-1])
                    pred_wear /= int(interval2[-1]) - int(interval1[-1])

                    gt_label = thresholds.shape[0] - 1 - (thresholds[::-1] <= gt_wear).argmax()
                    pred_label = thresholds.shape[0] - 1 - (thresholds[::-1] <= pred_wear).argmax()

                    out[i, 2*j + k, gt_label, pred_label] += 1

    metrics = determine_metrics(out.sum((0, 1)), ['No wear', 'physiological wear', 'pathological wear'])

    ConfusionMatrixDisplay(
        confusion_matrix=out.sum((0, 1)),
        display_labels=['Natural', 'Physiological', 'Pathological'][-thresholds.shape[0]:],
    ).plot()
    plt.yticks(*plt.yticks(), rotation=90)
    plt.title('Tooth wear classification between follow-ups')
    plt.show()

    return cms


def analysis_metrics(cms, verbose: bool=False):
    teeth_cms = np.zeros((5, 3, 2, 2))

    c_incisor_mask = np.zeros(32, dtype=bool)
    c_incisor_mask[[0, 8, 16, 24]] = True
    teeth_cms[0] = cms[c_incisor_mask].sum(0)

    l_incisor_mask = np.zeros(32, dtype=bool)
    l_incisor_mask[[1, 9, 17, 25]] = True
    teeth_cms[1] = cms[l_incisor_mask].sum(0)

    canine_mask = np.zeros(32, dtype=bool)
    canine_mask[[2, 10, 18, 26]] = True
    teeth_cms[2] = cms[canine_mask].sum(0)

    premolar_mask = np.zeros(32, dtype=bool)
    premolar_mask[[3, 4, 11, 12, 19, 20, 27, 28]] = True
    teeth_cms[3] = cms[premolar_mask].sum(0)

    molar_mask = np.zeros(32, dtype=bool)
    molar_mask[[5, 6, 13, 14, 21, 22, 29, 30]] = True
    teeth_cms[4] = cms[molar_mask].sum(0)

    teeth = ['Central Incisor', 'Lateral Incisor', 'Canine', 'Premolar', 'Molar', 'Total']
    intervals = ['0-1 years', '0-3 years', '0-5 years', 'Total']

    _, axs = plt.subplots(len(teeth), len(intervals), figsize=(12, 16))
    for i, tooth in enumerate(teeth):
        for j, interval in enumerate(intervals):
            tooth_cms = teeth_cms[i] if tooth != 'Total' else teeth_cms.sum(0)
            cm = tooth_cms[j] if interval != 'Total' else tooth_cms.sum(0)

            draw_confusion_matrix(
                cm=cm,
                labels=['Without lesion', 'With lesion'],
                ax=axs[i][j],
                title=f'{tooth} - {interval}',
                xaxis=i == len(teeth) - 1,
                yaxis=j == 0,
                colorbar=j == len(intervals) - 1,
            )
    plt.tight_layout()
    plt.savefig(f'figures/cms_table.png', dpi=500, bbox_inches='tight', pad_inches=None)
    if verbose: plt.show()
    plt.close()

    for metric_name in ['f1', 'prec', 'sens', 'spec', 'acc']:
        f1s = np.zeros((6, 4))
        for i, tooth_cms in enumerate(teeth_cms):
            f1s[i, -1] = determine_metrics(tooth_cms.sum(0), 'AB')[f'B_{metric_name}']
            for j, cm in enumerate(tooth_cms):
                f1s[i, j] = determine_metrics(cm, 'AB')[f'B_{metric_name}']
        for j, cm in enumerate(teeth_cms.sum(0)):
            f1s[-1, j] = determine_metrics(cm, 'AB')[f'B_{metric_name}']
        f1s[-1, -1] = determine_metrics(teeth_cms.sum((0, 1)), 'AB')[f'B_{metric_name}']

        print(metric_name)
        for i, f1 in enumerate(f1s.flatten()):
            if i % 4 == 0:
                print()
            print(f'{f1:.3f}')

    k = 3


if __name__ == '__main__':
    root = Path('methodology study')

    df_exclusions = pd.read_excel(root / 'overview.xlsx')
    # dfs_3dwa = extract_heights(df_exclusions)


    dfs_3dwa = pd.read_excel(root / '3DWA pairs' / '3dwa_max_heights.xlsx', sheet_name=None)
    dfs_ours = pd.read_excel(root / 'AI pairs' / 'ours_max_heights.xlsx', sheet_name=None)

    dfs_3dwa = apriori_exclusions(df_exclusions, dfs_3dwa)
    dfs_ours = apriori_exclusions(df_exclusions, dfs_ours)

    cms = direct_confusion_matrices(dfs_3dwa, dfs_ours)
    analysis_metrics(cms)


    exit()
    cms = indirect_confusion_matrices(dfs_3dwa, dfs_ours)

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
