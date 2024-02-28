import json
from pathlib import Path

import numpy as np


def compute_metrics(average='macro'):
    metrics = {k: {} for k in [
        'dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy'
    ]}
    total_teeth = 0
    for file1, file2 in zip(manual_files, automated_files):

        with open(file1, 'r') as f:
            labels1 = np.array(json.load(f)['labels'])

        with open(file2, 'r') as f:
            labels2 = np.array(json.load(f)['labels'])
            if np.unique(labels2)[1] < 30 and 'mandible' in file1.stem:
                labels2[labels2 > 0] += 20

        if np.all(labels1 == 0):
            continue

        fdis = np.unique(labels1)[1:]
        fdis = fdis[(fdis % 10) != 8]
        total_teeth += fdis.shape[0]
        case_metrics = {k: [] for k in metrics}
        for fdi in fdis:
            tp = ((labels1 == fdi) & (labels2 == fdi)).sum()
            fp = ((labels1 != fdi) & (labels2 == fdi)).sum()
            fn = ((labels1 == fdi) & (labels2 != fdi)).sum()
            tn = ((labels1 != fdi) & (labels2 != fdi)).sum()

            case_metrics['iou'].append(tp / (tp + fp + fn))
            case_metrics['dice'].append(2 * tp / (2 * tp + fp + fn))
            case_metrics['precision'].append(tp / (tp + fp))
            case_metrics['recall'].append(tp / (tp + fn))
            case_metrics['specificity'].append(tn / (tn + fp))
            case_metrics['accuracy'].append((tp + tn) / (tp + tn + fp + fn))

        for k in metrics:
            metrics[k][file1.stem] = np.mean(case_metrics[k])

    for k in metrics:
        metrics[k]['total_m'] = np.mean([v for v in metrics[k].values()])
        metrics[k]['total_s'] = np.std([v for v in metrics[k].values()])
        metrics[k]['mandible_m'] = np.mean([v for k, v in metrics[k].items() if 'mandible' in k])
        metrics[k]['mandible_s'] = np.std([v for k, v in metrics[k].items() if 'mandible' in k])
        metrics[k]['maxilla_m'] = np.mean([v for k, v in metrics[k].items() if 'maxilla' in k])
        metrics[k]['maxilla_s'] = np.std([v for k, v in metrics[k].items() if 'maxilla' in k])

    metrics = {f'{k}_{k_}': v for k in metrics for k_, v in metrics[k].items()}

    print('Total teeth:', total_teeth)

    return metrics


if __name__ == '__main__':
    manual_root = Path('methodology study/3DWA labels')
    automated_root = Path('methodology study/AI labels')

    manual_files = sorted(manual_root.glob('*.json'))
    # manual_files = [f for f in manual_files if '0' not in f.stem[4:6]]
    automated_files = sorted(automated_root.glob('*.json'))
    # automated_files = [f for f in automated_files if '0' not in f.stem[4:6]]

    for k, v in compute_metrics().items():
        print(f'{k}: {v:.3f}')
