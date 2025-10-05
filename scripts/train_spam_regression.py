from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.spam_detection.modeling import train_elasticnet_regressor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def main() -> None:
    logging.info('Training ElasticNet spam regression model...')
    metrics = train_elasticnet_regressor()
    summary = {
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'r2': metrics['r2'],
        'roc_auc': metrics['roc_auc'],
        'classification_accuracy': metrics['classification_view']['accuracy'],
    }
    logging.info('Evaluation summary: %s', json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
