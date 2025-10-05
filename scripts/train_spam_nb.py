from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.spam_detection.modeling import train_multinomial_nb

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def main() -> None:
    logging.info('Training Multinomial Naive Bayes spam classifier...')
    metrics = train_multinomial_nb()
    summary = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
    }
    logging.info('Evaluation summary: %s', json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
