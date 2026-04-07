from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.evaluation import (
    save_cluster_size_bar,
    save_confusion_matrix,
    save_metric_bar_chart,
    save_precision_recall_curve,
    save_regression_diagnostics,
    save_roc_curve,
)
from src.common.paths import reports_malware_path, reports_spam_path
from src.malware_detection.modeling import load_malware_dataset, make_train_test_split as malware_split
from src.spam_detection.modeling import load_merged_spam_dataset, make_train_test_split as spam_split

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def _spam_test_split():
    df = load_merged_spam_dataset()
    split = spam_split(df, test_size=0.2, random_state=42)
    return split


def _malware_ds1_split():
    df = load_malware_dataset('malware_ds1')
    drop_cols = {'label', 'classification', 'hash'}
    feature_cols = [col for col in df.columns if col not in drop_cols]
    split = malware_split(df, feature_cols=feature_cols, test_size=0.2, random_state=42)
    return split


def _malware_ds2_split():
    df = load_malware_dataset('malware_ds2')
    drop_cols = {'label', 'classification', 'hash', 'family_type'}
    feature_cols = [col for col in df.columns if col not in drop_cols]
    split = malware_split(df, feature_cols=feature_cols, test_size=0.2, random_state=42)
    return split


def _load_metrics_summary() -> pd.DataFrame:
    summary_path = REPO_ROOT / 'reports' / 'metrics_summary.csv'
    if summary_path.exists():
        return pd.read_csv(summary_path)
    logging.warning('metrics_summary.csv not found - run scripts/summarise_metrics.py first')
    return pd.DataFrame()


def main() -> None:
    logging.info('Generating evaluation plots...')

    # Spam classifiers
    spam_split_data = _spam_test_split()
    spam_eval_dir = reports_spam_path('evaluation')

    nb_model = joblib.load(REPO_ROOT / 'models/spam/multinomial_nb.joblib')
    save_confusion_matrix(
        nb_model,
        spam_split_data.X_test,
        spam_split_data.y_test,
        title='Spam Multinomial NB - Confusion',
        path=spam_eval_dir / 'multinomial_nb_confusion.png',
    )
    save_roc_curve(
        nb_model,
        spam_split_data.X_test,
        spam_split_data.y_test,
        title='Spam Multinomial NB - ROC',
        path=spam_eval_dir / 'multinomial_nb_roc.png',
    )
    save_precision_recall_curve(
        nb_model,
        spam_split_data.X_test,
        spam_split_data.y_test,
        title='Spam Multinomial NB - Precision-Recall',
        path=spam_eval_dir / 'multinomial_nb_pr.png',
    )

    # Spam regression diagnostics
    elastic_model = joblib.load(REPO_ROOT / 'models/spam/elasticnet_regressor.joblib')
    save_regression_diagnostics(
        elastic_model,
        spam_split_data.X_test,
        spam_split_data.y_test.astype(float),
        prefix=spam_eval_dir / 'elasticnet',
        title='Spam ElasticNet',
    )

    # Spam clustering summary (cluster assignment CSV)
    spam_cluster_path = REPO_ROOT / 'reports/spam/kmeans_assignments.csv'
    if spam_cluster_path.exists():
        clusters = pd.read_csv(spam_cluster_path)['cluster']
        save_cluster_size_bar(
            clusters,
            title='Spam KMeans Cluster Sizes',
            path=spam_eval_dir / 'kmeans_cluster_sizes.png',
        )

    # Malware DS1 evaluation
    malware_eval_dir = reports_malware_path('evaluation')
    ds1_split = _malware_ds1_split()
    rf_model = joblib.load(REPO_ROOT / 'models/malware/malware_ds1_random_forest.joblib')
    save_confusion_matrix(
        rf_model,
        ds1_split.X_test,
        ds1_split.y_test,
        title='Malware DS1 RandomForest - Confusion',
        path=malware_eval_dir / 'ds1_random_forest_confusion.png',
    )

    save_roc_curve(
        rf_model,
        ds1_split.X_test,
        ds1_split.y_test,
        title='Malware DS1 RandomForest - ROC',
        path=malware_eval_dir / 'ds1_random_forest_roc.png',
    )

    save_precision_recall_curve(
        rf_model,
        ds1_split.X_test,
        ds1_split.y_test,
        title='Malware DS1 RandomForest - Precision-Recall',
        path=malware_eval_dir / 'ds1_random_forest_pr.png',
    )

    # Malware DS2 classification models
    ds2_split = _malware_ds2_split()
    hist_gbdt_model = joblib.load(REPO_ROOT / 'models/malware/malware_ds2_hist_gbdt.joblib')
    save_confusion_matrix(
        hist_gbdt_model,
        ds2_split.X_test,
        ds2_split.y_test,
        title='Malware DS2 HistGBDT - Confusion',
        path=malware_eval_dir / 'ds2_hist_gbdt_confusion.png',
    )
    save_roc_curve(
        hist_gbdt_model,
        ds2_split.X_test,
        ds2_split.y_test,
        title='Malware DS2 HistGBDT - ROC',
        path=malware_eval_dir / 'ds2_hist_gbdt_roc.png',
    )
    save_precision_recall_curve(
        hist_gbdt_model,
        ds2_split.X_test,
        ds2_split.y_test,
        title='Malware DS2 HistGBDT - Precision-Recall',
        path=malware_eval_dir / 'ds2_hist_gbdt_pr.png',
    )


    ds2_cluster_assignments = REPO_ROOT / 'reports/malware/malware_ds2_kmeans_assignments.csv'
    if ds2_cluster_assignments.exists():
        clusters = pd.read_csv(ds2_cluster_assignments)['cluster']
        save_cluster_size_bar(
            clusters,
            title='Malware DS2 KMeans Cluster Sizes',
            path=malware_eval_dir / 'ds2_kmeans_cluster_sizes.png',
        )

    # Summary bar charts
    summary_df = _load_metrics_summary()
    if not summary_df.empty:
        save_metric_bar_chart(
            summary_df,
            domain='spam',
            metric='f1',
            path=spam_eval_dir / 'spam_f1_bar.png',
            types=('classification', 'regression'),
        )
        save_metric_bar_chart(
            summary_df,
            domain='spam',
            metric='accuracy',
            path=spam_eval_dir / 'spam_accuracy_bar.png',
            types=('classification', 'regression'),
        )
        save_metric_bar_chart(
            summary_df,
            domain='malware',
            metric='f1',
            path=malware_eval_dir / 'malware_f1_bar.png',
        )

    logging.info('Evaluation plots generated successfully.')


if __name__ == '__main__':
    main()



