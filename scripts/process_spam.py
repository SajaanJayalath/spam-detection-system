from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.paths import reports_spam_path
from src.spam_detection.datasets import run_full_processing_pipeline
from src.spam_detection.visualization import (
    plot_label_distribution,
    plot_length_distribution,
    plot_top_words,
    plot_wordcloud,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> None:
    logging.info("Processing raw spam datasets...")
    datasets = run_full_processing_pipeline()

    figure_plan = {
        "spam_ds1": "dataset1",
        "spam_ds2": "dataset2",
        "spam_ds3": "dataset3",
        "spam_merged": "merged",
    }

    for key, prefix in figure_plan.items():
        df = datasets[key]
        logging.info("Generating figures for %s", key)
        plot_label_distribution(
            df,
            label_col="label",
            title=f"Spam vs Ham Distribution ({key.replace('_', ' ').title()})",
            path=reports_spam_path(f"{prefix}_label_distribution.png"),
        )

        plot_length_distribution(
            df,
            text_col="clean_text",
            label_col="label",
            title=f"Email Length Distribution ({key.replace('_', ' ').title()})",
            path=reports_spam_path(f"{prefix}_length_distribution.png"),
        )

        spam_texts = df[df["label"] == 1]["clean_text"]
        ham_texts = df[df["label"] == 0]["clean_text"]

        plot_top_words(
            spam_texts,
            title=f"Top Words in Spam ({key.replace('_', ' ').title()})",
            palette="Reds_r",
            path=reports_spam_path(f"{prefix}_spam_top_words.png"),
        )
        plot_top_words(
            ham_texts,
            title=f"Top Words in Ham ({key.replace('_', ' ').title()})",
            palette="Blues_r",
            path=reports_spam_path(f"{prefix}_ham_top_words.png"),
        )

        plot_wordcloud(
            spam_texts,
            title=f"Spam WordCloud ({key.replace('_', ' ').title()})",
            path=reports_spam_path(f"{prefix}_spam_wordcloud.png"),
        )
        plot_wordcloud(
            ham_texts,
            title=f"Ham WordCloud ({key.replace('_', ' ').title()})",
            path=reports_spam_path(f"{prefix}_ham_wordcloud.png"),
        )

    logging.info("All artefacts saved under data/processed/spam and reports/spam")


if __name__ == "__main__":
    main()
