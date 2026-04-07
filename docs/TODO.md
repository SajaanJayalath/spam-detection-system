# Integration Checklist

- [x] Modularise the spam prototype into packages under src/:
  - [x] src/common/paths.py and src/common/text_processing.py for shared helpers.
  - [x] src/spam_detection/datasets.py for dataset-specific cleaning and merging.
  - [ ] src/spam_detection/models.py for the three spam classifiers.
  - [x] src/malware_detection/datasets.py for malware data preparation.
  - [ ] src/malware_detection/models.py for the two malware classifiers.
- [x] Create command-line entry points under scripts/ (process_spam.py, process_malware.py).
- [ ] Capture experiment configurations (train/test splits, feature sets) in a reusable format (YAML or JSON).
- [x] Update requirements.txt once dependencies are identified.
- [ ] Add notebooks that replicate exploratory visualisations for each dataset.
- [ ] Document dataset sources and preprocessing decisions in docs/.
