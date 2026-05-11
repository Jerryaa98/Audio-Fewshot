# Mixer Datasets Setup Guide

This folder is intentionally kept lightweight in git. Put the raw datasets here before running preprocessing and mixing.

Please download the datasets and unroll them in the specified order below.

## Required final tree (exact names)

```text
Audio-Fewshot/Mixer/
└── Datasets/
    ├── DECASE2016/
    │   ├── dcase2016_task2_train_dev/
    │   │   ├── README.txt
    │   │   ├── dcase2016_task2_train/
    │   │   │   ├── clearthroat*.wav
    │   │   │   ├── doorslam*.wav
    │   │   │   ├── drawer*.wav
    │   │   │   ├── keysDrop*.wav
    │   │   │   ├── pageturn*.wav
    │   │   │   ├── phone*.wav
    │   │   │   └── ...
    │   │   └── dcase2016_task2_dev/
    │   │       ├── annotation/
    │   │       └── sound/
    │   └── dcase2016_task2_test_public/
    │       ├── annotation/
    │       └── sound/
    ├── DESED/
    │   └── data/
    │       └── dataset/
    │           ├── audio/
    │           │   └── train/
    │           │       └── weak/
    │           └── metadata/
    │               └── train/
    │                   └── weak.tsv
    ├── ESC-50/
    │   ├── audio/
    │   │   └── *.wav
    │   └── meta/
    │       └── esc50.csv
    ├── Urban8k/
    │   ├── audio/
    │   │   ├── fold1/
    │   │   ├── fold2/
    │   │   ├── ...
    │   │   └── fold10/
    │   └── metadata/
    │       └── UrbanSound8K.csv
    ├── USM/
    │   ├── train/
    │   ├── eval/
    │   ├── val/
    │   └── urls.txt
    └── VocalSound/
        ├── audio_16k/
        │   └── *.wav
        ├── class_labels_indices_vs.csv
        └── meta/
```

## Where to get each dataset

1. DECASE2016 (Task 2 - SASED)
- Source: http://www.cs.tut.fi/sgn/arg/dcase2016/
- Task page: http://www.cs.tut.fi/sgn/arg/dcase2016/task-synthetic-sound-event-detection
- Put extracted folders under `Datasets/DECASE2016/` as shown above.

2. DESED
- Project page: https://project.inria.fr/desed/
- Main repository: https://github.com/turpaultn/DESED
- Zenodo links from DESED README:
  - synthetic: https://zenodo.org/record/3702397
  - public eval: https://zenodo.org/record/3588172
- For this mixer, at minimum ensure:
  - `Datasets/DESED/data/dataset/audio/train/weak/`
  - `Datasets/DESED/data/dataset/metadata/train/weak.tsv`

3. ESC-50
- Source repository: https://github.com/karoldvl/ESC-50
- Direct zip (from upstream README): https://github.com/karoldvl/ESC-50/archive/master.zip
- Keep `audio/` and `meta/esc50.csv` in `Datasets/ESC-50/`.

4. UrbanSound8K
- Project page: https://urbansounddataset.weebly.com/urbansound8k.html
- Mirror referenced in local README: http://serv.cusp.nyu.edu/projects/urbansounddataset
- Keep `audio/fold1...fold10` and `metadata/UrbanSound8K.csv` in `Datasets/Urban8k/`.

5. VocalSound
- Source repository: https://github.com/YuanGongND/vocalsound
- Keep at least `audio_16k/` and `class_labels_indices_vs.csv` in `Datasets/VocalSound/`.

6. USM
- Zenodo record used in this project: https://zenodo.org/records/6413788
- Project includes direct links in `Datasets/USM/urls.txt`.
- Download and extract so that `train/`, `eval/`, and `val/` exist directly under `Datasets/USM/`.

## What these key files/folders contain

1. `ESC-50/meta/esc50.csv`
- Per clip metadata.
- Important columns: `filename`, `fold`, `target`, `category`.
- Used to map each file in `ESC-50/audio/` to a class label.

2. `Urban8k/metadata/UrbanSound8K.csv`
- Per clip metadata for UrbanSound8K.
- Important columns: `slice_file_name`, `fold`, `class`.
- Used to build file paths like `audio/fold{fold}/{slice_file_name}`.

3. `DESED/data/dataset/metadata/train/weak.tsv`
- Weak labels for DESED training clips.
- Key fields used by preprocessing: `filename`, `event_labels`.
- Combined with audio files in `audio/train/weak/`.

4. `DECASE2016/dcase2016_task2_train_dev/dcase2016_task2_train/*.wav`
- Isolated office events (class encoded in filename prefix).
- Mixer preprocessing infers class names from file names.

5. `VocalSound/audio_16k/*.wav`
- Vocal/non-speech sound clips.
- Current preprocessing expects category encoded in filename pattern with underscores.

6. `USM/train`, `USM/eval`, `USM/val`
- USM split data used as mixed source material in this codebase.
- This repository already has a `urls.txt` with exact download URLs.