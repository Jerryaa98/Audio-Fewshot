# Mixer (SpurAudio Mixer)

This directory is the data mixer and preprocessing pipeline used to build SpurAudio-style datasets for few-shot audio experiments.

If you keep this as a subdirectory inside `Audio-Fewshot`, this README documents the expected layout, required inputs, and end-to-end run commands.

The merger algorithm is deterministic, so following the setup and running the written commands should give exactly SpurAudio.

## What This Repository Does

`Mixer` builds mixed audio datasets where:

- foreground classes are paired with correlated background classes,
- background strength is automatically controlled during mixing,
- metadata CSV files are generated for downstream training/evaluation,
- optional OOD query configurations can be generated.

Core scripts:

- `main.py`: main mixer entry point.
- `merger.py`: pairing policy and audio mixing implementation.
- `dataset_preprocessing/*.py`: builds per-dataset CSV indexes used by the mixer.
- `Results/full_stack_ESC.py`: converts mixed WAV outputs into class folders, `.npy`, and mel-spectrogram `.npy` for training.

## Structure

tree:

```text
Audio-Fewshot/
├── config/
├── libfewshot_core/
├── run_trainer.py
├── requirements.txt
└── matcher/
		├── main.py
		├── merger.py
		├── merger_ood_configs.py
		├── configs/
		├── dataset_preprocessing/
		├── Matcher/
		├── Datasets/
		├── Preprocessed_CSVs/
		└── Results/
```

All commands below assume your working directory is:

```bash
cd Audio-Fewshot/matcher
```

## Requirements

### Python

- Python `3.9+` recommended.
- Works with virtual environments (`venv` or conda).

### Requirements

Minimum packages needed for mixer/preprocessing scripts:

```bash
pip install numpy pandas scipy librosa soundfile tqdm fastargs pyyaml
```

## Data Requirements

The mixer expects:

1. Raw downloaded datasets under `Datasets/`.
2. Preprocessed CSVs under `Preprocessed_CSVs/`.
3. Foreground-background correlation definitions in `Matcher/`.

## Expected Downloaded Dataset Tree

```text
matcher/
└── Datasets/
		├── ESC-50/
		│   ├── audio/
		│   └── meta/esc50.csv
		├── DESED/
		│   └── data/dataset/
		│       ├── audio/train/weak/
		│       └── metadata/train/weak.tsv
		├── DECASE2016/
		│   └── dcase2016_task2_train_dev/dcase2016_task2_train/
		├── Urban8k/
		│   ├── audio/fold1 ... fold10
		│   └── metadata/UrbanSound8K.csv
		├── VocalSound/
		│   └── audio_16k/
		└── USM/
				├── train/
				├── eval/
				└── val/
```

These names must align with `configs/datasets.json` and preprocessing scripts.

## Required Matcher Files

`Matcher/` should include, at minimum:

- `FBC.csv`: foreground to candidate backgrounds mapping.
- `FBC.json`: optional per-foreground override for primary spurious background.

OOD generation may also use a variant JSON (for example `FBC_clean_queries.json`).

## Preprocessed CSV Requirements

`utils.read_datasets_into_foreground_background()` expects per-dataset CSV files under `Preprocessed_CSVs/<dataset_folder>/`. (We already provide them)

Accepted patterns:

- Split files:
	- `*foreground*.csv` with columns like `audio_path`, `category`
	- `*background*.csv` with columns like `audio_path`, `category`
- Combined file:
	- `*all_dataset*.csv` with columns including:
		- file path column: `file_path` or `audio_path`
		- labels: `foreground_label`, `background_label`, `category`

Current repository already uses folders similar to:

```text
Preprocessed_CSVs/
├── decase2016_csvs/
├── desed_csv/
├── esc50_csvs/
├── urban8k_csvs/
└── vocalsound_csvs/
```

## End-to-End Pipeline

### 1) Build/Refresh Preprocessed CSVs

Run all preprocessing scripts:

```bash
bash dataset_preprocessing/run_all.sh
```

or run one by one:

```bash
python dataset_preprocessing/preprocess_desed.py
python dataset_preprocessing/preprocess_vocalsound.py
python dataset_preprocessing/preprocess_urban8k.py
python dataset_preprocessing/preprocess_esc50.py
python dataset_preprocessing/preprocess_decase.py
python dataset_preprocessing/update_csvs.py
```

Important: several preprocessing scripts contain legacy absolute paths under `/root/SC/...` in their example blocks. Update those paths for your machine if needed.

### 2) Normalize CSV Audio Paths (If Needed)

If generated CSVs still contain `/root/SC/Datasets/...`, run:

```bash
python scripts/update_csv_paths.py
```

This rewrites CSV paths to your local `matcher/Datasets` directory.

### 3) Configure Mixer Inputs

Primary config file: `configs/merger.yaml`

Default example in this repo:

- `Datasets.file_name: configs/datasets.json`
- `Datasets.dataset_target_name: Mixed-dataset`
- `Alterations.combination_coffiecents: auto`
- `Alterations.loop_background: Yes`
- `Alterations.policy: 1:1`

You can either edit YAML or pass flags directly.

### 4) Run Main Mixer

Typical run (explicit args):

```bash
python main.py --config ./configs/merger.yaml
```

Output location:

```text
Results/<dataset_target_name>/
├── Clean/
├── With_loop/
├── Without_loop/
├── meta.csv
└── meta_debug.csv
```


### 5) Convert Mixed WAVs to Spectrogram `.npy` (Optional for Training)

The conversion stack script is `Results/full_stack_ESC.py`.

Before running, set `CURR_SET` inside that script to your generated dataset folder name, then run:

```bash
python Results/full_stack_ESC.py
```

This creates:

- sorted class folders (`Sorted/`),
- normalized waveform `.npy` tree (`Sorted_npy/`),
- mel-spectrogram `.npy` dataset (`<dataset>_spec`).

## Typical File Tree After Full Pipeline

```text
matcher/
├── Datasets/
│   └── ... raw downloaded data ...
├── Preprocessed_CSVs/
│   ├── esc50_csvs/
│   ├── desed_csv/
│   ├── urban8k_csvs/
│   ├── vocalsound_csvs/
│   └── decase2016_csvs/
├── Matcher/
│   ├── FBC.csv
│   └── FBC.json
└── Results/
		├── Mixed-datasets/
		│   ├── Clean/
		│   ├── With_loop/
		│   ├── Without_loop/
		│   ├── meta.csv
		│   └── meta_debug.csv
		├── Mixed-datasets_spec/
		│   └── <class_name>/*.npy
		└── Mixed-datasets_secondary_OOD/
				├── foreground_queries/
				└── background_queries/
```

## Common Pitfalls

1. Missing columns in CSV files.
Use the expected columns listed above (`audio_path` or `file_path`, and category/foreground/background labels).

2. Legacy absolute paths (`/root/SC/...`).
Adjust script constants or run `python scripts/update_csv_paths.py`.

3. Matcher file confusion.
`main.py` currently reads `Matcher/FBC.csv` and `Matcher/FBC.json` directly in code. Keep those files up to date with the mapping you want.

4. Dataset naming mismatch.
Ensure dataset names in `configs/datasets.json` correspond to folders under `Preprocessed_CSVs/`.

5. Audio loading failures.
Validate paths in generated CSVs and verify the source dataset folders exist.


## Citation and Licenses

- Respect licenses of all upstream datasets (ESC-50, DESED, UrbanSound8K, etc.).
- Respect the parent `Audio-Fewshot` project license and dataset usage terms.