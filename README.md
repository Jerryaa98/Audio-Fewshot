# SpurAudio: An Audio Few-Shot Classification Library

Official implementation for:

**SpurAudio: A Benchmark for Studying Shortcut Learning in Few-Shot Audio Classification**

Authors: Anonymous (under review)

![SpurAudio logo](images/logo.png)

![IID vs OOD illustration](images/illustration_iid_vs_ood.png)

---

## Overview

This repository extends LibFewShot for few-shot **audio** classification experiments on SpurAudio, with support for:

- IID and OOD episode evaluation.
- Multiple few-shot paradigms (fine-tuning, meta-learning, metric-learning, transductive).
- Backbone and classifier modular configurations.

---

## SpurAudio Dataset Layout

### Top-level directory layout

```
LibFewShot/
├── SpurAudio_dataset/            # per-class spectrogram tensors (.npy)
│   ├── air_conditioner/
│   ├── blender/
│   ├── cat/
│   ├── dog+dog_bark/
│   ├── …                         # 38 class folders total
│   └── laughter/
├── Auxiliary/                    # normalization stats + class-split definition
│   ├── Clean_Mean_Std.npy
│   ├── Spurious_Mean_Std.npy
│   └── SpurAudio_paper_splits.npy
├── data/
│   └── fewshot/
├── config/                       # YAML configs
├── libfewshot_core/              # framework source
└── results/                      # training outputs
```

### Per-class file naming convention

Each `.npy` inside a class folder is a spectrogram tensor for one sample. Filenames encode the mixing of two classes:

```
SpurAudio_dataset/cat/cat-crying_baby_alpha=0.00013126751582603902_loop=1_id=37.npy
                       │   │           │                                  │       │
                       │   │           │                                  │       └── sample id
                       │   │           │                                  └── loop index
                       │   │           └── mixing coefficient (α) between foreground and background
                       │   └── background class
                       └── foreground class
```

The directory the file lives in is the **foreground** class (the label); the second class in the filename is the **background** spurious co-occurrence.

### Class splits file

`Auxiliary/SpurAudio_paper_splits.npy` is a NumPy array (loaded with `np.load(..., allow_pickle=True)`) whose three top-level entries are the class-name lists for each split:

- index `0` → list of **train** class names
- index `1` → list of **validation** class names
- index `2` → list of **test** class names

Each entry is a list of strings matching subfolder names under `SpurAudio_dataset/` (e.g. `"cat"`, `"dog+dog_bark"`). To replicate the paper splits, leave this file untouched; to define your own splits, write a 3-element object array in the same `[train, val, test]` order and point `class_per_split` in your YAML at it.

### Downloading the dataset

In the meantime, you can download the SpurAudio dataset (with its splits) from the anonymous Hugging Face repository: [`spuraudioNips/SpurAudio-neurips-anonym`](https://huggingface.co/datasets/spuraudioNips/SpurAudio-neurips-anonym).

To run the pipeline on it:

1. Unpack all audio files into a single flat directory, e.g. `all_audios_unpacked/`.
2. Run the ESC-50 full-stack script on that folder to produce the `SpurAudio_dataset/` spectrogram layout shown above.

### Generating the dataset (and using a custom one)

The SpurAudio spectrograms are produced from ESC-50 by the ESC-50 full-stack script. If you want to use a different audio dataset, run that same script on your dataset to produce a `SpurAudio_dataset/`-shaped folder — i.e. one subfolder per foreground class, with `.npy` filenames following the `{foreground}-{background}_alpha={float}_loop={int}_id={int}.npy` schema shown above. The rest of the pipeline (configs, dataloaders, normalization) expects exactly this format.

---

## Supported Methods

### Non-episodic (fine-tuning based)
- [Baseline (ICLR 2019)](https://arxiv.org/abs/1904.04232)
- [Baseline++ (ICLR 2019)](https://arxiv.org/abs/1904.04232)
- [Meta-Baseline (ICCV 2021)](https://arxiv.org/abs/2003.04390)
- [DiffKendall (NeurIPS 2023)](https://arxiv.org/abs/2307.15317)

### Meta-learning based
- [MAML (ICML 2017)](https://arxiv.org/abs/1703.03400)
- [Versa (NeurIPS 2018)](https://openreview.net/forum?id=HkxStoC5F7)
- [R2D2 (ICLR 2019)](https://arxiv.org/abs/1805.08136)
- [LEO (ICLR 2019)](https://arxiv.org/abs/1807.05960)
- [MTL (CVPR 2019)](https://arxiv.org/abs/1812.02391)
- [ANIL (ICLR 2020)](https://arxiv.org/abs/1909.09157)
- [BOIL (ICLR 2021)](https://arxiv.org/abs/2008.08882)
- [MeTAL (ICCV 2021)](https://arxiv.org/abs/2110.03909)

### Metric-learning based
- [ProtoNet (NeurIPS 2017)](https://arxiv.org/abs/1703.05175)
- [RelationNet (CVPR 2018)](https://arxiv.org/abs/1711.06025)
- [DN4 (CVPR 2019)](https://arxiv.org/abs/1903.12290)
- [ATL-Net (IJCAI 2020)](https://www.ijcai.org/proceedings/2020/0100.pdf)
- [ADM (IJCAI 2020)](https://arxiv.org/abs/2002.00153)
- [FRN (CVPR 2021)](https://arxiv.org/abs/2012.01506)
- [DeepBDC (CVPR 2022)](https://arxiv.org/abs/2204.04567)
- [MCL (CVPR 2022)](http://openaccess.thecvf.com/content/CVPR2022/html/Liu_Learning_To_Affiliate_Mutual_Centralized_Learning_for_Few-Shot_Classification_CVPR_2022_paper.html)

### Transductive Methods
- [LaplacianShot (2020)](https://arxiv.org/abs/2006.15486)
- [BDCSPN (2020)](https://arxiv.org/pdf/1911.10713)
- [PADDLE (2022)](https://arxiv.org/abs/2210.14545)
- [Proto-LP (2023)](https://arxiv.org/abs/1703.05175)
- [BPA (2024)](https://arxiv.org/abs/2407.01467)
- [ECPE (2026)](https://www.sciencedirect.com/science/article/abs/pii/S0167865526000231)

---

## Reproducibility Guide

### 1) Environment setup

```bash
cd /path/to/LibFewShot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Dataset and metadata preparation

Make sure these assets exist and match your local paths:

- `SpurAudio_dataset/` (or your configured data root).
- `Auxiliary/Spurious_Mean_Std.npy` (or selected mean/std file).
- `Auxiliary/KOS_paper_splits.npy` (class split definition).

If your machine paths differ, update the relevant YAML entries such as `data_root`, `mean_std_file`, and `class_per_split`.

### 3) Choose or create a YAML config

Configs are under `config/` (examples: `config/proto_5shot_iid.yaml`, `config/maml_5shot_ood.yaml`).

Run training with:

```bash
python run_trainer.py --yaml_path ./config/proto_5shot_iid.yaml # or any other yaml file
```

### 4) IID vs OOD evaluation toggle (important)

For IID/OOD experiments, set the YAML key **`ood`** explicitly:

- `ood: False`  → IID evaluation
- `ood: True`   → OOD evaluation

Example:

```yaml
ood: False  # IID
# ood: True # OOD
```

> Note: the key in this codebase is lowercase `ood`.

### 5) Test a trained experiment

`run_test.py` currently loads a result directory via the `PATH` variable inside the script.

1. Open `run_test.py`.
2. Set `PATH` to your trained run directory under `results/`.
3. Run:

```bash
python run_test.py --yaml_path ./config/proto_5shot_iid.yaml # or any other yaml file
```
---

## Quick Run Examples

### ProtoNet 5-shot IID

```bash
python run_trainer.py --yaml_path ./config/proto_5shot_iid.yaml
```

### ProtoNet 5-shot OOD

```bash
python run_trainer.py --yaml_path ./config/proto_5shot_ood.yaml
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgement

LibFewShot is an open-source framework for few-shot learning research. Contributions, feedback, and extensions are welcome.

## Citation

If you use this repository in your research, please cite the corresponding SpurAudio/LibFewShot paper(s).



