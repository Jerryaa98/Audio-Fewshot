# Large Model Frozen Evaluation Commands

Run these from the `LibFewShot` root.

For BEATs, download `BEATs_iter3_plus_AS2M.pt` from the official Microsoft UniLM BEATs page:

```text
https://github.com/microsoft/unilm/tree/master/beats
```

Place it at:

```text
large_models/BEATs_iter3_plus_AS2M.pt
```

## CLAP

```bash
python large_models/clap_hela_vfa_episode_eval.py --data-root /root/SC/Results/KOS_1_alpha/KOS_1_alpha_spec --sorted-root /root/SC/Results/KOS_1_alpha/Sorted --split-file /root/LibFewShot/analysis/KOS_paper_splits.npy --n-way 5 --k-shot 1 --q-query 10 --episodes 1000 --scenarios iid ood --algorithms hela_vfa proto protolp laplacian_shot bdcspn paddle ecpe --device auto --cache-path large_models/cache/clap_1shot_test_embeddings.npz --recompute-cache --output-csv large_models/results/clap_1shot_iid_ood_test_eval.csv

python large_models/clap_hela_vfa_episode_eval.py --data-root /root/SC/Results/KOS_1_alpha/KOS_1_alpha_spec --sorted-root /root/SC/Results/KOS_1_alpha/Sorted --split-file /root/LibFewShot/analysis/KOS_paper_splits.npy --n-way 5 --k-shot 5 --q-query 10 --episodes 1000 --scenarios iid ood --algorithms hela_vfa proto protolp laplacian_shot bdcspn paddle ecpe --device auto --cache-path large_models/cache/clap_5shot_test_embeddings.npz --recompute-cache --output-csv large_models/results/clap_5shot_iid_ood_test_eval.csv
```

## AudioMAE

```bash
python large_models/audiomae_episode_eval.py --data-root /root/SC/Results/KOS_1_alpha/KOS_1_alpha_spec --sorted-root /root/SC/Results/KOS_1_alpha/Sorted --split-file /root/LibFewShot/analysis/KOS_paper_splits.npy --n-way 5 --k-shot 1 --q-query 10 --episodes 1000 --scenarios iid ood --algorithms proto protolp laplacian_shot bdcspn paddle ecpe --device auto --cache-path large_models/cache/audiomae_1shot_test_embeddings.npz --recompute-cache --output-csv large_models/results/audiomae_1shot_iid_ood_test_eval.csv

python large_models/audiomae_episode_eval.py --data-root /root/SC/Results/KOS_1_alpha/KOS_1_alpha_spec --sorted-root /root/SC/Results/KOS_1_alpha/Sorted --split-file /root/LibFewShot/analysis/KOS_paper_splits.npy --n-way 5 --k-shot 5 --q-query 10 --episodes 1000 --scenarios iid ood --algorithms proto protolp laplacian_shot bdcspn paddle ecpe --device auto --cache-path large_models/cache/audiomae_5shot_test_embeddings.npz --recompute-cache --output-csv large_models/results/audiomae_5shot_iid_ood_test_eval.csv
```

## BEATs

```bash
python large_models/beats_episode_eval.py --data-root /root/SC/Results/KOS_1_alpha/KOS_1_alpha_spec --sorted-root /root/SC/Results/KOS_1_alpha/Sorted --split-file /root/LibFewShot/analysis/KOS_paper_splits.npy --beats-ckpt large_models/BEATs_iter3_plus_AS2M.pt --n-way 5 --k-shot 1 --q-query 10 --episodes 1000 --scenarios iid ood --algorithms proto protolp laplacian_shot bdcspn paddle ecpe --device auto --cache-path large_models/cache/beats_1shot_test_embeddings.npz --recompute-cache --output-csv large_models/results/beats_1shot_iid_ood_test_eval.csv

python large_models/beats_episode_eval.py --data-root /root/SC/Results/KOS_1_alpha/KOS_1_alpha_spec --sorted-root /root/SC/Results/KOS_1_alpha/Sorted --split-file /root/LibFewShot/analysis/KOS_paper_splits.npy --beats-ckpt large_models/BEATs_iter3_plus_AS2M.pt --n-way 5 --k-shot 5 --q-query 10 --episodes 1000 --scenarios iid ood --algorithms proto protolp laplacian_shot bdcspn paddle ecpe --device auto --cache-path large_models/cache/beats_5shot_test_embeddings.npz --recompute-cache --output-csv large_models/results/beats_5shot_iid_ood_test_eval.csv
```

## Qwen2-Audio

```bash
python large_models/qwen2audio_episode_eval.py --data-root /root/SC/Results/KOS_1_alpha/KOS_1_alpha_spec --sorted-root /root/SC/Results/KOS_1_alpha/Sorted --split-file /root/LibFewShot/analysis/KOS_paper_splits.npy --n-way 5 --k-shot 1 --q-query 10 --episodes 1000 --scenarios iid ood --algorithms proto protolp laplacian_shot bdcspn paddle ecpe --device auto --cache-path large_models/cache/qwen2audio_1shot_test_embeddings.npz --recompute-cache --output-csv large_models/results/qwen2audio_1shot_iid_ood_test_eval.csv

python large_models/qwen2audio_episode_eval.py --data-root /root/SC/Results/KOS_1_alpha/KOS_1_alpha_spec --sorted-root /root/SC/Results/KOS_1_alpha/Sorted --split-file /root/LibFewShot/analysis/KOS_paper_splits.npy --n-way 5 --k-shot 5 --q-query 10 --episodes 1000 --scenarios iid ood --algorithms proto protolp laplacian_shot bdcspn paddle ecpe --device auto --cache-path large_models/cache/qwen2audio_5shot_test_embeddings.npz --recompute-cache --output-csv large_models/results/qwen2audio_5shot_iid_ood_test_eval.csv
```

## AST

```bash
python large_models/ast_hela_vfa_episode_eval.py --data-root /root/SC/Results/KOS_1_alpha/KOS_1_alpha_spec --sorted-root /root/SC/Results/KOS_1_alpha/Sorted --split-file /root/LibFewShot/analysis/KOS_paper_splits.npy --n-way 5 --k-shot 1 --q-query 10 --episodes 1000 --scenarios iid ood --algorithms hela_vfa proto protolp laplacian_shot bdcspn paddle ecpe --device auto --batch-size 16 --output-csv large_models/results/ast_1shot_iid_ood_test_eval.csv

python large_models/ast_hela_vfa_episode_eval.py --data-root /root/SC/Results/KOS_1_alpha/KOS_1_alpha_spec --sorted-root /root/SC/Results/KOS_1_alpha/Sorted --split-file /root/LibFewShot/analysis/KOS_paper_splits.npy --n-way 5 --k-shot 5 --q-query 10 --episodes 1000 --scenarios iid ood --algorithms hela_vfa proto protolp laplacian_shot bdcspn paddle ecpe --device auto --batch-size 16 --output-csv large_models/results/ast_5shot_iid_ood_test_eval.csv
```
