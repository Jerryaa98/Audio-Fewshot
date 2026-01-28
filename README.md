# [LibFewShot](https://arxiv.org/abs/2109.04898)
Make few-shot learning easy.

<p align="center"><a href="https://libfewshot-en.readthedocs.io/en/latest/"><img src="./logo.png" alt="LibFewShot logo" width="40%"></a></p>


## Supported Methods
### Non-episodic methods (a.k.a Fine-tuning based methods)
+ [Baseline (ICLR 2019)](https://arxiv.org/abs/1904.04232)
+ [Baseline++ (ICLR 2019)](https://arxiv.org/abs/1904.04232)
+ [Meta-Baseline (ICCV 2021)](https://arxiv.org/abs/2003.04390)
+ [Diffkendall(NeurIPS 2023)](https://arxiv.org/abs/2307.15317)
### Meta-learning based methods
+ [MAML (ICML 2017)](https://arxiv.org/abs/1703.03400)
+ [Versa (NeurIPS 2018)](https://openreview.net/forum?id=HkxStoC5F7)
+ [R2D2 (ICLR 2019)](https://arxiv.org/abs/1805.08136)
+ [LEO (ICLR 2019)](https://arxiv.org/abs/1807.05960)
+ [MTL (CVPR 2019)](https://arxiv.org/abs/1812.02391)
+ [ANIL (ICLR 2020)](https://arxiv.org/abs/1909.09157)
+ [BOIL (ICLR 2021)](https://arxiv.org/abs/2008.08882)
+ [MeTAL (ICCV 2021)](https://arxiv.org/abs/2110.03909)
### Metric-learning based methods
+ [ProtoNet (NeurIPS 2017)](https://arxiv.org/abs/1703.05175)
+ [RelationNet (CVPR 2018)](https://arxiv.org/abs/1711.06025)
+ [DN4 (CVPR 2019)](https://arxiv.org/abs/1903.12290)
+ [ATL-Net (IJCAI 2020)](https://www.ijcai.org/proceedings/2020/0100.pdf)
+ [ADM (IJCAI 2020)](https://arxiv.org/abs/2002.00153)
+ [FRN (CVPR 2021)](https://arxiv.org/abs/2012.01506)
+ [DeepBDC (CVPR 2022)](https://arxiv.org/abs/2204.04567)
+ [MCL (CVPR 2022)](http://openaccess.thecvf.com/content/CVPR2022/html/Liu_Learning_To_Affiliate_Mutual_Centralized_Learning_for_Few-Shot_Classification_CVPR_2022_paper.html)


## How To Run

### Train 
```python
python run_trainer.py --yaml_path ./config/***yaml path to algorithm***
```

### Test
modify the PATH in run_test.py to the path of your pre-trained model then run:
```python
python run_test.py --yaml_path ./config/***yaml path to algorithm***
```

## License
This project is licensed under the MIT License. See LICENSE for more details.

## Acknowledgement
LibFewShot is an open source project designed to help few-shot learning researchers quickly understand the classic methods and code structures. We welcome other contributors to use this framework to implement their own or other impressive methods and add them to LibFewShot. This library can only be used for academic research. We welcome any feedback during using LibFewShot and will try our best to continually improve the library.

## Citation
If you use this code for your research, please cite our paper.



