# Distribution Aligned Semantics Adaption for Lifelong Person Re-Identification
> Official PyTorch implementation of ["Distribution Aligned Semantics Adaption for Lifelong Person Re-Identification"](https://arxiv.org/abs/2405.19695).
>
> Accepted by Machine Learning, Long Oral presentation at ACML 2024
>
> Qizao Wang, Xuelin Qian, Bin Li, Xiangyang Xue
>
> Fudan University, Northwestern Polytechnical University


## Getting Started

### Environment

- Python == 3.7.16
- PyTorch == 1.13.1

### Preparation

1) The dataset preparation details are provided in Sec. 4.1 of our [paper](https://arxiv.org/abs/2405.19695).
Please download person re-identification datasets and place them in any path `DATASET_ROOT`:

```
DATASET_ROOT
    │── market1501
    │   │── bounding_box_train
    │   │── bounding_box_test
    │   └── query
    │── DukeMTMC-reID
    │   │── bounding_box_train
    │   │── bounding_box_test
    │   └── query
    │── cuhksysu4reid
    │   │── train
    │   │── gallery
    │   └── query
    │── MSMT17
    │   │── train
    │   │── test
    │   └── ...
    └── viper
        │── cam_a
        │── cam_b 
        └── splits.json
```

You can prepare "cuhksysu4reid" following the instructions in [AKA](https://github.com/TPCD/LifelongReID/) and split "viper" using "[splits.json](https://github.com/cly234/LReID-KRKC/blob/main/docs/splits.json)".

2) Please download the LUPerson-NL pre-trained weight file "[lupws_r50.pth](https://github.com/DengpanFu/LUPerson-NL)" and place it in the project directory.

### Training

```sh
# Order 1
CUDA_VISIBLE_DEVICES=0 python continual_train.py --training-order=1 --data-dir=DATASET_ROOT --logs-dir=LOG_DIR

# Order 2
CUDA_VISIBLE_DEVICES=0 python continual_train.py --training-order=2 --data-dir=DATASET_ROOT --logs-dir=LOG_DIR
```

`--data_dir` : replace `DATASET_ROOT` with your dataset root path

`--logs_dir`: replace `LOG_DIR` with the path to save log file and, if using `--save_checkpoint`, the checkpoints

For implementation simplicity, we save all model parameters instead of just *BN<sup>(t)</sup>* and *SA<sup>(t)</sup>*. 
However, for resource efficiency, it is recommended to follow our paper and save only the shared parameters and domain-specific extra parameters.

### Evaluation

Evaluation is conducted during training. It is worth noting that in our DASA framework, the performance on the previously learned domains remains unchanged throughout the lifelong learning process.

### Results

- Training order: Market-1501 &rarr; DukeMTMC-reID &rarr; CUHK-SYSU &rarr; MSMT17

<table>
  <tr>
    <th colspan="2">Market-1501</th>
    <th colspan="2">DukeMTMC</th>
    <th colspan="2">CUHK-SYSU</th>
    <th colspan="2">MSMT17</th>
    <th colspan="2">Average</th>
  </tr>
  <tr>
    <td>mAP</td>
    <td>R-1</td>
    <td>mAP</td>
    <td>R-1</td>
    <td>mAP</td>
    <td>R-1</td>
    <td>mAP</td>
    <td>R-1</td>
    <td>mAP</td>
    <td>R-1</td>
  </tr>
  <tr>
    <td>86.2</td>
    <td>94.6</td>
    <td>77.3</td>
    <td>87.6</td>
    <td>93.8</td>
    <td>94.7</td>
    <td>49.3</td>
    <td>73.6</td>
    <td>76.7</td>
    <td>87.6</td>
  </tr>
</table>

More results can be found in our [paper](https://arxiv.org/abs/2405.19695). You can achieve similar results with the released code.

## Citation

Please cite the following paper in your publications if it helps your research:

```
@article{wang2025distribution,
  title={Distribution aligned semantics adaption for lifelong person re-identification},
  author={Wang, Qizao and Qian, Xuelin and Li, Bin and Xue, Xiangyang},
  journal={Machine Learning},
  volume={114},
  number={3},
  pages={1--22},
  year={2025},
  publisher={Springer}
}
```


## Contact

Any questions or discussions are welcome!

Qizao Wang (<qzwang22@m.fudan.edu.cn>)
