# FewX

**FewX** is an open source toolbox on top of Detectron2 for data-limited instance-level recognition tasks, e.g., few-shot object detection, few-shot instance segmentation, partially supervised instance segmentation and so on. 

All data-limited instance-level recognition works from **Qi Fan**  (HKUST, qfanaa@connect.ust.hk) are open-sourced here.

To date, FewX implements the following algorithms:

- [FSOD](https://arxiv.org/abs/1908.01998): few-shot object detection with [FSOD dataset](https://github.com/fanq15/Few-Shot-Object-Detection-Dataset).
- [CPMask](https://arxiv.org/abs/2007.12387): partially supervised/fully supervised/few-shot instance segmentation (to be released).
- [FSVOD](https://arxiv.org/abs/2104.14805): few-shot video object detection with [FSVOD-500 dataset](https://drive.google.com/drive/folders/1DDQ81A8yVj7D8vLUS01657ATr2sK1zgC?usp=sharing) and [FSYTV-40 dataset](https://drive.google.com/drive/folders/1a1PpfAxeYL7AbxYViDDnx7ACFtRohVL5?usp=sharing).

## Highlights
- **State-of-the-art performance.**  
  - FSOD is the best few-shot object detection model. (This model can be directly applied to novel classes without finetuning. And finetuning can bring better performance.)
  - CPMask is the best partially supervised/few-shot instance segmentation model.
- **Easy to use.** You only need to run 3 code lines to conduct the entire experiment.
  - Install Pre-Built Detectron2 in one code line.
  - Prepare dataset in one code line. (You need to first download the dataset and change the **data path** in the script.)
  - Training and evaluation in one code line.

## Updates
- FewX has been released. (09/08/2020)

## Results on MS COCO

### Few Shot Object Detection

|Method|Training Dataset|Evaluation way&shot|box AP|download|
|:--------:|:--------:|:--------:|:--------:|:--:|
|FSOD (paper)|COCO (non-voc)|full-way 10-shot|11.1|-|
|FSOD (this implementation)|COCO (non-voc)|full-way 10-shot|**12.0**|<a href="https://drive.google.com/file/d/1VO1XMKtiU4pMNPfIvw5iZRqlO9dr5BhN/view?usp=sharing">model</a>&nbsp;\|&nbsp;<a href="https://drive.google.com/file/d/18eC5Nn1HBJcDf75CoLWOwncYFXzHGXFD/view?usp=sharing">metrics</a>|

The results are reported on the COCO voc subset with **ResNet-50** backbone.

The model only trained on base classes is <a href="https://drive.google.com/file/d/1VdGVmcufa2JBmZUfwAcDj1OL5tKTFhQ1/view?usp=sharing"> base model</a>&nbsp;\.

You can reference the [original FSOD implementation](https://github.com/fanq15/FSOD-code) on the [Few-Shot-Object-Detection-Dataset](https://github.com/fanq15/Few-Shot-Object-Detection-Dataset).

## Step 1: Installation
You only need to install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). We recommend the Pre-Built Detectron2 (Linux only) version with pytorch 1.7. I use the Pre-Built Detectron2 with CUDA 10.1 and pytorch 1.7 and you can run this code to install it.

```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
```

## Step 2: Prepare dataset
- Prepare for coco dataset following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

- `cd datasets`, change the `DATA_ROOT` in the `generate_support_data.sh` to your data path and run `sh generate_support_data.sh`.

``` 
cd FewX/datasets
sh generate_support_data.sh
```

## Step 3: Training and Evaluation

Run `sh all.sh` in the root dir. (This script uses `4 GPUs`. You can change the GPU number. If you use 2 GPUs with unchanged batch size (8), please [halve the learning rate](https://github.com/fanq15/FewX/issues/6#issuecomment-674367388).)

```
cd FewX
sh all.sh
```


## TODO
 - [ ] Add other dataset results to FSOD.
 - [ ] Add [CPMask](https://arxiv.org/abs/2007.12387) code with partially supervised instance segmentation, fully supervised instance segmentation and few-shot instance segmentation.

## Citing FewX
If you use this toolbox in your research or wish to refer to the baseline results, please use the following BibTeX entries.

  ```
  @inproceedings{fan2021fsvod,
    title={Few-Shot Video Object Detection},
    author={Fan, Qi and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={arxiv},
    year={2021}
  }
  @inproceedings{fan2020cpmask,
    title={Commonality-Parsing Network across Shape and Appearance for Partially Supervised Instance Segmentation},
    author={Fan, Qi and Ke, Lei and Pei, Wenjie and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={ECCV},
    year={2020}
  }
  @inproceedings{fan2020fsod,
    title={Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector},
    author={Fan, Qi and Zhuo, Wei and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={CVPR},
    year={2020}
  }
  ```

## Special Thanks
[Detectron2](https://github.com/facebookresearch/detectron2), [AdelaiDet](https://github.com/aim-uofa/AdelaiDet), [centermask2](https://github.com/youngwanLEE/centermask2)
