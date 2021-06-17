# Self-Ensembling Attention Networks: Addressing Domain Shift for Semantic Segmentation

Pytorch implementation of our method for domain adaptation in semantic segmentation task.


![](Figure/Framework.jpg)

## Paper
[Self-Ensembling Attention Networks: Addressing Domain Shift for Semantic Segmentation](https://m.aaai.org/ojs/index.php/AAAI/article/view/4500)

Please cite our paper if you find it useful for your research.

```
@inproceedings{SEAN,
  title={Self-Ensembling Attention Networks: Addressing Domain Shift for Semantic Segmentation},
  author={Xu, Yonghao and Du, Bo and Zhang, Lefei and Zhang, Qian and Wang, Guoli and Zhang, Liangpei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={5581--5588},
  year={2019}
}
```

## Installation
* Install `Pytorch 0.4.0` from https://github.com/pytorch/pytorch with `Python 3.6`.

* Clone this repo.
```
git clone https://github.com/YonghaoXu/SEANet
```

## Dataset
* Download the [GTA-5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/).

* Download the [SYNTHIA-RAND-CITYSCAPES Dataset](http://synthia-dataset.net/download/808/).
  - Note: The ground-truth data in the original SYNTHIA-RAND-CITYSCAPES dataset should be adjusted to be consistent with those in the cityscapes dataset. Here we attach the transformed [ground-truth data](https://drive.google.com/open?id=1GvdXSG4nq8Px0xYs3ate0reNNKtci2dS) for the SYNTHIA-RAND-CITYSCAPES dataset.

* Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com/).

## Pretrained Model
* Download the pretrained [VGG-16 Model](https://drive.google.com/file/d/1PGuOb-ZIOc10aMGOxj5xFSubi8mkVXaq/view?usp=sharing).

## Training
* Training for GTA-5 to Cityscapes.
  - Change the default path of `--data_dir_source` in `SEAN_GTA5.py` with your GTA-5 dataset folder.
  - Change the default path of `--data_dir_target` in `SEAN_GTA5.py` with your Cityscapes dataset folder.
  - Change the default path of `--restore_from` in `SEAN_GTA5.py` with your pretrained VGG model path.
  - Refer to `dataset/gta5_dataset.py` and `dataset/cityscapes_dataset.py` for further guidance about how the images and ground-truth files are organized.

```
python SEAN_GTA5.py
```

* Training for Synthia to Cityscapes.
  - Change the default path of `--data_dir_source` in `SEAN_Synthia.py` with your Synthia dataset folder.
  - Change the default path of `--data_dir_target` in `SEAN_Synthia.py` with your Cityscapes dataset folder.
  - Change the default path of `--restore_from` in `SEAN_Synthia.py` with your pretrained VGG model path.
  - Refer to `dataset/synthia_dataset.py` and `dataset/cityscapes16_dataset.py` for further guidance about how the images and ground-truth files are organized.

```
python SEAN_Synthia.py
```  

## Evaluation
* Test for GTA-5 to Cityscapes.
  - Change the default path of `--data_dir` in `evaluation.py` with your Cityscapes dataset folder.
  - Change the default path of `--restore_from` in `evaluation.py` with your trained model path. You can also download our [GTA-5 to Cityscapes model](https://drive.google.com/open?id=1g-NSAaHxkvru4G0lBNolmcioH8elCoqo) for a look.

```
python evaluation.py
```

* Test for Synthia to Cityscapes.
  - For evaluation on Synthia to Cityscapes case, please replace the `test_mIoU` function in `evaluation.py` with the `test_mIoU16` function. Since there are only 16 categories in common in this case, the code for writing the segmentation maps parts needs to be further modified. If you want to share your implementation for this issue, please pull a request.


## Empirical Observations 
* Following the previous research setting in this task, we check the mIoU value on the target domain after every 500 iterations. A lower frequency for the checking would accelerate the network training, but may also miss the best performance.
* A large `--attention_threshold` would be detrimental to the performance of the framework. Empirically, 0 to 0.3 is a suitable range for this parameter.
* Best performance is usually obtained within 6 epochs. For the GTA-5 to Cityscapes case, the mIoU can reach about 34% to 35%. For the Synthia to Cityscapes case, the mIoU can reach about 36% to 37%.

## Multi-GPU Training
* This repo is tested with a batch size of 1 using a single GPU. For a larger batch size with multi-GPU training, the code may need to be modified. If you want to share your implementation for this issue, please pull a request.
