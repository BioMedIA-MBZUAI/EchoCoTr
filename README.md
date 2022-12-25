# EchoCoTr: Estimation of the Left Ventricular Ejection Fraction from Spatiotemporal Echocardiography
**Authors:** Rand Muhtaseb, Mohammad Yaqub

**Institution:** Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI)

[arXiv](https://arxiv.org/abs/2209.04242) | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_36)

## :page_facing_up: Abstract
Learning spatiotemporal features is an important task for efficient video understanding especially in medical images such as echocardiograms. Convolutional neural networks (CNNs) and more recent vision transformers (ViTs) are the most commonly used methods with limitations per each. CNNs are good at capturing local context but fail to learn global information across video frames. On the other hand, vision transformers can incorporate global details and long sequences but are computationally expensive and typically require more data to train. In this paper, we propose a method that addresses the limitations we typically face when training on medical video data such as echocardiographic scans. The algorithm we propose (EchoCoTr) utilizes the strength of vision transformers and CNNs to tackle the problem of estimating the left ventricular ejection fraction (LVEF) on ultrasound videos. We demonstrate how the proposed method outperforms state-of-the-art work to-date on the EchoNet-Dynamic dataset with MAE of 3.95 and 𝑅2 of 0.82. These results show noticeable improvement compared to all published research. In addition, we show extensive ablations and comparisons with several algorithms, including ViT and BERT.

This work has been accepted at [MICCAI 2022](https://conferences.miccai.org/2022/en/).
### Keywords
Transformers, Deep learning, Echocardiography, Ejection fraction, Heart failure

## Data
You need to request an access to download the [EchoNet-Dynamic dataset](https://echonet.github.io/dynamic/index.html#dataset)

You also need to download the [pretrained UniFormer models](https://huggingface.co/Sense-X/uniformer_video/tree/main)

## Requirements

You can install all requirements using `pip` by running this command:

``` pip install -r requirements.txt```

Generally speaking, our code uses the following core packages: 
- Python 3.8.12
- PyTorch 1.10.2
- [wandb](https://wandb.ai): you need to create an account for logging purposes

## Training/Testing

```
python main.py --exp_no [EXP_NO] --exp_name [EXP_NAME] \
                                 --model_name uniformer_small \
                                 --batch_size 25 \
                                 --epochs 45 \
                                 --pretrained True \
                                 --frames 36 \
                                 --frequency 4 \
                                 --data_dir [PATH_TO_DATASET_FOLDER] \
                                 --weights  [PATH_TO_PRETRAINED_MODEL]
```

## Citation

```
@InProceedings{10.1007/978-3-031-16440-8_36,
	author="Muhtaseb, Rand and Yaqub, Mohammad",
	editor="Wang, Linwei and Dou, Qi and Fletcher, P. Thomas and Speidel, Stefanie and Li, Shuo",
	title="EchoCoTr: Estimation of the Left Ventricular Ejection Fraction from Spatiotemporal Echocardiography",
	booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022",
	year="2022",
	publisher="Springer Nature Switzerland",
	address="Cham",
	pages="370--379",
	abstract="Learning spatiotemporal features is an important task for efficient video understanding especially in medical images such as echocardiograms. Convolutional neural networks (CNNs) and more recent vision transformers (ViTs) are the most commonly used methods with limitations per each. CNNs are good at capturing local context but fail to learn global information across video frames. On the other hand, vision transformers can incorporate global details and long sequences but are computationally expensive and typically require more data to train. In this paper, we propose a method that addresses the limitations we typically face when training on medical video data such as echocardiographic scans. The algorithm we propose (EchoCoTr) utilizes the strength of vision transformers and CNNs to tackle the problem of estimating the left ventricular ejection fraction (LVEF) on ultrasound videos. We demonstrate how the proposed method outperforms state-of-the-art work to-date on the EchoNet-Dynamic dataset with MAE of 3.95 and {\$}{\$}R^2{\$}{\$}R2of 0.82. These results show noticeable improvement compared to all published research. In addition, we show extensive ablations and comparisons with several algorithms, including ViT and BERT. The code is available at https://github.com/BioMedIA-MBZUAI/EchoCoTr.",
isbn="978-3-031-16440-8"
}

```

## Disclaimer
UniFormer models are taken from [this repository](https://huggingface.co/Sense-X/uniformer_video). Our implementation code is inspired by [EchoNet-Dynamic](https://github.com/echonet/dynamic) and [UVT](https://github.com/HReynaud/UVT).

## Questions?
For all code related questions, please create a GitHub Issue above and our team will respond to you as soon as possible.

