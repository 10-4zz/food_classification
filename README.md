# A deep learning project for Food Classification

[![Python >= 3.9](https://img.shields.io/badge/python->=3.9-blue.svg)](https://www.python.org/downloads/release/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit/)

This project implements a food classification model using deep learning techniques. The model is trained to classify images of food into various categories.

It contains the following models:
- **MobileViTv2**: A lightweight model designed for mobile devices, optimized for speed and efficiency.
- **EHFR-Net**: A model that combines Inverted Residual Blocks and LP-ViT for enhanced feature extraction and classification performance in food classification task.
- **AF-Net**: A model that uses the aggregate operation and combines the shuffle module form GS-Net to improve classification accuracy for food classification task.

## Installation

Get the code:

```bash
git clone https://github.com/10-4zz/food_classification.git
cd food_classification
```

Create environment:

```bash
conda create -n food_classification python=3.9
conda activate food_classification
```

Install torch package:

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```
We use CUDA 11.8, if you want to use other versions of CUDA, please refer to the official website: https://pytorch.org/get-started/locally/.

Install other dependencies:

```bash
pip install -r requirements.txt
```

## Paper Cite
If you find this project useful in your research, please consider citing:
MobileViTv2:
```
@article{mehta2022separable,
  title={Separable self-attention for mobile vision transformers},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2206.02680},
  year={2022}
}
```

EHFR-Net:
```
@article{sheng2024lightweight,
  title={A lightweight hybrid model with location-preserving ViT for efficient food recognition},
  author={Sheng, Guorui and Min, Weiqing and Zhu, Xiangyi and Xu, Liang and Sun, Qingshuo and Yang, Yancun and Wang, Lili and Jiang, Shuqiang},
  journal={Nutrients},
  volume={16},
  number={2},
  pages={200},
  year={2024},
  publisher={MDPI}
}
```

AF-Net:
```
@article{yang2024lightweight,
  title={Lightweight food recognition via aggregation block and feature encoding},
  author={Yang, Yancun and Min, Weiqing and Song, Jingru and Sheng, Guorui and Wang, Lili and Jiang, Shuqiang},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  volume={20},
  number={10},
  pages={1--25},
  year={2024},
  publisher={ACM New York, NY}
}
```

## License

Food Classification is MIT licensed. See the [LICENSE](LICENSE) for details.