<div align="center">
<h1>Bridging Adversarial and Collaborative Learning for AI-Generated Image Quality Assessment</h1>

<div>
    <a href='https://github.com/LQAMEI/ACL-IQA?tab=repositories' target='_blank'>Baoliang Chen</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Qing Lin</a><sup>2</sup>&emsp;
    <a href='' target='_blank'>Shijie Mai</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>South China Normal University, China&emsp;
    <sup>2</sup>Hong Kong University of Science and Technology (Guangzhou), China&emsp;
</div>
<div>
    <em>*denotes Corresponding author</em>
</div>

<div>
    <h4 align="center">
        • <a href="" target='_blank'>[arXiv]</a> • 
        <a href="https://acliqa-github-io.vercel.app/" target='_blank'>[Project Page]</a> •
    </h4>
</div>

<img src="assets/frame.jpg" width="1500px"/>

<strong>The proposed Adversarial and Collaborative  Learning based IQA model (ACL-IQA) provides Dual-Gated Mixture-of-Experts (DG-MoE) module that dynamically routes expert features through collaborative or adversarial paths within a unified transformer architecture. It jointly models adversarial and collaborative interactions, enabling complementary reasoning about perceptual fidelity and prompt alignment.</strong>

<div>
    If you find ACL-IQA useful for your projects, please consider ⭐ this repo. Thank you! 😉
</div>


---

</div>

## :postbox: Updates
<!-- - 2023.12.04: Add an option to speed up the inference process by adjusting the number of denoising steps. -->

- 2025.11.21: This repo is created. :yum:

## :diamonds: Installation

### Codes and Environment

```
# git clone this repository
git clone https://github.com/LQAMEI/ACL-IQA.git

cd ACL-IQA

# create new anaconda env
conda create -n acliqa python=3.8 -y
conda activate acliqa

# install python dependencies
pip install -r requirements.txt
```


## :circus_tent: Data Preparation

Download [AGIQA-1K](https://github.com/lcysyzxdxc/AGIQA-1k-Database), [AGIQA-3K](https://github.com/lcysyzxdxc/AGIQA-3k-Database) and [AIGCIQA2023](https://github.com/wangjiarui153/AIGCIQA2023) datasets and unzip them into the "<u>./data</u>" directory.

Prepare the training and testing CSV annotation files into the "<u>./Database</u>" directory.

**Configure Dataset Paths**  
Modify the dataset paths in `train_aigc_agiqa1k.py/train_aigc_agiqa3k_xx.py/train_aigc_aigciqa2023_xx.py`:

- `aigc_set`: List of root directories for each dataset.
- `aigc_train_csv/aigc_val_csv`: Mapping from dataset names to their corresponding CSV annotation files (containing image paths and ground-truth quality scores).

## :zap: Training and Testing

After preparing the code environment and downloading the data, run the following codes to train and test model.

```
# AGIQA-1K
python train_aigc_agiqa1k.py

# AGIQA-3K-perception
python train_aigc_agiqa3k_perception.py

# AGIQA-3K-alignment
python train_aigc_aigciqa2023_align.py

# AIGCIQA2023-perception
python train_aigc_aigciqa2023_perception.py

# AIGCIQA2023-alignment
python train_aigc_aigciqa2023_align.py

# AIGCIQA2023-authenticity
python train_aigc_aigciqa2023_authenticity.py
```

## :love_you_gesture: Citation
If you find our work useful for your research, please consider citing the paper:
```

```

### Contact
If you have any questions, please feel free to reach out at `2803951059@qq.com, blchen@m.scnu.edu.cn`. 