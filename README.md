
## StarGAN v2 - Official PyTorch Implementation

<p align="left"><img width="95%" src="assets/teaser.jpg" /></p>

> **StarGAN v2: Diverse Image Synthesis for Multiple Domains**<br>
> [Yunjey Choi](https://github.com/yunjey)\*, [Youngjung Uh](https://github.com/youngjung)\*, [Jaejun Yoo](http://jaejunyoo.blogspot.com/search/label/kr)\*, [Jung-Woo Ha](https://www.facebook.com/jungwoo.ha.921)<br>
> In CVPR 2020. (* indicates equal contribution)<br>

> Paper: https://arxiv.org/abs/1912.01865<br>
> Video: https://youtu.be/0EVh5Ki4dIY<br>

> **Abstract:** *A good image-to-image translation model should learn a mapping between different visual domains while satisfying the following properties: 1) diversity of generated images and 2) scalability over multiple domains. Existing methods address either of the issues, having limited diversity or multiple models for all domains. We propose StarGAN v2, a single framework that tackles both and shows significantly improved results over the baselines. Experiments on CelebA-HQ and a new animal faces dataset (AFHQ) validate our superiority in terms of visual quality, diversity, and scalability. To better assess image-to-image translation models, we release AFHQ, high-quality animal faces with large inter- and intra-domain variations. The code, pre-trained models, and dataset are available at clovaai/stargan-v2.*

## TensorFlow implementation
The TensorFlow implementation of StarGAN v2 by our team member junho can be found at [clovaai/stargan-v2-tensorflow](https://github.com/clovaai/stargan-v2-tensorflow).

Install the dependencies:
```bash
conda create -n stargan-v2 python=3.6.7
conda activate stargan-v2
conda install -y pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
pip install opencv-python==4.1.2.30 ffmpeg-python==0.2.0 scikit-image==0.16.2
pip install pillow==7.0.0 scipy==1.2.1 tqdm==4.43.0 munch==2.5.0
```

## Datasets and pre-trained networks
We provide a script to download datasets used in StarGAN v2 and the corresponding pre-trained networks. The datasets and network checkpoints will be downloaded and stored in the `data` and `expr/checkpoints` directories, respectively.

<b>CelebA-HQ.</b> To download the [CelebA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs) dataset and the pre-trained network, run the following commands:
```bash
bash download.sh wing
```

## Training networks
To train StarGAN v2 from scratch, run the following commands. Generated images and network checkpoints will be stored in the `expr/samples` and `expr/checkpoints` directories, respectively. Training takes about three days on a single Tesla V100 GPU. Please see [here](https://github.com/clovaai/stargan-v2/blob/master/main.py#L86-L179) for training arguments and a description of them. 


```bash
python main.py
```


Feat : 새로운 기능에 대한 커밋
Fix : 버그 수정에 대한 커밋
Build : 빌드 관련 파일 수정에 대한 커밋
Chore : 그 외 자잘한 수정에 대한 커밋
Ci : CI관련 설정 수정에 대한 커밋
Docs : 문서 수정에 대한 커밋
Style : 코드 스타일 혹은 포맷 등에 관한 커밋
Refactor : 코드 리팩토링에 대한 커밋
Test : 테스트 코드 수정에 대한 커밋