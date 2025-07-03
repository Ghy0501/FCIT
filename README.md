# 🎸 Federated Continual Instruction Tuning [ICCV 2025]

[![🤗 Dataset (HuggingFace)](https://huggingface.co/datasets/MLLM-CL/FCIT)  [![📑 Paper (arXiv:2503.12897)](https://img.shields.io/badge/arXiv-2503.12941-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2503.12897)

This repo is the official implementation of ICCV 2025 paper: **[Federated Continual Instruction Tuning](https://arxiv.org/pdf/2503.12897)**

> Federated Continual Instruction Tuning
>
> Haiyang Guo, Fanhu Zeng, Fei Zhu, Wenzhuo Liu, Da-Han Wang, Jian Xu, Xu-Yao Zhang, Cheng-Lin Liu

## Installation

The installation of our environment is the same as [CoIN](https://github.com/zackschen/CoIN) and [HiDe-LLaVA](https://github.com/Ghy0501/HiDe-LLaVA).

```bash
conda create -n FCIT python=3.10 -y
conda activate hide
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

To measure the metrics of caption tasks, please install the following three packages:

```bash
pip install nltk==3.9.1
pip install pycocotools==2.0.8
pip install pycocoevalcap==1.2
```
We recommend replacing the eval.py file under that path `/envs/FCIT/lib/python3.10/site-packages/pycocoevalcap/` in your environment with the eval.py file that we have provided in the repository to avoid unwanted error reporting and time overhead.


Technical issues can be reported and addressed through the official GitHub issue trackers for both projects: [CoIN](https://github.com/zackschen/CoIN) and [LLaVA](https://github.com/haotian-liu/LLaVA).

## FCIT Benchmark

Please download the images from the constituting dataset：

|Image Source | Download Path|
| :-: | :-: |
|ArxivQA|[images](https://huggingface.co/datasets/MMInstruction/ArxivQA/tree/main)|
|ImageNet-R|[images](https://huggingface.co/datasets/HaiyangGuo/UCIT/tree/main/UCIT/ImageNet-R)|
|IconQA|[images](https://iconqa.github.io/)|
|CLEVR-Math|[images](https://huggingface.co/datasets/dali-does/clevr-math/tree/main)|
|super-CLEVR|[images](https://github.com/Lizw14/Super-CLEVR)|
|Flickr30k|[images](https://huggingface.co/datasets/HaiyangGuo/UCIT/tree/main/UCIT/Flickr30k)|
|DVQA|[images](/mnt/haiyangguo/mywork/CL-MLLM/LLaVA-ModalPrompt/scripts/ModalPrompt/Train_DCL/train_all.sh)|
|Grounding, AOKVQA|[train](http://images.cocodataset.org/zips/train2014.zip) [val](http://images.cocodataset.org/zips/val2014.zip) [test](http://images.cocodataset.org/zips/test2014.zip)|
|OCR-VQA|[images](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_)|
|TabMWP|[images](https://github.com/lupantech/PromptPG)|
|FigureQA|[images]|

After downloading all of them, organize the data as follows:
```
|-- datasets
    |-- ArxivQA
        |-- images/
    |-- CLEVR
        |-- images
            |-- train/
            |-- test/
            |-- val/
    |-- Flickr30k
        |-- train/
        |-- val/
    |-- IconQA
        |-- iconqa_data/
            |-- iconqa/
    |-- ImageNet-R
        |-- train/
        |-- test/
    |-- COCO2014
        |-- train2014/
        |-- test2014/
        |-- val2014/
    |-- super-CLEVR
        |-- images/
    |-- FigureQA
        |-- images/
    |-- OCR-VQA
        |-- images/
    |-- DVQA
        |-- images/
    |-- TabMWP
        |-- tables/
```

Please download the instructions from our [HuggingFace](https://huggingface.co/datasets/HaiyangGuo/UCIT) page, then, organize the instructions as follows:
```
|-- instructions
    |-- ArxivQA
    |-- CLEVR-Math
    |-- Flickr30k-cap
    |-- IconQA
    |-- ImageNet-R
    |-- super-CLEVR
    |-- DVQA
    |-- FigureQA
    |-- Grounding
    |-- OCRVQA
    |-- AOKVQA
    |-- TabMWP
|-- partitioned_data
    |-- Capability-related
    |-- Task-related
```

## Pre-trained Weights

Please download [LLaVA](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336), and use the **config.json** provided in this repository replace the original config.json in LLaVA.

## Citation

```bibtex
@article{guo2025federated,
  title={Federated continual instruction tuning},
  author={Guo, Haiyang and Zeng, Fanhu and Zhu, Fei and Liu, Wenzhuo and Wang, Da-Han and Xu, Jian and Zhang, Xu-Yao and Liu, Cheng-Lin},
  journal={arXiv preprint arXiv:2503.12897},
  year={2025}
}
```

## Acknowledgememnt

This repository is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA), [CoIN](https://github.com/zackschen/CoIN), [Shepherd](https://github.com/JayZhang42/FederatedGPT-Shepherd), and [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM) projects. We sincerely thank the authors for their valuable contributions to the research community.
