# RAPL: Relation-Aware Prototype Learning for Few-Shot DocRE

This repository contains the data, code and trained models for paper [RAPL: A Relation-Aware Prototype Learning Approach for Few-Shot Document-Level Relation Extraction](https://aclanthology.org/2023.emnlp-main.316/).

## Quick Links

- [Overview](#Overview)
- [Setup](#Setup)
  - [Install dependencies](#Install-dependencies)
  - [Trained models](#Trained-models)
- [Datasets](#Datasets)
  - [FREDo](#FREDo)
  - [ReFREDo](#ReFREDo)
- [Quick Start](#Quick-Start)
- [Citation](#Citation)

## Overview

![](model.png)

In this work, we present a relation-aware prototype learning method (RAPL) for few-shot document-level relation extraction. We reframe the construction of relation prototypes into instance level and further propose a relation-weighted contrastive learning method to jointly refine the relation prototypes. We also design a task-specific NOTA prototype generation strategy to better capture the NOTA semantics in each task.

You can find more details of this work in our [paper](https://aclanthology.org/2023.emnlp-main.316/).

## Setup

### Install dependencies

To run the code, please install the following dependency packages:

- apex (experiment on 0.1)
- numpy (experiment on 1.19.4)
- opt_einsum (experiment on 3.3.0)
- torch (experiment on 1.9.0+cu111)
- tqdm (experiment on 4.64.0)
- transformers (experiment on 3.4.0)
- wandb (experiment on 0.12.21)

### Trained models

We release the sample trained models for each task setting on [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/ea949a7000b64fb9a7f8/). To reproduce the results in the paper, you can download the corresponding models and place them in [checkpoints](checkpoints) directory.

## Datasets

Our experiments are based on two benchmarks: FREDo and ReFREDo, and all relevant data files are located in [dataset](dataset) directory.

### FREDo

FREDo is a few-shot document-level relation extraction benchmark consisting of two main tasks (in-domain / cross-domain) with a 1-Doc and a 3-Doc subtask each. The relevant data files include:

- dataset/[train_docred, dev_docred, test_docred, test_scierc].json contain all annotated documents used for training and testing.
- dataset/[test_docred_1_doc_indices, test_docred_3_doc_indices, test_scierc_1_doc_indices, test_scierc_3_doc_indices].json contain sampled episodes (only the indices of the documents and which relations are to be annotated/extracted).

### ReFREDo

ReFREDo is a revised version of FREDo, which replaces the training, development and in-domain test document corpus with Re-DocRED, leading to more complete annotations. The relevant data files include:

- dataset/[train_redocred, dev_redocred, test_redocred, test_scierc].json contain all annotated documents used for training and testing.
- dataset/[test_redocred_1_doc_indices, test_redocred_3_doc_indices, test_scierc_1_doc_indices, test_scierc_3_doc_indices].json contain sampled episodes (only the indices of the documents and which relations are to be annotated/extracted).

## Quick Start

In [scripts](scripts) directory, we provide the example scripts for running experiments under each task setting. For example, you can use the following command to run the trained model on in-domain 3-Doc test tasks in ReFREDo benchmark:

```bash
sh scripts/refredo_indomain_3doc.sh
```

You can also comment the `--load_checkpoint` argument and set `--num_epochs` argument to 25 for training. The following command can be used to display the details about each argument:

```bash
python src/main.py -h
```

## Citation

Please kindly cite our paper if you use the data, code or models of RAPL in your work:

```bibtex
@inproceedings{meng-etal-2023-rapl,
    title = "{RAPL}: A Relation-Aware Prototype Learning Approach for Few-Shot Document-Level Relation Extraction",
    author = "Meng, Shiao  and
      Hu, Xuming  and
      Liu, Aiwei  and
      Li, Shuang  and
      Ma, Fukun  and
      Yang, Yawen  and
      Wen, Lijie",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
    url = "https://aclanthology.org/2023.emnlp-main.316",
    doi = "10.18653/v1/2023.emnlp-main.316",
    pages = "5208--5226"
}
```

