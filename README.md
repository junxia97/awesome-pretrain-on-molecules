# A Survey of Pretraining on Graphs: Taxonomy, Methods, and Applications
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/JaydenXia-tech/awesome-self-supervised-learning-for-graphs?color=yellow)  ![GitHub forks](https://img.shields.io/github/forks/SXKDZ/awesome-self-supervised-learning-for-graphs?color=green&label=Fork)  ![visitors](https://visitor-badge.glitch.me/badge?page_id=SXKDZ.awesome-self-supervised-learning-for-graphs)

This is a repository to help all readers who are interested in pre-training on graphs. 
If your papers are missing or you have other requests, please contact to <xiajun@westlake.edu.cn>. Note that we only collect papers that follows pretrain-then-finetune paradigm. We will update this repository and paper on a regular basis to maintain up-to-date.
> **Last update date: 2022-2-10**

## Contents
- [Papers List](#papers)
  <a name="prestrategies"></a>
  + Pretraining Strategies
  <a name="tunestrategies"></a>
  + Tuning Strategies
  <a name="others"></a>
  + Others
- [Open-Sourced Graph Pretrained Models](#PGMs)
- [Pretraing Datasets](#Datasets)
- [Citation](#Cite)

<a name="papers"></a>
## Papers List
All Papers are sorted chronologically according to the year when they are published or released below.
### Year 2022
1. [WWW 2022] **SimGRACE: A Simple Framework for Graph Contrastive Learning without Data Augmentation** [[paper]](https://arxiv.org/pdf/2202.03104.pdf) [[code]](https://github.com/junxia97/SimGRACE)
4. [WSDM 2022]**Bringing Your Own View: Graph Contrastive Learning without Prefabricated Data Augmentations** [[paper]](https://arxiv.org/abs/2201.01702) [[code]](https://github.com/Shen-Lab/GraphCL_Automated)
5. [SDM 2022] **Structure-Enhanced Heterogeneous Graph Contrastive Learning** [[paper]](https://sxkdz.github.io/files/publications/SDM/STENCIL/STENCIL.pdf)
6. [AAAI 2022] **Self-supervised Graph Neural Networks via Diverse and Interactive Message Passing** [[paper]](https://yangliang.github.io/pdf/aaai22.pdf)
9. [AAAI 2022] **Augmentation-Free Self-Supervised Learning on Graphs** [[paper]](https://arxiv.org/pdf/2112.02472.pdf)[[code]](https://github.com/Namkyeong/AFGRL)
10. [AAAI 2022] **Molecular Contrastive Learning with Chemical Element Knowledge Graph** [[paper]](https://arxiv.org/pdf/2112.00544.pdf)
13. [AAAI 2022] **Deep Graph Clustering via Dual Correlation Reduction** [[paper]](https://arxiv.org/pdf/2112.14772)[[code]](https://github.com/yueliu1999/DCRN)
14. [ICOIN 2022] **Adaptive Self-Supervised Graph Representation Learning** [[paper]](https://ieeexplore.ieee.org/abstract/document/9687176)
15. [BioRxiv 2022] **Towards Effective and Generalizable Fine-tuning for Pre-trained Molecular Graph Models** (Tuning Strategies) [[paper]](https://www.biorxiv.org/content/10.1101/2022.02.03.479055v1)
16. [arXiv 2022] **Graph Self-supervised Learning with Accurate Discrepancy Learning** [[paper]](https://arxiv.org/pdf/2202.02989.pdf)
17. [arXiv 2022] **Dual Space Graph Contrastive Learning** [[paper]](https://arxiv.org/pdf/2201.07409.pdf)

## Year 2021
## Year 2020
## Year 2019

<a name="PGMs"></a>
## Open-Sourced Graph Pretrained Models
| oprule \textbf{PGMs}                | \textbf{Input}   | \textbf{Architecture}                   | \textbf{Pretraining Task} | \textbf{Pretraining Database}                   | \textbf{\# Params.} |
|-------------------------------------|------------------|-----------------------------------------|---------------------------|-------------------------------------------------|---------------------|
| \hline \citet{Hu*2020Strategies}    | Graph            | 5-layer GIN                             | GCP + MCM                 | ZINC15 (2M) + ChEMBL (456K)                     | $\sim$ 2M           |
| Graph-BERT~\cite{zhang2020graph}    | Graph            | Graph Transformer~\cite{zhang2020graph} | GAEs                      | Cora + CiteSeer + PubMed                        | N/A                 |
| GraphCL~\cite{You2020GraphCL}       | Graph            | 5-layer GIN                             | IND                       | ZINC15 (2M) + ChEMBL (456K)                     | $\sim$ 2M           |
| GPT-GNN~\cite{hu2020gpt}            | Graph            | HGT~\cite{hu2020heterogeneous}          | GAM                       | OAG + Amazon                                    | N/A                 |
| GCC~\cite{qiu2020gcc}               | Graph            | 5-layer GIN                             | IND                       | Academia + DBLP + IMDB + Facebook + LiveJournal | \textless1M         |
| JOAO~\cite{you2021graph}            | Graph            | 5-layer GIN                             | IND                       | ZINC15 (2M) + ChEMBL (456K)                     | $\sim$ 2M           |
| AD-GCL~\cite{suresh2021adversarial} | Graph            | 5-layer GIN                             | IND                       | ZINC15 (2M) + ChEMBL (456K)                     | $\sim$ 2M           |
| GraphLog~\cite{xu2021self}          | Graph            | 5-layer GIN                             | IND                       | ZINC15 (2M) + ChEMBL (456K)                     | $\sim$ 2M           |
| GROVER~\cite{rong2020self}          | Graph            | GTransformer~\cite{rong2020self}        | GCP + MCM                 | ZINC + ChEMBL (10M)                             | 48M$\sim$100M       |
| MGSSL~\cite{zhang2021motif}         | Graph            | 5-layer GIN                             | MCM + GAM                 | ZINC15 (250K)                                   | $\sim$ 2M           |
| CPT-HG~\cite{jiang2021contrastive}  | Graph            | HGT~\cite{hu2020heterogeneous}          | IND                       | DBLP + YELP + Aminer                            | N/A                 |
| PGM~\cite{li2021effective}          | Graph            | MolGNet~\cite{li2021effective}          | RCD + MCM                 | ZINC + ChEMBL (11M)                             | 53M                 |
| LP-Info~\cite{you2022bringing}      | Graph            | 5-layer GIN                             | IND                       | ZINC15 (2M) + ChEMBL (456K)                     | $\sim$ 2M           |
| SimGRACE~\cite{xia2022simgrace}     | Graph            | 5-layer GIN                             | IND                       | ZINC15 (2M) + ChEMBL (456K)                     | $\sim$ 2M           |
| MolCLR~\cite{Wang2021MolCLRMC}      | Graph + SMILES   | GCN + GIN                               | IND                       | PubChem (10M)                                   | N/A                 |
| DMP~\cite{Zhu2021DualviewMP}        | Graph + SMILES   | DeeperGCN + Transformer                 | MCM + IND                 | PubChem (110M)                                  | 104.1 M             |
| ChemRL-GEM~\cite{Fang2021geo}       | Graph + Geometry | GeoGNN~\cite{Fang2021geo}               | MCM+GCP                   | ZINC15 (20M)                                    | N/A                 |
| KCL~\cite{fang2021molecular}        | Graph + KG       | GCN + KMPNN~\cite{fang2021molecular}    | IND                       | ZINC15 (250K)                                   | \textless 1M        |
| 3D Infomax~\cite{stark20213d}       | 2D and 3D graph  | PNA~\cite{corso2020principal}           | IND                       | QM9(50K) + GEOM-drugs(140K) + QMugs(620K)       | N/A                 |
| GraphMVP~\cite{liu2022pretraining}  | 2D and 3D graph  | GIN + SchNet~\cite{NIPS2017_303ed4c6}   | IND + GAEs                | GEOM (50k)             | $\sim$ 2M  |



<a name="Datasets"></a>
## Pretraing Datasets
| 名称           | 数据来源     | 训练数据大小 | 词表大小 | 模型大小 | 下载地址 |
| :----------:  | :---------: | :---------:| :------: | :------: | :------: |
| RoBERTa Tiny  | 百科,新闻 等  |     35G    | 21128    | 27MB | [下载链接](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-4_H-312_A-12.zip) |
| RoBERTa Small | 百科,新闻 等  |     35G    | 21128  | 48MB  | [下载链接](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-6_H-384_A-12.zip) |
| SimBERT Tiny  | [百度知道](http://zhidao.baidu.com/) | 2200万相似句组 | 13685  | 26MB  | [下载链接](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_simbert_L-4_H-312_A-12.zip) |
| SimBERT Small  | [百度知道](http://zhidao.baidu.com/) | 2200万相似句组 | 13685  | 49MB  | [下载链接](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_simbert_L-6_H-384_A-12.zip) |
| SimBERT Base  | [百度知道](http://zhidao.baidu.com/) | 2200万相似句组 | 13685  | 344MB  | [下载链接](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_simbert_L-12_H-768_A-12.zip) |
| RoBERTa<sup>+</sup> Tiny  | 百科,新闻 等  |     35G    | 21128    | 35MB | [下载链接](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-4_H-312_A-12_K-104.zip) |
| RoBERTa<sup>+</sup> Small | 百科,新闻 等  |     35G    | 21128  | 67MB  | [下载链接](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-6_H-384_A-12_K-128.zip) |
| WoBERT | 百科,新闻 等  |     35G    | 33586/50000  | 400M  | [WoBERT项目](https://github.com/ZhuiyiTechnology/WoBERT) |
| T5 PEGASUS | 百科,新闻 等  |     35G    | 50000  | 971M  | [T5 PEGASUS项目](https://github.com/ZhuiyiTechnology/t5-pegasus) |


## __Citation (.bib)__ </br>
```
@article{song2020learning,
title={A Survey of Pretraining on Graphs: Taxonomy, Methods, and Applications},
author={Song, Hwanjun and Kim, Minseok and Park, Dongmin and Shin, Yooju and Lee, Jae-Gil},
journal={arXiv preprint arXiv:2007.08199},
year={2020}}
```

