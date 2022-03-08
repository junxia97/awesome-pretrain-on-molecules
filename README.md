# A Survey of Pretraining on Graphs: Taxonomy, Methods, and Applications
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/junxia97/awesome-pretrain-on-graphs?color=yellow)  ![GitHub forks](https://img.shields.io/github/forks/junxia97/awesome-pretrain-on-graphs?color=green&label=Fork)  ![visitors](https://visitor-badge.glitch.me/badge?page_id=junxia97.awesome-pretrain-on-graphs)

This is a repository to help all readers who are interested in pre-training on graphs. 
If you find there are other resources with this topic missing, feel free to let us know via github issues, pull requests or my email: xiajun@westlake.edu.cn. We will update this repository and paper on a regular basis to maintain up-to-date.
> **Last update date: 2022-3-8**

## Contents
- [Papers List](#papers)
  + [Pretraining Strategies](#prestrategies)
  + [Knowledge-Enriched Pretraining Strategies](#know)
  + [Hard Negative Mining Strategies](#hard)
  + [Tuning Strategies](#tunestrategies)
  + [Applications](#Applications)
  + [Others](#others)
- [Open-Sourced Graph Pretrained Models](#PGMs)
- [Pretraining Datasets](#Datasets)
- [Citation](#Cite)
- [Acknowledgement](#Acknow)

<a name="papers"></a>
## Papers List
<a name="prestrategies"></a>

### Pretraining Strategies
1. [SDM 2022] **Neural Graph Matching for Pre-training Graph Neural Networks**[[paper]](https://arxiv.org/pdf/2203.01597.pdf)
1. [Nature Machine Intelligence 2022]  **Molecular contrastive learning of representations via graph neural networks** [[paper]](https://www.nature.com/articles/s42256-022-00447-x)
1. [WWW 2022] **SimGRACE: A Simple Framework for Graph Contrastive Learning without Data Augmentation** [[paper]](https://arxiv.org/pdf/2202.03104.pdf) [[code]](https://github.com/junxia97/SimGRACE)
2. [WWW 2022] **Dual Space Graph Contrastive Learning** [[paper]](https://arxiv.org/pdf/2201.07409.pdf)
3. [WWW 2022] **Adversarial Graph Contrastive Learning with Information Regularization** [[paper]](https://arxiv.org/pdf/2202.06491.pdf)
4. [WWW 2022] **The Role of Augmentations in Graph Contrastive Learning: Current Methodological Flaws & Improved Practices** [[paper]](https://arxiv.org/pdf/2111.03220.pdf)
5. [WWW 2022] **ClusterSCL: Cluster-Aware Supervised Contrastive Learning on Graphs** [[paper]](https://xiaojingzi.github.io/publications/WWW22-Wang-et-al-ClusterSCL.pdf)
6. [WWW 2022] **Graph Communal Contrastive Learning** [[paper]](https://arxiv.org/pdf/2110.14863.pdf)
7. [TKDE 2022] **CCGL: Contrastive Cascade Graph Learning** [[paper]](https://arxiv.org/pdf/2107.12576.pdf)[[code]](https://arxiv.org/pdf/2107.12576.pdf)
8. [BIBM 2021] **Molecular Graph Contrastive Learning with Parameterized Explainable Augmentations** [[paper]](https://www.biorxiv.org/content/10.1101/2021.12.03.471150v1)
9. [WSDM 2022]**Bringing Your Own View: Graph Contrastive Learning without Prefabricated Data Augmentations** [[paper]](https://arxiv.org/abs/2201.01702) [[code]](https://github.com/Shen-Lab/GraphCL_Automated)
10. [SDM 2022] **Structure-Enhanced Heterogeneous Graph Contrastive Learning** [[paper]](https://sxkdz.github.io/files/publications/SDM/STENCIL/STENCIL.pdf)
11. [AAAI 2022] **GeomGCL: Geometric Graph Contrastive Learning for Molecular Property Prediction** [[paper]](https://arxiv.org/pdf/2109.11730.pdf)
12. [AAAI 2022] **Self-supervised Graph Neural Networks via Diverse and Interactive Message Passing** [[paper]](https://yangliang.github.io/pdf/aaai22.pdf)
13. [AAAI 2022] **Augmentation-Free Self-Supervised Learning on Graphs** [[paper]](https://arxiv.org/pdf/2112.02472.pdf)[[code]](https://github.com/Namkyeong/AFGRL)
14. [AAAI 2022] **Deep Graph Clustering via Dual Correlation Reduction** [[paper]](https://arxiv.org/pdf/2112.14772)[[code]](https://github.com/yueliu1999/DCRN)
15. [ICOIN 2022] **Adaptive Self-Supervised Graph Representation Learning** [[paper]](https://ieeexplore.ieee.org/abstract/document/9687176)
16. [arXiv 2022] **Graph Masked Autoencoder** [[paper]](https://arxiv.org/abs/2202.08391)
17. [arXiv 2022] **Structural and Semantic Contrastive Learning for Self-supervised Node Representation Learning** [[paper]](https://arxiv.org/pdf/2202.08480.pdf)
18. [arXiv 2022] **Graph Self-supervised Learning with Accurate Discrepancy Learning** [[paper]](https://arxiv.org/pdf/2202.02989.pdf)
19. [arXiv 2021] **Multilayer Graph Contrastive Clustering Network** [[paper]](https://arxiv.org/pdf/2112.14021.pdf)
20. [arXiv 2021] **Graph Representation Learning via Contrasting Cluster Assignments** [[paper]](https://arxiv.org/pdf/2112.07934.pdf)
21. [arXiv 2021] **Graph-wise Common Latent Factor Extraction for Unsupervised Graph Representation Learning** [[paper]](https://arxiv.org/pdf/2112.08830.pdf)
22. [arXiv 2021] **Bayesian Graph Contrastive Learning** [[paper]](https://arxiv.org/pdf/2112.07823.pdf)
23. [NeurIPS 2021 Workshop] **Self-Supervised GNN that Jointly Learns to Augment** [[paper]](https://www.researchgate.net/profile/Zekarias-Kefato/publication/356997993_Self-Supervised_GNN_that_Jointly_Learns_to_Augment/links/61b75d88a6251b553ab64ff4/Self-Supervised-GNN-that-Jointly-Learns-to-Augment.pdf)
24. [NeurIPS 2021] **Enhancing Hyperbolic Graph Embeddings via Contrastive Learning** [[paper]](https://sslneurips21.github.io/files/CameraReady/NeurIPS_2021_workshop_version2.pdf)
25. [NeurIPS 2021] **Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization** [[paper]](https://openreview.net/forum?id=J_pvI6ap5Mn)
26. [NeurIPS 2021] **Motif-based Graph Self-Supervised Learning for Molecular Property Prediction** [[paper]](https://arxiv.org/pdf/2110.00987.pdf)
27. [NeurIPS 2021] **Graph Adversarial Self-Supervised Learning** [[paper]](https://proceedings.neurips.cc/paper/2021/file/7d3010c11d08cf990b7614d2c2ca9098-Paper.pdf)
28. [NeurIPS 2021] **Contrastive laplacian eigenmaps** [[paper]](https://papers.nips.cc/paper/2021/file/2d1b2a5ff364606ff041650887723470-Paper.pdf)
29. [NeurIPS 2021] **Directed Graph Contrastive Learning** [[paper]](https://zekuntong.com/files/digcl_nips.pdf)[[code]](https://github.com/flyingtango/DiGCL)
30. [NeurIPS 2021] **Multi-view Contrastive Graph Clustering** [[paper]](https://arxiv.org/pdf/2110.11842.pdf)[[code]](https://github.com/Panern/MCGC)
31. [NeurIPS 2021] **From Canonical Correlation Analysis to Self-supervised Graph Neural Networks** [[paper]](https://arxiv.org/pdf/2106.12484.pdf)[[code]](https://github.com/hengruizhang98/CCA-SSG)
32. [NeurIPS 2021] **InfoGCL: Information-Aware Graph Contrastive Learning** [[paper]](https://arxiv.org/pdf/2110.15438.pdf)
33. [NeurIPS 2021] **Adversarial Graph Augmentation to Improve Graph Contrastive Learning** [[paper]](https://arxiv.org/abs/2106.05819)[[code]](https://github.com/susheels/adgcl)
34. [NeurIPS 2021] **Disentangled Contrastive Learning on Graphs** [[paper]](https://openreview.net/pdf?id=C_L0Xw_Qf8M)
35. [arXiv 2021] **Subgraph Contrastive Link Representation Learning** [[paper]](https://arxiv.org/pdf/2112.01165.pdf)
36. [arXiv 2021] **Augmentations in Graph Contrastive Learning: Current Methodological Flaws & Towards Better Practices** [[paper]](https://arxiv.org/pdf/2111.03220.pdf)
37. [arXiv 2021] **Collaborative Graph Contrastive Learning: Data Augmentation Composition May Not be Necessary for Graph Representation Learning** [[paper]](https://arxiv.org/pdf/2111.03262.pdf)
38. [CIKM 2021] **Contrastive Pre-Training of GNNs on Heterogeneous Graphs** [[paper]](https://yuanfulu.github.io/publication/CIKM-CPT.pdf)
39. [CIKM 2021] **Self-supervised Representation Learning on Dynamic Graphs** [[paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482389)
40. [CIKM 2021] **SGCL: Contrastive Representation Learning for Signed Graphs** [[paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482478)
41. [CIKM 2021] **Semi-Supervised and Self-Supervised Classification with Multi-View Graph Neural Networks** [[paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482477)
42. [arXiv 2021] **Graph Communal Contrastive Learning** [[paper]](https://arxiv.org/pdf/2110.14863.pdf)
43. [arXiv 2021] **Self-supervised Contrastive Attributed Graph Clustering** [[paper]](https://arxiv.org/pdf/2110.08264.pdf)
44. [arXiv 2021] **Adaptive Multi-layer Contrastive Graph Neural Networks** [[paper]](https://arxiv.org/pdf/2109.14159.pdf)
45. [arXiv 2021] **Graph-MVP: Multi-View Prototypical Contrastive Learning for Multiplex Graphs** [[paper]](https://arxiv.org/pdf/2109.03560.pdf)
46. [arXiv 2021] **Spatio-Temporal Graph Contrastive Learning** [[paper]](https://arxiv.org/pdf/2108.11873.pdf)
47. [IJCAI 2021] **Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning** [[paper]](https://www.ijcai.org/proceedings/2021/0204.pdf)
48. [IJCAI 2021] **Pairwise Half-graph Discrimination: A Simple Graph-level Self-supervised Strategy for Pre-training Graph Neural Networks** [[paper]](https://www.ijcai.org/proceedings/2021/0371.pdf)
49. [arXiv 2021] **RRLFSOR: An Efficient Self-Supervised Learning Strategy of Graph Convolutional Networks** [[paper]](https://arxiv.org/ftp/arxiv/papers/2108/2108.07481.pdf)
50. [ICML 2021] **Graph Contrastive Learning Automated** [[paper]](https://arxiv.org/abs/2106.07594) [[code]](https://github.com/Shen-Lab/GraphCL_Automated)
51. [ICML 2021] **Self-supervised Graph-level Representation Learning with Local and Global Structure** [[paper]](https://arxiv.org/pdf/2106.04113) [[code]](https://github.com/DeepGraphLearning/GraphLoG)
52.[arXiv 2021] **Group Contrastive Self-Supervised Learning on Graphs** [[paper]](https://arxiv.org/abs/2107.09787) 
53. [arXiv 2021] **Multi-Level Graph Contrastive Learning** [[paper]](https://arxiv.org/abs/2107.02639)
54. [KDD 2021] **Pre-training on Large-Scale Heterogeneous Graph** [[paper]](http://www.shichuan.org/doc/111.pdf)
55. [KDD 2021] **Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning** [[paper]](https://arxiv.org/abs/2105.09111) [[code]](https://github.com/liun-online/HeCo)
56. [arXiv 2021] **Prototypical Graph Contrastive Learning** [[paper]](https://arxiv.org/pdf/2106.09645.pdf)
57. [arXiv 2021] **Graph Barlow Twins: A self-supervised representation learning framework for graphs** [[paper]](https://arxiv.org/pdf/2106.02466.pdf)
58. [arXiv 2021] **Self-Supervised Graph Learning with Proximity-based Views and Channel Contrast** [[paper]](https://arxiv.org/pdf/2106.03723.pdf)
59. [arXiv 2021] **FedGL: Federated Graph Learning Framework with Global Self-Supervision** [[paper]](https://arxiv.org/pdf/2105.03170.pdf)
60. [IJCNN 2021] **Node Embedding using Mutual Information and Self-Supervision based Bi-level Aggregation** [[paper]](https://arxiv.org/abs/2104.13014v1)
61. [arXiv 2021] **Graph Representation Learning by Ensemble Aggregating Subgraphs via Mutual Information Maximization** [[paper]](https://arxiv.org/abs/2103.13125)
62. [arXiv 2021] **Self-supervised Auxiliary Learning for Graph Neural Networks via Meta-Learning** [[paper]](https://arxiv.org/abs/2103.00771)
63. [arXiv 2021] **Towards Robust Graph Contrastive Learning** [[paper]](https://arxiv.org/pdf/2102.13085.pdf)
64. [arXiv 2021] **Pre-Training on Dynamic Graph Neural Networks** [[paper]](https://arxiv.org/abs/2102.12380)
65. [WWW 2021] **Graph Contrastive Learning with Adaptive Augmentation** [[paper]](https://arxiv.org/abs/2010.14945) [[code]](https://github.com/CRIPAC-DIG/GCA)
66. [Arxiv 2020] **Distance-wise Graph Contrastive Learning** [[paper]](https://arxiv.org/abs/2012.07437)
67. [Openreview 2020] **Motif-Driven Contrastive Learning of Graph Representations** [[paper]](https://openreview.net/forum?id=qcKh_Msv1GP)
68. [Openreview 2020] **SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks** [[paper]](https://openreview.net/forum?id=a5KvtsZ14ev)
69. [Openreview 2020] **TopoTER: Unsupervised Learning of Topology Transformation Equivariant Representations** [[paper]](https://openreview.net/forum?id=9az9VKjOx00)
70. [Openreview 2020] **Graph-Based Neural Network Models with Multiple Self-Supervised Auxiliary Tasks** [[paper]](https://openreview.net/forum?id=hnJSgY7p33a)
71. [NeurIPS 2020] **Self-Supervised Graph Transformer on Large-Scale Molecular Data** [[paper]](https://drug.ai.tencent.com/publications/GROVER.pdf)
72. [NeurIPS 2020] **Self-supervised Auxiliary Learning with Meta-paths for Heterogeneous Graphs** [[paper]](https://arxiv.org/abs/2007.08294) [[code]](https://github.com/mlvlab/SELAR)
73. [NeurIPS 2020] **Graph Contrastive Learning with Augmentations** [[paper]](https://arxiv.org/abs/2010.13902) [[code]](https://github.com/Shen-Lab/GraphCL)
74. [Arxiv 2020] **Deep Graph Contrastive Representation Learning** [[paper]](https://arxiv.org/abs/2006.04131)
75. [ICML 2020] **When Does Self-Supervision Help Graph Convolutional Networks?** [[paper]](https://arxiv.org/abs/2006.09136) [[code]](https://github.com/Shen-Lab/SS-GCNs)
76. [ICML 2020] **Contrastive Multi-View Representation Learning on Graphs.** [[paper]](https://arxiv.org/abs/2006.05582) [[code]](https://github.com/kavehhassani/mvgrl)
77. [ICML 2020 Workshop] **Self-supervised edge features for improved Graph Neural Network training.** [[paper]](https://arxiv.org/abs/2007.04777)
78. [Arxiv 2020] **Self-supervised Training of Graph Convolutional Networks.** [[paper]](https://arxiv.org/abs/2006.02380)
79. [Arxiv 2020] **Self-Supervised Graph Representation Learning via Global Context Prediction.** [[paper]](https://arxiv.org/abs/2003.01604)
80. [KDD 2020] **GPT-GNN: Generative Pre-Training of Graph Neural Networks.** [[pdf]](https://arxiv.org/abs/2006.15437) [[code]](https://github.com/acbull/GPT-GNN)
81. [KDD 2020] **GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training.** [[pdf]](https://arxiv.org/abs/2006.09963) [[code]](https://github.com/THUDM/GCC) 
82. [Arxiv 2020] **Graph-Bert: Only Attention is Needed for Learning Graph Representations.** [[paper]](https://arxiv.org/abs/2001.05140) [[code]](https://github.com/anonymous-sourcecode/Graph-Bert)
83. [ICLR 2020] **InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization.** [[paper]](https://arxiv.org/abs/1908.01000) [[code]](https://github.com/fanyun-sun/InfoGraph)
84. [ICLR 2020] **Strategies for Pre-training Graph Neural Networks.** [[paper]](https://arxiv.org/abs/1905.12265) [[code]](https://github.com/snap-stanford/pretrain-gnns)
85. [KDD 2019 Workshop] **SGR: Self-Supervised Spectral Graph Representation Learning.** [[paper]](https://arxiv.org/abs/1811.06237)
86. [ICLR 2019 workshop] **Pre-Training Graph Neural Networks for Generic Structural Feature Extraction.** [[paper]](https://arxiv.org/abs/1905.13728)
87. [Arxiv 2019] **Heterogeneous Deep Graph Infomax** [[paper]](https://arxiv.org/abs/1911.08538) [[code]](https://github.com/YuxiangRen/Heterogeneous-Deep-Graph-Infomax)
88. [ICLR 2019] **Deep Graph Informax.** [[paper]](https://arxiv.org/abs/1809.10341) [[code]](https://github.com/PetarV-/DGI)

<a name="know"></a>
### Knowledge-Enriched Pretraining Strategies
1. [Nature Machine Itelligence 2022] Geometry-enhanced molecular representation learning for property prediction[[paper]](https://www.nature.com/articles/s42256-021-00438-4)
2. [ICLR 2022] **PRE-TRAINING MOLECULAR GRAPH REPRESENTATION WITH 3D GEOMETRY** [[paper]](https://wyliu.com/papers/GraphMVP.pdf) [[code]](https://github.com/chao1224/GraphMVP)
3. [AAAI 2022] **Molecular Contrastive Learning with Chemical Element Knowledge Graph** [[paper]](https://arxiv.org/pdf/2112.00544.pdf)
4. [KDD 2021] **MoCL: Data-driven Molecular Fingerprint via Knowledge-aware Contrastive Learning from Molecular Graph** [[paper]](https://dl.acm.org/doi/abs/10.1145/3447548.3467186) [[code]](https://github.com/illidanlab/MoCL-DK)
9. [arXiv 2021] **3D Infomax improves GNNs for Molecular Property Prediction** [[paper]](https://arxiv.org/abs/2110.04126v1) [[code]](https://github.com/HannesStark/3DInfomax)

<a name="hard"></a>
### Hard Negative Mining Strategies
1. [SDM 2022] **Structure-Enhanced Heterogeneous Graph Contrastive Learning** [[paper]](https://sxkdz.github.io/files/publications/SDM/STENCIL/STENCIL.pdf)
3. [Signal Processing 2021] **Negative Sampling Strategies for Contrastive Self-Supervised Learning of Graph Representations** [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0165168421003479)
4. [IJCAI 2021] **Graph Debiased Contrastive Learning with Joint Representation Clustering** [[paper]](https://www.ijcai.org/proceedings/2021/0473.pdf)
5. [IJCAI 2021] **CuCo: Graph Representation with Curriculum Contrastive Learning** [[paper]](https://www.ijcai.org/proceedings/2021/0317.pdf)
6. [arXiv 2021] **Debiased Graph Contrastive Learning** [[paper]](https://arxiv.org/pdf/2110.02027.pdf)

<a name="tunestrategies"></a>
### Tuning Strategies
1. [KDD 2021] **Adaptive Transfer Learning on Graph Neural Networks** [[paper]](https://arxiv.org/pdf/2107.08765.pdf)
3. [BioRxiv 2022] **Towards Effective and Generalizable Fine-tuning for Pre-trained Molecular Graph Models**[[paper]](https://www.biorxiv.org/content/10.1101/2022.02.03.479055v1)
4. [AAAI 2022] **CODE: Contrastive Pre-training with Adversarial Fine-tuning for Zero-shot Expert Linking** [[paper]](https://arxiv.org/abs/2012.11336) [[code]](https://github.com/BoChen-Daniel/Expert-Linking)


<a name="Applications"></a>
### Applications
1. [Nature Communications 2021] **Masked graph modeling for molecule generation** [[paper]](https://www.nature.com/articles/s41467-021-23415-2)
1. [NPL 2022] **How Does Bayesian Noisy Self-Supervision Defend Graph Convolutional Networks?** [[paper]](https://link.springer.com/article/10.1007/s11063-022-10750-8)
2. [arXiv 2022] **Self-supervised Graphs for Audio Representation Learning with Limited Labeled Data** [[paper]](https://arxiv.org/pdf/2202.00097.pdf)
3. [arXiv 2022] **Link Prediction with Contextualized Self-Supervision** [[paper]](https://arxiv.org/pdf/2201.10069.pdf)
4. [arXiv 2022] **Learning Robust Representation through Graph Adversarial Contrastive Learning** [[paper]](https://arxiv.org/pdf/2201.13025.pdf)
5. [WWW 2021] **Multi-view Graph Contrastive Representation Learning for Drug-drug Interaction Prediction** [[paper]](https://arxiv.org/pdf/2010.11711.pdf)
6. [BIBM 2021] **SGAT: a Self-supervised Graph Attention Network for Biomedical Relation Extraction** [[paper]](https://ieeexplore.ieee.org/abstract/document/9669699)
7. [ICBD 2021] **Session-based Recommendation via Contrastive Learning on Heterogeneous Graph** [[paper]](https://ieeexplore.ieee.org/abstract/document/9671296)
8. [arXiv 2021] **Graph Augmentation-Free Contrastive Learning for Recommendation** [[paper]](https://arxiv.org/pdf/2112.08679.pdf)
9. [arXiv 2021] **TCGL: Temporal Contrastive Graph for Self-supervised Video Representation Learning** [[paper]](https://arxiv.org/pdf/2112.03587.pdf)
10. [NeurIPS 2021 Workshop] **Contrastive Embedding of Structured Space for Bayesian Optimisation** [[paper]](https://openreview.net/pdf?id=xFpkJUMS9te)
11. [ICCSNT 2021] **Graph Data Augmentation based on Adaptive Graph Convolution for Skeleton-based Action Recognition** [[paper]](https://ieeexplore.ieee.org/abstract/document/9615451)
12. [arXiv 2021] **Pre-training Graph Neural Network for Cross Domain Recommendation** [[paper]](https://arxiv.org/pdf/2111.08268.pdf)
13. [CIKM 2021] **Social Recommendation with Self-Supervised Metagraph Informax Network** [[paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482480) [[code]](https://github.com/SocialRecsys/SMIN)
14. [arXiv 2021] **Self-Supervised Learning for Molecular Property Prediction** [[paper]](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/61677becaa918db6bf2a31cb/original/self-supervised-learning-for-molecular-property-prediction.pdf)
15. [arXiv 2021] **Contrastive Graph Convolutional Networks for Hardware Trojan Detection in Third Party IP Cores** [[paper]](https://people.cs.vt.edu/~ramakris/papers/Hardware_Trojan_Trigger_Detection__HOST2021.pdf)
16. [KBS 2021] **Multi-aspect self-supervised learning for heterogeneous information network** [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S095070512100736X)
17. [arXiv 2021] **Hyper Meta-Path Contrastive Learning for Multi-Behavior Recommendation** [[paper]](https://arxiv.org/pdf/2109.02859.pdf)
18. [arXiv 2021] **Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection** [[paper]](https://arxiv.org/pdf/2108.09896.pdf)
19. [IJCAI 2021] **CSGNN: Contrastive Self-Supervised Graph Neural Network for Molecular Interaction Prediction** [[paper]](https://www.ijcai.org/proceedings/2021/0517.pdf)
20. [arXiv 2021] **GCCAD: Graph Contrastive Coding for Anomaly Detection** [[paper]](https://arxiv.org/pdf/2108.07516.pdf)
21. [arXiv 2021] **Contrastive Self-supervised Sequential Recommendation with Robust Augmentation** [[paper]](https://arxiv.org/pdf/2108.06479.pdf)
22. [KDD 2021] **Contrastive Multi-View Multiplex Network Embedding with Applications to Robust Network Alignment** [[paper]](https://dl.acm.org/doi/abs/10.1145/3447548.3467227)
23. [arXiv 2021] **Hop-Count Based Self-Supervised Anomaly Detection on Attributed Networks** [[paper]](https://arxiv.org/abs/2104.07917)
24. [arXiv 2021] **Representation Learning for Networks in Biology and Medicine: Advancements, Challenges, and Opportunities** [[paper]](https://arxiv.org/abs/2104.04883)
25. [arXiv 2021] **Drug Target Prediction Using Graph Representation Learning via Substructures Contrast** [[paper]](https://www.preprints.org/manuscript/202103.0337/v1)
26. [Arxiv 2021] **Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation** [[paper]](https://arxiv.org/abs/2101.06448) [[code]](https://github.com/Coder-Yu/RecQ)
27. [ICLR 2021] **How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision** [[paper]](https://openreview.net/forum?id=Wi5KUNlqWty) [[code]](https://github.com/dongkwan-kim/SuperGAT)
28. [WSDM 2021] **Pre-Training Graph Neural Networks for Cold-Start Users and Items Representation** [[paper]](https://arxiv.org/abs/2012.07064) [[code]](https://github.com/jerryhao66/Pretrain-Recsys)
29. [ICML 2020] **Graph-based, Self-Supervised Program Repair from Diagnostic Feedback.** [[paper]](https://arxiv.org/abs/2005.10636)

<a name="others"></a>

### Others
1. [arXiv 2022] **A Survey of Pretraining on Graphs: Taxonomy, Methods, and Applications** [[paper]](https://arxiv.org/abs/2202.07893)
1. [NeurIPS 2021 datasets and benchmark track] **An Empirical Study of Graph Contrastive Learning** [[paper]](https://openreview.net/forum?id=fYxEnpY-__G)
2. [arXiv 2021] **Evaluating Modules in Graph Contrastive Learning** [[paper]](https://arxiv.org/abs/2106.08171) [[code]](https://github.com/thunlp/OpenGCL)
3. [arXiv 2021] **Graph Self-Supervised Learning: A Survey** [[paper]](https://arxiv.org/abs/2103.00111)
4. [arXiv 2021] **Self-Supervised Learning of Graph Neural Networks: A Unified Review** [[paper]](https://arxiv.org/abs/2102.10757)
5. [Arxiv 2020] **Self-supervised Learning on Graphs: Deep Insights and New Direction.** [[paper]](https://arxiv.org/abs/2006.10141) [[code]](https://github.com/ChandlerBang/SelfTask-GNN)
6. [ICLR 2019 Workshop] **Can Graph Neural Networks Go "Online"? An Analysis of Pretraining and Inference.** [[paper]](https://arxiv.org/abs/1905.06018)

<a name="PGMs"></a>
## Open-Sourced Pretrained Graph Models
|PGMs| Architecture | Pretraining Database| \# Params. | Download Link |
|-------------------------------------|-----------------------------------------|---------------------------|-------------------------------------------------|---------------------|
|[Hu et al.](https://arxiv.org/pdf/2001.05140.pdf)| 5-layer GIN| ZINC15 (2M) + ChEMBL (456K)| ~ 2M |[Link](https://github.com/snap-stanford/pretrain-gnns/tree/master/chem/model_gin)|
| [Graph-BERT](https://arxiv.org/pdf/2001.05140.pdf)| [Graph Transformer](https://arxiv.org/pdf/2001.05140.pdf) | Cora + CiteSeer + PubMed | N/A |[Link](https://github.com/jwzhanggy/Graph-Bert/tree/master/result/PreTrained_GraphBert)|
| [GraphCL](https://arxiv.org/abs/2010.13902)| 5-layer GIN| ZINC15 (2M) + ChEMBL (456K) | ~ 2M|[Link](https://github.com/Shen-Lab/GraphCL/tree/master/transferLearning_MoleculeNet_PPI)|
| [GPT-GNN](https://arxiv.org/pdf/2006.15437.pdf)       | [HGT](https://arxiv.org/pdf/2003.01332.pdf)    | OAG + Amazon| N/A|[Link](https://github.com/acbull/GPT-GNN)|
| [GCC](https://arxiv.org/pdf/2006.09963.pdf)              | 5-layer GIN      | Academia + DBLP + IMDB + Facebook + LiveJournal | <1M|[Link](https://github.com/THUDM/GCC#download-pretrained-models)|
| [JOAO](http://proceedings.mlr.press/v139/you21a.html)          | 5-layer GIN       | ZINC15 (2M) + ChEMBL (456K) | ~ 2M |[Link](https://github.com/Shen-Lab/GraphCL_Automated/tree/master/transferLearning_MoleculeNet_PPI)|
| [AD-GCL](https://openreview.net/forum?id=ioyq7NsR1KJ)  | 5-layer GIN       | ZINC15 (2M) + ChEMBL (456K) | ~ 2M |N/A|
| [GraphLog](https://arxiv.org/pdf/2106.04113.pdf) | 5-layer GIN        | ZINC15 (2M) + ChEMBL (456K)| ~ 2M |[Link](https://github.com/DeepGraphLearning/GraphLoG/tree/main/models)|
| [GROVER](https://arxiv.org/abs/2007.02835)    | [GTransformer](https://arxiv.org/abs/2007.02835)     | ZINC + ChEMBL (10M)| 48M ~ 100M|[Link](https://github.com/tencent-ailab/grover)|
| [MGSSL](https://arxiv.org/pdf/2110.00987.pdf)     | 5-layer GIN   | ZINC15 (250K)  | ~ 2M |[Link](https://github.com/zaixizhang/MGSSL/tree/main/motif_based_pretrain/saved_model)|
| [CPT-HG](https://yuanfulu.github.io/publication/CIKM-CPT.pdf)    | [HGT](https://arxiv.org/pdf/2003.01332.pdf)  | DBLP + YELP + Aminer | N/A |N/A|
| [MPG](https://pubmed.ncbi.nlm.nih.gov/33940598/)       | [MolGNet](https://pubmed.ncbi.nlm.nih.gov/33940598/)    | ZINC + ChEMBL (11M) | 53M|N/A|
| [LP-Info](https://arxiv.org/pdf/2201.01702.pdf)    | 5-layer GIN     | ZINC15 (2M) + ChEMBL (456K)  | ~ 2M|[Link](https://github.com/Shen-Lab/GraphCL_Automated/tree/master/transferLearning_MoleculeNet_PPI_LP)|
| [SimGRACE](https://arxiv.org/pdf/2202.03104.pdf)  | 5-layer GIN       | ZINC15 (2M) + ChEMBL (456K)  | ~ 2M |[Link](https://github.com/junxia97/SimGRACE)|
| [MolCLR](https://arxiv.org/pdf/2102.10056.pdf)     | GCN + GIN     | PubChem (10M)| N/A|[Link](https://github.com/yuyangw/MolCLR/tree/master/ckpt)|
| [DMP](https://arxiv.org/pdf/2106.10234.pdf)       | DeeperGCN + Transformer   | PubChem (110M) | 104.1 M|N/A|
| [ChemRL-GEM](https://www.nature.com/articles/s42256-021-00438-4)  | GeoGNN     | ZINC15 (20M)| N/A                 |[Link](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/pretrained_compound/ChemRL/GEM)|
| [KCL](https://arxiv.org/pdf/2112.00544.pdf)     | GCN + KMPNN    | ZINC15 (250K)   | < 1M        |N/A|
| [3D Infomax](https://arxiv.org/pdf/2110.04126.pdf)    | PNA   | QM9(50K) + GEOM-drugs(140K) + QMugs(620K)       | N/A |[Link](https://github.com/HannesStark/3DInfomax)|
| [GraphMVP](https://openreview.net/pdf?id=xQUe1pOKPam)  | GIN + SchNet | GEOM (50k)             | ~ 2M  |[Link](https://github.com/chao1224/GraphMVP)|



<a name="Datasets"></a>
## Pretraing Datasets
| Name           | Category  | Download Link |
| :----------:  | :---------: | :------: |
| ZINC  | Molecular Graph |    [Link](https://zinc15.docking.org/) |
| CheMBL | Molecular Graph |   [Link](https://chembl.gitbook.io/chembl-interface-documentation/downloads) |
| PubChem | Molecular Graph |  [Link](https://pubchem.ncbi.nlm.nih.gov) |
| QM9  | Molecular Graph| [Link](http://quantum-machine.org/datasets/) |
| QMugs  |Molecular Graph |   [Link](https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM) |
| GEOM | Molecular Graph |  [Link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF) |

<a name="Cite"></a>
## __Citation (.bib)__ </br>
```
@article{xia2022survey,
title={A Survey of Pretraining on Graphs: Taxonomy, Methods, and Applications},
author={Xia, Jun and Zhu, Yanqiao and Du, Yuanqi and Stan Z., Li},
journal={arXiv preprint arXiv:2202.07893},
year={2022}}
```
<a name="Acknow"></a>
## Acknowledgements
+ [awesome-pretrained-chinese-nlp-models](https://github.com/lonePatient/awesome-pretrained-chinese-nlp-models)
+ [awesome-self-supervised-gnn](https://github.com/ChandlerBang/awesome-self-supervised-gnn)
+ [awesome-self-supervised-learning-for-graphs](https://github.com/SXKDZ/awesome-self-supervised-learning-for-graphs)

