# PAS-OGB
We apply our novel framework PAS to automatically learn data-specific pooling architectures for graph classification task, which has been published in CIKM 2021: [Pooling Architecture Search for Graph Classification](https://arxiv.org/pdf/2108.10587.pdf).
We verified the effect on [Open Graph Benchmark (OGB) datasets](https://ogb.stanford.edu/docs/leader_graphprop/) (ogbg-molhiv, ogbg-molpcba and ogbg-ppa) based on OGB examples. Thanks for their contributions.

## Requirements
```
Python>=3.7 Pytorch==1.8.0 pytorch_geometric==2.0.1 numpy==1.21.2 pandas==1.3.4 
scikit-learn==0.24.2 ogb>=1.3.2 deep_gcns_torch LibAUC 
```

## Results on OGB
### ogbg-molhiv dataset

|  Dataset   | Method  | Test AUC   | Validation AUC  |Hardware  |
|  ----  | ----  | ----  | ----  |----  |
| ogbg-molhiv  | PAS | 0.8221 ± 0.0017  | 0.8178 ± 0.0031 | RTX3090 |
| ogbg-molhiv  | PAS+HIG | 0.8416 ± 0.0019  | 0.8186 ± 0.0039 |RTX3090 |

### ogbg-molpcba dataset

|  Dataset   | Method  | Test AP   | Validation AP  |Hardware  |
|  ----  | ----  | ----  | ----  |----  |
| ogbg-molhiv  | PAS+HIG | 0.3012 ± 0.0039  | 0.3151 ± 0.0047 | RTX3090 |

### ogbg-molppa dataset


|  Dataset   | Method  | Test ACC   | Validation ACC  |Hardware  |
|  ----  | ----  | ----  | ----  |----  |
| ogbg-ppa  | PAS | 0.7828 ± 0.0024  | 0.7523 ± 0.0028 | RTX3090 |
