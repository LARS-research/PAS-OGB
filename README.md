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
| ogbg-molhiv  | PAS | 0.8221 $\pm$ 0.0021  | 单元格 |单元格 |
| 单元格  | 单元格 | 单元格  | 单元格 |单元格 |


|  Dataset   |  Method | |Test AUROC| |Validation AUROC| |Hardware|
|  ----  | ---- |  ----  | ----  |  ----  |



| ogbg-molhiv  | PAS | 
| 单元格  | 单元格 |
### ogbg-molpcba dataset
|  表头   | 表头  | 表头   | 表头  |表头  |
|  ----  | ----  | ----  | ----  |----  |
| 单元格  | 单元格 | 单元格  | 单元格 |单元格 |
| 单元格  | 单元格 | 单元格  | 单元格 |单元格 |
### ogbg-molppa dataset
