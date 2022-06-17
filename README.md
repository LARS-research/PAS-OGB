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

|  Dataset   | Method  | Test AUC   | Validation AUC  | #Parameters |Hardware  |
|  ----  | ----  | ----  | ----  |----  |----  |
| ogbg-molhiv  | PAS | 0.8221 ± 0.0017  | 0.8178 ± 0.0031 |26,706,952| RTX3090 |
| ogbg-molhiv  | PAS+FingerPrint | 0.8420 ± 0.0015  | 0.8238 ± 0.0028 | 26,706,953| RTX3090 |


### ogbg-molpcba dataset

|  Dataset   | Method  | Test AP   | Validation AP   |Hardware  |
|  ----  | ----  | ----  | ----  |----  |
| ogbg-molpcba  | PAS | 0.3012 ± 0.0039  | 0.3151 ± 0.0047 | RTX3090 |

### ogbg-molppa dataset


|  Dataset   | Method  | Test ACC   | Validation ACC  |Hardware  |
|  ----  | ----  | ----  | ----  |----  |
| ogbg-ppa  | PAS | 0.7828 ± 0.0024  | 0.7523 ± 0.0028 | RTX3090 |
| ogbg-ppa  | PAS+F2GNN(hidden_size 128) | 0.7842 ± 0.0023  | 0.7373 ± 0.0021 | RTX3090 |
| ogbg-ppa  | PAS+F2GNN(hidden_size 512) | 0.8201 ± 0.0019  | 0.7720 ± 0.0023 | RTX3090 |


### Training Process for ogbg-molhiv
 1. Search Architecture

```
python hiv_train_search.py --gpu 0 --num_layers 14 --epochs 50 --data ogbg-molhiv
--remove_pooling True
```
2. Extract finerprints and train Random Forest by following [PaddleHelix](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/competition/ogbg_molhiv)
```
python extract_fingerprint.py
python random_forest.py
```
3. Finetune the model.

```
python -u finetune.py --data ogbg-molhiv --gpu 0 --dropout 0.2 --lr 0.1 
--batch_size 256 --gamma 700 --epochs 400 --hidden_size 512 
--arch_filename ./exp_res/ogbg-molhiv-searched_res-20220120-220405-eps0.0-reg1e-05.txt
```
If you want to use the model framework you searched for, please enter your model address after ```--arch_filename```

4. Finetune the model with FingerPrints, the FT model can be found in the release, its name is ```BS_256-NF_full_valid_best_AUC-FP_E_341_R0.pth```.
Create the folder ```model_0206_gamma_500``` and drage the model file into it.

```
python -u finetune_Drop.py --data ogbg-molhiv --gpu 3 --dropout 0.1 --lr 0.005 --batch_size 256 --gamma 100 --epochs 40 --hidden_size 512 --arch_filename ./exp_res/ogbg-molhiv-searched_res-20220120-220405-eps0.0-reg1e-05.txt
```

### Training Process for ogbg-molpcba

 1. Search Architecture
```
python train_search.py --gpu 0 --num_layers 5 --epochs 5 --data ogbg-molpcba
--remove_pooling True
```
2. Finetune the model.

```
python finetune.py --gpu 0 --dropout 0.5 --lr 0.001 --batch_size 100 --num_layers 5 --epochs 100 --hidden_size 384  --arch_filename ./exp_res/ogbg-molpcba-searched_res-20220316-235183-eps0.0-reg1e-05.txt
```


### Training Process for ogbg-ppa(PAS)

 1. Search Architecture
```
python train_search.py --gpu 0 --num_layers 3 --epochs 5 --data ogbg-ppa
--remove_pooling True
```
2. Finetune the model.

```
python finetune.py --gpu 1 --dropout 0.5 --lr 0.0005 --batch_size 24 --num_layers 3 --epochs 200 --hidden_size 512 --arch_filename ./exp_res/ogbg-ppa-searched_res-20220217-214538-eps0.0-reg1e-05.txt
```

### Training Process for ogbg-ppa(PAS+F2GNN)

 1. Search Architecture
```
python train_search.py --gpu 0 --num_layers 5 --epochs 10 --batch_size 24 --hidden_size 64 --data ogbg-ppa
--remove_pooling True
```
2. Finetune the model.

```
python finetune.py --gpu 1 --dropout 0.5 --lr 0.0005 --batch_size 24 --num_layers 5 --epochs 200 --warmup_epochs 20 --hidden_size 512 --arch_filename ./exp_res/ogbg-ppa-searched_res-20220415-112841-eps0.0-reg1e-05.tx
```

### Cite
Please kindly cite our paper if you use this code:
```
@inproceedings{wei2021pooling,
  title={Pooling Architecture Search for Graph Classification},
  author={Wei, Lanning and Zhao, Huan and Yao, Quanming and He, Zhiqiang},
  booktitle={CIKM},
  year={2021}
}
```
