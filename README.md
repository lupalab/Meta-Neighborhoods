# Meta-Neighborhoods

This is the official repository for our NeurIPS 2020 paper [Meta-Neighborhoods](https://arxiv.org/abs/1909.09140).
```
@article{shan2019meta,
  title={Meta-Neighborhoods},
  author={Shan, Siyuan and Li, Yang and Oliva, Junier},
  journal={arXiv preprint arXiv:1909.09140},
  year={2019}
}
```

## prerequisites
clone our repository and install TensorFlow 1.12.0

## Prepare datasets
the script below help to prepare CIFAR100 and save it to , based on which you can prepare other datasets.
```
python prepare_cifar100.py --save_dir=path_to_save_data
```

## Experiments

### MN+iFilm model
```
(Train) python main.py --logdir=path_to_save_log --data_dir=path_to_save_data --dataset=cifar100 --meta_batch_size=128 --dropout_ratio=0.5 --vanilla=False --dict_size=10000 --update_lr=0.1 --num_updates=1 --fix_v=False --alpha=5.0 --meta_lr=1e-3 --backbone=resnet56 --scalar_lr=True --dot=False --modulate=all --film_dict_size=10 --visualize=False
```
python main.py --logdir=path_to_save_log --data_dir=path_to_save_data --dataset=cifar100 --meta_batch_size=128 --finetune=False --dropout_ratio=0.5 --vanilla=False --update_batch_size=5000 --update_lr=0.1 --num_updates=1 --fix_v=False --alpha=5.0 --meta_lr=1e-3 --backbone=resnet56 --scalar_lr=True --dot=False --modulate=all --film_dict_size=10
```
- for `synthetic` dataset
```
(Train) python scripts/train.py --cfg_file=./exp/syn/params.json
(ACFlow DFA) python scripts/test_cls_dfa.py --cfg_file=./exp/syn/params.json
(ACFlow SFA) python scripts/test_cls_sfa.py --cfg_file=./exp/syn/params.json
```
- for `gas` dataset
```
(Train) python scripts/train.py --cfg_file=./exp/gas/params.json
(ACFlow DFA) python scripts/test_cls_dfa.py --cfg_file=./exp/gas/params.json
(ACFlow SFA) python scripts/test_cls_sfa.py --cfg_file=./exp/gas/params.json
```
