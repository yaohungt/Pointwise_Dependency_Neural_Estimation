# CIFAR10 Experiments

The code is adapted from [here](https://github.com/Philip-Bachman/amdim-public)
## Usage

### Self-supervised Representation Learning

CPC

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --loss nce --epochs 300 --ndf 192 --n_rkhs 1536 \
                                      --batch_size 480 --tclip 20.0 --n_depth 8 --dataset C10 \
                                      --rkhs --l2_reg 0.05 --use_tanh_clip
```

PCC 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --loss JS --epochs 300 --ndf 192 --n_rkhs 1536 \
                                      --batch_size 480 --tclip 20.0 --n_depth 8 --dataset C10 \
                                      --rkhs --l2_reg 0.05 --grad_clip_value 1.0 --use_tanh_clip \
                                      --use_bn 1
```

D-RFC

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --loss ours --epochs 300 --ndf 192 --n_rkhs 1536 \
                                      --batch_size 480 --tclip 20.0 --n_depth 8 --dataset C10 --rkhs \
                                      --relative_ratio 0.01 --grad_clip_value 1.0 --use_bn 1 
```


