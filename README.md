# SAT: Stylized Adversarial Training

 **Under Review** ([arXiv link](https://arxiv.org/abs/2007.14672))

![Learning Algo](/assests/method_fig.jpg)


### Table of Contents  
1) [Download Pretrained SAT](#Download-Pretrained-SAT)
1) [Evaluate SAT against ROA (unrestricted attack)](#Evaluate-SAT-against-ROA)
2) [Evaluate SAT against restricted attack (PGD, CW, FGSM, MIFGSM)](#Evaluate-SAT-against-restricted-attack ) 
3) [Evaluate SAT against Common Corruptions](#Evaluate-SAT-against-Common-Corruptions)


## Pretrained SAT

Download the Pretrained SAT model from [here](https://drive.google.com/file/d/1wbCaKW0S8aK0BC0knpnxE_A9YfYQFW91/view?usp=sharing) and put it int the folder "pretrained_models"

## SAT against ROA

*Note that SAT is not trained against ROA but it still performs better than Trades/Feature scaterring.

```
  python test_roa.py 
```


## SAT against Restricted attacks: PGD, MIFGSM etc.
```
  python test.py --attack_type pgd --eps 8 --iters 100 --random_restart
```

## SAT against Common Corruptions
Download corrupted CIFAR10 dataset from [augmix](https://github.com/google-research/augmix) and extract to the folder "CIFAR-10-C". Run the following command to observe the robustness gains.

```
  python test_common_corruptions.py 
```

