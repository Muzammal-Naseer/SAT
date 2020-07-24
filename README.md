# SAT: Stylized Adversarial Training


### Table of Contents  
1) [Evaluate SAT against ROA (unrestricted attack)](#Evaluate-SAT-against-ROA)
2) [Evaluate SAT against restricted attack (PGD, CW, FGSM, MIFGSM)](#Evaluate-SAT-against-restricted-attack ) 
3) [Evaluate SAT against Common Corruptions](#Evaluate-SAT-against-Common-Corruptions)

## SAT against ROA
```
  python test_roa.py 
```


## SAT against restricted attack
```
  python test.py --attack_type pgd --eps 8 --iters 100 --random_restart
```

## SAT against Common Corruptions
Download corrupted CIFAR10 dataset from [augmix](https://github.com/google-research/augmix) and extract to the folder "CIFAR-10-C". Run the following command to observe the robustness gains.

```
  python test_common_corruptions.py 
```

