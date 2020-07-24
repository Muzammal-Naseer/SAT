# SAT: Stylized Adversarial Training


### Table of Contents  
1) [Evaluate SAT against ROA (unrestricted attack)](#Evaluate-SAT-against-ROA)
2) [Evaluate SAT against restricted attack (PGD, CW, FGSM, MIFGSM)](#Evaluate-SAT-against-restricted-attack ) 


## Evaluate SAT against ROA
```
  python test_roa.py 
```


## Evaluate SAT against restricted attack
```
  python test.py --attack_type pgd --eps 8 --iters 100 --random_restart
```

