# SAT: Stylized Adversarial Training

[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en), [Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en), [Munawar Hayat](https://scholar.google.ch/citations?user=Mx8MbWYAAAAJ&hl=en&oi=ao), [Fahad Shahbaz Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao), and [Fatih Porikli](https://scholar.google.com/citations?user=VpB8NZ8AAAAJ&hl=en)

**Paper**: https://arxiv.org/abs/2007.14672

> **Abstract:** *Deep Convolution Neural Networks (CNNs) can easily be fooled by subtle, imperceptible changes to the input images. To address this vulnerability, adversarial training creates perturbation patterns and includes them in the training set to robustify the model. In contrast to existing adversarial training methods that only use class-boundary information (e.g., using a cross entropy loss), we propose to exploit additional information from the feature space to craft stronger adversaries that are in turn used to learn a robust model. Specifically, we use the style and content information of the target sample from another class, alongside its class boundary information to create adversarial perturbations. We apply our proposed multi-task objective in a deeply supervised manner, extracting multi-scale feature knowledge to create maximally separating adversaries. Subsequently, we propose a max-margin adversarial training approach that minimizes the distance between source image and its adversary and maximizes the distance between the adversary and the target image. Our adversarial training approach demonstrates strong robustness compared to state of the art defenses, generalizes well to naturally occurring corruptions and data distributional shifts, and retains the model accuracy on clean examples.*
> 

## Citation

```bibtex
    @InProceedings{naseer2020stylized,
        title={Stylized Adversarial Defense},
        author={Naseer, Muzammal and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Porikli, Fatih},
        journal={arXiv preprint arXiv:2007.14672},
        year={2020}
    }
```


![Learning Algo](/assests/method_fig.jpg)


### Table of Contents  
1) [Download Pretrained SAT](#Download-Pretrained-SAT)
1) [Evaluate SAT against ROA (unrestricted attack)](#Evaluate-SAT-against-ROA)
2) [Evaluate SAT against restricted attack (PGD, CW, FGSM, MIFGSM)](#Evaluate-SAT-against-restricted-attack ) 
3) [Evaluate SAT against Common Corruptions](#Evaluate-SAT-against-Common-Corruptions)


## Pretrained SAT

Download the Pretrained SAT model from [here](https://drive.google.com/file/d/1wbCaKW0S8aK0BC0knpnxE_A9YfYQFW91/view?usp=sharing) and put it int the folder "pretrained_models"

## SAT against ROA

**Note that SAT is not trained against ROA but it still performs better than Trades/Feature scaterring.**

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
![Results](/assests/robustness_against_common_corruptions.png)
