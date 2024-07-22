# symbiotic-gnn-robustness

Code for the paper [Poisoning × Evasion: Symbiotic Adversarial Robustness for Graph Neural Networks](https://arxiv.org/abs/2312.05502)
> It is well-known that deep learning models are vulnerable to small input perturbations. Such perturbed instances are called adversarial examples. Adversarial examples are commonly crafted to fool a model either at training time (poisoning) or test time (evasion). In this work, we study the symbiosis of poisoning and evasion. We show that combining both threat models can substantially improve the devastating efficacy of adversarial attacks. Specifically, we study the robustness of Graph Neural Networks (GNNs) under structure perturbations and devise a memory-efficient adaptive end-to-end attack for the novel threat model using first-order optimization.

## Directory contents

`attacks.py` and `models.py` contain the implementations of the attacks and models used in the comparisons in the paper.

`scripts/main.py` is the main script to reproduce the experiments. The command line arguments enables choice of attacks, models, and various other params. 

