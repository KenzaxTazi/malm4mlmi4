# Objective

Use the MAML algorithm to meta-train a classifier, using the Mini ImageNet dataset.

# Requirements

### python3.6 or above
### pytorch > 1.0
Quick install: pip insall torch, torchvision, torchmeta


# Experiments

The below results are single system (single seed) results on the MiniImageNet dataset.


| Setup | Accuracy (%) |
| ------ | ----------- |
| 1-shot, 5-way | 43.9 |
| 5-shot, 5-way | 60.7 |


Note, the number of inner_loop iterations per task was 1 in both of these experiments - increasing the number of inner iterations could boost performance.
