import torch
import torch.nn.functional as F

def log(*args, **kwargs):
    print(*args, **kwargs)

    with open('./training_log.txt', 'a') as f:
        print(*args, **kwargs, file=f)

def info_nce_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates the InfoNCE loss for a batch of logits.
    Assumes logits are (N, N) and labels are just torch.arange(N).
    """
    n = logits.shape[0]
    labels = torch.arange(n, device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_j) / 2.0