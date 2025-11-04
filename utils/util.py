from pathlib import Path
import torch
import torch.nn.functional as F
from models.multimodal_graph_encoder import MultimodalGraphEncoder
from config import CLIP_MODEL_NAME, FUSED_DIM, PROJ_DIM

# Buffer for log messages to reduce file I/O
_log_buffer = []
_LOG_BUFFER_SIZE = 10

def log(*args, **kwargs):
    """
    Optimized logging function with buffering to reduce file I/O.
    Flushes to file every _LOG_BUFFER_SIZE messages or when explicitly called with flush=True.
    """
    message = ' '.join(str(arg) for arg in args)
    print(message, **kwargs)
    
    _log_buffer.append(message)
    
    # Flush to file if buffer is full or if flush is explicitly requested
    should_flush = kwargs.pop('flush', False) or len(_log_buffer) >= _LOG_BUFFER_SIZE
    
    if should_flush:
        flush_log()

def flush_log():
    """Flush buffered log messages to file."""
    global _log_buffer
    if _log_buffer:
        with open('./training_log.txt', 'a') as f:
            for msg in _log_buffer:
                print(msg, file=f)
        _log_buffer = []


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


def load_pretrained_encoder(
    checkpoint_path: str,
    device: str = "cuda"
) -> MultimodalGraphEncoder:
    """
    Initializes the MultimodalGraphEncoder, loads the pretrained 
    state_dict, and returns it in evaluation mode.

    Args:
        checkpoint_path (str): Path to the .pt file.
        device (str): Device to load the model onto ('cuda' or 'cpu').

    Returns:
        MultimodalGraphEncoder: The loaded, pretrained model.
    """
    log(f"Loading pretrained encoder from {checkpoint_path}...")

    model = MultimodalGraphEncoder(
        clip_name=CLIP_MODEL_NAME,
        fused_dim=FUSED_DIM,
        proj_dim=PROJ_DIM
    )

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Please run 'pretrain_end_to_end.py' first."
        )
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except RuntimeError as e:
        log(f"Error loading state_dict: {e}")
        log("This might happen if the model architecture in util.py "
              "does not match the one used during training.")
        raise e

    model.to(device)
    model.eval()

    log("Pretrained encoder loaded successfully and set to eval mode.")
    return model
