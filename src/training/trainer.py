from typing import Any

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import get_dataloaders
from src.model.model import get_model
from src.utils.config_loader import load_configs