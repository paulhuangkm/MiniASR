
import logging
from os.path import join
import time
import torch
from torch import nn


class TransformerLM(nn.Module):
    def __init__(self, tokenizer, args):
        super().__init__()
        
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size