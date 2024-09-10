# https://arena3-chapter1-transformer-interp.streamlit.app/~/+/[1.1]_Transformer_from_Scratch

import os
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict, Callable
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import webbrowser

# Make sure exercises are in the path
chapter = r"chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_transformer_from_scratch"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
import part1_transformer_from_scratch.solutions as solutions
import part1_transformer_from_scratch.tests as tests

device = t.device("mps" if t.backends.mps.is_available() else "cpu")

MAIN = __name__ == '__main__'

# HookedTransformer docs: https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.HookedTransformer.html
reference_gpt2 : HookedTransformer = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device
)

def examine_vocab():
    """Didactic only"""
    sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])
    print(sorted_vocab[:20])
    print()
    print(sorted_vocab[250:270])
    print()
    print(sorted_vocab[500:520])
    print()
    print(sorted_vocab[990:1010])
    print()
    print(sorted_vocab[-20:])
    print()

    lengths = dict.fromkeys(range(3, 8), "")
    for tok, idx in sorted_vocab:
        if not lengths.get(len(tok), True):
            lengths[len(tok)] = tok

    for length, tok in lengths.items():
        print(f"{length}: {tok}")

# examine_vocab()

out = reference_gpt2('Hello World')[0]
print(out.shape)

# Note that everywhere below the batch size is 1 because we're dealing with a 
# single text snippet. It's there as the leading dimension but should be ignored.

reference_text = "The most likely way for AI to take over the world is"
tokens: t.Tensor
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(f'tokens.shape: {tokens.shape} ({type(tokens)})') # [batch, seq_len]. Each element is an int < d_vocab.
print(reference_gpt2.to_str_tokens(tokens))

# From our input of shape [batch, seq_len], 
#  we get output of shape [batch, seq_len, vocab_size] (plus an activation cache)
# The [i, j, :]-th element of our output is a vector of logits representing 
#   our prediction for the j+1-th token in the i-th sequence.
logits: t.Tensor
logits, activation_cache = reference_gpt2.run_with_cache(tokens, device=device)
print(f'logits.shape: {logits.shape}') # [batch, seq_len, vocab_size]

# Convert the logits to a probability distribution with softmax
probs = logits.softmax(dim=-1)
print(f'probs.shape: {probs.shape}') # [batch, seq_len, vocab_size]

# Show predictions at each location
most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])
print(f'Most likely next tokens:')
print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))
print()

# Map distribution to a token
logits_at_last_position = logits[0, -1]
next_token = logits_at_last_position.argmax(dim=-1)
next_token_s = reference_gpt2.to_string([next_token])
print(f'Predicted token: {repr(next_token_s)}')
print()

# Autoregress 10 steps:
for i in range(10):
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)
