import torch
import functools

from transformer_lens import HookedTransformer
from huggingface_hub import login
from typing import List
from torch import Tensor
from transformers import AutoTokenizer
from jaxtyping import Int

from config import TOKEN, MODEL_PATH, MODEL1_PATH, DEVICE, GEMMA_CHAT_TEMPLATE, QWEN_CHAT_TEMPLATE


def load_gemma(model_path=MODEL_PATH, device=DEVICE):
    login(token=TOKEN)

    model = HookedTransformer.from_pretrained_no_processing(
        model_path,
        device=device,
        dtype=torch.float32,
        default_padding_side='left',
    )

    model.tokenizer.padding_side = 'left'
    model.tokenizer.pad_token = '<|extra_0|>'
    return model


def load_qwen(model_path=MODEL1_PATH, device=DEVICE):
    login(token=TOKEN)

    model = HookedTransformer.from_pretrained_no_processing(
        model_path,
        device=device,
        dtype=torch.float32,
        default_padding_side='left'
    )

    model.tokenizer.padding_side = 'left'
    model.tokenizer.pad_token = '<|extra_0|>'
    return model


def tokenize_instructions_gemma_chat(
        instructions: List[str],
        tokenizer: AutoTokenizer
) -> Int[Tensor, 'batch_size seq_len']:
    prompts = [GEMMA_CHAT_TEMPLATE.format(instruction=instruction) for instruction in instructions]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors='pt').input_ids


def tokenize_instructions_qwen_chat(
        instructions: List[str],
        tokenizer: AutoTokenizer
) -> Int[Tensor, 'batch_size seq_len']:
    prompts = [QWEN_CHAT_TEMPLATE.format(instruction=instruction) for instruction in instructions]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors='pt').input_ids


def get_tokenize_fn(model, template='gemma'):
    if template == 'gemma':
        return functools.partial(tokenize_instructions_gemma_chat, tokenizer=model.tokenizer)
    elif template == 'qwen':
        return functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer)
    else:
        raise ValueError(f"Unknown template: {template}")
