import torch
import functools
import einops
import gc

from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
import torch.nn.functional as F

from config import REFUSAL_PHRASES


# ---- Similarity ----

def cos_sim(dir1, dir2):
    return F.cosine_similarity(dir1, dir2, dim=0).item()


# ---- Hook functions ----

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return (activation - proj)


def cone_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    B
):
    for i in range(B.shape[0]):
        direction = B[i, :]
        proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
        activation = activation - proj
    return activation


def direction_steering_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    coef: float,
    steering_vector: Float[Tensor, "d_act"]
):
    return activation + coef * steering_vector


# ---- Generation ----

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 30,
    fwd_hooks=[],
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)  # greedy sampling (temperature=0)
            all_toks[:, -max_tokens_generated + i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)


def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks=[],
    max_tokens_generated: int = 30,
    batch_size: int = 4,
) -> List[str]:

    generations = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i:i + batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)

    return generations


def get_intervention_generations(model, tokenizer, N, prompts, hook_fn, intervention_layers, act_names=['resid_pre', 'resid_mid', 'resid_post']):
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in act_names]
    intervention_generations = get_generations(model, prompts[:N], tokenizer, fwd_hooks=fwd_hooks)
    baseline_generations = get_generations(model, prompts[:N], tokenizer, fwd_hooks=[])
    return intervention_generations, baseline_generations


# ---- Intervention helpers ----

def run_with_ablation(model, tokenizer, r, prompt, type, layer='all', act_names=['resid_post']):
    intervention_layers = list(range(model.cfg.n_layers))
    if layer != 'all':
        intervention_layers = layer
    hook_fn = functools.partial(direction_ablation_hook, direction=r)
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in act_names]
    logits = model.run_with_hooks(tokenizer(prompt), fwd_hooks=fwd_hooks)
    logits = logits[torch.arange(logits.shape[0]), -1, :150000]
    torch.cuda.empty_cache()
    if type == 'logits':
        return logits
    if type == 'prob':
        return F.softmax(logits, dim=-1)


def run_with_steering(model, tokenizer, r, coef, prompt, type, layer='all', act_names=['resid_post']):
    intervention_layers = list(range(model.cfg.n_layers))
    if layer != 'all':
        intervention_layers = layer
    hook_fn = functools.partial(direction_steering_hook, coef=coef, steering_vector=r)
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in act_names]
    logits = model.run_with_hooks(tokenizer(prompt), fwd_hooks=fwd_hooks)
    logits = logits[torch.arange(logits.shape[0]), -1, :150000]
    torch.cuda.empty_cache()
    if type == 'logits':
        return logits
    if type == 'prob':
        return F.softmax(logits, dim=-1)


def generate_data(model, tokenizer, r, coef, pos_prompt, neg_prompt):
    t_answer = run_with_ablation(model, tokenizer, r, pos_prompt, type='prob')
    t_refusal = run_with_steering(model, tokenizer, r, coef, neg_prompt, type='prob')
    t_retain_log = model(neg_prompt)
    t_retain = F.softmax(t_retain_log[torch.arange(t_retain_log.shape[0]), -1, :], dim=-1)
    torch.cuda.empty_cache()
    return (pos_prompt, neg_prompt, t_answer, t_refusal, t_retain)


# ---- Evaluation ----

def detect_refusal(response: str) -> int:
    return int(any(phrase.lower() in response.lower() for phrase in REFUSAL_PHRASES))


def refusal_rate(generations: list[str], detect_refusal) -> float:
    total = len(generations)
    if total == 0:
        return 0.0
    num_refusals = sum(detect_refusal(gen) for gen in generations)
    return (num_refusals / total) * 100


# ---- DIM direction ----

def dim_direction(model, tokenizer, N, pos_prompts, neg_prompts, layer, pos=-1):
    pos_toks = tokenizer(instructions=pos_prompts[:N])
    neg_toks = tokenizer(instructions=neg_prompts[:N])

    _, pos_cache = model.run_with_cache(pos_toks, names_filter=lambda hook_name: 'resid' in hook_name)
    _, neg_cache = model.run_with_cache(neg_toks, names_filter=lambda hook_name: 'resid' in hook_name)

    pos_mean_act = pos_cache['resid_pre', layer][:, pos, :].mean(dim=0)
    neg_mean_act = neg_cache['resid_pre', layer][:, pos, :].mean(dim=0)

    dim_dir = pos_mean_act - neg_mean_act
    dim_dir = dim_dir / dim_dir.norm()

    del pos_cache, neg_cache
    gc.collect(); torch.cuda.empty_cache()

    return dim_dir
