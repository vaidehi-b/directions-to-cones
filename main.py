"""
Concept Cones: Refusal Direction and Cone Optimization

Main experiment script. Reproduces the notebook pipeline:
1. Load models (Gemma, Qwen)
2. Load harmful/harmless instruction datasets
3. Compute DIM refusal directions
4. Generate target distributions
5. Run RDO (Refusal Direction Optimization)
6. Run RCO (Refusal Cone Optimization)
7. Evaluate via ablation and refusal rate
"""

import torch
import functools
import textwrap
import gc

from colorama import Fore
from transformer_lens import utils

from config import DEVICE
from models import load_gemma, load_qwen, get_tokenize_fn
from data import get_harmful_instructions, get_harmless_instructions
from utils import (
    cos_sim, dim_direction, direction_ablation_hook, cone_ablation_hook,
    get_intervention_generations, generate_data, detect_refusal, refusal_rate,
)
from optimization import (
    refusal_direction_optimization, refusal_cone_optimization, sample,
)


def print_generations(intervention_generations, baseline_generations, instructions):
    for i in range(len(intervention_generations)):
        print(f"INSTRUCTION {i}: {repr(instructions[i])}")
        print(Fore.GREEN + f"BASELINE COMPLETION:")
        print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RED + f"INTERVENTION COMPLETION:")
        print(textwrap.fill(repr(intervention_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RESET)


def main():
    # ---- Load models ----
    print("Loading Gemma model...")
    model = load_gemma()
    tokenize_instructions_fn = get_tokenize_fn(model, template='gemma')

    print("Loading Qwen model...")
    model1 = load_qwen()
    tokenize_instructions_fn1 = get_tokenize_fn(model1, template='qwen')

    # ---- Load datasets ----
    print("Loading datasets...")
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()

    print("Harmful instructions:")
    for i in range(4):
        print(f"\t{repr(harmful_inst_train[i])}")
    print("Harmless instructions:")
    for i in range(4):
        print(f"\t{repr(harmless_inst_train[i])}")

    # ---- Compute DIM refusal directions ----
    print("Computing DIM refusal directions...")
    ref_dir_q = dim_direction(model1, tokenize_instructions_fn1, 32, harmful_inst_train, harmless_inst_train, 20)
    print(f"Qwen DIM direction shape: {ref_dir_q.shape}")

    ref_dir_g = dim_direction(model, tokenize_instructions_fn, 32, harmful_inst_train, harmless_inst_train, 20)
    print(f"Gemma DIM direction shape: {ref_dir_g.shape}")

    # ---- Generate target data ----
    print("Generating target distributions...")
    N = 10
    data = []
    for i in range(N):
        d = generate_data(model, tokenize_instructions_fn, ref_dir_g, 3, [harmful_inst_train[i]], [harmless_inst_train[i]])
        data.append(d)

    torch.cuda.empty_cache(); gc.collect()

    # ---- RDO ----
    print("Running Refusal Direction Optimization (RDO)...")
    rdo_r = refusal_direction_optimization(model1, tokenize_instructions_fn1, data, coef=0.31, num_steps=20)
    print(f"RDO direction: {rdo_r}")
    print(f"Cosine similarity with DIM direction: {cos_sim(rdo_r, ref_dir_q)}")

    # Evaluate RDO
    print("Evaluating RDO ablation...")
    intervention_layers = list(range(model1.cfg.n_layers))
    hook_fn = functools.partial(direction_ablation_hook, direction=rdo_r)
    intervention_generations, baseline_generations = get_intervention_generations(
        model1, tokenize_instructions_fn1, 20, harmful_inst_test, hook_fn,
        intervention_layers, act_names=['resid_pre', 'resid_mid', 'resid_post']
    )
    print_generations(intervention_generations, baseline_generations, harmful_inst_test)
    print(f"RDO refusal rate: {refusal_rate(intervention_generations, detect_refusal)}")

    torch.cuda.empty_cache(); gc.collect()

    # ---- RCO ----
    print("Running Refusal Cone Optimization (RCO)...")
    rco_b = refusal_cone_optimization(model1, tokenize_instructions_fn1, data, d=model1.cfg.d_model, n=5, steps=50)
    print(f"RCO basis shape: {rco_b.shape}")

    rco_r = sample(rco_b)
    print(f"Sampled RCO direction: {rco_r}")

    # Evaluate RCO
    print("Evaluating RCO ablation...")
    intervention_layers = list(range(model1.cfg.n_layers))
    hook_fn = functools.partial(direction_ablation_hook, direction=rco_b[0, :])
    intervention_generations, baseline_generations = get_intervention_generations(
        model1, tokenize_instructions_fn1, 20, harmful_inst_test, hook_fn,
        intervention_layers, act_names=['resid_pre', 'resid_mid', 'resid_post']
    )
    print_generations(intervention_generations, baseline_generations, harmful_inst_test)
    print(f"RCO refusal rate: {refusal_rate(intervention_generations, detect_refusal)}")

    torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
