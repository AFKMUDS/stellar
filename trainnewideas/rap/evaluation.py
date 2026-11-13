"""Held-out next-symbol scoring and clearly labeled frozen-model interventions."""
import math
import numpy as np
import torch
from .data import StreamBatcher
from .runtime import autocast

def _entropy(prob):
    prob = np.asarray(prob, dtype=np.float64)
    return float(-np.sum(prob*np.log2(np.maximum(prob, 1e-300))))

@torch.no_grad()
def evaluate(model, corpus, cfg, device, precision, *, target_budget=None,
             input_limit=None, transition_policy="native", retrieval_disabled=False,
             collect_diagnostics=False):
    model.eval()
    budget = target_budget or cfg.validation_target_tokens
    limit = input_limit or cfg.validation_input_limit
    stream = StreamBatcher(corpus, 1, cfg.chunk_length, cfg.seed+9000, shuffle=False)
    state = model.initial_state(1, device)
    n = n_bytes = correct = byte_correct = inputs = 0
    nll = byte_nll = 0.0
    core_counts = np.zeros(8, dtype=np.int64)
    head_count = 1 if model.cfg.expression_head == "direct" else 8
    expression_counts = np.zeros(head_count, dtype=np.int64)
    core_mass = np.zeros(8, dtype=np.float64)
    expression_mass = np.zeros(head_count, dtype=np.float64)
    core_entropy = expression_entropy = 0.0
    rows_g, rows_expert = [], []
    gate_sum = hits = age_max = 0.0
    while n < budget and inputs < limit:
        batch = {k: v.to(device) for k, v in stream.next().items()}
        with autocast(device, precision):
            output = model(batch["input"], state, transition_policy=transition_policy,
                           retrieval_disabled=retrieval_disabled)
        state = output["state"].detached()
        selected = batch["mask"].clone()
        positions = selected.flatten().nonzero().flatten()
        if len(positions) > budget-n:
            selected.flatten()[positions[budget-n:]] = False
        target = batch["target"]
        loss = -output["log_probs"].gather(-1, target.unsqueeze(-1)).squeeze(-1)
        chosen = output["log_probs"].argmax(-1)
        raw_selected = selected & (target < 256)
        count = int(selected.sum().item())
        raw_count = int(raw_selected.sum().item())
        n += count; n_bytes += raw_count; inputs += batch["input"].numel()
        nll += float(loss[selected].sum().item())
        byte_nll += float(loss[raw_selected].sum().item())
        correct += int((chosen[selected] == target[selected]).sum().item())
        byte_correct += int((chosen[raw_selected] == target[raw_selected]).sum().item())
        core_prob = output['transition_probs'][selected].double()
        expression_prob = output['expression_probs'][selected].double()
        core_mass += core_prob.sum(0).cpu().numpy()
        expression_mass += expression_prob.sum(0).cpu().numpy()
        core_entropy += float(-(core_prob*core_prob.clamp_min(1e-300).log2()).sum().item())
        expression_entropy += float(-(expression_prob*expression_prob.clamp_min(1e-300).log2()).sum().item())
        core = core_prob.argmax(-1).cpu().numpy()
        expression = expression_prob.argmax(-1).cpu().numpy()
        core_counts += np.bincount(core, minlength=8)
        expression_counts += np.bincount(expression, minlength=head_count)
        gate_sum += float(output["gate_mean"][selected].sum().item())
        hits += float(output["retrieval_hits"][selected].sum().item())
        age_max = max(age_max, float(output["retrieval_ages"].max().item()))
        if collect_diagnostics and count:
            maps = output["map_indices"][selected]
            true = target[selected]
            encoded = maps.gather(-1, true[:, None, None].expand(-1, head_count, 1)).squeeze(-1)
            expert = output["code_log_probs"][selected].gather(-1, encoded)
            rows_g.append(output["expression_probs"][selected].cpu().double().numpy())
            rows_expert.append(expert.cpu().double().numpy())
    result = dict(supervised_symbols=n, raw_bytes=n_bytes, input_symbols=inputs,
        language_nats=nll/n if n else None, bits_per_byte=byte_nll/max(n_bytes, 1)/math.log(2),
        symbol_accuracy=correct/max(n, 1), byte_accuracy=byte_correct/max(n_bytes, 1),
        transition_counts=core_counts.tolist(), expression_counts=expression_counts.tolist(),
        mean_transition_weights=(core_mass/max(n, 1)).tolist(),
        mean_expression_weights=(expression_mass/max(n, 1)).tolist(),
        transition_entropy_bits=core_entropy/max(n, 1),
        expression_entropy_bits=expression_entropy/max(n, 1),
        effective_transition_count=2**_entropy(core_counts/max(n, 1)),
        effective_expression_count=2**_entropy(expression_counts/max(n, 1)),
        mean_slice_gate=gate_sum/max(n, 1), retrieval_hits_per_symbol=hits/max(n, 1),
        oldest_read_age_symbols=int(age_max), transition_policy=transition_policy,
        retrieval_disabled=retrieval_disabled,
        complete_target_budget=(n >= budget))
    if not n:
        result["note"] = "No supervised targets reached within the input limit; increase validation_input_limit."
    if collect_diagnostics and n:
        gate = np.concatenate(rows_g)
        expert_log = np.concatenate(rows_expert)
        expert = np.exp(expert_log)
        native = -np.log(np.maximum((gate*expert).sum(1), 1e-300)).mean()
        shuffled_gate = gate[np.random.default_rng(391).permutation(n)]
        shuffled = -np.log(np.maximum((shuffled_gate*expert).sum(1), 1e-300)).mean()
        hard = -expert_log[np.arange(n), gate.argmax(1)].mean()
        result["expression_intervention"] = dict(native_nats=float(native),
            shuffled_gate_nats=float(shuffled), penalty_bits=float((shuffled-native)/math.log(2)),
            hard_path_nats=float(hard),
            note="Post-hoc gate shuffle preserves marginal usage; diagnostic only, not a deployable predictive score.")
    return result
