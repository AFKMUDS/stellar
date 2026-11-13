import torch
from torch.nn import functional as F
from .tokens import BOS, EOS

def _masked_mean(values, mask):
    return (values*mask).sum()/mask.sum().clamp_min(1)

def training_loss(model, output, batch, cfg):
    target, mask = batch["target"], batch["mask"]
    n = mask.sum()
    nll = -output["log_probs"].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    language_sum = (nll*mask).sum()
    zero = language_sum*0
    h = output["hidden"]
    terms = {}
    future = []
    segments = (batch["input"] == BOS).long().cumsum(dim=1)
    for horizon in model.cfg.future_horizons:
        shift = horizon-1
        if shift >= target.shape[1]:
            continue
        labels = target[:, shift:]
        valid = (mask[:, :target.shape[1]-shift] & mask[:, shift:] & (labels < 256)
                 & (segments[:, :target.shape[1]-shift] == segments[:, shift:]))
        scores = model.future_heads[str(horizon)](h[:, :target.shape[1]-shift]).float()
        by_position = F.cross_entropy(scores.reshape(-1, 256), labels.clamp(max=255).reshape(-1),
                                      reduction="none").reshape_as(labels)
        future.append(_masked_mean(by_position, valid))
    terms["future"] = torch.stack(future).mean() if future else zero
    past = []
    for index, lag in enumerate(model.cfg.past_lags):
        labels = output["past_targets"][:, :, index]
        valid = mask & (labels >= 0) & (labels < 256)
        scores = model.past_heads[str(lag)](h).float()
        losses = F.cross_entropy(scores.reshape(-1, 256), labels.clamp(0, 255).reshape(-1),
                                 reduction="none").reshape_as(labels)
        past.append(_masked_mean(losses, valid))
    terms["past"] = torch.stack(past).mean() if past else zero
    boundary = ((batch["input"] == 32) | (batch["input"] == 10)
                | (batch["input"] == 9) | (batch["input"] == 13))
    boundary_mask = mask & boundary & (target < 256)
    boundary_loss = F.cross_entropy(model.boundary_head(h).float().reshape(-1, 256),
                                    target.clamp(max=255).reshape(-1), reduction="none").reshape_as(target)
    terms["boundary"] = _masked_mean(boundary_loss, boundary_mask)
    stop_loss = F.binary_cross_entropy_with_logits(output["stop_logits"].float(),
        (target == EOS).float(), reduction="none",
        pos_weight=output["stop_logits"].new_tensor(cfg.stop_positive_weight))
    terms["stop"] = _masked_mean(stop_loss, mask)
    total = language_sum + n*(cfg.future_weight*terms["future"]
        + cfg.past_weight*terms["past"] + cfg.boundary_weight*terms["boundary"]
        + cfg.stop_weight*terms["stop"])
    return total, n, {"language_nats": language_sum/n.clamp_min(1), **terms}

