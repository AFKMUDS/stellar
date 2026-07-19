import argparse
import math
import os
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from raplib.model import ModelConfig, PrimitiveManifoldLM
from raplib.pathfinder import build_k_hop_tables


def _load_model(ckpt_path: str, device: torch.device) -> tuple[PrimitiveManifoldLM, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("cfg") if isinstance(ckpt, dict) else None

    if isinstance(cfg_dict, dict):
        valid_keys = ModelConfig.__annotations__.keys()
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = ModelConfig(**filtered)
    else:
        cfg = ModelConfig()

    model = PrimitiveManifoldLM(cfg).to(device)
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else ckpt
    if state_dict is None:
        state_dict = ckpt.get("student_state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        raise KeyError("Checkpoint must contain either 'state_dict' or 'student_state_dict'.")
    model.load_state_dict(state_dict)
    model.eval()

    meta = {
        "ckpt": ckpt_path,
        "cfg": asdict(cfg),
        "step": int(ckpt.get("step", -1)) if isinstance(ckpt, dict) else -1,
    }
    return model, meta


def _byte_label(b: int) -> str:
    b = int(b) & 0xFF
    if b == 9:
        return "\\t"
    if b == 10:
        return "\\n"
    if b == 13:
        return "\\r"
    if b == 32:
        return "sp"
    if b in {128, 129, 130}:
        return f"<{b}>"
    if 33 <= b <= 126:
        return chr(b)
    return f"x{b:02X}"


def _pca(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    var = (s ** 2) / max(float(np.sum(s ** 2)), 1e-12)
    return coords, var[:n_components]


def _fit_pca_transform(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    comp = vt[:n_components].T
    coords = centered @ comp
    var = (s ** 2) / max(float(np.sum(s ** 2)), 1e-12)
    return coords, mean, comp


def _project_with_pca(x: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - mean) @ comp


def _trace_runtime(model: PrimitiveManifoldLM, ids: list[int], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    st_vec = model.init_state(batch_size=1, device=device)
    states = []
    prim_acts = []
    last_pos = None
    state_hist = None
    batch_ix = torch.arange(1, device=device)

    if bool(getattr(model, "retrieval_enabled", False)):
        last_pos = torch.full((1, model.cfg.vocab_size), -1, device=device, dtype=torch.long)
        state_hist = []

    with torch.no_grad():
        for i, tok in enumerate(ids):
            x = torch.tensor([tok], device=device, dtype=torch.long)
            retrieval = None
            if last_pos is not None and state_hist is not None:
                retrieval = torch.zeros_like(st_vec)
                cnt = torch.zeros((1, 1), device=device, dtype=st_vec.dtype)
                is_reserved = (x == model.reserved_bytes[0]) | (x == model.reserved_bytes[1]) | (x == model.reserved_bytes[2])
                if i > 0:
                    if bool(getattr(model, "path_retrieval_enabled", False)) and bool((~is_reserved).any().item()):
                        neigh = model.path_neigh_lut[x]
                        dist = model.path_dist_lut[x].to(dtype=st_vec.dtype)
                        neigh_clamped = neigh.clamp(min=0)
                        pos = last_pos.gather(1, neigh_clamped)
                        valid = (neigh >= 0) & (pos >= 0) & (~is_reserved).unsqueeze(1)
                        if bool(valid.any().item()):
                            pos_clamped = pos.clamp(min=0)
                            hist_tensor = torch.stack(state_hist, dim=1)
                            got = torch.gather(
                                hist_tensor,
                                dim=1,
                                index=pos_clamped.unsqueeze(-1).expand(-1, -1, hist_tensor.shape[-1]),
                            )
                            alpha = torch.tensor(float(getattr(model, "path_alpha", 1.0)), device=device, dtype=st_vec.dtype)
                            w = torch.exp(-alpha * dist) * valid.to(dtype=st_vec.dtype)
                            num = (got * w.unsqueeze(-1)).sum(dim=1)
                            den = torch.clamp(w.sum(dim=1, keepdim=True), min=1.0)
                            retrieval = retrieval + (num / den)
                            cnt = cnt + 1.0

                    if bool(getattr(model, "word_map_retrieval_enabled", False)) and bool((~is_reserved).any().item()):
                        neigh = model.word_neigh_lut[x]
                        ww = model.word_w_lut[x].to(dtype=st_vec.dtype)
                        neigh_clamped = neigh.clamp(min=0)
                        pos = last_pos.gather(1, neigh_clamped)
                        valid = (neigh >= 0) & (pos >= 0) & (~is_reserved).unsqueeze(1)
                        if bool(valid.any().item()):
                            pos_clamped = pos.clamp(min=0)
                            hist_tensor = torch.stack(state_hist, dim=1)
                            got = torch.gather(
                                hist_tensor,
                                dim=1,
                                index=pos_clamped.unsqueeze(-1).expand(-1, -1, hist_tensor.shape[-1]),
                            )
                            w = ww * valid.to(dtype=st_vec.dtype)
                            num = (got * w.unsqueeze(-1)).sum(dim=1)
                            den = torch.clamp(w.sum(dim=1, keepdim=True), min=1e-6)
                            retrieval = retrieval + (num / den)
                            cnt = cnt + 1.0

                    if bool(getattr(model, "symmetry_retrieval_enabled", False)) and bool((~is_reserved).any().item()):
                        inv = x ^ 0xFF
                        rev = model.bit_reverse_lut[x]
                        pos_inv = last_pos.gather(1, inv.unsqueeze(1)).squeeze(1)
                        pos_rev = last_pos.gather(1, rev.unsqueeze(1)).squeeze(1)
                        for pos1 in (pos_inv, pos_rev):
                            m = (pos1 >= 0) & (~is_reserved)
                            if bool(m.any().item()):
                                p = pos1.clamp(min=0)
                                hist_tensor = torch.stack(state_hist, dim=1)
                                got = hist_tensor[batch_ix, p]
                                retrieval = retrieval + got * m.to(dtype=st_vec.dtype).unsqueeze(1)
                                cnt = cnt + m.to(dtype=st_vec.dtype).unsqueeze(1)

                retrieval = retrieval / torch.clamp(cnt, min=1.0)

            x_emb = model._get_emb(x)
            if retrieval is not None:
                ctrl_in = torch.cat([x_emb, st_vec, retrieval], dim=-1)
            else:
                ctrl_in = torch.cat([x_emb, st_vec], dim=-1)
            w = torch.softmax(model.controller(ctrl_in), dim=-1)
            logits, st_vec = model.step(x_emb, st_vec, retrieval=retrieval)
            _ = logits
            if last_pos is not None and state_hist is not None:
                state_hist.append(st_vec.detach().clone())
                last_pos.scatter_(1, x.unsqueeze(1), torch.full((1, 1), i, device=device, dtype=torch.long))
            states.append(st_vec[0].detach().cpu().float().numpy())
            prim_acts.append(w[0].detach().cpu().float().numpy())

    return np.stack(states, axis=0), np.stack(prim_acts, axis=0)


def _sample_next_id(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(float(temperature), 1e-6)
    if int(top_k) > 0 and int(top_k) < logits.shape[-1]:
        v, ix = torch.topk(logits, k=int(top_k), dim=-1)
        probs = torch.softmax(v, dim=-1)
        pick = torch.multinomial(probs, num_samples=1)
        return int(ix.gather(-1, pick).item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _generate_ids(
    model: PrimitiveManifoldLM,
    prompt_ids: list[int],
    device: torch.device,
    max_new_bytes: int,
    temperature: float,
    top_k: int,
) -> list[int]:
    if int(max_new_bytes) <= 0:
        return []
    model.eval()
    ids = list(prompt_ids)
    state = None
    with torch.no_grad():
        if ids:
            x = torch.tensor([ids], device=device, dtype=torch.long)
            out = model(x, state=None)
            state = out[1]
            last_id = int(ids[-1])
        else:
            state = model.init_state(batch_size=1, device=device)
            last_id = 0

        gen = []
        for _ in range(int(max_new_bytes)):
            x = torch.tensor([[last_id]], device=device, dtype=torch.long)
            out = model(x, state=state)
            logits, state = out[0], out[1]
            next_id = _sample_next_id(logits[:, -1, :], temperature=float(temperature), top_k=int(top_k))
            gen.append(int(next_id))
            last_id = int(next_id)
    return gen


def _byte_colors(ids: list[int]) -> list[str]:
    colors = []
    for i in ids:
        if i in {128, 129, 130}:
            colors.append("#ff5a5f")
        elif 32 <= i <= 126:
            colors.append("#66ccff")
        else:
            colors.append("#9a9a9a")
    return colors


def _get_neighbor_tables(move_k: int) -> tuple[np.ndarray, np.ndarray]:
    neigh_t, dist_t = build_k_hop_tables(k=int(move_k), reserved=(128, 129, 130), device=torch.device("cpu"))
    return neigh_t.cpu().numpy(), dist_t.cpu().numpy()


def _draw_neighbor_edges(ax, coords: np.ndarray, edge_mode: str, focus_byte: int, move_k: int) -> None:
    if str(edge_mode) == "none":
        return
    neigh, dist = _get_neighbor_tables(move_k=int(move_k))
    if str(edge_mode) == "focus":
        srcs = [int(focus_byte) & 0xFF]
    else:
        srcs = [i for i in range(256) if i not in {128, 129, 130}]
    for src in srcs:
        row_n = neigh[src]
        row_d = dist[src]
        for nb, dd in zip(row_n.tolist(), row_d.tolist()):
            if int(nb) < 0:
                continue
            nb = int(nb)
            if nb in {128, 129, 130}:
                continue
            x0, y0 = coords[src]
            x1, y1 = coords[nb]
            alpha = 0.16 if str(edge_mode) == "all" else 0.5
            lw = 0.5 if float(dd) > 1.0 else 1.0
            ax.plot([x0, x1], [y0, y1], color="#444444", alpha=alpha, linewidth=lw, zorder=1)


def _render_composite_page(
    coords: np.ndarray,
    ids: list[int],
    labels: list[str],
    page_ids: list[int],
    out_path: str,
    title: str,
    annotate: bool,
    trajectory_ids: list[int],
    trajectory_xy: np.ndarray,
    prim_acts: np.ndarray,
    prompt_len: int,
    edge_mode: str,
    focus_byte: int,
    move_k: int,
) -> None:
    fig = plt.figure(figsize=(18, 10), dpi=180)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1.0], height_ratios=[1.0, 1.0], wspace=0.16, hspace=0.18)
    ax_left = fig.add_subplot(gs[:, 0])
    ax_rt = fig.add_subplot(gs[0, 1])
    ax_rb = fig.add_subplot(gs[1, 1])

    colors = _byte_colors(ids)
    _draw_neighbor_edges(ax_left, coords, edge_mode=edge_mode, focus_byte=focus_byte, move_k=move_k)
    ax_left.scatter(coords[:, 0], coords[:, 1], s=26, c=colors, alpha=0.35, zorder=2)
    if page_ids:
        page_arr = np.asarray(page_ids, dtype=np.int64)
        ax_left.scatter(coords[page_arr, 0], coords[page_arr, 1], s=46, c=np.asarray(colors, dtype=object)[page_arr], alpha=0.95, zorder=3)
    if annotate:
        for byte_id in page_ids:
            x, y = coords[int(byte_id)]
            ax_left.text(x, y, f"{byte_id}:{labels[int(byte_id)]}", fontsize=6.5, alpha=0.85, zorder=4)
    if trajectory_ids:
        traj_byte_xy = coords[np.asarray(trajectory_ids, dtype=np.int64)]
        p_len = max(0, min(int(prompt_len), len(trajectory_ids)))
        if p_len > 0:
            ax_left.plot(traj_byte_xy[:p_len, 0], traj_byte_xy[:p_len, 1], color="#111111", linewidth=1.3, alpha=0.75, zorder=4)
            ax_left.scatter(traj_byte_xy[:p_len, 0], traj_byte_xy[:p_len, 1], c=np.arange(p_len), cmap="Blues", s=28, zorder=5)
        if p_len < len(trajectory_ids):
            ax_left.plot(traj_byte_xy[p_len - 1 :, 0], traj_byte_xy[p_len - 1 :, 1], color="#c2185b", linewidth=1.3, alpha=0.8, zorder=5)
            ax_left.scatter(traj_byte_xy[p_len:, 0], traj_byte_xy[p_len:, 1], c=np.arange(len(trajectory_ids) - p_len), cmap="plasma", s=30, zorder=6)
        for i, (x, y) in enumerate(traj_byte_xy.tolist()):
            ax_left.text(x, y, str(i), fontsize=6, color="#7a0026", zorder=6)
    ax_left.set_title("Byte space")
    ax_left.set_xlabel("PC1")
    ax_left.set_ylabel("PC2")

    if trajectory_xy.size > 0:
        p_len = max(0, min(int(prompt_len), len(trajectory_ids)))
        if p_len > 0:
            ax_rt.plot(trajectory_xy[:p_len, 0], trajectory_xy[:p_len, 1], color="#333333", linewidth=1.2, alpha=0.7)
            ax_rt.scatter(trajectory_xy[:p_len, 0], trajectory_xy[:p_len, 1], c=np.arange(p_len), cmap="Blues", s=36)
        if p_len < len(trajectory_ids):
            ax_rt.plot(trajectory_xy[p_len - 1 :, 0], trajectory_xy[p_len - 1 :, 1], color="#c2185b", linewidth=1.2, alpha=0.8)
            ax_rt.scatter(trajectory_xy[p_len:, 0], trajectory_xy[p_len:, 1], c=np.arange(len(trajectory_ids) - p_len), cmap="plasma", s=38)
        for i, tok in enumerate(trajectory_ids):
            ax_rt.text(trajectory_xy[i, 0], trajectory_xy[i, 1], _byte_label(tok), fontsize=7, alpha=0.85)
    ax_rt.set_title("Runtime state trajectory")
    ax_rt.set_xlabel("State PC1")
    ax_rt.set_ylabel("State PC2")

    if prim_acts.size > 0:
        im = ax_rb.imshow(prim_acts.T, aspect="auto", cmap="magma")
        ax_rb.set_title("Primitive activity")
        ax_rb.set_xlabel("Timestep")
        ax_rb.set_ylabel("Primitive")
        ax_rb.set_yticks(np.arange(prim_acts.shape[1]))
        ax_rb.set_yticklabels([f"P{i}" for i in range(prim_acts.shape[1])])
        xt = np.arange(len(trajectory_ids))
        ax_rb.set_xticks(xt)
        ax_rb.set_xticklabels([_byte_label(v) for v in trajectory_ids], rotation=90, fontsize=6)
        fig.colorbar(im, ax=ax_rb, fraction=0.046, pad=0.04)
        if 0 < int(prompt_len) < prim_acts.shape[0]:
            ax_rb.axvline(float(int(prompt_len) - 0.5), color="#00bcd4", linewidth=2.0, alpha=0.9)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path)
    plt.close(fig)


def _get_space_matrix(model: PrimitiveManifoldLM, mode: str, device: torch.device) -> np.ndarray:
    mode = str(mode)
    with torch.no_grad():
        idx = torch.arange(256, device=device, dtype=torch.long)
        if mode == "input_emb":
            mat = model._get_emb(idx)
        elif mode == "tok_emb":
            if getattr(model, "tok_emb", None) is None:
                raise ValueError("Checkpoint does not use tok_emb; try --space input_emb or bit_lookup.")
            mat = model.tok_emb(idx)
        elif mode == "bit_lookup":
            if not hasattr(model, "bit_lookup"):
                raise ValueError("Checkpoint does not have bit_lookup; try --space input_emb or tok_emb.")
            mat = torch.nn.functional.embedding(idx, model.bit_lookup)
        elif mode == "numeric":
            if not getattr(model, "numeric_proj", None):
                raise ValueError("Checkpoint does not use numeric features; try --space input_emb or tok_emb.")
            nf = model._numeric_features(idx)
            mat = model.numeric_proj(nf)
        elif mode == "out_rows":
            mat = model.out.weight[:, :]
        else:
            raise ValueError(f"Unknown --space: {mode}")
    return mat.detach().float().cpu().numpy()


def _write_manifest(out_dir: str, meta: dict, coords: np.ndarray, labels: list[str], ids: list[int], space: str, variance: np.ndarray) -> None:
    manifest_path = os.path.join(out_dir, f"manifest_{space}.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"ckpt={meta['ckpt']}\n")
        f.write(f"step={meta['step']}\n")
        f.write(f"space={space}\n")
        f.write(f"pc_var_explained={variance.tolist()}\n")
        f.write(f"cfg={meta['cfg']}\n")
        f.write("\n")
        f.write("byte_id\tlabel\tpc1\tpc2\n")
        for byte_id, label, xy in zip(ids, labels, coords.tolist()):
            f.write(f"{byte_id}\t{label}\t{xy[0]:.6f}\t{xy[1]:.6f}\n")


def _render_page(
    coords: np.ndarray,
    ids: list[int],
    labels: list[str],
    out_path: str,
    title: str,
    page_idx: int,
    page_count: int,
    global_limits: tuple[float, float, float, float],
    grid_cols: int,
) -> None:
    fig, axes = plt.subplots(grid_cols, grid_cols, figsize=(18, 18), dpi=180)
    axes = np.asarray(axes)
    x_min, x_max, y_min, y_max = global_limits
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    pad_x = dx * 0.08
    pad_y = dy * 0.08

    for cell_idx, ax in enumerate(axes.reshape(-1)):
        if cell_idx < len(ids):
            x, y = coords[cell_idx]
            byte_id = ids[cell_idx]
            label = labels[cell_idx]
            ax.scatter(coords[:, 0], coords[:, 1], s=10, c="#cfcfcf", alpha=0.25)
            ax.scatter([x], [y], s=80, c="#ff5a5f", alpha=0.95)
            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{byte_id:03d} {label}", fontsize=9)
        else:
            ax.axis("off")

    fig.suptitle(f"{title} | page {page_idx + 1}/{page_count}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path)
    plt.close(fig)


def _render_scatter_page(
    coords: np.ndarray,
    ids: list[int],
    labels: list[str],
    out_path: str,
    title: str,
    annotate: bool,
    edge_mode: str,
    focus_byte: int,
    move_k: int,
) -> None:
    plt.figure(figsize=(12, 10), dpi=180)
    ax = plt.gca()
    _draw_neighbor_edges(ax, coords, edge_mode=edge_mode, focus_byte=focus_byte, move_k=move_k)
    plt.scatter(coords[:, 0], coords[:, 1], s=38, c="#66ccff", alpha=0.85)
    if annotate:
        for byte_id, label, (x, y) in zip(ids, labels, coords.tolist()):
            plt.text(x, y, f"{byte_id}:{label}", fontsize=7, alpha=0.9)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="byte_pages")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--space", type=str, default="input_emb", choices=["input_emb", "tok_emb", "bit_lookup", "numeric", "out_rows"])
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--max_prompt_bytes", type=int, default=64)
    p.add_argument("--trace_response", action="store_true")
    p.add_argument("--max_new_bytes", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--grid_cols", type=int, default=16)
    p.add_argument("--annotate_scatter", action="store_true")
    p.add_argument("--edge_mode", type=str, default="none", choices=["none", "focus", "all"])
    p.add_argument("--focus_byte", type=int, default=65)
    p.add_argument("--move_k", type=int, default=1, choices=[1, 2, 3, 4])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    model, meta = _load_model(args.ckpt, device=device)
    mat = _get_space_matrix(model, mode=args.space, device=device)
    coords, variance = _pca(mat, n_components=2)

    all_ids = list(range(coords.shape[0]))
    all_labels = [_byte_label(i) for i in all_ids]
    _write_manifest(args.out_dir, meta, coords, all_labels, all_ids, args.space, variance)

    scatter_title = f"RAP byte projection | {args.space} | step={meta['step']} | {os.path.basename(meta['ckpt'])}"
    _render_scatter_page(
        coords=coords,
        ids=all_ids,
        labels=all_labels,
        out_path=os.path.join(args.out_dir, f"scatter_{args.space}.png"),
        title=scatter_title,
        annotate=bool(args.annotate_scatter),
        edge_mode=str(args.edge_mode),
        focus_byte=int(args.focus_byte),
        move_k=int(args.move_k),
    )

    prompt_ids = list(str(args.prompt).encode("utf-8", errors="replace"))[: max(1, int(args.max_prompt_bytes))] if str(args.prompt) else []
    traj_states = np.zeros((0, model.cfg.d_state), dtype=np.float64)
    traj_xy = np.zeros((0, 2), dtype=np.float64)
    prim_acts = np.zeros((0, model.cfg.n_primitives), dtype=np.float64)
    trace_ids = list(prompt_ids)
    prompt_len = len(prompt_ids)
    if prompt_ids and bool(args.trace_response):
        gen_ids = _generate_ids(
            model,
            prompt_ids,
            device=device,
            max_new_bytes=int(args.max_new_bytes),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
        )
        trace_ids = list(prompt_ids) + list(gen_ids)
    if prompt_ids:
        traj_states, prim_acts = _trace_runtime(model, trace_ids, device=device)
        traj_xy, _traj_mean, _traj_comp = _fit_pca_transform(traj_states, n_components=2)

    chunk_size = max(1, int(args.chunk_size))
    grid_cols = max(1, int(args.grid_cols))
    page_count = int(math.ceil(len(all_ids) / float(chunk_size)))

    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    y_min = float(coords[:, 1].min())
    y_max = float(coords[:, 1].max())
    global_limits = (x_min, x_max, y_min, y_max)

    for page_idx in range(page_count):
        start = page_idx * chunk_size
        end = min(len(all_ids), start + chunk_size)
        page_ids = all_ids[start:end]
        page_labels = all_labels[start:end]
        page_coords = coords[start:end]
        title = f"RAP byte tiles | {args.space} | bytes {start}-{end - 1}"
        _render_page(
            coords=page_coords,
            ids=page_ids,
            labels=page_labels,
            out_path=os.path.join(args.out_dir, f"tiles_{args.space}_page_{page_idx:03d}.png"),
            title=title,
            page_idx=page_idx,
            page_count=page_count,
            global_limits=global_limits,
            grid_cols=grid_cols,
        )
        if prompt_ids:
            mode_label = "prompt+response" if bool(args.trace_response) else "prompt-only"
            comp_title = f"RAP byte/runtime composite | {args.space} | {mode_label} | bytes {start}-{end - 1}"
            _render_composite_page(
                coords=coords,
                ids=all_ids,
                labels=all_labels,
                page_ids=page_ids,
                out_path=os.path.join(args.out_dir, f"composite_{args.space}_page_{page_idx:03d}.png"),
                title=comp_title,
                annotate=bool(args.annotate_scatter),
                trajectory_ids=trace_ids,
                trajectory_xy=traj_xy,
                prim_acts=prim_acts,
                prompt_len=prompt_len,
                edge_mode=str(args.edge_mode),
                focus_byte=int(args.focus_byte),
                move_k=int(args.move_k),
            )

    print(f"Loaded: {meta['ckpt']}")
    print(f"Space:  {args.space}")
    print(f"Wrote:  {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
