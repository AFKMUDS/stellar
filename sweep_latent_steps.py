import argparse

from probe_latent_step import (
    capture_state,
    decode_ids,
    load_model,
    rollout_from_state,
    summarize_weights,
)


def parse_int_list(s: str) -> list[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints1/b.pt")
    p.add_argument("--prompt", type=str, default="What is gravity?")
    p.add_argument("--steps", type=str, default="0,4,8,12")
    p.add_argument("--primitives", type=str, default="")
    p.add_argument("--forbid_primitives", type=str, default="")
    p.add_argument("--pulse_primitives", type=str, default="")
    p.add_argument("--force_strength", type=float, default=1.0)
    p.add_argument("--pulse_strength", type=float, default=1.0)
    p.add_argument("--pulse_steps", type=int, default=0)
    p.add_argument("--max_new", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    model = load_model(args.ckpt, args.device)
    prompt_ids = list(str(args.prompt).encode("utf-8", errors="replace"))
    steps = parse_int_list(args.steps)
    prims = parse_int_list(args.primitives) if str(args.primitives).strip() else []
    forbid_prims = parse_int_list(args.forbid_primitives) if str(args.forbid_primitives).strip() else []
    pulse_prims = parse_int_list(args.pulse_primitives) if str(args.pulse_primitives).strip() else []

    print(f"Checkpoint: {args.ckpt}")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {steps}")
    print(f"Forced primitives: {prims if prims else '[none]'}")
    print(f"Forbidden primitives: {forbid_prims if forbid_prims else '[none]'}")
    print(f"Pulsed primitives: {pulse_prims if pulse_prims else '[none]'}")
    print()

    for step in steps:
        cap_state, cap_last_id, cap_weights, cap_last_pos, cap_state_hist = capture_state(model, prompt_ids, int(step), args.device)
        actual_step = min(max(int(step), 0), max(len(prompt_ids) - 1, 0))
        print("=" * 100)
        print(f"[capture_step={actual_step}] last_byte={cap_last_id}")
        print("[capture primitive summary]")
        print(summarize_weights(cap_weights))
        print()

        normal_ids, normal_weights = rollout_from_state(
            model,
            cap_state.clone(),
            cap_last_id,
            last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
            state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
            max_new=int(args.max_new),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            force_primitive=None,
            force_strength=0.0,
        )
        print("[normal]")
        print(decode_ids(normal_ids))
        print("[normal primitive summary]")
        print(summarize_weights(normal_weights))
        print()

        for prim in forbid_prims:
            forbidden_ids, forbidden_weights = rollout_from_state(
                model,
                cap_state.clone(),
                cap_last_id,
                last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
                state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
                max_new=int(args.max_new),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                force_primitive=None,
                force_strength=0.0,
                forbid_primitive=int(prim),
                pulse_primitive=None,
                pulse_strength=0.0,
                pulse_steps=0,
            )
            print(f"[forbidden primitive {int(prim)}]")
            print(decode_ids(forbidden_ids))
            print("[forbidden primitive summary]")
            print(summarize_weights(forbidden_weights))
            print()

        for prim in prims:
            forced_ids, forced_weights = rollout_from_state(
                model,
                cap_state.clone(),
                cap_last_id,
                last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
                state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
                max_new=int(args.max_new),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                force_primitive=int(prim),
                force_strength=float(args.force_strength),
                forbid_primitive=None,
                pulse_primitive=None,
                pulse_strength=0.0,
                pulse_steps=0,
            )
            print(f"[forced primitive {int(prim)} strength={float(args.force_strength):.2f}]")
            print(decode_ids(forced_ids))
            print("[forced primitive summary]")
            print(summarize_weights(forced_weights))
            print()

        for prim in pulse_prims:
            pulsed_ids, pulsed_weights = rollout_from_state(
                model,
                cap_state.clone(),
                cap_last_id,
                last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
                state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
                max_new=int(args.max_new),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                force_primitive=None,
                force_strength=0.0,
                forbid_primitive=None,
                pulse_primitive=int(prim),
                pulse_strength=float(args.pulse_strength),
                pulse_steps=int(args.pulse_steps),
            )
            print(f"[pulsed primitive {int(prim)} strength={float(args.pulse_strength):.2f} steps={int(args.pulse_steps)}]")
            print(decode_ids(pulsed_ids))
            print("[pulsed primitive summary]")
            print(summarize_weights(pulsed_weights))
            print()


if __name__ == "__main__":
    main()
