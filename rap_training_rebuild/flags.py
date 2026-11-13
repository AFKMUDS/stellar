"""vNext pilot flag contract: reject mechanisms not implemented by the original checkpoint."""


CHOICES = {
    "core": ("original", "fresh"),
    "primitive_mode": ("original", "fresh"),
    "encoding": ("original", "bitviews", "multi"),
    "head": ("original", "direct", "expression", "joint"),
    "retrieval": ("original", "none", "ordered", "random"),
    "memory_readout": ("off", "residual", "joint"),
    "expression_loss": ("off", "byte-delta", "sequence-delta"),
    "phase_clock": ("off", "observed", "learned"),
    "route_regularization": ("off", "entropy", "balance"),
    "state_persistence": ("record", "stream"),
    "curriculum": ("off", "mixed", "staged"),
    "diagnostics": ("basic", "full"),
}


def add_flags(parser):
    for name, choices in CHOICES.items():
        parser.add_argument("--" + name.replace("_", "-"), choices=choices, default=None)


def validate_flags(args):
    # Explicitly reject unimplemented choices rather than silently switching architectures.
    supported = {
        "core": "original",
        "primitive_mode": "original",
        "encoding": "original",
        "head": "joint" if args.variant in ("B", "D") else "original",
        "retrieval": "original",
        "memory_readout": "joint" if args.variant in ("B", "D") else "off",
        "expression_loss": "off",
        "phase_clock": "off",
        "route_regularization": "off",
        "state_persistence": "record",
        "curriculum": "off",
        "diagnostics": "basic",
    }
    for name, choice in supported.items():
        provided = getattr(args, name)
        if provided is not None and provided != choice:
            raise ValueError(
                f"--{name.replace('_', '-')}={provided} is not implemented for variant "
                f"{args.variant}; only {choice} is supported by this original-core pilot"
            )
    return supported
