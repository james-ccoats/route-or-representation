"""Model loading and MoE structure inspection.

Role in the project: the single place where a Hugging Face MoE model is
loaded. Used from Phase 0 (smoke test) onward. Phase 4 reuses it with a second
MoE model, so nothing here may hard-code OLMoE.

Methodological note:
    Architecture facts (layer count, expert count, top-k, router module names)
    must be *read off the loaded model* by :func:`inspect_moe_structure` and
    compared with the config via :func:`verify_structure`, not assumed. Module
    names and router implementations can differ across ``transformers`` versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rr.routing._hooks import normalizes_topk_probs

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from rr.config import ArchitectureConfig


@dataclass(frozen=True)
class MoEStructure:
    """MoE architecture facts discovered by inspecting a loaded model.

    Attributes:
        num_layers: Number of MoE layers.
        num_experts: Routed experts per layer.
        top_k: Experts activated per token.
        router_module_names: Fully qualified router module name for each layer.
        normalizes_topk_probs: Whether top-k weights are renormalized to sum
            to 1 (affects what "mixing weights" means when forcing).
    """

    num_layers: int
    num_experts: int
    top_k: int
    router_module_names: tuple[str, ...]
    normalizes_topk_probs: bool


def load_model(
    name: str,
    dtype: str = "bfloat16",
    device: str = "auto",
    revision: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal-LM MoE model and its tokenizer in eval mode.

    Args:
        name: Hugging Face model id, e.g. ``"allenai/OLMoE-1B-7B-0924"``.
        dtype: ``"float32"``, ``"float16"`` or ``"bfloat16"``.
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
        revision: Optional commit hash to pin.

    Returns:
        ``(model, tokenizer)``. The model is in ``eval()`` mode with gradients
        disabled by the caller (use ``torch.inference_mode()``).
    """
    # TODO(Phase 0): map dtype string to torch dtype, resolve "auto" device,
    # AutoModelForCausalLM / AutoTokenizer.from_pretrained(...). Beware of
    # bf16 on MPS/CPU. Do not run until the user approves the weights download.
    raise NotImplementedError("TODO(Phase 0): load model and tokenizer")


def inspect_moe_structure(model: PreTrainedModel) -> MoEStructure:
    """Discover MoE structure by walking ``model.named_modules()``.

    Args:
        model: A loaded MoE model.

    Returns:
        The discovered :class:`MoEStructure`.

    Raises:
        RuntimeError: If no router modules are found.
    """
    routers = [(name, m) for name, m in model.named_modules() if _is_router(m)]
    if not routers:
        raise RuntimeError(
            "no router modules found: expected modules with integer 'top_k' and "
            "'num_experts' attributes and a router 'weight'"
        )
    modules = [m for _, m in routers]
    facts = {(m.num_experts, m.top_k, normalizes_topk_probs(m)) for m in modules}
    if len(facts) != 1:
        raise RuntimeError(f"routers disagree on (num_experts, top_k, normalize): {sorted(facts)}")
    num_experts, top_k, normalizes = facts.pop()
    for name, module in routers:
        if module.weight.shape[0] != num_experts:
            raise RuntimeError(
                f"router {name!r} has weight rows {module.weight.shape[0]} "
                f"but num_experts={num_experts}"
            )
    _check_against_model_config(model, num_experts, top_k)
    return MoEStructure(
        num_layers=len(routers),
        num_experts=num_experts,
        top_k=top_k,
        router_module_names=tuple(name for name, _ in routers),
        normalizes_topk_probs=normalizes,
    )


def _is_router(module: nn.Module) -> bool:
    return (
        isinstance(getattr(module, "top_k", None), int)
        and isinstance(getattr(module, "num_experts", None), int)
        and hasattr(module, "weight")
    )


def _check_against_model_config(model: PreTrainedModel, num_experts: int, top_k: int) -> None:
    """Cross-check module-derived facts with ``model.config`` where it exposes them."""
    config = model.config
    for attr, value in (("num_experts", num_experts), ("num_experts_per_tok", top_k)):
        configured = getattr(config, attr, None)
        if configured is not None and configured != value:
            raise RuntimeError(f"model.config.{attr}={configured} but the routers imply {value}")


def verify_structure(structure: MoEStructure, expected: ArchitectureConfig) -> None:
    """Assert that discovered structure matches the (unverified) config values.

    Raises:
        ValueError: Listing every mismatch, so the config can be corrected.
    """
    mismatches = [
        f"{field}: config={want!r}, model={got!r}"
        for field, want, got in (
            ("num_layers", expected.num_layers, structure.num_layers),
            ("num_experts", expected.num_experts, structure.num_experts),
            ("top_k", expected.top_k, structure.top_k),
            (
                "router_module_names",
                expected.router_module_names(),
                structure.router_module_names,
            ),
        )
        if want != got
    ]
    if mismatches:
        raise ValueError(
            "architecture config does not match the loaded model:\n  " + "\n  ".join(mismatches)
        )
