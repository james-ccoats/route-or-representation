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

if TYPE_CHECKING:
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
    # TODO(Phase 0): find router modules (e.g. Linear layers named *.mlp.gate),
    # read num_experts/top_k from model.config AND from module shapes, and
    # check the top-k/normalization logic in the installed transformers source.
    raise NotImplementedError("TODO(Phase 0): inspect router modules and config")


def verify_structure(structure: MoEStructure, expected: ArchitectureConfig) -> None:
    """Assert that discovered structure matches the (unverified) config values.

    Raises:
        ValueError: Listing every mismatch, so the config can be corrected.
    """
    # TODO(Phase 0): compare fields and raise with a readable diff.
    raise NotImplementedError("TODO(Phase 0): compare discovered vs configured architecture")
