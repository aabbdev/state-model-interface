"""Loading helpers for the canonical SMI chat template."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def load_chat_template(path: str | Path | None = None) -> str:
    """Load an explicit template or the packaged canonical template."""
    if path is not None:
        candidate = Path(path).expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError(f"SMI chat template does not exist: {candidate}")
        return candidate.read_text(encoding="utf-8")

    source_checkout = Path(__file__).resolve().parents[2] / "chat_template.jinja"
    if source_checkout.is_file() and not source_checkout.is_symlink():
        return source_checkout.read_text(encoding="utf-8")
    packaged = resources.files("state_model_interface").joinpath("chat_template.jinja")
    return packaged.read_text(encoding="utf-8")


__all__ = ["load_chat_template"]
