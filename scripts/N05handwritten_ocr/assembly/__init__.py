"""N05 handwriting assembly layer.

This package turns existing expert/proposer outputs into a shared decision
surface. It does not make the final OCR decision yet; v0.1 builds the skeleton
that later formula-selection AI can learn from.
"""

from .orchestrator import build_assembly_map

__all__ = ["build_assembly_map"]
