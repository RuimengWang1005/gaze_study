"""Independent BBox-only teacher sufficiency and causal-utility audit."""

from .metrics import normalize_answer
from .token_mapping import pixel_bboxs_to_merged_mask
from .utility import grad_input_utility

__all__ = ["grad_input_utility", "normalize_answer", "pixel_bboxs_to_merged_mask"]
