"""
InvariantOne: Permutation-Equivariant Natural-Language Decision Model.

Version: 1.0.0
Research Designation: D5-L4 / Nop=16 / Seed 42
"""

from invariantone.calibration import calibrated_softmax
from invariantone.config import InvariantOneConfig
from invariantone.formatting import (
    compute_option_spans,
    format_candidate_prompt,
    format_candidate_prompts,
    format_prefix,
    pool_option_representations,
)
from invariantone.ensemble import InvariantOneEnsemble
from invariantone.model import InvariantOne
from invariantone.profiles import InvariantOneProfile
from invariantone.result import (

    BatchDecisionResult,
    DecisionResult,
    EnsembleDecisionResult,
)

__version__ = "1.0.0"

__all__ = [
    "BatchDecisionResult",
    "DecisionResult",
    "EnsembleDecisionResult",
    "InvariantOne",
    "InvariantOneConfig",
    "InvariantOneEnsemble",
    "InvariantOneProfile",
    "__version__",
    "calibrated_softmax",
    "compute_option_spans",
    "format_candidate_prompt",
    "format_candidate_prompts",
    "format_prefix",
    "pool_option_representations",
]

