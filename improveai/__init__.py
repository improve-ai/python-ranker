from warnings import warn

from improveai.decision import Decision
from improveai.decision_context import DecisionContext
from improveai.feature_encoder import FeatureEncoder
from improveai.decision_model import DecisionModel
from improveai.givens_provider import GivensProvider
# import gcc and python3-dev detection utility
from improveai.settings import CYTHON_BACKEND_AVAILABLE

if CYTHON_BACKEND_AVAILABLE:
    # if gcc and python3 dev installed use cythonized FeatureEcoder
    from improveai.cythonized_feature_encoding import cfe
    FeatureEncoder = cfe.FeatureEncoder
else:
    # no cythonized FeatureEncoder
    warn('No build tools detected (gcc, python3-dev) -> FastFeatureEncoder falls back to normal FeatureEncoder')
