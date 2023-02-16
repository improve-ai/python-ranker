from concurrent.futures import ThreadPoolExecutor
from pkg_resources import get_distribution, DistributionNotFound
from warnings import warn

from improveai.feature_encoder import FeatureEncoder
from improveai.ranker import Ranker
from improveai.scorer import Scorer
from improveai.reward_tracker import RewardTracker
# import gcc and python3-dev detection utility
from improveai.settings import CYTHON_BACKEND_AVAILABLE, MAX_TRACK_THREADS

if CYTHON_BACKEND_AVAILABLE:
    # if gcc and python3 dev installed use cythonized FeatureEcoder
    from improveai.cythonized_feature_encoding import cfe
    FeatureEncoder = cfe.FeatureEncoder
else:
    # no cythonized FeatureEncoder
    warn('No build tools detected (gcc, python3-dev) -> FastFeatureEncoder falls back to normal FeatureEncoder')

# set __version__ package attribute; If package is not installed with pip set INFO string to __version__
try:
    __version__ = get_distribution('improveai').version
except DistributionNotFound as enferr:
    __version__ = 'Distribution not found - please install `improveai` wit pip'

track_improve_executor = ThreadPoolExecutor(max_workers=MAX_TRACK_THREADS)
