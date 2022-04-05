# from improveai.cythonized_feature_encoding import cfe
from improveai.decision import Decision
from improveai.decision_context import DecisionContext
from improveai.feature_encoder import FeatureEncoder
from improveai.decision_model import DecisionModel
from improveai.givens_provider import GivensProvider
from improveai.settings import USE_CYTHON_BACKEND

# import improveai.cythonized_feature_encoding.cythonized_feature_encoder as cfe
from improveai.cythonized_feature_encoding import cfe  # import FeatureEncoder as FastFeatureEncoder
FastFeatureEncoder = cfe.FeatureEncoder
