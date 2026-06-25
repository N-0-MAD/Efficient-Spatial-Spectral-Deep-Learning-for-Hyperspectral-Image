"""Model definitions for dense forest monitoring experiments."""

from dfm.models.cnn_baseline import CNNBaseline
from dfm.models.hybrid import HybridSpatialSpectralClassifier
from dfm.models.transformer_baseline import IndependentBandTransformerClassifier

__all__ = [
    "CNNBaseline",
    "HybridSpatialSpectralClassifier",
    "IndependentBandTransformerClassifier",
]

