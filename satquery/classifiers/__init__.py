from satquery.classifiers.land_cover import LandCoverModel, BIGEARTHNET_19_CLASSES, NUM_CLASSES, get_transforms
from satquery.classifiers.predict import predict, LandCoverPredictor

__all__ = [
    "LandCoverModel",
    "BIGEARTHNET_19_CLASSES",
    "NUM_CLASSES",
    "get_transforms",
    "predict",
    "LandCoverPredictor",
]
