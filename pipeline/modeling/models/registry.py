from .ridge import RidgeSpec
from .random_forest import RandomForestSpec

MODEL_REGISTRY = {spec.name: spec for spec in (
    RidgeSpec(), RandomForestSpec(),
)}


def get_model(name):
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}. Choose from {list(MODEL_REGISTRY)}") from exc
