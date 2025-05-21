import json
from dataclasses import asdict, dataclass
from inspect import signature
from typing import Self, Type

import numpy as np


class _KalmanFilterNumpyEncoder:
    """ Special json encoder for writing numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class BaseClassExtended:
    """Extend the functionality of a base class, allowing it to be:
    - subscriptable
    - constructed from a dictionary (e.g., kwargs)
    - coerced to a dictionary
    - constructed from a `.json` file
    - written to a `.json` file
    """

    # Make subscriptable:
    def __getitem__(self, item):
        return getattr(self, item)

    # Build class from dict, ignoring additional kwargs:
    @classmethod
    def from_dict(cls, data) -> Type[Self]:
        """Class constructor from dict, ignoring additional kwargs."""
        class_attributes = signature(cls).parameters
        return cls(**{k: v for k, v in data.items() if k in class_attributes})

    @classmethod
    def from_json(cls, path: str) -> Type[Self]:
        """Class constructor from `.json` path, ignoring additional kwargs, and coercing to `np.ndarray` types."""
        with open(path, "r") as f:
            json_input = json.load(f)
        raw_output = {key: np.array(value) if hasattr(
            value, "__len__") else value for key, value in json_input.items()}
        # Construct:
        return cls.from_dict(raw_output)

    def to_dict(self) -> dict[str, float | np.ndarray]:
        """Return the coerced attributes of a DataClass object as a dictionary"""
        return asdict(self)

    def to_json(self, path: str, **kwargs) -> None:
        """Write class object attributes to a `.json` path.

        - `path` (str): path to the written `.json` file. The `.json` suffix is not necessary.
        - `**kwargs`: additional kwargs to pass to the `json.dump` method.
        """
        if not path.endswith(".json"):
            path += ".json"
        output = self.to_dict()
        with open(path, "w") as f:
            json.dump({key: value.tolist() if isinstance(value, np.ndarray) else value for key,
                      value in output.items()}, f, cls=kwargs.pop("cls", _KalmanFilterNumpyEncoder), **kwargs)
