from dataclasses import asdict
from inspect import signature


class BaseClassExtended:
    """Extend the functionality of a base class, allowing it to be:
    - subscritable
    - built from a dictionary (e.g., kwargs)
    - coerced to a dictionary
    """

    def to_dict(self):
        """Return the coerced attributes of a DataClass object as a dictionary"""
        return asdict(self)

    # Make subscriptable:
    def __getitem__(self, item):
        return getattr(self, item)

    # Build class from dict, ignoring additional kwargs:
    @classmethod
    def from_dict(cls, input):
        class_attributes = signature(cls).parameters
        return cls(**{k: v for k, v in input.items() if k in class_attributes})
