from dataclasses import dataclass


@dataclass
class ControlPoint:
    coordinates: tuple[float, float, float]
    is_reproduction_point: bool
