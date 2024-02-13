from shapely.geometry import Point
from dataclasses import dataclass


@dataclass
class MedialAxisPoint:
    point: Point
    to_convert: bool
