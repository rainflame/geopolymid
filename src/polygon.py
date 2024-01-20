
from shapely.geometry import Polygon

def reduce_polygon_dimensions(polygon):
    exterior_coords = [(x, y) for x, y, *_ in polygon.exterior.coords]
    interiors = []
    for interior in polygon.interiors:
        interior_coords = [(x, y) for x, y, *_ in interior.coords]
        interiors.append(interior_coords)

    # Create a new 2D polygon
    return Polygon(exterior_coords, interiors)