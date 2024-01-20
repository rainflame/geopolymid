import networkx as nx
import skgeom as sg
import numpy as np

from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, LineString, MultiLineString

from .graph import dfs_sum_weights, get_heaviest_path


def reduce_polygon_dimensions(polygon):
    exterior_coords = [(x, y) for x, y, *_ in polygon.exterior.coords]
    interiors = []
    for interior in polygon.interiors:
        interior_coords = [(x, y) for x, y, *_ in interior.coords]
        interiors.append(interior_coords)

    # Create a new 2D polygon
    return Polygon(exterior_coords, interiors)


# create an approximation of medial axes from the input polygons
# first, we create a skeleton of each polygon
# then, we find the weight of each node, which is the sum of the distance to all child nodes
# then, we find the two heaviest paths from the center node. this gives us the path from the cetner
# that is the furthest from the polygon boundary, an approxmiation for the most visually massive section of the polygon
# then, we join the two paths together to get the medial axis
def get_weighted_medial_axis(args):
    (
        polygon,
        skip_spline,
        presimplification_percentage,
        spline_degree,
        spline_points,
        debug,
    ) = args
    geom, properties = polygon
    try:
        polygon = sg.Polygon(geom.exterior.coords)
        # simplify the geometry to speed up the medial axis calculation
        polygon = sg.simplify(polygon, presimplification_percentage)
        skeleton = sg.skeleton.create_interior_straight_skeleton(polygon)

        graph = nx.Graph()
        debug_skeleton = []

        for h in skeleton.halfedges:
            if h.is_bisector:
                p1 = h.vertex.point
                p2 = h.opposite.vertex.point
                graph.add_edge(
                    (float(p1.x()), float(p1.y())), (float(p2.x()), float(p2.y()))
                )
                debug_skeleton.append(
                    LineString(
                        [(float(p1.x()), float(p1.y())), (float(p2.x()), float(p2.y()))]
                    )
                )

        center = nx.center(graph)[0]
        node_weights = {}
        dfs_sum_weights(node_weights, graph, center, set())

        neighbors = graph.neighbors(center)

        # get the two neighbors with the highest weights
        neighbor_weights = [(n, node_weights[n]) for n in neighbors]
        neighbor_weights.sort(key=lambda x: x[1], reverse=True)

        # get the two heaviest paths
        heaviest_paths = []
        for n, _ in neighbor_weights[:2]:
            heaviest_paths.append(
                get_heaviest_path(graph, node_weights, n, set([center]))
            )

        joined_line = LineString(heaviest_paths[0] + [center] + heaviest_paths[1][::-1])
        debug_medial_axis = joined_line

        if not skip_spline:
            x, y = joined_line.xy
            # ensure there are enough points to create a spline
            if len(x) >= spline_degree + 1:
                # create a B-spline representation of the line
                tck, _ = splprep([x, y], k=spline_degree)
                new_x, new_y = splev(np.linspace(0, 1, spline_points), tck)
                joined_line = LineString([(x, y) for x, y in zip(new_x, new_y)])

        # trim the line to the original polygon
        intersection = joined_line.intersection(geom)

        if intersection.is_empty:
            result = MultiLineString([])
        elif isinstance(intersection, LineString):
            result = MultiLineString([intersection])
        elif isinstance(intersection, MultiLineString):
            result = intersection
        else:
            raise Exception(f"Unexpected intersection type: {type(intersection)}")

        if debug:
            return (
                (
                    MultiLineString(debug_skeleton),
                    MultiLineString([debug_medial_axis]),
                    result,
                ),
                properties,
            )

        return (result, properties)

    except Exception as e:
        print(e)
        print(f"Error processing polygon with properties: {properties}")
        print("Skipped polygon")
        return (None, properties)
