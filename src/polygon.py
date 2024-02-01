import networkx as nx
import skgeom as sg
import numpy as np

import traceback
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, LineString, MultiLineString

from rtree import index
from pyproj import Geod

from .graph import dfs_sum_weights, get_heaviest_path
from .smoothing import chaikins_corner_cutting


def reduce_polygon_dimensions(polygon):
    exterior_coords = [(x, y) for x, y, *_ in polygon.exterior.coords]
    interiors = []
    for interior in polygon.interiors:
        interior_coords = [(x, y) for x, y, *_ in interior.coords]
        interiors.append(interior_coords)

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
        smoothing_iterations,
        spline_degree,
        spline_distance_threshold,
        spline_distance_allowable_variance,
        debug,
    ) = args
    geom, properties = polygon
    try:
        # TODO: set these as args from cli
        MIN_AREA_LOW_SIMPLIFICATION = 0.0000001
        MAX_AREA_HIGH_SIMPLIFICATION = 0.001
        LOW_SIMPLIFICATION = 0.00001
        HIGH_SIMPLIFICATION = 0.0001

        area = geom.area
        poly_simplification = LOW_SIMPLIFICATION

        if area <= MIN_AREA_LOW_SIMPLIFICATION:
            poly_simplification = LOW_SIMPLIFICATION
        elif area >= MAX_AREA_HIGH_SIMPLIFICATION:
            poly_simplification = HIGH_SIMPLIFICATION
        else:
            # Calculate the ratio of the area within the threshold range
            ratio = (area - MIN_AREA_LOW_SIMPLIFICATION) / (
                MAX_AREA_HIGH_SIMPLIFICATION - MIN_AREA_LOW_SIMPLIFICATION
            )
            # Interpolate simplification value
            poly_simplification = np.interp(
                ratio, [0, 1], [LOW_SIMPLIFICATION, HIGH_SIMPLIFICATION]
            )

        simple = geom.simplify(poly_simplification)
        polygon = sg.Polygon(simple.exterior.coords)
        polygon = sg.simplify(polygon, 0.5)  # skgeom hack to make poly simple
        if not polygon.is_simple():
            # try on a non-simplified version
            polygon = sg.Polygon(geom.exterior.coords)
            polygon = sg.simplify(polygon, 0.5)
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

        medial_axis = LineString(heaviest_paths[0] + [center] + heaviest_paths[1][::-1])
        debug_medial_axis = medial_axis

        geod = Geod(ellps="WGS84")
        idx = index.Index()

        # load each coordinate of the geometry into the index
        for i, (x, y) in enumerate(geom.exterior.coords):
            idx.insert(i, [x, y, x, y])

        points = []
        prev_point = None
        segments = []

        # create sections of the line that are far enough from the polygon boundary
        for i, (x, y) in enumerate(medial_axis.coords):
            # for each point in the line, find the nearest point in the original polygon
            nearest_index = list(idx.nearest((x, y, x, y), num_results=1))[0]
            nearest_point = geom.exterior.coords[nearest_index]
            _, _, distance = geod.inv(x, y, nearest_point[0], nearest_point[1])

            if i == 0:
                segments.append([i])
                # print("new segment")
            else:
                if distance < spline_distance_threshold:
                    if segments[-1][-1] == i - 1:
                        segments[-1].append(i)
                        # print("adding to previous")
                    else:
                        segments.append([i])
                        # print("new segment")

        def line_too_close_to_edge(line, idx):
            for x, y in line.coords:
                nearest_index = list(idx.nearest((x, y, x, y), num_results=1))[0]
                nearest_point = geom.exterior.coords[nearest_index]
                _, _, distance = geod.inv(x, y, nearest_point[0], nearest_point[1])
                if distance < spline_distance_threshold:
                    return True
            return False

        while len(segments) > 1:
            segment_a_point_index = (
                segments[0][-1] + 1
            )  # start at the first point that's away from the edge
            segment_b_point_index = segments[1][0] - 1

            while segment_b_point_index != segment_a_point_index:
                # Generate 100 evenly spaced points along the line
                linspace_points = np.linspace(
                    medial_axis.coords[segment_a_point_index],
                    medial_axis.coords[segment_b_point_index],
                    100,
                )
                candidate_line = LineString([tuple(p) for p in linspace_points])

                # if the invariant does not hold, try the next point down the medial axis
                if line_too_close_to_edge(candidate_line, idx):
                    segment_b_point_index -= 1
                else:
                    # if the last point in the segment a is not the current segment a point index, then we
                    # are at the start and need to add the current point to the end of segment a
                    if segments[0][-1] != segment_a_point_index:
                        segments[0].append(segment_a_point_index)
                    # add the located point to the end of segment a
                    segments[0].append(segment_b_point_index)
                    # reset segment b starting point
                    segment_b_point_index = segments[1][0] - 1
                    # now start segment a at the new point
                    segment_a_point_index = segments[0][-1]

            # add segment b to a
            segments[0] += segments[1]
            # remove segment b
            segments.pop(1)

        lines = []
        # create lines from the sections
        for segment in segments:
            if len(segment) > 1:
                x, y = zip(*[medial_axis.coords[i] for i in segment])
                line = LineString([(x, y) for x, y in zip(x, y)])
                lines.append(line)

        # join the lines together
        result = MultiLineString(lines)

        # coords = list(sections[0].coords)
        # for line in sections[1:]:
        #     coords += list(line.coords)[1:]

        # line = LineString(coords)
        # line = line.simplify(poly_simplification)

        # # apply chaikins corner cutting to smooth the line
        # result = LineString(chaikins_corner_cutting(line.coords, smoothing_iterations))

        # result = MultiLineString([result])

        # # trim the line to the original polygon
        # intersection = joined_line.intersection(geom)

        # if intersection.is_empty:
        #     result = MultiLineString([])
        # elif isinstance(intersection, LineString):
        #     result = MultiLineString([intersection])
        # elif isinstance(intersection, MultiLineString):
        #     result = intersection
        # else:
        #     result = MultiLineString(
        #         [geom for geom in intersection.geoms if isinstance(geom, LineString)]
        #     )

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
        traceback.print_exc()
        print(f"Error processing polygon with properties: {properties}")
        print("Skipped polygon")
        return (None, properties)
