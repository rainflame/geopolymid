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
        spline_points,
        debug,
    ) = args
    geom, properties = polygon
    try:
        # print the name and area
        if properties["name"] == "Green Lakes" or properties["name"] == "Waldo Lake":
            print(f"Processing polygon {properties['name']} with area {geom.area}")

        MIN_AREA_LOW_SIMPLIFICATION = 0.0000001  # example value, define your own
        MAX_AREA_HIGH_SIMPLIFICATION = 0.001  # example value, define your own
        LOW_SIMPLIFICATION = 0.0001
        HIGH_SIMPLIFICATION = 0.001

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

        joined_line = LineString(heaviest_paths[0] + [center] + heaviest_paths[1][::-1])
        debug_medial_axis = joined_line

        # # simplify the joined line
        # joined_line = joined_line.simplify(0.001)
        # # smooth the line with chaikins
        # joined_line = LineString(
        #     chaikins_corner_cutting(joined_line.coords, smoothing_iterations)
        # )

        geod = Geod(ellps="WGS84")
        idx = index.Index()

        # load each coordinate of the geometry into the index
        for i, (x, y) in enumerate(geom.exterior.coords):
            idx.insert(i, [x, y, x, y])

        DIST_THRESHOLD = 600  # points must be 600m from the shore
        ALLOWABLE_VARIANCE = 50  # allow 50m variance once a segment has started

        points = []
        prev_point = None

        # create sections of the line that are far enough from the polygon boundary
        for i, (x, y) in enumerate(joined_line.coords):
            # for each point in the line, find the nearest point in the original polygon
            nearest_index = list(idx.nearest((x, y, x, y), num_results=1))[0]
            nearest_point = geom.exterior.coords[nearest_index]
            _, _, distance = geod.inv(x, y, nearest_point[0], nearest_point[1])

            if i == 0:
                this_point = {
                    "away_from_edge": distance > DIST_THRESHOLD,
                    "point": (x, y),
                }
            else:
                this_point = {
                    "away_from_edge": (
                        distance > DIST_THRESHOLD
                        or (
                            prev_point["away_from_edge"]
                            and distance > DIST_THRESHOLD - ALLOWABLE_VARIANCE
                        )
                    )
                    and not (
                        i == 1 and (not prev_point["away_from_edge"])
                    ),  # if the first point is by the edge, don't let it be a singleton
                    "point": (x, y),
                }

            # print(f"Point {i} away from edge: {this_point['away_from_edge']}")
            points.append(this_point)
            prev_point = this_point

        sections = []
        # construct sections from the points. points are grouped into sections with the
        # same away_from_edge value. In the case that a point is away_from_edge, but the
        # previous and next points are not, the point is added to the previous section
        for i, point in enumerate(points):
            if len(sections) == 0:
                # print(f"adding point {i} to new section {point['away_from_edge']}")
                sections.append([point])
            elif point["away_from_edge"] == sections[-1][-1]["away_from_edge"]:
                # print(
                #     f"adding point {i} to section away from edge: {point['away_from_edge']} (1)"
                # )
                sections[-1].append(point)
            elif i < len(points) - 1 and (
                point["away_from_edge"]
                and not sections[-1][-1]["away_from_edge"]
                and not points[i + 1]["away_from_edge"]
            ):
                # print(
                #     f"adding point {i} to section away from edge: {point['away_from_edge']} (2)"
                # )
                sections[-1].append(point)
            else:
                # print(f"adding point {i} to new section {point['away_from_edge']}")
                amended_previous_point = {
                    "away_from_edge": point["away_from_edge"],
                    "point": sections[-1][-1]["point"],
                }
                sections.append([amended_previous_point, point])

        # sections far from the borders get another round of simplification to smooth out
        # # any interior bends in the medial axis
        # for i, section in enumerate(sections):
        #     if section[0]["away_from_edge"]:
        #         # get the first and last points of the sectino
        #         first_point = section[0]["point"]
        #         last_point = section[-1]["point"]

        #         interpolated_x = np.linspace(first_point[0], last_point[0], 6)
        #         interpolated_y = np.linspace(first_point[1], last_point[1], 6)
        #         line = LineString(
        #             [(x, y) for x, y in zip(interpolated_x, interpolated_y)]
        #         )
        #         sections[i] = line

        #         # line = LineString([point["point"] for point in section])
        #         # # simplify the section more aggressively
        #         # line = line.simplify(0.001)
        #         # sections[i] = line

        if not skip_spline:
            for i, section in enumerate(sections):
                if section[0]["away_from_edge"]:
                    # create a line from the points
                    line = LineString([point["point"] for point in section])
                    line = line.simplify(poly_simplification)
                    x, y = line.xy
                    # ensure there are enough points to create a spline
                    if len(x) >= spline_degree + 1:
                        # create a B-spline representation of the line
                        tck, *_ = splprep([x, y], k=spline_degree)
                        new_x, new_y = splev(np.linspace(0, 1, len(x)), tck)
                        # make the first and last points the same as the original line
                        new_x[0] = x[0]
                        new_y[0] = y[0]
                        new_x[-1] = x[-1]
                        new_y[-1] = y[-1]

                        sections[i] = LineString([(x, y) for x, y in zip(new_x, new_y)])

        # # filter out sectiosn away from edge
        # sections = [section for section in sections if not section[0]["away_from_edge"]]

        # create lines from any remaining sections
        for i, section in enumerate(sections):
            if isinstance(section, list):
                x, y = zip(*[point["point"] for point in section])
                sections[i] = LineString([(x, y) for x, y in zip(x, y)])

        coords = list(sections[0].coords)
        for line in sections[1:]:
            coords += list(line.coords)[1:]

        line = LineString(coords)
        line = line.simplify(poly_simplification)

        # apply chaikins corner cutting to smooth the line
        result = LineString(chaikins_corner_cutting(line.coords, smoothing_iterations))

        result = MultiLineString([result])

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
        # print(e)
        traceback.print_exc()
        print(f"Error processing polygon with properties: {properties}")
        print("Skipped polygon")
        return (None, properties)
