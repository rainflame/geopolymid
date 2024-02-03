import networkx as nx
import numpy as np

import traceback
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, LineString, MultiLineString, Point
from typing import List

from rtree import index
from pyproj import Geod

from .graph import find_graph_medial_axis, make_skeleton_graph_from_poly
from .smoothing import chaikins_corner_cutting
from .line import trim_line
from .point import MedialAxisPoint


def reduce_polygon_dimensions(polygon):
    exterior_coords = [(x, y) for x, y, *_ in polygon.exterior.coords]
    interiors = []
    for interior in polygon.interiors:
        interior_coords = [(x, y) for x, y, *_ in interior.coords]
        interiors.append(interior_coords)

    return Polygon(exterior_coords, interiors)


def construct_polygon_rtree_index(polygon):
    idx = index.Index()
    for i, (x, y) in enumerate(polygon.exterior.coords):
        idx.insert(i, (x, y, x, y))

    return idx


def get_nearest_point_on_polygon(polygon, polygon_rtree_index, x, y):
    geod = Geod(ellps="WGS84")
    nearest_index = list(polygon_rtree_index.nearest((x, y, x, y), num_results=1))[0]
    nearest_point = polygon.exterior.coords[nearest_index]
    # get dist between points in meters
    _, _, distance = geod.inv(x, y, nearest_point[0], nearest_point[1])

    return distance


# create an approximation of medial axes from the input polygons
# first, we create a skeleton of each polygon
# then, we find the weight of each node, which is the sum of the distance to all child nodes
# then, we find the two heaviest paths from the center node. this gives us the path from the cetner
# that is the furthest from the polygon boundary, an approxmiation for the most visually massive section of the polygon
# then, we join the two paths together to get the medial axis
def get_weighted_medial_axis(args):
    (
        polygon,
        simplification_factor,
        smoothing_iterations,
        spline_degree,
        spline_distance_threshold,
        trim_output_lines_by_percent,
        debug,
    ) = args
    geom, properties = polygon
    try:
        graph = nx.Graph()
        graph = make_skeleton_graph_from_poly(geom, graph)

        # convert the graph's edges to lines
        debug_skeleton = []
        for edge in graph.edges:
            debug_skeleton.append(LineString([edge[0], edge[1]]))

        medial_axis = find_graph_medial_axis(graph)
        debug_medial_axis = medial_axis

        idx = construct_polygon_rtree_index(geom)

        medial_axis_points: List[MedialAxisPoint] = []
        for i, (x, y) in enumerate(medial_axis.coords):
            # for each point in the line, find the nearest point in the original polygon
            distance = get_nearest_point_on_polygon(geom, idx, x, y)
            p = MedialAxisPoint(Point(x, y), distance > spline_distance_threshold)
            medial_axis_points.append(p)

        sections = []
        for i, point in enumerate(medial_axis_points):
            if i == 0:
                sections.append([i])
            elif point.to_convert == medial_axis_points[i - 1].to_convert:
                # both are the same, so we can add to the current section
                sections[-1].append(i)
            elif point.to_convert != medial_axis_points[i - 1].to_convert:
                # if the previous section is one point only, add the current point to it
                # this is to ensure each section has at least two points and can be a line
                if len(sections[-1]) == 1:
                    sections[-1].append(i)
                    # set this point to match the previous to_convert value
                    medial_axis_points[i].to_convert = medial_axis_points[
                        i - 1
                    ].to_convert
                else:
                    sections.append([i])

        smoothed_section_lines = []
        for i, section in enumerate(sections):
            line = LineString([medial_axis_points[i].point for i in section])
            if medial_axis_points[section[0]].to_convert:
                # convert the line to a B-spline
                x, y = line.xy
                if len(x) > spline_degree:
                    tck, *_ = splprep([x, y], k=spline_degree)
                    new_x, new_y = splev(np.linspace(0, 1, len(x)), tck)
                    # make the first and last points the same as the original line
                    new_x[0] = x[0]
                    new_y[0] = y[0]
                    new_x[-1] = x[-1]
                    new_y[-1] = y[-1]
                    smoothed_section_lines.append(
                        LineString([(x, y) for x, y in zip(new_x, new_y)])
                    )
            else:
                smoothed_section_lines.append(line)

        # combine the lines into one
        coords = list(smoothed_section_lines[0].coords)
        for line in smoothed_section_lines[1:]:
            coords += list(line.coords)[1:]
        smoothed_medial_axis = LineString(coords)

        # finalize with smoothing, trimming, and simplification
        smoothed_medial_axis = LineString(
            chaikins_corner_cutting(smoothed_medial_axis.coords, smoothing_iterations)
        )
        smoothed_medial_axis = trim_line(
            smoothed_medial_axis, trim_output_lines_by_percent
        )
        smoothed_medial_axis = smoothed_medial_axis.simplify(
            geom.area * simplification_factor
        )

        if debug:
            return (
                (
                    MultiLineString(debug_skeleton),
                    MultiLineString([debug_medial_axis]),
                    smoothed_medial_axis,
                ),
                properties,
            )

        return (smoothed_medial_axis, properties)

    except Exception as e:
        traceback.print_exc()
        print(f"Error processing polygon with properties: {properties}")
        print("Skipped polygon")
        return (None, properties)
