import click
import os
import multiprocessing

import networkx as nx
import skgeom as sg
import numpy as np
import fiona

from shapely.geometry import Polygon, LineString, MultiLineString, mapping
from shapely.geometry.polygon import orient
from scipy.interpolate import splprep, splev
from tqdm import tqdm

from .graph import dfs_sum_weights, get_heaviest_path
from .polygon import reduce_polygon_dimensions


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

        # # return the skeleton as a LineString
        # string_skel = []
        # for h in skeleton.halfedges:
        #     if h.is_bisector:
        #         p1 = h.vertex.point
        #         p2 = h.opposite.vertex.point
        #         string_skel.append(LineString([(float(p1.x()), float(p1.y())), (float(p2.x()), float(p2.y()))]))

        # return (MultiLineString(string_skel), properties)

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
            # ensure there's enough points to create a spline
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
                    MultiLineString(debug_medial_axis),
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


@click.command()
@click.option(
    "--workers", default=multiprocessing.cpu_count(), help="Number of workers to use"
)
@click.option(
    "--input-file",
    help="The input gpkg file of polygons",
    required=True,
)
@click.option(
    "--output-file",
    help="The output gpkg file of lines",
    required=True,
)
@click.option(
    "--skip-spline",
    help="Don't smooth the medial axis",
    default=False,
    required=False,
)
@click.option(
    "--presimplification-percentage",
    help="The simplificication percentage to apply to the input polygons. This speeds up the medial axis calculation, but may result in a less accurate medial axis.",
    default=0.5,
    type=click.FloatRange(0, 1),
    required=False,
)
@click.option(
    "--spline-degree",
    help="The degree of the spline. See scipy.interpolate.splprep for more info",
    default=3,
    type=click.IntRange(1, 5),
    required=False,
)
@click.option(
    "--spline-points",
    help="The number of points to use in the spline.",
    default=100,
    type=click.IntRange(25, 400),
    required=False,
)
@click.option(
    "--debug",
    help="Output debug geometry",
    default=False,
    required=False,
)
def cli(
    workers,
    input_file,
    output_file,
    skip_spline,
    presimplification_percentage,
    spline_degree,
    spline_points,
    debug,
):
    # check input exists
    if not os.path.exists(input_file):
        raise Exception(f"Cannot open {input_file}")

    # check path to output exists
    output_dir = os.path.dirname(output_file)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    geoms = []
    with fiona.open(input_file, "r") as f:
        crs = f.crs
        schema = f.schema
        for feature in f:
            geom = feature["geometry"]

            if geom["type"] != "Polygon" and geom["type"] != "MultiPolygon":
                print(f"Skipping non-polygon geometry...")

            elif geom["type"] == "MultiPolygon":
                for polygon in geom["coordinates"]:
                    # reduce coordinates to two dimensions
                    poly_geom = reduce_polygon_dimensions(Polygon(polygon[0]))
                    # ensure points are oriented counter-clockwise, otherwise skeleton will be inverted
                    poly_geom = orient(poly_geom, sign=1.0)
                    geoms.append((poly_geom, feature["properties"]))
            else:
                poly_geom = reduce_polygon_dimensions(Polygon(geom["coordinates"][0]))
                poly_geom = orient(poly_geom, sign=1.0)
                geoms.append((poly_geom, feature["properties"]))

    print(f"Calculating smoothed medial axes for {len(geoms)} polygons...")
    lines = []
    with multiprocessing.Pool(workers) as p:
        for line, properties in tqdm(
            p.imap_unordered(
                get_weighted_medial_axis,
                [
                    (
                        g,
                        skip_spline,
                        presimplification_percentage,
                        spline_degree,
                        spline_points,
                        debug,
                    )
                    for g in geoms
                ],
            ),
            total=len(geoms),
        ):
            if line is None:
                continue
            lines.append((line, properties))

    print(f"Sucessfully created smoothed medial axes for {len(lines)} polygons")

    schema["geometry"] = "MultiLineString"

    if debug:
        skeleton_output_file = output_file.replace(".gpkg", "_skeleton.gpkg")
        medial_axis_output_file = output_file.replace(".gpkg", "_medial_axis.gpkg")

        with fiona.open(
            skeleton_output_file,
            "w",
            driver="GPKG",
            crs=crs,
            schema=schema,
        ) as f:
            for line, properties in lines:
                skeleton, _, _ = line
                if len(skeleton) > 0:
                    f.write(
                        {
                            "geometry": mapping(skeleton),
                            "properties": properties,
                        }
                    )

        with fiona.open(
            medial_axis_output_file,
            "w",
            driver="GPKG",
            crs=crs,
            schema=schema,
        ) as f:
            for line, properties in lines:
                _, medial_axis, _ = line
                if len(medial_axis) > 0:
                    f.write(
                        {
                            "geometry": mapping(medial_axis),
                            "properties": properties,
                        }
                    )

    with fiona.open(
        output_file,
        "w",
        driver="GPKG",
        crs=crs,
        schema=schema,
    ) as f:
        for line, properties in lines:
            f.write(
                {
                    "geometry": mapping(line),
                    "properties": properties,
                }
            )

    print("Done!")


if __name__ == "__main__":
    cli()
