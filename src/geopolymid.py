import click
import os
import multiprocessing

import networkx as nx
import skgeom as sg
import numpy as np
import fiona

from shapely.geometry import Polygon, LineString, Point
from scipy.interpolate import splprep, splev
from tqdm import tqdm

from .graph import dfs_sum_weights, get_heaviest_path

# create an approximation of medial axes from the input polygons
# first, we create a skeleton of each polygon
# then, we find the weight of each node, which is the sum of the distance to all child nodes
# then, we find the two heaviest paths from the center node. this gives us the path from the cetner
# that is the furthest from the polygon boundary, an approxmiation for the most visually massive section of the polygon
# then, we join the two paths together to get the medial axis
def get_weighted_medial_axis(args):
    polygon, presimplification_percentage, spline_degree, spline_points = args
    geom, properties = polygon
    try:
        polygon = sg.Polygon(geom.exterior.coords)
        # simplify the geometry to speed up the medial axis calculation
        polygon = sg.simplify(polygon, presimplification_percentage)
        skeleton = sg.skeleton.create_interior_straight_skeleton(polygon)

        graph = nx.Graph()

        for h in skeleton.halfedges:
            if h.is_bisector:
                p1 = h.vertex.point
                p2 = h.opposite.vertex.point
                graph.add_edge(
                    (float(p1.x()), float(p1.y())), (float(p2.x()), float(p2.y()))
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
        
        x, y = joined_line.xy

        if len(x) < spline_degree + 1:
            # not enough points to smooth
            return (joined_line, properties)

        # create a B-spline representation of the line
        tck, _ = splprep([x, y], k=spline_degree)
        new_x, new_y = splev(np.linspace(0, 1, spline_points), tck)
        smoothed_line = LineString([(x, y) for x, y in zip(new_x, new_y)])

        return (smoothed_line, properties)
    
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
def cli(workers, input_file, output_file, presimplification_percentage, spline_degree, spline_points):
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
                continue

            if geom["type"] == "MultiPolygon":
                for polygon in geom["coordinates"]:
                    geoms.append((Polygon(polygon[0]), feature["properties"]))
                continue

            geom = Polygon(geom["coordinates"][0])
            geoms.append((geom, feature["properties"]))
        
    print(f"Calculating smoothed medial axes for {len(geoms)} polygons...")
    lines = []
    with multiprocessing.Pool(workers) as p:
        for line, properties in tqdm(
            p.imap_unordered(get_weighted_medial_axis, 
                             [(g, presimplification_percentage, spline_degree, spline_points, ) for g in geoms]
                            ), total=len(geoms)
        ):
            if line is None:
                continue
            lines.append((line, properties))

    print(f"Sucessfully created smoothed medial axes for {len(lines)} polygons")

    schema["geometry"] = "LineString"

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
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            (float(x), float(y)) for x, y in line.coords
                        ],
                    },
                    "properties": properties,
                }
            )

    print("Done!")

if __name__ == "__main__":
    cli()
