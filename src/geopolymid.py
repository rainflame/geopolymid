import click
import os
import multiprocessing
import fiona

from shapely.geometry import Polygon, mapping
from shapely.geometry.polygon import orient
from tqdm import tqdm

from .polygon import reduce_polygon_dimensions, get_weighted_medial_axis


@click.command()
@click.option(
    "--workers",
    default=multiprocessing.cpu_count(),
    help="Number of workers to use. Defaults to all available cores.",
)
@click.option(
    "--input-file",
    help="The input gpkg file of polygons.",
    required=True,
)
@click.option(
    "--output-file",
    help="The output gpkg file of smoothed medial axes.",
    required=True,
)
@click.option(
    "--output-file-centroids",
    help="The output gpkg file of centroids, if --min-area > 0.",
    required=False,
)
@click.option(
    "--min-area",
    help="Minimum area of polygons to process. Smaller polygons will have centroids calculated instead of medial axes.",
    default=0,
    type=click.FloatRange(0, 1000000),
    required=False,
)
@click.option(
    "--simplification-factor",
    help="Amount the output medial axes should be simplified.",
    default=0.5,
    type=click.FloatRange(0, 1),
    required=False,
)
@click.option(
    "--smoothing-iterations",
    help="The number of smoothing iterations to apply to the medial axis (non-spline sections only).",
    default=5,
    type=click.IntRange(1, 10),
    required=False,
)
@click.option(
    "--spline-degree",
    help="The degree of the spline. See scipy.interpolate.splprep for more info.",
    default=3,
    type=click.IntRange(1, 5),
    required=False,
)
@click.option(
    "--spline-start-percent",
    help="How far from the side of the polygon must the medial axis be to convert it to a spline? Expressed as a percentage of the length of the small side of the bounding box enclosing the polygon. Recommended to be between 0.05 and 0.3",
    default=0.2,
    type=click.FloatRange(0, 1),
    required=False,
)
@click.option(
    "--trim-output-lines-by-percent",
    help="Trim the output from each end by this percent.",
    default=0,
    type=click.IntRange(0, 99),
    required=False,
)
@click.option(
    "--debug",
    help="Output debug geometry of the skeleton and medial axis in a separate file.",
    default=False,
    required=False,
    is_flag=True,
)
def cli(
    workers,
    input_file,
    output_file,
    output_file_centroids,
    min_area,
    simplification_factor,
    smoothing_iterations,
    spline_degree,
    spline_start_percent,
    trim_output_lines_by_percent,
    debug,
):
    if not os.path.exists(input_file):
        raise Exception(f"Cannot open {input_file}")

    output_dir = os.path.dirname(output_file)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if min_area > 0 and output_file_centroids is None:
        raise Exception("Output file for centroids is required when --min-area > 0")

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
    results = []
    with multiprocessing.Pool(workers) as p:
        for result in tqdm(
            p.imap_unordered(
                get_weighted_medial_axis,
                [
                    (
                        g,
                        min_area,
                        simplification_factor,
                        smoothing_iterations,
                        spline_degree,
                        spline_start_percent,
                        trim_output_lines_by_percent,
                    )
                    for g in geoms
                ],
            ),
            total=len(geoms),
        ):
            if result is None:
                continue
            else:
                results.append(result)

    print(f"Sucessfully created smoothed medial axes for {len(results)} polygons")

    if debug:
        res_skeletons = list(filter(lambda x: x.debug_skeleton is not None, results))
        res_medial_axes = list(
            filter(lambda x: x.debug_medial_axis is not None, results)
        )
        skeleton_output_file = output_file.replace(".gpkg", "_skeleton.gpkg")
        medial_axis_output_file = output_file.replace(".gpkg", "_medial_axis.gpkg")
        print(f"Writing {len(res_skeletons)} debug skeletons to file...")
        schema["geometry"] = "MultiLineString"
        with fiona.open(
            skeleton_output_file,
            "w",
            driver="GPKG",
            crs=crs,
            schema=schema,
        ) as f:
            for res in res_skeletons:
                f.write(
                    {
                        "geometry": mapping(res.debug_skeleton),
                        "properties": res.properties,
                    }
                )

        print(f"Writing {len(res_medial_axes)} debug medial axes to file...")
        schema["geometry"] = "MultiLineString"
        with fiona.open(
            medial_axis_output_file,
            "w",
            driver="GPKG",
            crs=crs,
            schema=schema,
        ) as f:
            for res in res_medial_axes:
                f.write(
                    {
                        "geometry": mapping(res.debug_medial_axis),
                        "properties": res.properties,
                    }
                )

    res_lines = list(filter(lambda x: x.axis is not None, results))
    res_centroids = list(filter(lambda x: x.centroid is not None, results))
    print(f"Writing {len(res_lines)} medial axes to file...")
    schema["geometry"] = "LineString"
    with fiona.open(
        output_file,
        "w",
        driver="GPKG",
        crs=crs,
        schema=schema,
    ) as f:
        for res in res_lines:
            f.write(
                {
                    "geometry": mapping(res.axis),
                    "properties": res.properties,
                }
            )

    if len(res_centroids) > 0 and output_file_centroids is not None:
        print(f"Writing {len(res_centroids)} centroids to file...")
        schema["geometry"] = "Point"
        with fiona.open(
            output_file_centroids,
            "w",
            driver="GPKG",
            crs=crs,
            schema=schema,
        ) as f:
            for res in res_centroids:
                f.write(
                    {
                        "geometry": mapping(res.centroid),
                        "properties": res.properties,
                    }
                )

    print("Done!")


if __name__ == "__main__":
    cli()
