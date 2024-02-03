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
    "--simplification-factor",
    help="Amount the output medial axes should be simplified.",
    default=0.5,
    type=click.FloatRange(0, 1),
    required=False,
)
@click.option(
    "--skip-spline",
    help="Don't smooth the medial axis with a B-spline.",
    default=False,
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
    "--spline-distance-threshold",
    help="The distance in meters from the edge of the polygon the centerline must be to smooth with a B-spline.",
    default=600,
    type=click.IntRange(0, 100000),
    required=False,
)
@click.option(
    "--spline-distance-allowable-variance",
    help="Once a section is greater than --spline-distance-threshold, how much can the distance vary less than --spline-distance-threshold before the spline is terminated?",
    default=50,
    type=click.IntRange(0, 100000),
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
    simplification_factor,
    skip_spline,
    smoothing_iterations,
    spline_degree,
    spline_distance_threshold,
    spline_distance_allowable_variance,
    trim_output_lines_by_percent,
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
                        simplification_factor,
                        skip_spline,
                        smoothing_iterations,
                        spline_degree,
                        spline_distance_threshold,
                        spline_distance_allowable_variance,
                        trim_output_lines_by_percent,
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

    if debug:
        skeleton_output_file = output_file.replace(".gpkg", "_skeleton.gpkg")
        medial_axis_output_file = output_file.replace(".gpkg", "_medial_axis.gpkg")
        schema["geometry"] = "MultiLineString"
        with fiona.open(
            skeleton_output_file,
            "w",
            driver="GPKG",
            crs=crs,
            schema=schema,
        ) as f:
            for line, properties in lines:
                skeleton, _, _ = line
                f.write(
                    {
                        "geometry": mapping(skeleton),
                        "properties": properties,
                    }
                )
        schema["geometry"] = "LineString"
        with fiona.open(
            medial_axis_output_file,
            "w",
            driver="GPKG",
            crs=crs,
            schema=schema,
        ) as f:
            for line, properties in lines:
                _, medial_axis, _ = line
                f.write(
                    {
                        "geometry": mapping(medial_axis),
                        "properties": properties,
                    }
                )

    schema["geometry"] = "LineString"

    with fiona.open(
        output_file,
        "w",
        driver="GPKG",
        crs=crs,
        schema=schema,
    ) as f:
        for line, properties in lines:
            if debug:
                _, _, line = line

            f.write(
                {
                    "geometry": mapping(line),
                    "properties": properties,
                }
            )

    print("Done!")


if __name__ == "__main__":
    cli()
