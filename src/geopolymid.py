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
    "--skip-spline",
    help="Don't smooth the medial axis.",
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
    "--smoothing-iterations",
    help="The number of smoothing iterations to apply to the medial axis.",
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
    "--spline-points",
    help="The number of points to create the spline line from.",
    default=100,
    type=click.IntRange(25, 400),
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
    skip_spline,
    presimplification_percentage,
    smoothing_iterations,
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
                        smoothing_iterations,
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
