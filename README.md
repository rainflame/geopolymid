# Geopolymid

Create smoothed medians through geographic polygons.

## Install

We use conda/mamba to manage dependencies. First, create the environment:

```
mamba env create -f environment.yml
```

Then activate it:

```
mamba activate geopolymid
```

## Usage

To create smoothed centerlines for a set of polygons, run:

```
python -m src.geopolymid \
    --input-file=data/polygons.gpkg \
    --output-file=data/centerlines.gpkg
```

## Options

```
Usage: python -m src.geopolymid [OPTIONS]

Options:
  --input-file TEXT               The input gpkg file of polygons.  [required]
  --output-file TEXT              The output gpkg file of smoothed medial
                                  axes.  [required]
  --output-file-centroids TEXT    The output gpkg file of centroids. Only
                                  required if --min-area > 0.
  --min-area FLOAT RANGE          Minimum area of polygons to process.
                                  Polygons smaller than this will have
                                  centroids calculated instead of medial axes.
                                  [0<=x<=1000000]
  --simplification-factor FLOAT RANGE
                                  Amount the output medial axes should be
                                  simplified.  [0<=x<=1]
  --smoothing-iterations INTEGER RANGE
                                  The number of smoothing iterations to apply
                                  to the medial axis.  [1<=x<=10]
  --spline-degree INTEGER RANGE   For sections of the medial axes that are
                                  smoothed with a spline, the degree of the
                                  spline. See scipy.interpolate.splprep for
                                  more info.  [1<=x<=5]
  --spline-start-percent FLOAT RANGE
                                  How far from the side of the polygon must
                                  the medial axis be to use a spline to smooth
                                  it? Expressed as a percentage of the length
                                  of the small side of the bounding box
                                  enclosing the polygon. Somewhere betwee 0.05
                                  and 0.3 often works well.  [0<=x<=1]
  --trim-output-lines-by-percent INTEGER RANGE
                                  Trim the output lines from each end by this
                                  percent.  [0<=x<=99]
  --workers INTEGER               Number of workers to use. Defaults to all
                                  available CPU cores.
  --debug                         Output debug geometry of the skeleton and
                                  medial axis in a separate file for
                                  debugging.
  --help                          Show this message and exit.
```
