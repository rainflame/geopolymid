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
  --workers INTEGER               Number of workers to use. Defaults to all
                                  available cores.
  --input-file TEXT               The input gpkg file of polygons.  [required]
  --output-file TEXT              The output gpkg file of smoothed medial
                                  axes.  [required]
  --skip-spline BOOLEAN           Don't smooth the medial axis with a
                                  B-spline.
  --smoothing-iterations INTEGER RANGE
                                  The number of smoothing iterations to apply
                                  to the medial axis (non-spline sections
                                  only).  [1<=x<=10]
  --spline-degree INTEGER RANGE   The degree of the spline. See
                                  scipy.interpolate.splprep for more info.
                                  [1<=x<=5]
  --spline-distance-threshold INTEGER RANGE
                                  The distance in meters from the edge of the
                                  polygon the centerline must be to smooth
                                  with a B-spline.  [0<=x<=100000]
  --spline-distance-allowable-variance INTEGER RANGE
                                  Once a section is greater than --spline-
                                  distance-threshold, how much can the
                                  distance vary less than --spline-distance-
                                  threshold before the spline is terminated?
                                  [0<=x<=100000]
  --debug                         Output debug geometry of the skeleton and
                                  medial axis in a separate file.
  --help                          Show this message and exit.
```
