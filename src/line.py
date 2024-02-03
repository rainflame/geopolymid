from shapely.geometry import LineString


def trim_line(linestring, reduction_percent):
    original_length = linestring.length
    trim_length_each_side = original_length * (reduction_percent / 100) / 2

    start_dist = trim_length_each_side
    end_dist = original_length - trim_length_each_side

    # Find the points along the line that correspond to these distances
    coords = list(linestring.coords)
    new_coords = []
    accumulated_length = 0

    for i in range(len(coords) - 1):
        segment = LineString(coords[i : i + 2])
        segment_length = segment.length
        next_accumulated_length = accumulated_length + segment_length

        # Add points to new_coords based on accumulated_length
        if next_accumulated_length > start_dist and accumulated_length < end_dist:
            if accumulated_length <= start_dist:
                # Add the interpolated start point
                exact_start_point = segment.interpolate(start_dist - accumulated_length)
                new_coords.append(exact_start_point.coords[0])

            new_coords.append(coords[i + 1])

        if next_accumulated_length >= end_dist:
            if accumulated_length < end_dist:
                # Add the interpolated end point
                exact_end_point = segment.interpolate(end_dist - accumulated_length)
                new_coords.append(exact_end_point.coords[0])
            break

        accumulated_length = next_accumulated_length

    return LineString(new_coords)
