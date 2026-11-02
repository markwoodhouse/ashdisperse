from numba import njit

webmerc = {'init': 'epsg:3857'}
wgs84 = {'init': 'epsg:4326'}


@njit
def utm_zone_number(lat, lon):
    """UTM zone number for a point, following the same rules as utm.from_latlon.

    Includes the Norway and Svalbard exceptions so that the EPSG code always
    matches the eastings/northings computed with the ``utm`` package.
    """
    # normalize longitude to [-180, 180)
    lon = (lon % 360 + 540) % 360 - 180

    # special zone for Norway
    if 56 <= lat < 64 and 3 <= lon < 12:
        return 32
    # special zones for Svalbard
    if 72 <= lat <= 84 and lon >= 0:
        if lon < 9:
            return 31
        elif lon < 21:
            return 33
        elif lon < 33:
            return 35
        elif lon < 42:
            return 37

    return int((lon + 180) / 6) + 1


@njit
def latlon_point_to_utm_code(lat, lon):
    """EPSG code of the WGS84 UTM zone containing (lat, lon)."""
    zone = utm_zone_number(lat, lon)
    return 32600 + zone if (lat >= 0) else 32700 + zone
