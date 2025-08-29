from numba import njit

webmerc = {'init': 'epsg:3857'}
wgs84 = {'init': 'epsg:4326'}

@njit
def latlon_point_to_utm_code(lat, lon):
    offset = int(round((183+lon)/6.0))
    return 32600+offset if (lat > 0) else 32700+offset
