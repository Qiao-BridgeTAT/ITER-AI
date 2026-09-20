"""Local GCJ-02 geometry for meal recall, independent of route feasibility."""

from __future__ import annotations

from math import cos, degrees, hypot, radians

from backend.contracts.enums import CoordinateSystem
from backend.contracts.places import Gcj02Coordinates

EARTH_RADIUS_M = 6_371_008.8
CORRIDOR_MARGIN_M = 3_000.0


def corridor_polygon(
    a: Gcj02Coordinates | None, b: Gcj02Coordinates | None
) -> tuple[Gcj02Coordinates, ...]:
    """Return a closed rectangle, 3 km past each endpoint and on either side.

    A midpoint equirectangular projection keeps this local calculation fast and
    deterministic. The resulting search region is not a route-access promise.
    One endpoint, or coincident endpoints, produces a 6 by 6 km square.
    """
    if a is None and b is None:
        raise ValueError("meal corridor requires at least one reliable endpoint")
    a = a or b
    b = b or a
    assert a is not None and b is not None
    if any(point.coord_system is not CoordinateSystem.GCJ_02 for point in (a, b)):
        raise ValueError("meal corridor requires GCJ-02 coordinates")
    latitude = (a.latitude + b.latitude) / 2
    longitude = (a.longitude + b.longitude) / 2
    longitude_scale = cos(radians(latitude))
    if abs(longitude_scale) < 1e-8:
        raise ValueError("meal corridor cannot be projected at the poles")

    def project(point: Gcj02Coordinates) -> tuple[float, float]:
        return (
            EARTH_RADIUS_M * radians(point.longitude - longitude) * longitude_scale,
            EARTH_RADIUS_M * radians(point.latitude - latitude),
        )

    ax, ay = project(a)
    bx, by = project(b)
    length = hypot(bx - ax, by - ay)
    ux, uy = ((bx - ax) / length, (by - ay) / length) if length > 1e-6 else (1.0, 0.0)
    nx, ny = -uy, ux
    margin = CORRIDOR_MARGIN_M
    start = (ax - ux * margin, ay - uy * margin)
    end = (bx + ux * margin, by + uy * margin)
    corners = (
        (start[0] + nx * margin, start[1] + ny * margin),
        (end[0] + nx * margin, end[1] + ny * margin),
        (end[0] - nx * margin, end[1] - ny * margin),
        (start[0] - nx * margin, start[1] - ny * margin),
    )
    points = tuple(
        Gcj02Coordinates(
            longitude=round(longitude + degrees(x / (EARTH_RADIUS_M * longitude_scale)), 6),
            latitude=round(latitude + degrees(y / EARTH_RADIUS_M), 6),
            coord_system=CoordinateSystem.GCJ_02,
        )
        for x, y in corners
    )
    return (*points, points[0])


def point_in_polygon(point: Gcj02Coordinates, polygon: tuple[Gcj02Coordinates, ...]) -> bool:
    """Include boundary points; used only to filter stored replay facts."""
    x, y = point.longitude, point.latitude
    inside = False
    for start, end in zip(polygon, polygon[1:], strict=False):
        x1, y1 = start.longitude, start.latitude
        x2, y2 = end.longitude, end.latitude
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-12
            and min(x1, x2) <= x <= max(x1, x2)
            and min(y1, y2) <= y <= max(y1, y2)
        ):
            return True
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside
