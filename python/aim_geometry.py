"""Coordinate conversions shared by the aim trainer and its unit tests."""

from __future__ import annotations

from typing import Any

import math


def target_yaw_error_degrees(frame: dict[str, Any]) -> float | None:
    """Return horizontal target-center error in degrees (right positive).

    `target_dx` is the normalized screen-center offset reported by the mod;
    the horizontal FOV converts that perspective offset to an angular error.
    The result is None when no target point can be projected in front of the
    camera.
    """
    features = frame.get("features", {})
    if not features.get("target_valid", False):
        return None
    if "target_bbox_projection_valid" in features:
        if not bool(features["target_bbox_projection_valid"]):
            return None
    elif not bool(features.get("target_on_screen", False)):
        return None
    try:
        # A missing offset must not silently look like perfect centering.
        if "target_dx" not in features:
            return None
        dx = float(features["target_dx"])
        horizontal_fov = float(features.get("camera_horizontal_fov_degrees", 90.0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(dx) or not math.isfinite(horizontal_fov) or not 0.0 < horizontal_fov < 180.0:
        return None
    return math.degrees(math.atan(2.0 * dx * math.tan(math.radians(horizontal_fov) / 2.0)))


def head_aim_error(frame: dict[str, Any]) -> float | None:
    """Return the projected head point's vertical error from the screen center.

    The mod reports absolute normalized screen Y: 0 is the top, 0.5 is center,
    and 1 is the bottom. The trainer uses a signed error where alignment is 0.
    """
    features = frame.get("features", {})
    if not features.get("target_valid", False):
        return None
    if "target_aim_projection_valid" in features:
        if not bool(features["target_aim_projection_valid"]):
            return None
        # A missing or malformed point is unavailable, not perfect aim.
        try:
            aim_dy = float(features["target_aim_dy"])
        except (KeyError, TypeError, ValueError):
            return None
        return aim_dy - 0.5 if math.isfinite(aim_dy) else None

    # Compatibility with older mod builds whose target_dy is already a
    # reticle-relative center offset and whose box can approximate head Y.
    if not bool(features.get("target_on_screen", False)):
        return None
    try:
        center_dy = float(features["target_dy"])
        bbox_height = float(features["target_bbox_height"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(center_dy) or not math.isfinite(bbox_height) or bbox_height <= 1e-6:
        return None
    return center_dy - 0.32 * bbox_height
