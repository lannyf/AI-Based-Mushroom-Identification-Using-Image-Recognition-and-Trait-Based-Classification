"""
Unit tests for models/visual_trait_extractor.py

Tests cover:
  - _hsv_to_name: colour naming from HSV values
  - analyse_colours: returns expected keys and valid ratios
  - analyse_shape: returns expected keys and valid shape names
  - analyse_texture: returns expected keys
  - extract: end-to-end with a synthetic PNG image
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import Dict

import numpy as np
import pytest

from models.visual_trait_extractor import (
    _hsv_to_name,
    analyse_brightness,
    analyse_colours,
    analyse_shape,
    analyse_texture,
    extract,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic images
# ---------------------------------------------------------------------------

def _make_png(r: int, g: int, b: int, w: int = 64, h: int = 64) -> bytes:
    """Create a minimal solid-colour RGB PNG in memory."""
    raw_rows = b""
    for _ in range(h):
        row = b"\x00" + bytes([r, g, b] * w)
        raw_rows += row
    compressed = zlib.compress(raw_rows)
    # PNG header + IHDR + IDAT + IEND
    def chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        header
        + chunk(b"IHDR", ihdr_data)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


RED_PNG      = _make_png(200, 30, 30)    # clearly red
WHITE_PNG    = _make_png(230, 230, 230)  # white / pale
BROWN_PNG    = _make_png(120, 70, 30)    # brown
YELLOW_PNG   = _make_png(220, 190, 30)   # yellow / chanterelle-ish
BLACK_PNG    = _make_png(20,  20,  20)   # very dark


# ---------------------------------------------------------------------------
# _hsv_to_name
# ---------------------------------------------------------------------------

class TestHsvToName:
    def test_white(self):
        assert _hsv_to_name(0, 10, 220) == "white"

    def test_black(self):
        assert _hsv_to_name(0, 0, 20) == "black"

    def test_grey(self):
        assert _hsv_to_name(0, 20, 100) == "grey"

    def test_red_low_hue(self):
        name = _hsv_to_name(5, 200, 180)
        assert name == "red"

    def test_orange(self):
        name = _hsv_to_name(15, 200, 200)
        assert name in {"orange", "orange-yellow"}

    def test_brown(self):
        name = _hsv_to_name(18, 80, 90)
        assert name in {"brown", "olive-brown", "tan", "orange"}


# ---------------------------------------------------------------------------
# analyse_colours
# ---------------------------------------------------------------------------

class TestAnalyseColours:
    def _bgr(self, r, g, b, size=128):
        import cv2
        arr = np.full((size, size, 3), [b, g, r], dtype=np.uint8)
        return arr

    def test_returns_required_keys(self):
        import cv2
        bgr = self._bgr(200, 30, 30)
        result = analyse_colours(bgr)
        for key in ("dominant_color", "secondary_color", "red_ratio",
                    "orange_yellow_ratio", "brown_ratio", "white_ratio",
                    "dark_ratio"):
            assert key in result, f"Missing key: {key}"

    def test_red_image_has_high_red_ratio(self):
        import cv2
        bgr = self._bgr(220, 30, 30)
        result = analyse_colours(bgr)
        assert result["red_ratio"] > 0.5

    def test_white_image_has_high_white_ratio(self):
        import cv2
        bgr = self._bgr(235, 235, 235)
        result = analyse_colours(bgr)
        assert result["white_ratio"] > 0.5

    def test_brown_image_has_brown_ratio(self):
        import cv2
        bgr = self._bgr(120, 70, 30)
        result = analyse_colours(bgr)
        assert result["brown_ratio"] > 0.0

    def test_dark_image_has_dark_ratio(self):
        import cv2
        bgr = self._bgr(15, 15, 15)
        result = analyse_colours(bgr)
        assert result["dark_ratio"] > 0.5

    def test_ratios_are_in_0_1(self):
        import cv2
        bgr = self._bgr(100, 150, 80)
        result = analyse_colours(bgr)
        for key in ("red_ratio", "orange_yellow_ratio", "brown_ratio",
                    "white_ratio", "dark_ratio"):
            val = result[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# analyse_shape
# ---------------------------------------------------------------------------

class TestAnalyseShape:
    def _bgr(self, size=128):
        import cv2
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(arr, (size // 2, size // 2), size // 3, (255, 255, 255), -1)
        return arr

    def test_returns_required_keys(self):
        result = analyse_shape(self._bgr())
        for key in ("cap_shape", "aspect_ratio", "circularity"):
            assert key in result

    def test_cap_shape_is_valid_string(self):
        result = analyse_shape(self._bgr())
        valid = {"convex", "flat", "bell-shaped", "wavy", "funnel-shaped", "irregular", "unknown"}
        assert result["cap_shape"] in valid

    def test_aspect_ratio_positive(self):
        result = analyse_shape(self._bgr())
        assert result["aspect_ratio"] > 0

    def test_circle_shape_is_convex(self):
        result = analyse_shape(self._bgr())
        assert result["cap_shape"] == "convex"


# ---------------------------------------------------------------------------
# analyse_texture
# ---------------------------------------------------------------------------

class TestAnalyseTexture:
    def _uniform_bgr(self):
        return np.full((64, 64, 3), 128, dtype=np.uint8)

    def test_returns_required_keys(self):
        result = analyse_texture(self._uniform_bgr())
        for key in ("surface_texture", "edge_density", "has_ridges"):
            assert key in result

    def test_surface_texture_is_valid(self):
        result = analyse_texture(self._uniform_bgr())
        assert result["surface_texture"] in {"smooth", "fibrous", "scaly"}

    def test_uniform_image_is_smooth(self):
        result = analyse_texture(self._uniform_bgr())
        assert result["surface_texture"] == "smooth"

    def test_edge_density_in_range(self):
        result = analyse_texture(self._uniform_bgr())
        assert 0.0 <= result["edge_density"] <= 1.0


# ---------------------------------------------------------------------------
# analyse_brightness
# ---------------------------------------------------------------------------

class TestAnalyseBrightness:
    def test_bright_image(self):
        bgr = np.full((32, 32, 3), 220, dtype=np.uint8)
        assert analyse_brightness(bgr) == "bright"

    def test_dark_image(self):
        bgr = np.full((32, 32, 3), 20, dtype=np.uint8)
        assert analyse_brightness(bgr) == "dark"

    def test_medium_image(self):
        bgr = np.full((32, 32, 3), 120, dtype=np.uint8)
        assert analyse_brightness(bgr) == "medium"


# ---------------------------------------------------------------------------
# extract (end-to-end, no trained CNN)
# ---------------------------------------------------------------------------

class TestExtract:
    def test_returns_visible_traits(self):
        result = extract(RED_PNG)
        assert "visible_traits" in result

    def test_visible_traits_has_required_keys(self):
        result = extract(RED_PNG)
        vt = result["visible_traits"]
        for key in ("dominant_color", "cap_shape", "surface_texture",
                    "has_ridges", "brightness", "colour_ratios"):
            assert key in vt

    def test_visible_traits_has_no_ml_fields(self):
        """visible_traits must not contain CNN prediction fields."""
        result = extract(RED_PNG)
        vt = result["visible_traits"]
        assert "ml_top_species" not in vt
        assert "ml_confidence" not in vt

    def test_colour_ratios_present(self):
        result = extract(RED_PNG)
        cr = result["visible_traits"]["colour_ratios"]
        for key in ("red", "orange_yellow", "brown", "white", "dark"):
            assert key in cr

    def test_red_image_ratings(self):
        """A strongly red image should report high red_ratio."""
        result = extract(RED_PNG)
        assert result["visible_traits"]["colour_ratios"]["red"] > 0.5
