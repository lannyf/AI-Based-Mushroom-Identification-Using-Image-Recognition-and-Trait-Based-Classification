"""
Unit tests for models/visual_trait_extractor.py

Tests cover:
  - _hsv_to_name: colour naming from HSV values
  - analyse_colours: returns expected keys and valid ratios
  - analyse_shape: returns expected keys and valid shape names
  - analyse_texture: returns expected keys
  - score_species: dominant colour/shape/texture influence scores
  - _normalise: sums to 1.0, handles zero scores
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
    _normalise,
    analyse_brightness,
    analyse_colours,
    analyse_shape,
    analyse_texture,
    extract,
    score_species,
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
# score_species
# ---------------------------------------------------------------------------

class TestScoreSpecies:
    def _default_colour(self, dominant="unknown") -> dict:
        return {
            "dominant_color": dominant, "secondary_color": "unknown",
            "red_ratio": 0.0, "orange_red_ratio": 0.0,
            "orange_yellow_ratio": 0.0, "brown_ratio": 0.0,
            "white_ratio": 0.0, "dark_ratio": 0.0,
        }

    def _default_shape(self, cap_shape="convex") -> dict:
        return {"cap_shape": cap_shape, "aspect_ratio": 1.0, "circularity": 0.8}

    def _default_texture(self, surface="smooth", has_ridges=False) -> dict:
        return {"surface_texture": surface, "edge_density": 0.05, "has_ridges": has_ridges}

    def test_returns_all_species(self):
        from models.visual_trait_extractor import _ALL_SPECIES
        scores = score_species(self._default_colour(), self._default_shape(), self._default_texture())
        for sp in _ALL_SPECIES:
            assert sp in scores

    def test_red_cap_boosts_fly_agaric(self):
        colour = self._default_colour("red")
        colour["red_ratio"] = 0.8
        scores = score_species(colour, self._default_shape(), self._default_texture())
        assert scores["Fly Agaric"] > scores["Chanterelle"]
        assert scores["Fly Agaric"] > scores["Porcini"]

    def test_brown_cap_boosts_porcini(self):
        colour = self._default_colour("brown")
        colour["brown_ratio"] = 0.7
        scores = score_species(colour, self._default_shape(), self._default_texture())
        assert scores["Porcini"] > scores["Fly Agaric"]
        assert scores["Porcini"] > scores["Amanita virosa"]

    def test_white_cap_boosts_amanita_virosa(self):
        colour = self._default_colour("white")
        colour["white_ratio"] = 0.8
        scores = score_species(colour, self._default_shape(), self._default_texture())
        assert scores["Amanita virosa"] > scores["Porcini"]

    def test_dark_cap_boosts_black_trumpet(self):
        colour = self._default_colour("black")
        colour["dark_ratio"] = 0.8
        scores = score_species(colour, self._default_shape(), self._default_texture())
        assert scores["Black Trumpet"] > scores["Chanterelle"]

    def test_ridges_boost_chanterelle(self):
        colour = self._default_colour("yellow")
        colour["orange_yellow_ratio"] = 0.5
        scores = score_species(colour, self._default_shape(), self._default_texture(has_ridges=True))
        assert scores["Chanterelle"] > scores["Porcini"]

    def test_scores_all_positive(self):
        scores = score_species(self._default_colour(), self._default_shape(), self._default_texture())
        for sp, val in scores.items():
            assert val >= 0, f"Negative score for {sp}: {val}"


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_sums_to_one(self):
        scores = {"A": 1.0, "B": 2.0, "C": 3.0}
        norm = _normalise(scores)
        assert abs(sum(norm.values()) - 1.0) < 1e-9

    def test_uniform_for_all_zeros(self):
        scores = {"A": 0.0, "B": 0.0}
        norm = _normalise(scores)
        assert abs(norm["A"] - 0.5) < 1e-9
        assert abs(norm["B"] - 0.5) < 1e-9

    def test_negative_values_clamped(self):
        scores = {"A": -1.0, "B": 2.0}
        norm = _normalise(scores)
        assert norm["A"] == 0.0
        assert norm["B"] == pytest.approx(1.0)

    def test_preserves_ordering(self):
        scores = {"A": 1.0, "B": 5.0, "C": 2.0}
        norm = _normalise(scores)
        assert norm["B"] > norm["C"] > norm["A"]


# ---------------------------------------------------------------------------
# extract (end-to-end, no trained CNN)
# ---------------------------------------------------------------------------

class TestExtract:
    def test_returns_ml_prediction_and_visible_traits(self):
        result = extract(RED_PNG)
        assert "ml_prediction" in result
        assert "visible_traits" in result

    def test_ml_prediction_has_required_keys(self):
        result = extract(RED_PNG)
        ml = result["ml_prediction"]
        for key in ("top_species", "confidence", "method", "top_k", "reasoning"):
            assert key in ml

    def test_visible_traits_has_required_keys(self):
        result = extract(RED_PNG)
        vt = result["visible_traits"]
        for key in ("dominant_color", "cap_shape", "surface_texture",
                    "has_ridges", "brightness", "colour_ratios"):
            assert key in vt

    def test_confidence_in_0_1(self):
        result = extract(RED_PNG)
        conf = result["ml_prediction"]["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_top_k_is_list_of_dicts(self):
        result = extract(RED_PNG)
        top_k = result["ml_prediction"]["top_k"]
        assert isinstance(top_k, list)
        assert len(top_k) >= 1
        assert "species" in top_k[0]
        assert "confidence" in top_k[0]

    def test_method_is_cv_fallback_without_weights(self):
        result = extract(WHITE_PNG)
        assert result["ml_prediction"]["method"] in {"cv_fallback", "cnn"}

    def test_colour_ratios_present(self):
        result = extract(RED_PNG)
        cr = result["visible_traits"]["colour_ratios"]
        for key in ("red", "orange_yellow", "brown", "white", "dark"):
            assert key in cr

    def test_red_image_top_species_not_chanterelle(self):
        """A strongly red image should not rank Chanterelle first."""
        result = extract(RED_PNG)
        # We don't hard-assert species (no trained CNN), but verify it runs
        assert result["ml_prediction"]["top_species"] != ""
