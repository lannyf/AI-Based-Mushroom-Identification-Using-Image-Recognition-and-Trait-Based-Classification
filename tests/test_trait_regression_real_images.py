from __future__ import annotations

from pathlib import Path
from statistics import mean

import pytest

from models.key_tree_traversal import KeyTreeEngine
from models.visual_trait_extractor import extract


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "images"
KEY_XML = PROJECT_ROOT / "data" / "raw" / "key.xml"
USER_FLY_AGARIC_IMAGE = Path("/home/iannyf/projekt/Mushroom_examples/flugsvamptest.jpg")


def _sample_paths(folder: str, limit: int = 5) -> list[Path]:
    directory = IMAGE_ROOT / folder
    if not directory.exists():
        pytest.skip(f"{directory} not found")
    paths = sorted(directory.glob("*.jpg"))[:limit]
    if not paths:
        pytest.skip(f"No jpg images found in {directory}")
    return paths


def _traits(paths: list[Path]) -> list[dict]:
    return [extract(path.read_bytes())["visible_traits"] for path in paths]


def _first_auto_answer(
    engine: KeyTreeEngine, traits: dict, ml_hint: dict | None = None
) -> str | None:
    result = engine.start_session(None, traits, ml_hint)
    for item in result.get("auto_answered", []):
        if item.get("question") == "Hur ser svampen ut?":
            return item.get("answer")
    return None


@pytest.fixture(scope="module")
def engine() -> KeyTreeEngine:
    if not KEY_XML.exists():
        pytest.skip("data/raw/key.xml not found — skipping real-image regression tests")
    return KeyTreeEngine(str(KEY_XML))


class TestRealImageTraitRegression:
    def test_false_chanterelle_samples_keep_warm_trait_signal(self):
        traits = _traits(_sample_paths("HY.PS"))
        assert mean(t["colour_ratios"]["orange_yellow"] for t in traits) > 0.10
        assert mean(t["colour_ratios"]["orange_yellow"] for t in traits) > mean(
            t["colour_ratios"]["white"] for t in traits
        )

    def test_black_trumpet_samples_keep_dark_trait_signal(self):
        traits = _traits(_sample_paths("CR.CO"))
        assert mean(t["colour_ratios"]["dark"] for t in traits) > mean(
            t["colour_ratios"]["orange_red"] for t in traits
        )
        darkish = sum(
            1
            for t in traits
            if t["dominant_color"] in {"black", "grey", "olive-brown", "yellow-green"}
            or t["secondary_color"] in {"black", "grey"}
        )
        assert darkish >= 3

    def test_other_bolete_samples_keep_brown_trait_signal(self):
        traits = _traits(_sample_paths("BO.BA"))
        assert mean(t["colour_ratios"]["brown"] for t in traits) > 0.10
        assert mean(t["colour_ratios"]["brown"] for t in traits) > mean(
            t["colour_ratios"]["white"] for t in traits
        )

    def test_porcini_samples_keep_brown_trait_signal(self):
        traits = _traits(_sample_paths("BO.ED"))
        assert mean(t["colour_ratios"]["brown"] for t in traits) > 0.10
        assert mean(t["colour_ratios"]["brown"] for t in traits) > mean(
            t["colour_ratios"]["orange_red"] for t in traits
        )

    def test_fly_agaric_samples_keep_red_signal(self):
        traits = _traits(_sample_paths("AM.MU"))
        warm_red = sum(
            1
            for t in traits
            if (
                t["dominant_color"] == "red"
                or t["colour_ratios"]["red"] >= 0.08
                or t["colour_ratios"]["orange_red"] >= 0.03
            )
        )
        assert warm_red >= 3

    def test_user_fly_agaric_image_extracts_traits(self):
        """Pure extractor no longer forces colours; just verify it runs."""
        if not USER_FLY_AGARIC_IMAGE.exists():
            pytest.skip(f"{USER_FLY_AGARIC_IMAGE} not found")
        traits = extract(USER_FLY_AGARIC_IMAGE.read_bytes())["visible_traits"]
        assert "dominant_color" in traits
        assert "colour_ratios" in traits
        assert traits["colour_ratios"]["red"] > 0.0


class TestTreeTraversalRegression:
    @pytest.mark.parametrize(
        ("image_path", "expected_species", "hint_species"),
        [
            ("data/raw/images/CA.CI/Cantharellus_cibarius_1.jpg", "Kantarell", "Chanterelle"),
            ("data/raw/images/CR.CO/Craterellus_cornucopioides_1.jpg", "Svart trumpetsvamp", "Black Trumpet"),
            ("data/raw/images/BO.ED/Boletus_edulis_1.jpg", "Stensopp (karljohanssvamp)", "Porcini"),
            ("data/raw/images/BO.BA/Boletus_badius_10.jpg", "Brunsopp", "Other Boletus"),
        ],
    )
    def test_supported_species_images_traverse_to_matching_species(
        self, engine: KeyTreeEngine, image_path: str, expected_species: str, hint_species: str
    ):
        step1 = extract((PROJECT_ROOT / image_path).read_bytes())
        ml_hint = {"top_species": hint_species, "confidence": 0.95}
        result = engine.start_session(None, step1["visible_traits"], ml_hint)
        assert result["status"] == "conclusion"
        assert result["species"] == expected_species

    @pytest.mark.parametrize("folder", ["BO.ED", "BO.BA"])
    def test_bolete_images_do_not_auto_answer_ridges(self, engine: KeyTreeEngine, folder: str):
        for path in _sample_paths(folder, limit=3):
            step1 = extract(path.read_bytes())
            ml_hint = {"top_species": "Porcini", "confidence": 0.95}
            answer = _first_auto_answer(engine, step1["visible_traits"], ml_hint)
            assert answer != "Undersidan har åsar eller ådror", path.name

    @pytest.mark.parametrize("folder", ["CA.CI", "CR.CO"])
    def test_chanterelle_like_images_do_not_auto_answer_pores(self, engine: KeyTreeEngine, folder: str):
        for path in _sample_paths(folder, limit=3):
            step1 = extract(path.read_bytes())
            ml_hint = {"top_species": "Chanterelle", "confidence": 0.95}
            answer = _first_auto_answer(engine, step1["visible_traits"], ml_hint)
            assert answer != "Undersidan har rör", path.name

    def test_fly_agaric_images_do_not_auto_answer_ridges(self, engine: KeyTreeEngine):
        for path in _sample_paths("AM.MU", limit=3):
            step1 = extract(path.read_bytes())
            ml_hint = {"top_species": "Fly Agaric", "confidence": 0.95}
            answer = _first_auto_answer(engine, step1["visible_traits"], ml_hint)
            assert answer != "Undersidan har åsar eller ådror", path.name

    def test_user_fly_agaric_image_uses_ml_hint_for_precheck(self, engine: KeyTreeEngine):
        if not USER_FLY_AGARIC_IMAGE.exists():
            pytest.skip(f"{USER_FLY_AGARIC_IMAGE} not found")
        traits = extract(USER_FLY_AGARIC_IMAGE.read_bytes())["visible_traits"]
        result = engine.start_session(
            None, traits, ml_hint={"top_species": "Fly Agaric", "confidence": 0.95}
        )
        assert result["status"] == "conclusion"
        assert result["species"] == "Flugsvamp"
        assert result["tree_compatibility"]["tree_policy"] == "skip_tree"

    @pytest.mark.parametrize(
        ("image_path", "expected_species", "ml_species"),
        [
            ("data/raw/images/AM.MU/Amanita_muscaria_1.jpg", "Flugsvamp", "Fly Agaric"),
            ("data/raw/images/AM.VI/Amanita_virosa_1.jpg", "Änglsvamp", "Amanita virosa"),
            ("data/raw/images/HY.PS/Hygrophoropsis_aurantiaca_12.jpg", "Falsk kantarell", "False Chanterelle"),
        ],
    )
    def test_species_missing_from_tree_use_precheck_instead_of_wrong_tree_species(
        self, engine: KeyTreeEngine, image_path: str, expected_species: str, ml_species: str
    ):
        traits = extract((PROJECT_ROOT / image_path).read_bytes())["visible_traits"]
        result = engine.start_session(
            None, traits, ml_hint={"top_species": ml_species, "confidence": 0.95}
        )
        assert result["status"] == "conclusion"
        assert result["species"] == expected_species
        assert result["tree_compatibility"]["tree_policy"] == "skip_tree"
