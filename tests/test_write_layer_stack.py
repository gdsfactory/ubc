"""Tests for the generated layer-stack documentation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from ubcpdk import PDK


def _load_generator():
    path = Path(__file__).parents[1] / ".github" / "write_layer_stack.py"
    spec = spec_from_file_location("write_layer_stack", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_generator()


def test_gds_layers_are_resolved_to_physical_tuples() -> None:
    PDK.activate()

    assert generator._gds_layer_tuple(PDK.layer_stack.layers["substrate"].layer) == (
        999,
        0,
    )
    assert generator._gds_layer_tuple(PDK.layer_stack.layers["core2"].layer) == (
        31,
        0,
    )
    assert generator._gds_layer_tuple(PDK.layer_stack.layers["heater"].layer) == (
        11,
        0,
    )


def test_cross_sections_include_metal_layers_without_duplicates() -> None:
    PDK.activate()
    cross_sections = generator._extract_cross_sections(PDK, PDK.layer_stack)
    by_name = {cross_section["name"]: cross_section for cross_section in cross_sections}

    assert by_name["heater_metal"]["layers"][0]["gds"] == 11
    assert by_name["metal_routing"]["layers"][0]["gds"] == 12
    for cross_section in cross_sections:
        layers = cross_section["layers"]
        identities = {
            (layer["gds"], layer["zmin"], layer["zmax"], layer["width"])
            for layer in layers
        }
        assert len(layers) == len(identities)


def test_background_layers_come_from_the_layer_stack() -> None:
    backgrounds = generator._extract_background_layers(PDK.layer_stack)
    by_name = {layer["name"]: layer for layer in backgrounds}

    assert (by_name["substrate"]["zmin"], by_name["substrate"]["zmax"]) == (
        -13.0,
        -3.0,
    )
    assert (by_name["box"]["zmin"], by_name["box"]["zmax"]) == (-3.0, 0.0)
    assert (by_name["clad"]["zmin"], by_name["clad"]["zmax"]) == (0.0, 1.8)
