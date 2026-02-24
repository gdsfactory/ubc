"""Shared test fixtures and custom difftest with visual XOR diff output."""

import filecmp
import pathlib
import shutil

import gdsfactory as gf
from gdsfactory.name import clean_name, get_name_short

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DIFF_DIR = PROJECT_ROOT / "test_diffs"

_config = {"update_gds_refs": False}


def pytest_addoption(parser):
    """Add --update-gds-refs option to pytest."""
    parser.addoption(
        "--update-gds-refs",
        action="store_true",
        default=False,
        help="Update GDS reference files instead of failing on mismatch.",
    )


def pytest_configure(config):
    """Read --update-gds-refs flag and create diff output directory."""

    DIFF_DIR.mkdir(exist_ok=True)
    _config["update_gds_refs"] = config.getoption("--update-gds-refs", default=False)


def _render_gds_to_png(gds_path: pathlib.Path, png_path: pathlib.Path) -> None:
    """Render a GDS file to a PNG image using klayout's headless LayoutView."""
    from klayout.lay import LayoutView  # noqa: PLC0415

    view = LayoutView()
    view.load_layout(str(gds_path))
    view.max_hier()
    view.zoom_fit()
    view.save_image(str(png_path), 1024, 1024)


def difftest(  # noqa: C901
    component: gf.Component,
    test_name: str | None = None,
    dirpath: pathlib.Path = pathlib.Path.cwd(),  # noqa: B008
    xor: bool = True,
    dirpath_run: pathlib.Path | None = None,
    ignore_sliver_differences: bool | None = None,
    sliver_tolerance: int = 1,
) -> None:
    """Custom difftest that saves XOR diff images on failure instead of prompting."""
    from gdsfactory.difftest import diff  # noqa: PLC0415

    if test_name is None:
        test_name = component.name

    filename = get_name_short(clean_name(test_name))

    dirpath_ref = pathlib.Path(dirpath)
    dirpath_ref.mkdir(exist_ok=True, parents=True)

    if dirpath_run is None:
        dirpath_run = dirpath_ref.parent / "gds_run"
    dirpath_run.mkdir(exist_ok=True, parents=True)

    ref_file = dirpath_ref / f"{filename}.gds"
    run_file = dirpath_run / f"{filename}.gds"

    component.write_gds(gdspath=run_file)

    if not ref_file.exists():
        shutil.copy(run_file, ref_file)
        raise AssertionError(
            f"Reference GDS file not found. Created new reference: {ref_file}\n"
            "Run the test again to confirm it passes."
        )

    if filecmp.cmp(ref_file, run_file, shallow=False):
        return

    # Files differ - generate XOR diff
    diff_gds = DIFF_DIR / f"{filename}_diff.gds"
    is_different = diff(
        ref_file=ref_file,
        run_file=run_file,
        xor=xor,
        test_name=test_name,
        ignore_sliver_differences=ignore_sliver_differences,
        out_file=diff_gds,
        sliver_tolerance=sliver_tolerance,
    )

    if not is_different:
        return

    if _config["update_gds_refs"]:
        shutil.copy(run_file, ref_file)
        print(f"  Updated reference: {ref_file}")
        return

    # Render diff GDS to PNG
    diff_png = DIFF_DIR / f"{filename}_diff.png"
    try:
        _render_gds_to_png(diff_gds, diff_png)
        image_msg = f"XOR diff image saved to: {diff_png}"
    except Exception as e:  # noqa: BLE001
        image_msg = f"Failed to render diff image: {e}"

    raise AssertionError(
        f"GDS mismatch for {test_name!r}.\n"
        f"  Reference: {ref_file}\n"
        f"  Run file:  {run_file}\n"
        f"  Diff GDS:  {diff_gds}\n"
        f"  {image_msg}\n"
        f"To update the reference, run: pytest --update-gds-refs"
    )
