"""Symlink tech to klayout."""

import os
import pathlib
import sys
import subprocess

klayout_folder = "KLayout" if sys.platform == "win32" else ".klayout"
cwd = pathlib.Path(__file__).resolve().parent
home = pathlib.Path.home()
src = cwd / "ubcpdk" / "klayout" / "tech"
dest_folder = home / klayout_folder / "tech"
dest_folder.mkdir(exist_ok=True, parents=True)
dest = dest_folder / "ubcpdk"


def make_link(src, dest):
    if sys.platform == "win32":
        subprocess.check_call(f"mklink /J {dest} {src}", shell=True)
    else:
        os.symlink(src, dest)


def install_tech(src, dest):
    """Installs tech."""
    if dest.exists():
        print(f"tech already installed in {dest}")
        return

    try:
        make_link(src, dest)
    except Exception:
        os.remove(dest)
        make_link(src, dest)

    print(f"layermap installed to {dest}")


def install_drc(src, dest):
    """Installs drc."""
    if dest.exists():
        print("drc already installed")
        return

    dest_folder = dest.parent
    dest_folder.mkdir(exist_ok=True, parents=True)
    try:
        make_link(src, dest)
    except Exception:
        os.remove(dest)
        make_link(src, dest)

    print(f"layermap installed to {dest}")


if __name__ == "__main__":

    install_tech(src=src, dest=dest)

    # src = cwd / "klayout" / "tech" / "drc.lydrc"
    # dest = home / klayout_folder / "drc" / "drc.lydrc"
    # install_drc(src, dest)
