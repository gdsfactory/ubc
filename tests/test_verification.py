"""Test functional verification of simple layouts """
import os

import gdsfactory as gf

import ubcpdk.components as uc
from ubcpdk.verification import layout_check


def test_verification_import_gds():
    file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "tests/mmi2x2.oas"
    )
    c = gf.import_gds(file_path)
    layout_check(c)


def test_verification_mzi():
    splitter = uc.ebeam_y_1550(decorator=gf.port.auto_rename_ports)
    mzi = gf.components.mzi(splitter=splitter)
    c = uc.add_fiber_array(component=mzi)
    layout_check(c)
