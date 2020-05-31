# this code has been automatically generated from ubc/write_tests.py
import ubc


def test_gc_te1550(num_regression):
    c = ubc.gc_te1550()
    num_regression.check(c.get_ports_array())


def test_y_splitter(num_regression):
    c = ubc.y_splitter()
    num_regression.check(c.get_ports_array())


def test_mzi(num_regression):
    c = ubc.mzi()
    num_regression.check(c.get_ports_array())


def test_waveguide(num_regression):
    c = ubc.waveguide()
    num_regression.check(c.get_ports_array())


def test_bend90(num_regression):
    c = ubc.bend90()
    num_regression.check(c.get_ports_array())
