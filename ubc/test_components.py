# this code has been automatically generated from ubc/write_tests.py
import ubc


def test_y_splitter(data_regression):
    c = ubc.y_splitter()
    data_regression.check(c.get_settings())


def test_gc_te1550(data_regression):
    c = ubc.gc_te1550()
    data_regression.check(c.get_settings())


def test_bend90(data_regression):
    c = ubc.bend90()
    data_regression.check(c.get_settings())


def test_mzi(data_regression):
    c = ubc.mzi()
    data_regression.check(c.get_settings())


def test_mzi_te(data_regression):
    c = ubc.mzi_te()
    data_regression.check(c.get_settings())


def test_ring_single_te(data_regression):
    c = ubc.ring_single_te()
    data_regression.check(c.get_settings())


def test_waveguide(data_regression):
    c = ubc.waveguide()
    data_regression.check(c.get_settings())
