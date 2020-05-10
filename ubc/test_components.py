# this code has been automatically generated from pp/samples/pdk/write_tests.py
import ubc


def test_bend_circular(data_regression):
    c = ubc.bend_circular()
    data_regression.check(c.get_settings())


def test_y_splitter(data_regression):
    c = ubc.y_splitter()
    data_regression.check(c.get_settings())


def test_waveguide(data_regression):
    c = ubc.waveguide()
    data_regression.check(c.get_settings())


def test_mzi(data_regression):
    c = ubc.mzi()
    data_regression.check(c.get_settings())
