# this code has been automatically generated from pp/samples/pdk/write_tests.py
import ubc


def test_bend_circular(num_regression):
    c = ubc.bend_circular()
    num_regression.check(c.get_ports_array())


def test_y_splitter(num_regression):
    c = ubc.y_splitter()
    num_regression.check(c.get_ports_array())


def test_waveguide(num_regression):
    c = ubc.waveguide()
    num_regression.check(c.get_ports_array())


def test_mzi(num_regression):
    c = ubc.mzi()
    num_regression.check(c.get_ports_array())
