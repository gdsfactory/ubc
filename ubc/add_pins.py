from pp.add_pins import _add_pins, _add_pins_labels_and_outline
from ubc.layers import port_type2layer


def _add_pins_ubc(**kwargs):
    return _add_pins(port_type2layer=port_type2layer, **kwargs)


def _add_pins_labels_and_outline_ubc(**kwargs):
    return _add_pins_labels_and_outline(add_pins_function=_add_pins_ubc, **kwargs)


def add_pins(*args, **kwargs):
    from pp.add_pins import add_pins

    return add_pins(*args, **kwargs, function=_add_pins_labels_and_outline_ubc)


if __name__ == "__main__":
    import pp

    c = pp.c.waveguide()
    cc = add_pins(c)
    cc.show()
