from ubc.layers import port_type2layer


def add_pins(*args, **kwargs):
    from pp.add_pins import add_pins

    return add_pins(*args, **kwargs, port_type2layer=port_type2layer)
