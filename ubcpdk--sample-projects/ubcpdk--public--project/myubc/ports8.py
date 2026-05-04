import gdsfactory as gf


@gf.cell
def ports8() -> gf.Component:
    c = gf.Component()
    c.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=(1, 0))
    c.add_port("o1", center=(0, 3), layer=(1, 0), width=1, orientation=180)
    c.add_port("o2", center=(0, 7), layer=(1, 0), width=1, orientation=180)
    c.add_port("o3", center=(3, 10), layer=(1, 0), width=1, orientation=90)
    c.add_port("o4", center=(7, 10), layer=(1, 0), width=1, orientation=90)
    c.add_port("o5", center=(10, 7), layer=(1, 0), width=1, orientation=0)
    c.add_port("o6", center=(10, 3), layer=(1, 0), width=1, orientation=0)
    c.add_port("o7", center=(7, 0), layer=(1, 0), width=1, orientation=270)
    c.add_port("o8", center=(3, 0), layer=(1, 0), width=1, orientation=270)
    return c
