from dataclasses import dataclass


@dataclass
class Layer:
    WG = (1, 0)
    DEVREC = (68, 0)
    LABEL = (10, 0)
    PORT = (1, 10)
    FLOORPLAN = (99, 0)


LAYER = Layer()


if __name__ == "__main__":
    print(LAYER.WG)
