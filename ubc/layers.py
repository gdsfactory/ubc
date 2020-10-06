from dataclasses import dataclass


@dataclass
class Layer:
    WG = (1, 0)
    DEVREC = (68, 0)
    LABEL = (10, 0)
    PORT = (1, 10)
    FLOORPLAN = (99, 0)


LAYER = Layer()
port_layer2type = {LAYER.PORT: "optical"}
port_type2layer = {v: k for k, v in port_layer2type.items()}


if __name__ == "__main__":
    print(LAYER.WG)
