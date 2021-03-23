def chop(x, y, ymax=None, ymin=None, xmin=None, xmax=None):
    """Chops x, y."""

    if xmax:
        y = y[x < xmax]
        x = x[x < xmax]

    if xmin:
        y = y[x > xmin]
        x = x[x > xmin]

    if ymax:
        x = x[y < ymax]
        y = y[y < ymax]

    if ymin:
        x = x[y > ymin]
        y = y[y > ymin]

    return x, y
