import gdsfactory as gf


@gf.cell
def splitter_tree(
    noutputs=2**3, cross_section="strip", spacing=(120, 50), **kwargs
) -> gf.Component:
    c = gf.c.splitter_tree(
        noutputs=noutputs, cross_section=cross_section, spacing=spacing, **kwargs
    )
    return c
