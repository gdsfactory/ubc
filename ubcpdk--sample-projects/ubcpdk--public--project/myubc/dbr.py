import gdsfactory as gf


@gf.cell
def dbr(w1=0.5, w2=0.55, cross_section="strip") -> gf.Component:
    return gf.c.dbr(w1=w1, w2=w2, cross_section=cross_section)
