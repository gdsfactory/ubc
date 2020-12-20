import ubc
from lytest import contained_phidlDevice, difftest_it
from ubc.waveguide import waveguide as waveguide_function
from ubc.write_test_gds import container_factory


@contained_phidlDevice
def bend90(top):
    top.add_ref(ubc.bend90())


def test_gds_bend90():
    difftest_it(bend90)()


@contained_phidlDevice
def crossing_te(top):
    top.add_ref(ubc.crossing_te())


def test_gds_crossing_te():
    difftest_it(crossing_te)()


@contained_phidlDevice
def crossing_te_ring(top):
    top.add_ref(ubc.crossing_te_ring())


def test_gds_crossing_te_ring():
    difftest_it(crossing_te_ring)()


@contained_phidlDevice
def dbr_te(top):
    top.add_ref(ubc.dbr_te())


def test_gds_dbr_te():
    difftest_it(dbr_te)()


@contained_phidlDevice
def dcate(top):
    top.add_ref(ubc.dcate())


def test_gds_dcate():
    difftest_it(dcate)()


@contained_phidlDevice
def dcbte(top):
    top.add_ref(ubc.dcbte())


def test_gds_dcbte():
    difftest_it(dcbte)()


@contained_phidlDevice
def gc_te1310(top):
    top.add_ref(ubc.gc_te1310())


def test_gds_gc_te1310():
    difftest_it(gc_te1310)()


@contained_phidlDevice
def gc_te1550(top):
    top.add_ref(ubc.gc_te1550())


def test_gds_gc_te1550():
    difftest_it(gc_te1550)()


@contained_phidlDevice
def gc_te1550_broadband(top):
    top.add_ref(ubc.gc_te1550_broadband())


def test_gds_gc_te1550_broadband():
    difftest_it(gc_te1550_broadband)()


@contained_phidlDevice
def gc_tm1550(top):
    top.add_ref(ubc.gc_tm1550())


def test_gds_gc_tm1550():
    difftest_it(gc_tm1550)()


@contained_phidlDevice
def mzi(top):
    top.add_ref(ubc.mzi())


def test_gds_mzi():
    difftest_it(mzi)()


@contained_phidlDevice
def mzi_te(top):
    top.add_ref(ubc.mzi_te())


def test_gds_mzi_te():
    difftest_it(mzi_te)()


@contained_phidlDevice
def ring(top):
    top.add_ref(ubc.ring())


def test_gds_ring():
    difftest_it(ring)()


@contained_phidlDevice
def ring_single_te(top):
    top.add_ref(ubc.ring_single_te())


def test_gds_ring_single_te():
    difftest_it(ring_single_te)()


@contained_phidlDevice
def taper_factory(top):
    top.add_ref(ubc.taper_factory())


def test_gds_taper_factory():
    difftest_it(taper_factory)()


@contained_phidlDevice
def waveguide(top):
    top.add_ref(ubc.waveguide())


def test_gds_waveguide():
    difftest_it(waveguide)()


@contained_phidlDevice
def y_adiabatic(top):
    top.add_ref(ubc.y_adiabatic())


def test_gds_y_adiabatic():
    difftest_it(y_adiabatic)()


@contained_phidlDevice
def y_splitter(top):
    top.add_ref(ubc.y_splitter())


def test_gds_y_splitter():
    difftest_it(y_splitter)()


@contained_phidlDevice
def add_gc(top):
    component = waveguide_function()
    container_function = container_factory["add_gc"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_gc():
    difftest_it(add_gc)()
