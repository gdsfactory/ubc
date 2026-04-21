"""Test routing with layer_transitions and auto-taper."""

from ubcpdk import PDK
from ubcpdk.samples.sample_routing import sample_routing_different_widths


def test_sample_routing_different_widths() -> None:
    """Test that routing inserts tapers between straights of different widths."""
    PDK.activate()
    c = sample_routing_different_widths()
    assert len(c.insts) > 2, "Routing should insert taper/bend instances"
