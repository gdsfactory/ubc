"""Test routing with layer_transitions and auto-taper."""

from ubcpdk import PDK
from ubcpdk.samples.sample_routing import sample_routing_different_widths


def test_sample_routing_different_widths() -> None:
    """Test that routing two straights with different widths works."""
    PDK.activate()
    c = sample_routing_different_widths()
    assert c.ports, "Component should have ports"
