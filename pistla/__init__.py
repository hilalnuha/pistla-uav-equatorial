"""PI-STLA: Physics-Informed Spatio-Temporal Link-Availability model for UAV links
in equatorial regions (rain + ionospheric scintillation + radio interference)."""
from . import propagation, scintillation, interference, stgp, climatology  # noqa: F401

__version__ = "1.0.0"
