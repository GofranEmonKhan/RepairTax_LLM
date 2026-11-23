"""
Carbon accounting (spec §11): carbon_gco2e = energy_kwh * intensity_g_per_kwh.
The intensity value itself is validated (non-null) by config_loader before
a run is allowed to start, so this module can assume it's always set.
"""


def compute_carbon_gco2e(energy_kwh: float, intensity_g_per_kwh: float) -> float:
    return energy_kwh * intensity_g_per_kwh
