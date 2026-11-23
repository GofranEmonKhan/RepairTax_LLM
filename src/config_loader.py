"""
Loads config.yaml into a plain dict and validates the fields that must be
set before a run is allowed to start.
"""

import yaml


class ConfigError(RuntimeError):
    pass


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Spec §11: never silently run with a guessed carbon-intensity value.
    carbon = config.get("carbon", {})
    if carbon.get("intensity_g_per_kwh") is None or not carbon.get("intensity_source"):
        raise ConfigError(
            "carbon.intensity_g_per_kwh and carbon.intensity_source must be set "
            "in config.yaml before running (spec §11) — pick and cite one documented "
            "source for the Bangladesh grid; do not guess a value."
        )

    # Spec §4: max_attempts must be a positive integer.
    if config["experiment"]["max_attempts"] < 1:
        raise ConfigError("experiment.max_attempts must be >= 1.")

    return config
