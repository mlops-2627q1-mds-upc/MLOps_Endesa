import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import tracking


class FakeEmissionsTracker:
    """Stands in for CodeCarbon so tests never probe hardware or measure energy."""

    def __init__(self, output_dir, country_iso_code):
        self.output_dir = Path(output_dir)
        self.country_iso_code = country_iso_code
        self.started = self.stopped = False

    def start(self):
        # CodeCarbon sets its logger to INFO and reports its methods there.
        logger = logging.getLogger("codecarbon")
        logger.setLevel(logging.INFO)
        logger.info("CPU Tracking Method: TDP constant")
        logger.info("RAM Tracking Method: RAM power estimation model")
        self.started = True

    def stop(self):
        self.stopped = True
        (self.output_dir / "emissions.csv").write_text("energy_consumed\n0.002\n")
        self.final_emissions_data = SimpleNamespace(
            energy_consumed=0.002,
            cpu_energy=0.0015,
            gpu_energy=0.0,
            ram_energy=0.0005,
            emissions=0.0003,
            duration=1.5,
            country_iso_code=self.country_iso_code,
            tracking_mode="process",
            codecarbon_version="fake",
            cpu_model="Fake CPU",
            gpu_model="",
        )
        return self.final_emissions_data.emissions


@pytest.fixture(autouse=True)
def fake_codecarbon(monkeypatch):
    """Replace CodeCarbon in every test; returns the trackers created."""
    created = []

    def factory(output_dir, country_iso_code):
        tracker = FakeEmissionsTracker(output_dir, country_iso_code)
        created.append(tracker)
        return tracker

    monkeypatch.setattr(tracking, "_emissions_tracker", factory)
    return created
