from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from codecarbon import OfflineEmissionsTracker


@dataclass(frozen=True)
class CarbonSummary:
    emissions_kg_co2e: float | None
    duration_seconds: float | None
    energy_consumed_kwh: float | None
    cpu_energy_kwh: float | None
    gpu_energy_kwh: float | None
    ram_energy_kwh: float | None
    country_iso_code: str
    tracking_mode: str
    measure_power_secs: int
    codecarbon_output_file: str


def collect_platform_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
    }


def collect_nvidia_metadata() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi not found on PATH"}

    command = [
        executable,
        "--query-gpu=index,name,uuid,driver_version,memory.total,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, uuid, driver_version, memory_total_mb, power_limit_w = parts
        rows.append(
            {
                "index": index,
                "name": name,
                "uuid": uuid,
                "driver_version": driver_version,
                "memory_total_mb": memory_total_mb,
                "power_limit_w": power_limit_w,
            }
        )

    return {"available": True, "gpus": rows}


class CodeCarbonRunTracker:
    """Machine-level CodeCarbon tracker for an Ollama run.

    Machine tracking is intentional because Ollama inference executes in a separate
    server process rather than inside the Python client process.
    """

    def __init__(
        self,
        *,
        run_dir: str | Path,
        run_id: str,
        country_iso_code: str,
        measure_power_secs: int = 1,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.country_iso_code = country_iso_code.upper()
        self.measure_power_secs = measure_power_secs
        self.output_file = "codecarbon_emissions.csv"
        self.tracker = OfflineEmissionsTracker(
            project_name="reporting-validation-standards",
            country_iso_code=self.country_iso_code,
            output_dir=str(self.run_dir),
            output_file=self.output_file,
            save_to_file=True,
            save_to_api=False,
            log_level="error",
            tracking_mode="machine",
            measure_power_secs=self.measure_power_secs,
        )

    def start(self) -> None:
        self.tracker.start()

    def stop(self) -> CarbonSummary:
        emissions = self.tracker.stop()
        csv_path = self.run_dir / self.output_file

        row: dict[str, str] = {}
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                row = rows[-1]

        def number(name: str) -> float | None:
            value = row.get(name)
            if value is None or value == "":
                return None
            try:
                return float(value)
            except ValueError:
                return None

        summary = CarbonSummary(
            emissions_kg_co2e=float(emissions) if emissions is not None else number("emissions"),
            duration_seconds=number("duration"),
            energy_consumed_kwh=number("energy_consumed"),
            cpu_energy_kwh=number("cpu_energy"),
            gpu_energy_kwh=number("gpu_energy"),
            ram_energy_kwh=number("ram_energy"),
            country_iso_code=self.country_iso_code,
            tracking_mode="machine",
            measure_power_secs=self.measure_power_secs,
            codecarbon_output_file=self.output_file,
        )

        (self.run_dir / "carbon_summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
