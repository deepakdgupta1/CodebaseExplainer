import psutil
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class Profiler:
    def __init__(self) -> None:
        self.metrics: Dict[str, Any] = {
            'phases': {},
            'total_time': 0,
            'peak_memory_mb': 0
        }
        self.start_time = time.time()

    def start_phase(self, phase_name: str) -> None:
        self.metrics['phases'][phase_name] = {
            'start_time': time.time(),
            'start_memory_mb': self._get_memory_mb()
        }
        logging.info(f"Starting phase: {phase_name}")

    def end_phase(
        self,
        phase_name: str,
        extra_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        if phase_name not in self.metrics['phases']:
            return

        phase = self.metrics['phases'][phase_name]
        end_time = time.time()
        end_memory = self._get_memory_mb()

        phase['duration'] = end_time - phase['start_time']
        phase['end_memory_mb'] = end_memory
        phase['memory_delta_mb'] = end_memory - phase['start_memory_mb']

        if extra_metrics:
            phase.update(extra_metrics)

        logging.info(
            f"Completed phase: {phase_name} in "
            f"{phase['duration']:.2f}s"
        )

        # Update peak
        self.metrics['peak_memory_mb'] = max(
            self.metrics['peak_memory_mb'],
            end_memory
        )

    def save_metrics(self, path: Path) -> None:
        self.metrics['total_time'] = time.time() - self.start_time
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def _get_memory_mb(self) -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
