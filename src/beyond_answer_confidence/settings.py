"""Paths and run options, from the command line or the environment.

Nothing here hard-codes a location inside the repository: the data and
output directories default to ``./data`` and ``./outputs`` relative to the
working directory and can be moved with ``BEYOND_ANSWER_CONFIDENCE_DATA_DIR`` and
``BEYOND_ANSWER_CONFIDENCE_OUTPUT_DIR``.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from beyond_answer_confidence.backends.jev import DEFAULT_MODEL


@dataclass(frozen=True)
class Settings:
    """Where things live and how a run behaves.

    Attributes:
        data_dir: Downloaded datasets, caches and indexes (never commit).
        output_dir: Scored rows and summaries.
        model: Model alias to query.
        live: Whether paid API calls are allowed; offline runs answer only
            from the cache.
        max_input_tokens: Budget of new input tokens for a live run.
        concurrency: Maximum simultaneous API calls.
    """

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    model: str = DEFAULT_MODEL
    live: bool = False
    max_input_tokens: int = 0
    concurrency: int = 8

    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        """Build settings from the environment, then apply overrides.

        Args:
            **overrides: Field values that take precedence (``None`` values
                are ignored).

        Returns:
            The settings.
        """
        base: dict[str, object] = {
            "data_dir": Path(
                os.environ.get("BEYOND_ANSWER_CONFIDENCE_DATA_DIR", "data")
            ),
            "output_dir": Path(
                os.environ.get("BEYOND_ANSWER_CONFIDENCE_OUTPUT_DIR", "outputs")
            ),
        }
        base |= {k: v for k, v in overrides.items() if v is not None}
        return cls(**base)  # type: ignore[arg-type]

    @property
    def cache_dir(self) -> Path:
        """Directory of the request caches."""
        return self.data_dir / "cache"

    @property
    def hf_dir(self) -> Path:
        """Hugging Face download cache."""
        return self.data_dir / "hf"

    def cache_file(self, experiment: str) -> Path:
        """Request cache of one experiment.

        Args:
            experiment: Experiment name.

        Returns:
            ``<data_dir>/cache/<experiment>.jsonl``.
        """
        return self.cache_dir / f"{experiment}.jsonl"

    def out(self, experiment: str) -> Path:
        """Output directory of one experiment.

        Args:
            experiment: Experiment name.

        Returns:
            ``<output_dir>/<experiment>``.
        """
        return self.output_dir / experiment
