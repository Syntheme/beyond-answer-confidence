"""Loaders for standard multiple-choice sets: MMLU-Redux 2.0, MMLU-CF, ANLI.

Each source is pinned to a commit; downloads go to ``<data_dir>/hf`` and
hold dataset text (never commit them). The subject lists are fixed here so
that loading needs no repository listing (and works offline once cached).
"""

from pathlib import Path

import pandas as pd

# Public Hugging Face commit hashes (not secrets).
REVISIONS = {
    "edinburgh-dawg/mmlu-redux-2.0": "372ea425445d51e1ba1188c56e5e893f8138621f",  # pragma: allowlist secret
    "microsoft/MMLU-CF": "c25b89a968a2062e422dd96ddf0fe3387507f0dd",  # pragma: allowlist secret
    "facebook/anli": "8e4813d81f46d313dac7892e1c28076917cfcdf9",  # pragma: allowlist secret
}
"""Dataset repo -> pinned commit."""

MMLU_REDUX_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)
"""The 57 MMLU-Redux 2.0 subjects, sorted."""

MMLU_CF_SUBJECTS = (
    "Biology",
    "Business",
    "Chemistry",
    "Computer_Science",
    "Economics",
    "Engineering",
    "Health",
    "History",
    "Law",
    "Math",
    "Other",
    "Philosophy",
    "Physics",
    "Psychology",
)
"""The 14 MMLU-CF subjects."""

ANLI_ROUNDS = (1, 2, 3)


def pinned_file(repo: str, filename: str, data_dir: Path) -> Path:
    """Download one file of a repo pinned in :data:`REVISIONS` (cached).

    Args:
        repo: Dataset repo id.
        filename: Path inside the repo.
        data_dir: Data directory; files are cached under ``data_dir/hf``.

    Returns:
        Local path.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo,
            filename,
            repo_type="dataset",
            revision=REVISIONS[repo],
            cache_dir=str(data_dir / "hf"),
        )
    )


def load_mmlu_redux(data_dir: Path) -> pd.DataFrame:
    """All MMLU-Redux 2.0 subjects (Arrow files), in subject order.

    Args:
        data_dir: Data directory.

    Returns:
        One row per question with a ``subject`` column (index 0..n-1).
    """
    from datasets import Dataset

    repo = "edinburgh-dawg/mmlu-redux-2.0"
    frames = []
    for s in MMLU_REDUX_SUBJECTS:
        path = pinned_file(repo, f"{s}/data-00000-of-00001.arrow", data_dir)
        df = Dataset.from_file(str(path)).to_pandas()
        if not isinstance(df, pd.DataFrame):  # pragma: no cover - iterator form
            df = pd.concat(list(df))
        df["subject"] = s
        frames.append(df)
    out: pd.DataFrame = pd.concat(frames, ignore_index=True)
    return out


def load_mmlu_cf(data_dir: Path) -> pd.DataFrame:
    """MMLU-CF validation questions.

    Args:
        data_dir: Data directory.

    Returns:
        One row per question with a ``subject`` column (index 0..n-1).
    """
    frames = []
    for s in MMLU_CF_SUBJECTS:
        df = pd.read_parquet(
            pinned_file("microsoft/MMLU-CF", f"val/{s}_val.parquet", data_dir)
        )
        df["subject"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_anli(data_dir: Path) -> pd.DataFrame:
    """ANLI test rounds 1-3.

    Args:
        data_dir: Data directory.

    Returns:
        One row per item with a ``round`` column.
    """
    frames = []
    for r in ANLI_ROUNDS:
        df = pd.read_parquet(
            pinned_file(
                "facebook/anli",
                f"plain_text/test_r{r}-00000-of-00001.parquet",
                data_dir,
            )
        )
        df["round"] = r
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
