"""ACI-BENCH ICD-10-CM loader (T-094, not part of upstream codeseeker).

Reads the agentic-icd-materialized ACI-BENCH coding splits (Yuan & Shing 2025
ICD-10-CM annotations joined onto the public ACI-BENCH SOAP notes) into a flat
``datasets.Dataset`` for the codeseeker pipeline. The note text (not the
dialogue) is the coding input, mirroring how MDACE codes from the discharge
note and how Symphony reports on ACI-BENCH.

Each source jsonl record is ``{encounter_id, dataset, dialogue, note,
icd10_codes}`` where ``icd10_codes`` is a list of ``{code, description}``.
There are no evidence spans (ACI-BENCH carries note-level diagnosis codes only),
so the adapter emits empty ``evidence_spans`` and only the assign-stage,
target-based code-set metrics are meaningful.

The split directory defaults to the dobby path but can be overridden with the
``ACI_BENCH_CODING_DIR`` environment variable.
"""

import json
import os
import pathlib

import datasets

DEFAULT_CODING_DIR = (
    "/links/groups/mmoor/jlieberwirth/agentic-icd/data/preprocessed/aci-bench-coding"
)


def _coding_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("ACI_BENCH_CODING_DIR", DEFAULT_CODING_DIR))


def load_aci_bench(
    subset: str | None = None, split: str | None = None, **kws
) -> datasets.Dataset:
    """Load one ACI-BENCH coding split as a flat ``datasets.Dataset``.

    Conforms to the ``DatasetLoader`` protocol so it can be used directly as a
    ``DatasetConfig.name_or_path``.
    """
    split = split or "test"
    path = _coding_dir() / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"ACI-BENCH coding split not found: {path} "
            "(set ACI_BENCH_CODING_DIR or materialize the split)."
        )

    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            codes = [
                {
                    "code": str(c["code"]),
                    "description": str(c.get("description", "")),
                }
                for c in rec.get("icd10_codes", [])
                if c.get("code")
            ]
            rows.append(
                {
                    "encounter_id": str(rec.get("encounter_id", "")),
                    "note": rec["note"],
                    "icd10_codes": codes,
                }
            )

    if not rows:
        raise ValueError(f"No ACI-BENCH records loaded from {path}.")
    return datasets.Dataset.from_list(rows)
