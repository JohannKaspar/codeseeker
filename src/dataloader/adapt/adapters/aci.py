import typing as typ

import pydantic

from dataloader.adapt.base import Adapter, BaseModel
from dataloader.base import DatasetOptions


class AciCodeItem(pydantic.BaseModel):
    """One gold ICD-10-CM code on an ACI-BENCH note."""

    code: str
    description: str | None = None
    model_config = pydantic.ConfigDict(extra="ignore")


class AciBenchDataModel(pydantic.BaseModel):
    """A single ACI-BENCH coding record (note + note-level gold codes)."""

    encounter_id: str
    note: str
    icd10_codes: list[AciCodeItem]
    model_config = pydantic.ConfigDict(extra="ignore")

    @pydantic.field_validator("encounter_id", mode="before")
    @classmethod
    def _coerce_id(cls, value: str | int) -> str:
        return str(value)


class AciBenchAdapter(Adapter):
    """Adapter for the ACI-BENCH ICD-10-CM coding benchmark (Yuan & Shing 2025).

    ACI-BENCH has note-level gold diagnosis codes but no evidence spans, so
    ``evidence_spans`` is emitted empty; the span-grounded intermediate metrics
    are N/A and only the assign-stage code-set micro/macro-F1 + EMR over
    ``targets`` are meaningful.
    """

    input_model: typ.Type[AciBenchDataModel] = AciBenchDataModel
    output_model: typ.Type[BaseModel] = BaseModel

    @classmethod
    def translate_row(cls, row: dict[str, typ.Any], options: DatasetOptions) -> BaseModel:
        """Adapt a row."""
        struct_row = cls.input_model(**row)
        targets: list[str] = []
        for item in struct_row.icd10_codes:
            if item.code not in targets:
                targets.append(item.code)
        return cls.output_model(
            aid=struct_row.encounter_id,
            note=struct_row.note,
            note_type="aci-bench",
            evidence_spans=[],
            targets=targets,
        )
