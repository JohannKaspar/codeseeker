"""T-094 runner: CLH end-to-end on ACI-BENCH (outpatient) against a local vLLM (CLH-base 70B).
Added by reproduction work; not part of upstream codeseeker. Mirrors clh_run_70b.py
(MDACE) with the ACI-BENCH dataset + FY2025 catalog (ACI gold is ICD-10-CM FY2024+).
- ACI-BENCH has note-level gold codes only (no evidence spans): the assign-stage,
  target-based code-set micro/macro-F1 + EMR is the reportable number.
- catalog_year=2025 so all 225 ACI gold codes resolve (FY2022 misses I1A.0 + F10.90).
- all_codes left at the default False -> retrieval constrained to the ACI test gold
  label space (the "restricted" setting), matching every MDACE run (T-070/T-071).
- Qdrant via qdrant-client in-process LOCAL mode (no server); set QDRANT_LOCAL_PATH to
  a persistent dir so the FY2025 index is embedded once and reused across models.
- BYTE-LEVEL DETOK REPAIR kept (no-op for DeepSeek/stock-vLLM clean text; needed for vllm-q36).
Usage: python clh_aci_run_70b.py          # smoke debug=True (10 test notes)
       python clh_aci_run_70b.py --full   # full ACI-BENCH test split (120 notes)
"""
import os
import sys

import qdrant_client as _qcli
import retrieval.qdrant_search.client as _qc
import retrieval.qdrant_search.factory as _qf
import throughster.open_ai.client as _oaic

_QPATH = os.environ.get("QDRANT_LOCAL_PATH", "")


def _local_init(host, port, grpc_port, **kw):
    return _qcli.QdrantClient(path=_QPATH) if _QPATH else _qcli.QdrantClient(location=":memory:")


def _collection_exists_local(client, collection_name):
    return bool(client.collection_exists(collection_name=collection_name))


_qc._init_client = _local_init
_qf._collection_exists = _collection_exists_local


# --- byte-level detok repair (reverse GPT-2 byte->unicode surface) ---
def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BDEC = {v: k for k, v in _bytes_to_unicode().items()}


def _repair(t):
    if not t:
        return t
    try:
        return bytearray(_BDEC[c] for c in t).decode("utf-8", "replace")
    except KeyError:
        return t  # clean text (real spaces/newlines not in table) -> leave as-is


_orig_unpack = _oaic.OpenAiInterface.unpack_call


def _patched_unpack(self, response):
    br = _orig_unpack(self, response)
    for ch in br.choices:
        ch.content = _repair(ch.content)
    return br


_oaic.OpenAiInterface.unpack_call = _patched_unpack

import benchmark  # noqa: E402

FULL = "--full" in sys.argv
VLLM = {
    "provider": "openai",
    "api_base": "http://localhost:6539/v1",
    "endpoint": "chat/completions",
    "deployment": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "use_cache": False,
}
args = benchmark.Arguments(
    experiment_id="clh-repro",
    experiment_name=("clh-base-70b-aci-test-full" if FULL else "clh-base-70b-aci-smoke"),
    dataset="aci-bench-icd10cm",
    catalog_year=2025,
    base_model=VLLM,
    max_tokens=16384,
    debug=(not FULL),
)
print("RUNNER args:", args.model_dump(exclude={"embed_config", "qdrant_config"}), flush=True)
benchmark.run(args)
