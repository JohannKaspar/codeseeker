"""T-070 runner: CLH end-to-end on MDACE test split against a local vLLM (CLH-small 8B).
Added by reproduction work; not part of upstream codeseeker.
- Qdrant via qdrant-client in-process LOCAL mode (no server/docker; box glibc too old for binary).
- Local-mode-compatible collection-existence check.
- All agent stages via throughster openai provider -> the local vLLM OpenAI endpoint.
- BYTE-LEVEL DETOK REPAIR: vllm-q36 leaks GPT-2 byte-BPE markers in free generation; reverse them
  client-side so the model can THINK freely (paper-faithful) and agents parse clean text.
- Eval restricted to the MDACE *test* split (paper Table 1 set).
Usage: python clh_run_8b.py          # smoke debug=True (10 test notes)
       python clh_run_8b.py --full   # full test split (115 notes)
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
            bs.append(b); cs.append(256 + n); n += 1
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

import dataloader
dataloader.DATASET_CONFIGS["mdace-icd10cm"]["split"] = "test"

import benchmark

FULL = "--full" in sys.argv
VLLM = {
    "provider": "openai",
    "api_base": "http://localhost:6539/v1",
    "endpoint": "chat/completions",
    "deployment": "Qwen/Qwen3.6-27B",
    "use_cache": False,
}
args = benchmark.Arguments(
    experiment_id="clh-repro",
    experiment_name=("clh-qwen27b-test-full" if FULL else "clh-qwen27b-smoke"),
    dataset="mdace-icd10cm",
    base_model=VLLM,
    max_tokens=16384,
    debug=(not FULL),
)
print("RUNNER args:", args.model_dump(exclude={"embed_config", "qdrant_config"}), flush=True)
benchmark.run(args)
