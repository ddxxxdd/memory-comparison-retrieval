"""Frozen BGE and SparseCL encoders used in the comparisons."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from .data import sha256_file


MODEL_LOCK = json.loads(Path(__file__).with_name("model_lock.json").read_text(encoding="utf-8"))
MODEL_ROOT = Path(__file__).resolve().parents[2] / "baselines/SparseCL"
BATCH_SIZE = 4
CPU_THREADS = 4


def git_blob_sha1(path):
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(base, entries):
    verified = {}
    for directory, entry in entries:
        for item in entry['files']:
            path = base / directory / item['rfilename']
            if not path.is_file() or path.stat().st_size != item['size']:
                raise ValueError(f'Missing or incomplete model file: {path}')
            expected = (item.get('lfs') or {}).get('sha256')
            actual = sha256_file(path) if expected else git_blob_sha1(path)
            if actual != (expected or item['blobId']):
                raise ValueError(f'Hash mismatch: {path}')
            verified[path.relative_to(base).as_posix()] = actual if expected else sha256_file(path)
    return verified


def token_audit(tokenizer, texts, max_length):
    ids = tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']
    lengths = [len(row) + tokenizer.num_special_tokens_to_add() for row in ids]
    total = sum(map(len, ids))
    unknown = sum(row.count(tokenizer.unk_token_id) for row in ids)
    return {'texts': len(texts), 'tokens_excluding_special': total, 'unknown_tokens': unknown,
            'unknown_ratio': unknown / total if total else 0.0, 'max_tokens_with_special': max(lengths),
            'would_truncate': sum(n > max_length for n in lengths)}


def encode(texts, tokenizer, model, batch_size, max_length, sparse, label):
    import numpy as np
    import torch
    output = []
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[start:start + batch_size], max_length=max_length,
                               padding=True, truncation=True, return_tensors='pt')
            if sparse:
                result = model(**inputs, return_dict=True, sent_emb=True)
                vector = result.pooler_output
            else:
                result = model(**inputs, return_dict=True)
                vector = result.last_hidden_state[:, 0]
            output.append(vector.float().cpu())
            if (start + batch_size) % 80 == 0:
                print(f'{label}: {min(start + batch_size, len(texts))}/{len(texts)}', flush=True)
    vectors = torch.cat(output)
    norms = torch.norm(vectors, dim=1, keepdim=True)
    assert bool((norms > 0).all())
    vectors = (vectors / norms).numpy().astype(np.float32, copy=False)
    assert vectors.shape == (len(texts), 768) and np.isfinite(vectors).all()
    return vectors, time.perf_counter() - started


def encode_groups(name, groups, base):
    """Encode each group with the same batch boundaries as the frozen comparisons."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    spec = MODEL_LOCK[name]
    checkpoint = base / spec["directory"]
    hashes = verify_assets(base, [(spec["directory"], spec)])
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    sparse = name != "bge"
    if sparse:
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        from sparsecl.models import our_BertForCL
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        options = SimpleNamespace(pooler_type="avg", temp=0.05, hard_negative_weight=0.0,
                                  do_mlm=False, mlm_weight=0.1, mlp_only_train=False)
        model = our_BertForCL.from_pretrained(checkpoint, config=config, model_args=options, local_files_only=True)
    else:
        model = AutoModel.from_pretrained(checkpoint, local_files_only=True)
    vectors, audit, seconds = {}, {}, {}
    length = spec["max_seq_length"]
    for group, texts in groups.items():
        audit[group] = token_audit(tokenizer, texts, length)
        vectors[group], seconds[group] = encode(texts, tokenizer, model, BATCH_SIZE, length, sparse, f"{name} {group}")
    identity = {"model": spec["repo"], "revision": spec["revision"], "model_files_sha256": hashes,
                "pooling": "avg" if sparse else "CLS", "normalize": "L2", "max_length": length,
                "batch_size": BATCH_SIZE, "cpu_threads": CPU_THREADS, "seed": 0, "device": "cpu",
                "precision": "float32", "token_audit": audit}
    return vectors, identity, seconds
