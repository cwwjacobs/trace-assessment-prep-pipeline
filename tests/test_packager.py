#!/usr/bin/env python3
"""
Unit tests for GTDataworks Parquet & Loss-Masked Packager
=========================================================
Tests:
  1. Loss-masked conversions (OpenAI, ShareGPT, Anthropic formats)
  2. Typed Arrow & Parquet dataset creation & schema validation
  3. Compression codecs (zstd, gzip, uncompressed)
  4. Content-Addressable Storage (CAS) SHA-256 manifests and verification
  5. Multi-split datasets and partition sharding
  6. Tamper detection and integrity verification
"""

import gzip
import json
from pathlib import Path
import pytest

from goldtrace_mint.factory.loss_masking import MaskedMessage
from goldtrace_mint.factory.packager import (
    CASDatasetManifest,
    CompressionCodec,
    DatasetFormat,
    DatasetPackager,
    LossMaskingFormat,
    PartitionManifestEntry,
    build_arrow_table_from_records,
    format_for_anthropic,
    format_for_openai_chat,
    format_for_sharegpt,
    get_anthropic_arrow_schema,
    get_openai_chat_arrow_schema,
    get_sharegpt_arrow_schema,
    normalize_record_for_format,
    read_partition,
    verify_cas_dataset,
    write_arrow_dataset,
    write_loss_masked_jsonl,
    write_parquet_dataset,
    HAS_PYARROW,
    HAS_ZSTD,
)

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_events():
    """Sample Labyrinth agent trace events."""
    return [
        {
            "type": "system.prompt",
            "payload": {"instruction": "You are an expert autonomous security auditor."},
        },
        {
            "type": "user.input",
            "payload": {"prompt": "Analyze the repository for SSRF vulnerabilities."},
        },
        {
            "type": "model.response",
            "payload": {
                "reasoning": "I need to search for network requests handling user-controlled URLs.",
                "content": "Searching for HTTP request handlers...",
                "tool_calls": [
                    {
                        "id": "call_audit_01",
                        "type": "function",
                        "function": {
                            "name": "grep_search",
                            "arguments": json.dumps({"query": "requests.get("}),
                        },
                    }
                ],
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "output": "src/client.py:42: resp = requests.get(user_url)"
            },
        },
        {
            "type": "model.response",
            "payload": {
                "reasoning": "Found unsanitized user_url passed to requests.get at src/client.py line 42.",
                "content": "A high-severity SSRF was identified in `src/client.py:42` where `user_url` is fetched without validation.",
            },
        },
    ]


@pytest.fixture
def sample_masked_messages():
    """Sample pre-masked messages."""
    return [
        MaskedMessage(role="system", content="System instruction", loss_weight=0.0),
        MaskedMessage(role="user", content="User prompt", loss_weight=0.0),
        MaskedMessage(
            role="assistant",
            content="Assistant response",
            reasoning_content="Assistant step-by-step reasoning",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{}"}}],
            loss_weight=1.0,
        ),
        MaskedMessage(role="tool", content="Tool result output", loss_weight=0.0),
        MaskedMessage(role="assistant", content="Final answer", loss_weight=1.0),
    ]


# ---------------------------------------------------------------------------
# 1. Loss-Masked Conversions Tests
# ---------------------------------------------------------------------------

def test_format_for_openai_chat(sample_raw_events):
    formatted = format_for_openai_chat(sample_raw_events)
    assert "messages" in formatted
    msgs = formatted["messages"]
    assert len(msgs) == 5

    # System: loss_weight == 0.0
    assert msgs[0]["role"] == "system"
    assert msgs[0]["loss_weight"] == 0.0

    # User: loss_weight == 0.0
    assert msgs[1]["role"] == "user"
    assert msgs[1]["loss_weight"] == 0.0

    # Assistant Turn 1: loss_weight == 1.0, reasoning and tool_calls preserved
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["loss_weight"] == 1.0
    assert "reasoning_content" in msgs[2]
    assert msgs[2]["reasoning_content"] == "I need to search for network requests handling user-controlled URLs."
    assert len(msgs[2]["tool_calls"]) == 1
    assert msgs[2]["tool_calls"][0]["id"] == "call_audit_01"

    # Tool execution output: loss_weight == 0.0 (masked!)
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["loss_weight"] == 0.0
    assert "requests.get" in msgs[3]["content"]

    # Assistant Final Turn: loss_weight == 1.0
    assert msgs[4]["role"] == "assistant"
    assert msgs[4]["loss_weight"] == 1.0


def test_format_for_openai_chat_custom_weight_key(sample_raw_events):
    formatted = format_for_openai_chat(sample_raw_events, weight_key="weight")
    msgs = formatted["messages"]
    for m in msgs:
        assert "weight" in m
        assert "loss_weight" not in m
    assert msgs[0]["weight"] == 0.0
    assert msgs[2]["weight"] == 1.0


def test_format_for_sharegpt(sample_raw_events):
    formatted = format_for_sharegpt(sample_raw_events)
    assert "conversations" in formatted
    convs = formatted["conversations"]
    assert len(convs) == 5

    assert convs[0]["from"] == "system"
    assert convs[0]["loss_weight"] == 0.0

    assert convs[1]["from"] == "human"
    assert convs[1]["loss_weight"] == 0.0

    assert convs[2]["from"] == "gpt"
    assert convs[2]["loss_weight"] == 1.0
    assert "<thought>" in convs[2]["value"]
    assert "<tool_call>" in convs[2]["value"]

    assert convs[3]["from"] == "tool"
    assert convs[3]["loss_weight"] == 0.0

    assert convs[4]["from"] == "gpt"
    assert convs[4]["loss_weight"] == 1.0


def test_format_for_anthropic(sample_raw_events):
    formatted = format_for_anthropic(sample_raw_events)
    assert "system" in formatted
    assert "messages" in formatted
    assert "autonomous security auditor" in formatted["system"]

    msgs = formatted["messages"]
    # User turn
    assert msgs[0]["role"] == "user"
    assert msgs[0]["loss_weight"] == 0.0
    assert msgs[0]["content"][0]["type"] == "text"

    # Assistant turn with thinking and tool_use blocks
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["loss_weight"] == 1.0
    blocks = msgs[1]["content"]
    types = [b["type"] for b in blocks]
    assert "thinking" in types
    assert "tool_use" in types
    assert "text" in types

    # Tool result block in user turn
    assert msgs[2]["role"] == "user"
    assert msgs[2]["loss_weight"] == 0.0
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["loss_weight"] == 0.0

    # Final assistant turn (thinking and text blocks)
    assert msgs[3]["role"] == "assistant"
    assert msgs[3]["loss_weight"] == 1.0
    final_types = [b["type"] for b in msgs[3]["content"]]
    assert "thinking" in final_types
    assert "text" in final_types


# ---------------------------------------------------------------------------
# 2. JSONL Packaging & Compression Codecs
# ---------------------------------------------------------------------------

def test_write_loss_masked_jsonl_uncompressed(tmp_path, sample_raw_events):
    out_file = tmp_path / "train.jsonl"
    entry = write_loss_masked_jsonl(
        records=[sample_raw_events, sample_raw_events],
        output_path=out_file,
        loss_format=LossMaskingFormat.OPENAI,
        compression=CompressionCodec.UNCOMPRESSED,
        split="train",
    )

    assert out_file.exists()
    assert entry.record_count == 2
    assert entry.compression == "uncompressed"
    assert len(entry.sha256) == 64
    assert entry.byte_size == out_file.stat().st_size

    rows = read_partition(out_file)
    assert len(rows) == 2
    assert len(rows[0]["messages"]) == 5


def test_write_loss_masked_jsonl_gzip(tmp_path, sample_raw_events):
    out_file = tmp_path / "train.jsonl"
    entry = write_loss_masked_jsonl(
        records=[sample_raw_events],
        output_path=out_file,
        loss_format=LossMaskingFormat.SHAREGPT,
        compression=CompressionCodec.GZIP,
        split="train",
    )

    expected_gz_path = tmp_path / "train.jsonl.gz"
    assert expected_gz_path.exists()
    assert entry.partition_name == "train.jsonl.gz"
    assert entry.compression == "gzip"

    rows = read_partition(expected_gz_path)
    assert len(rows) == 1
    assert "conversations" in rows[0]
    assert len(rows[0]["conversations"]) == 5


@pytest.mark.skipif(not HAS_ZSTD, reason="zstandard library required")
def test_write_loss_masked_jsonl_zstd(tmp_path, sample_raw_events):
    out_file = tmp_path / "train.jsonl"
    entry = write_loss_masked_jsonl(
        records=[sample_raw_events],
        output_path=out_file,
        loss_format=LossMaskingFormat.ANTHROPIC,
        compression=CompressionCodec.ZSTD,
        split="train",
    )

    expected_zst_path = tmp_path / "train.jsonl.zst"
    assert expected_zst_path.exists()
    assert entry.partition_name == "train.jsonl.zst"
    assert entry.compression == "zstd"

    rows = read_partition(expected_zst_path)
    assert len(rows) == 1
    assert "system" in rows[0]
    assert "messages" in rows[0]


# ---------------------------------------------------------------------------
# 3. PyArrow & Parquet Dataset Export & Typed Schemas
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow required for Parquet tests")
def test_openai_chat_arrow_schema():
    schema = get_openai_chat_arrow_schema()
    assert "id" in schema.names
    assert "messages" in schema.names
    assert "split" in schema.names
    messages_field = schema.field("messages")
    assert pa.types.is_list(messages_field.type)
    struct_type = messages_field.type.value_type
    assert "loss_weight" in [f.name for f in struct_type]


@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow required for Parquet tests")
def test_write_parquet_dataset_openai(tmp_path, sample_raw_events):
    out_file = tmp_path / "train.parquet"
    records = [sample_raw_events, sample_raw_events, sample_raw_events]

    entry = write_parquet_dataset(
        records_or_table=records,
        output_path=out_file,
        loss_format=LossMaskingFormat.OPENAI,
        compression=CompressionCodec.ZSTD if HAS_ZSTD else CompressionCodec.UNCOMPRESSED,
        split="train",
    )

    assert out_file.exists()
    assert entry.format == "parquet"
    assert entry.record_count == 3
    assert len(entry.sha256) == 64

    # Read back table
    table = pq.read_table(out_file)
    assert table.num_rows == 3
    assert "messages" in table.column_names

    # Read back using packager helper
    rows = read_partition(out_file)
    assert len(rows) == 3
    assert len(rows[0]["messages"]) == 5
    assert rows[0]["messages"][2]["loss_weight"] == 1.0
    assert rows[0]["messages"][0]["loss_weight"] == 0.0


@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow required for Parquet tests")
def test_write_parquet_dataset_sharegpt(tmp_path, sample_raw_events):
    out_file = tmp_path / "validation.parquet"
    entry = write_parquet_dataset(
        records_or_table=[sample_raw_events],
        output_path=out_file,
        loss_format=LossMaskingFormat.SHAREGPT,
        split="validation",
    )
    assert out_file.exists()
    assert entry.record_count == 1

    rows = read_partition(out_file)
    assert len(rows) == 1
    assert "conversations" in rows[0]
    assert len(rows[0]["conversations"]) == 5


@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow required for Arrow IPC tests")
def test_write_arrow_ipc_dataset(tmp_path, sample_raw_events):
    out_file = tmp_path / "test.arrow"
    entry = write_arrow_dataset(
        records_or_table=[sample_raw_events, sample_raw_events],
        output_path=out_file,
        loss_format=LossMaskingFormat.OPENAI,
        split="test",
    )
    assert out_file.exists()
    assert entry.format == "arrow"
    assert entry.record_count == 2

    rows = read_partition(out_file)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# 4. Content-Addressable Storage (CAS) Manifest & Verification
# ---------------------------------------------------------------------------

def test_cas_dataset_manifest_computation():
    p1 = PartitionManifestEntry(
        partition_name="train.parquet",
        relative_path="train.parquet",
        split="train",
        format="parquet",
        compression="zstd",
        record_count=100,
        byte_size=50000,
        uncompressed_byte_size=120000,
        sha256="a" * 64,
        uncompressed_sha256="a" * 64,
    )
    p2 = PartitionManifestEntry(
        partition_name="validation.parquet",
        relative_path="validation.parquet",
        split="validation",
        format="parquet",
        compression="zstd",
        record_count=20,
        byte_size=10000,
        uncompressed_byte_size=24000,
        sha256="b" * 64,
        uncompressed_sha256="b" * 64,
    )

    manifest = CASDatasetManifest(
        dataset_name="agent_cyber_ops_v1",
        version="1.0.0",
        partitions=[p1, p2],
    )
    manifest.update_totals()

    assert manifest.total_records == 120
    assert manifest.total_bytes == 60000
    assert manifest.total_uncompressed_bytes == 144000
    assert len(manifest.root_manifest_hash) == 64


def test_dataset_packager_end_to_end(tmp_path, sample_raw_events):
    dataset_dir = tmp_path / "packaged_dataset"
    packager = DatasetPackager(
        dataset_name="agent_reasoning_gold",
        version="v1.0",
        loss_mask_format=LossMaskingFormat.OPENAI,
        compression=CompressionCodec.ZSTD if HAS_ZSTD else CompressionCodec.GZIP,
    )

    splits = {
        "train": [sample_raw_events] * 10,
        "validation": [sample_raw_events] * 2,
    }

    manifest = packager.package_dataset(
        splits=splits,
        output_dir=dataset_dir,
        formats=[DatasetFormat.JSONL, DatasetFormat.PARQUET] if HAS_PYARROW else [DatasetFormat.JSONL],
        write_manifest=True,
        write_sha256sums=True,
        provenance={"source": "Labyrinth-Mine", "curator": "GoldTrace-Mint"},
    )

    assert (dataset_dir / "manifest.json").exists()
    assert (dataset_dir / "SHA256SUMS").exists()
    assert manifest.total_records == 24 if HAS_PYARROW else 12  # 12 records * 2 formats

    # Run CAS integrity verification
    is_valid, errors = verify_cas_dataset(dataset_dir)
    assert is_valid, f"Verification failed with errors: {errors}"
    assert len(errors) == 0


def test_cas_tamper_detection(tmp_path, sample_raw_events):
    dataset_dir = tmp_path / "tamper_test"
    packager = DatasetPackager(
        dataset_name="tamper_check",
        version="v1",
        compression=CompressionCodec.UNCOMPRESSED,
    )

    splits = {"train": [sample_raw_events] * 3}
    packager.package_dataset(
        splits=splits,
        output_dir=dataset_dir,
        formats=[DatasetFormat.JSONL],
        write_manifest=True,
        write_sha256sums=True,
    )

    # Initial verification passes
    is_valid, errors = verify_cas_dataset(dataset_dir)
    assert is_valid

    # Now tamper with train.jsonl
    train_file = dataset_dir / "train.jsonl"
    content = train_file.read_text(encoding="utf-8")
    tampered_content = content + "\n{\"tampered\": true}\n"
    train_file.write_text(tampered_content, encoding="utf-8")

    # Verification must detect the mismatch!
    is_valid, errors = verify_cas_dataset(dataset_dir)
    assert not is_valid
    assert any("mismatch" in e for e in errors)


def test_sharded_packaging(tmp_path, sample_raw_events):
    dataset_dir = tmp_path / "sharded_dataset"
    packager = DatasetPackager(
        dataset_name="sharded_test",
        version="v1",
        compression=CompressionCodec.UNCOMPRESSED,
    )

    # 15 records with shard_size=5 -> 3 shards
    splits = {"train": [sample_raw_events] * 15}
    manifest = packager.package_dataset(
        splits=splits,
        output_dir=dataset_dir,
        formats=[DatasetFormat.JSONL],
        shard_size=5,
    )

    shards = [p for p in manifest.partitions if p.split == "train"]
    assert len(shards) == 3
    for s in shards:
        assert s.record_count == 5
        assert (dataset_dir / s.relative_path).exists()

    is_valid, errors = verify_cas_dataset(dataset_dir)
    assert is_valid


def test_masked_messages_input_directly(tmp_path, sample_masked_messages):
    out_file = tmp_path / "masked_direct.jsonl"
    entry = write_loss_masked_jsonl(
        records=[sample_masked_messages],
        output_path=out_file,
        loss_format=LossMaskingFormat.OPENAI,
        compression=CompressionCodec.UNCOMPRESSED,
    )
    assert entry.record_count == 1
    rows = read_partition(out_file)
    assert len(rows) == 1
    assert len(rows[0]["messages"]) == 5
    assert rows[0]["messages"][2]["loss_weight"] == 1.0
    assert rows[0]["messages"][2]["reasoning_content"] == "Assistant step-by-step reasoning"
    assert rows[0]["messages"][3]["loss_weight"] == 0.0


@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow required for Parquet tests")
def test_pre_structured_messages_dict(tmp_path):
    record = {
        "id": "item_001",
        "messages": [
            {"role": "system", "content": "You are helpful.", "loss_weight": 0.0},
            {"role": "user", "content": "Hi", "loss_weight": 0.0},
            {"role": "assistant", "content": "Hello!", "loss_weight": 1.0},
        ],
        "metadata": {"source": "unit_test"},
    }
    out_file = tmp_path / "structured.parquet"
    entry = write_parquet_dataset(
        records_or_table=[record],
        output_path=out_file,
        loss_format=LossMaskingFormat.OPENAI,
    )
    assert entry.record_count == 1
    rows = read_partition(out_file)
    assert len(rows) == 1
    assert rows[0]["id"] == "item_001"
    assert len(rows[0]["messages"]) == 3
    assert rows[0]["messages"][2]["loss_weight"] == 1.0


def test_compression_codec_normalization():
    assert CompressionCodec.normalize("zstd") == CompressionCodec.ZSTD
    assert CompressionCodec.normalize("ZSTANDARD") == CompressionCodec.ZSTD
    assert CompressionCodec.normalize("gzip") == CompressionCodec.GZIP
    assert CompressionCodec.normalize("gz") == CompressionCodec.GZIP
    assert CompressionCodec.normalize("uncompressed") == CompressionCodec.UNCOMPRESSED
    assert CompressionCodec.normalize("none") == CompressionCodec.UNCOMPRESSED
    assert CompressionCodec.normalize(CompressionCodec.ZSTD) == CompressionCodec.ZSTD
    with pytest.raises(ValueError):
        CompressionCodec.normalize("bzip2")


@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow required for Parquet tests")
def test_arrow_schemas():
    sg_schema = get_sharegpt_arrow_schema()
    assert "conversations" in sg_schema.names
    ant_schema = get_anthropic_arrow_schema()
    assert "system" in ant_schema.names
    assert "messages_json" in ant_schema.names


def test_invalid_partition_read_raises():
    with pytest.raises(FileNotFoundError):
        read_partition("non_existent_file.parquet")


@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow required for Parquet tests")
def test_dataset_packager_all_formats_combined(tmp_path, sample_raw_events):
    dataset_dir = tmp_path / "all_formats"
    packager = DatasetPackager(
        dataset_name="all_formats_pkg",
        version="2.0.0",
        loss_mask_format=LossMaskingFormat.OPENAI,
        compression=CompressionCodec.ZSTD if HAS_ZSTD else CompressionCodec.GZIP,
    )
    splits = {
        "train": [sample_raw_events] * 4,
        "validation": [sample_raw_events] * 2,
        "test": [sample_raw_events] * 1,
    }
    manifest = packager.package_dataset(
        splits=splits,
        output_dir=dataset_dir,
        formats=[DatasetFormat.JSONL, DatasetFormat.PARQUET, DatasetFormat.ARROW],
        write_manifest=True,
        write_sha256sums=True,
    )
    # 3 splits * 3 formats = 9 partitions
    assert len(manifest.partitions) == 9
    assert manifest.total_records == (4 + 2 + 1) * 3

    is_valid, errors = verify_cas_dataset(dataset_dir)
    assert is_valid
    assert len(errors) == 0

