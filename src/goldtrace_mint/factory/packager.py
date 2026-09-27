#!/usr/bin/env python3
"""
GTDataworks High-Performance Parquet & Loss-Masked Packager
==========================================================
Packages curated agent traces and instruction-tuning datasets into commercial-grade
training formats with exact selective loss weights, typed Arrow schemas, and
Content-Addressable Storage (CAS) manifest integrity:

Supported Formats:
  1. Loss-Masked JSONL:
     - OpenAI Chat Tool Format (system/user/tool masked @ 0.0, assistant reasoning/tools @ 1.0)
     - ShareGPT Format (human/tool @ 0.0, gpt @ 1.0 with thought/tool_call annotations)
     - Anthropic Messages Tool Format (system/user/tool_result @ 0.0, assistant text/thinking/tool_use @ 1.0)
  2. Parquet & Arrow Dataset Format:
     - Typed PyArrow schemas for zero-copy memory mapping in PyTorch, HF datasets, and Unsloth
     - Configurable row group sizes, dictionary encoding, and column metadata
  3. Compression Codecs:
     - zstd (Zstandard - high throughput, optimal ratio)
     - gzip (Deflate - universal compatibility)
     - uncompressed / none (raw text / table)
  4. Content-Addressable Storage (CAS) Manifest:
     - Streaming cryptographic SHA-256 calculation for both compressed artifacts and uncompressed payloads
     - Standard CAS manifest JSON and SHA256SUMS generation for complete provenance auditing
"""

from __future__ import annotations

import enum
import gzip
import hashlib
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    pa = None  # type: ignore
    pa_ipc = None  # type: ignore
    pq = None  # type: ignore
    HAS_PYARROW = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    zstd = None  # type: ignore
    HAS_ZSTD = False

from .loss_masking import MaskedMessage, format_trace_with_loss_mask


# ---------------------------------------------------------------------------
# Enums & Configurations
# ---------------------------------------------------------------------------

class LossMaskingFormat(str, enum.Enum):
    """Target loss-masking representation standard."""
    OPENAI = "openai"
    SHAREGPT = "sharegpt"
    ANTHROPIC = "anthropic"
    RAW = "raw"


class DatasetFormat(str, enum.Enum):
    """Storage container format."""
    JSONL = "jsonl"
    PARQUET = "parquet"
    ARROW = "arrow"


class CompressionCodec(str, enum.Enum):
    """Compression algorithm."""
    UNCOMPRESSED = "uncompressed"
    NONE = "none"
    ZSTD = "zstd"
    GZIP = "gzip"

    @classmethod
    def normalize(cls, codec: Union[str, "CompressionCodec"]) -> "CompressionCodec":
        if isinstance(codec, cls):
            if codec == cls.NONE:
                return cls.UNCOMPRESSED
            return codec
        c = str(getattr(codec, "value", codec)).lower().strip()
        if c.startswith("compressioncodec."):
            c = c.split(".", 1)[-1]
        if c in ("none", "uncompressed", "raw", ""):
            return cls.UNCOMPRESSED
        if c in ("zstd", "zstandard", "zst"):
            return cls.ZSTD
        if c in ("gzip", "gz"):
            return cls.GZIP
        raise ValueError(f"Unsupported compression codec: {codec}")


def _to_loss_format_str(loss_format: Union[LossMaskingFormat, str]) -> str:
    """Safely extract lowercase string representation of LossMaskingFormat."""
    if isinstance(loss_format, LossMaskingFormat):
        return loss_format.value.lower()
    val = getattr(loss_format, "value", str(loss_format))
    s = str(val).lower().strip()
    if s.startswith("lossmaskingformat."):
        s = s.split(".", 1)[-1]
    return s


def _to_dataset_format_str(dataset_format: Union[DatasetFormat, str]) -> str:
    """Safely extract lowercase string representation of DatasetFormat."""
    if isinstance(dataset_format, DatasetFormat):
        return dataset_format.value.lower()
    val = getattr(dataset_format, "value", str(dataset_format))
    s = str(val).lower().strip()
    if s.startswith("datasetformat."):
        s = s.split(".", 1)[-1]
    return s


# ---------------------------------------------------------------------------
# Manifest Data Models
# ---------------------------------------------------------------------------

@dataclass
class PartitionManifestEntry:
    """Content-Addressable Storage (CAS) manifest record for a single dataset partition."""
    partition_name: str
    relative_path: str
    split: str
    format: str
    compression: str
    record_count: int
    byte_size: int
    uncompressed_byte_size: int
    sha256: str
    uncompressed_sha256: str
    schema_info: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CASDatasetManifest:
    """Cryptographically bound Content-Addressable Storage manifest for an entire dataset."""
    dataset_name: str
    version: str
    manifest_version: str = "cas-dataset-manifest-v1"
    target_format: str = "mixed"
    loss_mask_format: str = "openai"
    partitions: List[PartitionManifestEntry] = field(default_factory=list)
    total_records: int = 0
    total_bytes: int = 0
    total_uncompressed_bytes: int = 0
    root_manifest_hash: str = ""
    schema_summary: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def update_totals(self) -> None:
        self.total_records = sum(p.record_count for p in self.partitions)
        self.total_bytes = sum(p.byte_size for p in self.partitions)
        self.total_uncompressed_bytes = sum(p.uncompressed_byte_size for p in self.partitions)
        self.compute_root_hash()

    def compute_root_hash(self) -> str:
        """Compute deterministic canonical SHA-256 of manifest content omitting self-referential hash."""
        data = self.to_dict()
        data.pop("root_manifest_hash", None)
        canonical_bytes = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.root_manifest_hash = hashlib.sha256(canonical_bytes).hexdigest()
        return self.root_manifest_hash

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["partitions"] = [p.to_dict() if isinstance(p, PartitionManifestEntry) else p for p in self.partitions]
        return d

    def to_json(self, indent: int = 2) -> str:
        self.compute_root_hash()
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def save(self, manifest_path: Path) -> str:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_json(indent=2)
        manifest_path.write_text(content + "\n", encoding="utf-8")
        return self.root_manifest_hash


# ---------------------------------------------------------------------------
# Formatting & Loss-Masking Normalizers
# ---------------------------------------------------------------------------

def _normalize_raw_trace(trace: Any) -> List[Dict[str, Any]]:
    """Normalize input into an event or message list."""
    if isinstance(trace, dict):
        if "messages" in trace and isinstance(trace["messages"], list):
            return trace["messages"]
        if "conversations" in trace and isinstance(trace["conversations"], list):
            return trace["conversations"]
        if "events" in trace and isinstance(trace["events"], list):
            return trace["events"]
        return [trace]
    if isinstance(trace, list):
        return trace
    return []


def format_for_openai_chat(
    trace: Union[List[Dict[str, Any]], Dict[str, Any], List[MaskedMessage]],
    weight_key: str = "loss_weight",
    include_reasoning: bool = True,
) -> Dict[str, Any]:
    """
    Format trace into standard OpenAI Chat tool format with exact loss weights.
    
    Structure:
      {
        "messages": [
          {"role": "system", "content": "...", "loss_weight": 0.0},
          {"role": "user", "content": "...", "loss_weight": 0.0},
          {
            "role": "assistant",
            "content": "...",
            "reasoning_content": "...",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            "loss_weight": 1.0
          },
          {"role": "tool", "tool_call_id": "call_1", "name": "f", "content": "...", "loss_weight": 0.0}
        ]
      }
    """
    if isinstance(trace, dict) and "messages" in trace and not isinstance(trace.get("events"), list):
        messages = trace["messages"]
        out_messages = []
        for msg in messages:
            if isinstance(msg, MaskedMessage):
                entry: Dict[str, Any] = {
                    "role": msg.role,
                    "content": msg.content,
                    weight_key: float(msg.loss_weight),
                }
                if include_reasoning and msg.reasoning_content:
                    entry["reasoning_content"] = msg.reasoning_content
                if msg.tool_calls:
                    entry["tool_calls"] = msg.tool_calls
                out_messages.append(entry)
            elif isinstance(msg, dict):
                entry = dict(msg)
                role = entry.get("role", "user")
                if weight_key not in entry:
                    entry[weight_key] = 1.0 if role == "assistant" else 0.0
                out_messages.append(entry)
        return {"messages": out_messages}

    raw_list = _normalize_raw_trace(trace)
    if raw_list and isinstance(raw_list[0], MaskedMessage):
        formatted = []
        for m in raw_list:
            item = {
                "role": m.role,
                "content": m.content,
                weight_key: float(m.loss_weight),
            }
            if include_reasoning and m.reasoning_content:
                item["reasoning_content"] = m.reasoning_content
            if m.tool_calls:
                item["tool_calls"] = m.tool_calls
            formatted.append(item)
        return {"messages": formatted}

    formatted_msgs = format_trace_with_loss_mask(raw_list)
    if weight_key != "loss_weight":
        for m in formatted_msgs:
            if "loss_weight" in m:
                m[weight_key] = m.pop("loss_weight")
    if not include_reasoning:
        for m in formatted_msgs:
            m.pop("reasoning_content", None)
    return {"messages": formatted_msgs}


def format_for_sharegpt(
    trace: Union[List[Dict[str, Any]], Dict[str, Any], List[MaskedMessage]],
    weight_key: str = "loss_weight",
    role_map: Optional[Dict[str, str]] = None,
    include_thought_tags: bool = True,
) -> Dict[str, Any]:
    """
    Format trace into ShareGPT conversation format with loss masking.
    
    Structure:
      {
        "conversations": [
          {"from": "system", "value": "...", "loss_weight": 0.0},
          {"from": "human", "value": "...", "loss_weight": 0.0},
          {"from": "gpt", "value": "<thought>...</thought>...", "loss_weight": 1.0},
          {"from": "tool", "value": "...", "loss_weight": 0.0}
        ]
      }
    """
    openai_form = format_for_openai_chat(trace, weight_key="loss_weight", include_reasoning=True)
    messages = openai_form.get("messages", [])

    mapping = {
        "system": "system",
        "user": "human",
        "human": "human",
        "assistant": "gpt",
        "gpt": "gpt",
        "tool": "tool",
        "function": "tool",
    }
    if role_map:
        mapping.update(role_map)

    conversations: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "human")
        from_role = mapping.get(role, role)
        loss_weight = float(msg.get("loss_weight", 1.0 if from_role == "gpt" else 0.0))

        content_parts = []
        reasoning = msg.get("reasoning_content")
        if include_thought_tags and reasoning:
            content_parts.append(f"<thought>\n{reasoning}\n</thought>")

        raw_content = msg.get("content") or ""
        if raw_content:
            content_parts.append(str(raw_content))

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name") or tc.get("name", "tool")
                fn_args = fn.get("arguments") or tc.get("arguments", "{}")
                if isinstance(fn_args, dict):
                    fn_args = json.dumps(fn_args)
                content_parts.append(f"<tool_call>\n{{\"name\": \"{fn_name}\", \"arguments\": {fn_args}}}\n</tool_call>")

        turn_value = "\n\n".join(content_parts).strip()
        if not turn_value and msg.get("role") == "tool":
            turn_value = str(msg.get("content", ""))

        conv_item: Dict[str, Any] = {
            "from": from_role,
            "value": turn_value,
            weight_key: loss_weight,
        }
        if "tool_call_id" in msg:
            conv_item["tool_call_id"] = msg["tool_call_id"]
        conversations.append(conv_item)

    return {"conversations": conversations}


def format_for_anthropic(
    trace: Union[List[Dict[str, Any]], Dict[str, Any], List[MaskedMessage]],
    weight_key: str = "loss_weight",
) -> Dict[str, Any]:
    """
    Format trace into Anthropic Messages API format with structured content blocks.
    
    Structure:
      {
        "system": "...",
        "messages": [
          {
            "role": "user",
            "content": [{"type": "text", "text": "...", "loss_weight": 0.0}],
            "loss_weight": 0.0
          },
          {
            "role": "assistant",
            "content": [
              {"type": "thinking", "thinking": "...", "loss_weight": 1.0},
              {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}, "loss_weight": 1.0},
              {"type": "text", "text": "...", "loss_weight": 1.0}
            ],
            "loss_weight": 1.0
          },
          {
            "role": "user",
            "content": [
              {"type": "tool_result", "tool_use_id": "toolu_1", "content": "...", "loss_weight": 0.0}
            ],
            "loss_weight": 0.0
          }
        ]
      }
    """
    openai_form = format_for_openai_chat(trace, weight_key="loss_weight", include_reasoning=True)
    messages = openai_form.get("messages", [])

    system_prompt = ""
    anthropic_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content")
        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")

        if role == "system":
            if system_prompt:
                system_prompt += "\n\n" + str(content)
            else:
                system_prompt = str(content)
            continue

        if role == "assistant":
            blocks: List[Dict[str, Any]] = []
            if reasoning:
                blocks.append({
                    "type": "thinking",
                    "thinking": str(reasoning),
                    weight_key: 1.0,
                })
            if tool_calls:
                for idx, tc in enumerate(tool_calls):
                    call_id = tc.get("id") or f"toolu_{idx}"
                    fn = tc.get("function", {})
                    fn_name = fn.get("name") or tc.get("name", "tool")
                    args = fn.get("arguments") or tc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    blocks.append({
                        "type": "tool_use",
                        "id": call_id,
                        "name": fn_name,
                        "input": args,
                        weight_key: 1.0,
                    })
            if content:
                blocks.append({
                    "type": "text",
                    "text": str(content),
                    weight_key: 1.0,
                })
            anthropic_messages.append({
                "role": "assistant",
                "content": blocks,
                weight_key: 1.0,
            })

        elif role == "tool":
            call_id = tool_call_id or "toolu_0"
            result_block = {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": str(content),
                weight_key: 0.0,
            }
            if anthropic_messages and anthropic_messages[-1].get("role") == "user":
                anthropic_messages[-1]["content"].append(result_block)
            else:
                anthropic_messages.append({
                    "role": "user",
                    "content": [result_block],
                    weight_key: 0.0,
                })

        else:  # role == "user"
            text_block = {
                "type": "text",
                "text": str(content),
                weight_key: 0.0,
            }
            anthropic_messages.append({
                "role": "user",
                "content": [text_block],
                weight_key: 0.0,
            })

    result: Dict[str, Any] = {"messages": anthropic_messages}
    if system_prompt:
        result["system"] = system_prompt
    return result


def normalize_record_for_format(
    record: Union[Dict[str, Any], List[Dict[str, Any]], MaskedMessage],
    loss_format: Union[LossMaskingFormat, str] = LossMaskingFormat.OPENAI,
    weight_key: str = "loss_weight",
    include_reasoning: bool = True,
) -> Dict[str, Any]:
    """Normalize arbitrary conversation/trace input into the specified target format."""
    fmt = _to_loss_format_str(loss_format)
    if fmt in ("openai", "chat", "messages"):
        return format_for_openai_chat(record, weight_key=weight_key, include_reasoning=include_reasoning)
    if fmt in ("sharegpt", "conversations"):
        return format_for_sharegpt(record, weight_key=weight_key)
    if fmt in ("anthropic", "claude"):
        return format_for_anthropic(record, weight_key=weight_key)
    if isinstance(record, dict):
        return record
    return {"data": record}


# ---------------------------------------------------------------------------
# PyArrow Typed Schema Definitions
# ---------------------------------------------------------------------------

def get_openai_chat_arrow_schema() -> "pa.Schema":
    """Return strict typed PyArrow schema for loss-masked OpenAI chat format."""
    if not HAS_PYARROW:
        raise RuntimeError("PyArrow is required for Arrow schemas. Please install pyarrow.")

    message_struct = pa.struct([
        pa.field("role", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
        pa.field("loss_weight", pa.float64(), nullable=False),
        pa.field("reasoning_content", pa.string(), nullable=True),
        pa.field("tool_calls_json", pa.string(), nullable=True),
        pa.field("tool_call_id", pa.string(), nullable=True),
        pa.field("name", pa.string(), nullable=True),
    ])

    return pa.schema([
        pa.field("id", pa.string(), nullable=True),
        pa.field("messages", pa.list_(message_struct), nullable=False),
        pa.field("split", pa.string(), nullable=True),
        pa.field("token_count", pa.int64(), nullable=True),
        pa.field("metadata_json", pa.string(), nullable=True),
    ])


def get_sharegpt_arrow_schema() -> "pa.Schema":
    """Return strict typed PyArrow schema for ShareGPT conversation format."""
    if not HAS_PYARROW:
        raise RuntimeError("PyArrow is required for Arrow schemas. Please install pyarrow.")

    turn_struct = pa.struct([
        pa.field("from", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
        pa.field("loss_weight", pa.float64(), nullable=False),
        pa.field("tool_call_id", pa.string(), nullable=True),
    ])

    return pa.schema([
        pa.field("id", pa.string(), nullable=True),
        pa.field("conversations", pa.list_(turn_struct), nullable=False),
        pa.field("split", pa.string(), nullable=True),
        pa.field("metadata_json", pa.string(), nullable=True),
    ])


def get_anthropic_arrow_schema() -> "pa.Schema":
    """Return strict typed PyArrow schema for Anthropic Messages format."""
    if not HAS_PYARROW:
        raise RuntimeError("PyArrow is required for Arrow schemas. Please install pyarrow.")

    return pa.schema([
        pa.field("id", pa.string(), nullable=True),
        pa.field("system", pa.string(), nullable=True),
        pa.field("messages_json", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=True),
        pa.field("metadata_json", pa.string(), nullable=True),
    ])


def build_arrow_table_from_records(
    records: List[Any],
    loss_format: Union[LossMaskingFormat, str] = LossMaskingFormat.OPENAI,
    schema: Optional["pa.Schema"] = None,
    split: Optional[str] = "train",
) -> "pa.Table":
    """Convert formatted records into a typed PyArrow Table compatible with Hugging Face & PyTorch."""
    if not HAS_PYARROW:
        raise RuntimeError("PyArrow is required to construct Arrow tables.")

    fmt = _to_loss_format_str(loss_format)

    if schema is None:
        if fmt in ("openai", "chat", "messages"):
            schema = get_openai_chat_arrow_schema()
        elif fmt in ("sharegpt", "conversations"):
            schema = get_sharegpt_arrow_schema()
        elif fmt in ("anthropic", "claude"):
            schema = get_anthropic_arrow_schema()

    pylist: List[Dict[str, Any]] = []

    for idx, raw_rec in enumerate(records):
        rec = normalize_record_for_format(raw_rec, loss_format=loss_format) if not (isinstance(raw_rec, dict) and (("messages" in raw_rec and fmt == "openai") or ("conversations" in raw_rec and fmt == "sharegpt") or ("system" in raw_rec and fmt == "anthropic"))) else raw_rec
        row_id = rec.get("id") or f"row_{idx:07d}" if isinstance(rec, dict) else f"row_{idx:07d}"
        rec_split = rec.get("split") or split if isinstance(rec, dict) else split
        meta_json = json.dumps(rec.get("metadata", {})) if isinstance(rec, dict) and "metadata" in rec else None

        if fmt in ("openai", "chat", "messages"):
            msgs = rec.get("messages", []) if isinstance(rec, dict) else []
            arrow_msgs = []
            for m in msgs:
                tc_raw = m.get("tool_calls")
                tc_json = json.dumps(tc_raw) if tc_raw is not None else None
                arrow_msgs.append({
                    "role": str(m.get("role", "user")),
                    "content": str(m.get("content", "")),
                    "loss_weight": float(m.get("loss_weight", 1.0 if m.get("role") == "assistant" else 0.0)),
                    "reasoning_content": m.get("reasoning_content"),
                    "tool_calls_json": tc_json,
                    "tool_call_id": m.get("tool_call_id"),
                    "name": m.get("name"),
                })
            pylist.append({
                "id": str(row_id),
                "messages": arrow_msgs,
                "split": str(rec_split) if rec_split else None,
                "token_count": rec.get("token_count") if isinstance(rec, dict) else None,
                "metadata_json": meta_json,
            })

        elif fmt in ("sharegpt", "conversations"):
            convs = rec.get("conversations", []) if isinstance(rec, dict) else []
            arrow_convs = []
            for c in convs:
                arrow_convs.append({
                    "from": str(c.get("from", "human")),
                    "value": str(c.get("value", "")),
                    "loss_weight": float(c.get("loss_weight", 1.0 if c.get("from") in ("gpt", "assistant") else 0.0)),
                    "tool_call_id": c.get("tool_call_id"),
                })
            pylist.append({
                "id": str(row_id),
                "conversations": arrow_convs,
                "split": str(rec_split) if rec_split else None,
                "metadata_json": meta_json,
            })

        elif fmt in ("anthropic", "claude"):
            pylist.append({
                "id": str(row_id),
                "system": rec.get("system") if isinstance(rec, dict) else None,
                "messages_json": json.dumps(rec.get("messages", [])) if isinstance(rec, dict) else "[]",
                "split": str(rec_split) if rec_split else None,
                "metadata_json": meta_json,
            })

        else:
            pylist.append(rec if isinstance(rec, dict) else {"data": rec})

    if schema is not None:
        return pa.Table.from_pylist(pylist, schema=schema)
    return pa.Table.from_pylist(pylist)


# ---------------------------------------------------------------------------
# High-Performance Streaming Writers & Compressors
# ---------------------------------------------------------------------------

class StreamingHashingWriter:
    """Streams data through an uncompressed hash accumulator and into an output destination."""

    def __init__(self, raw_file_handle: io.BufferedWriter, codec: CompressionCodec, compression_level: int = 3):
        self.raw_handle = raw_file_handle
        self.codec = codec
        self.uncompressed_hasher = hashlib.sha256()
        self.compressed_hasher = hashlib.sha256()
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0

        if self.codec == CompressionCodec.ZSTD:
            if not HAS_ZSTD:
                raise RuntimeError("zstandard package is required for zstd compression.")
            cctx = zstd.ZstdCompressor(level=compression_level)
            self._compressor = cctx.compressobj()
            self._write_func = self._write_zstd
        elif self.codec == CompressionCodec.GZIP:
            self._gzip_file = gzip.GzipFile(mode="wb", fileobj=self.raw_handle, compresslevel=compression_level)
            self._write_func = self._write_gzip
        else:
            self._write_func = self._write_raw

    def write(self, data: bytes) -> None:
        self.uncompressed_hasher.update(data)
        self.uncompressed_bytes += len(data)
        self._write_func(data)

    def _write_raw(self, data: bytes) -> None:
        self.compressed_hasher.update(data)
        self.compressed_bytes += len(data)
        self.raw_handle.write(data)

    def _write_zstd(self, data: bytes) -> None:
        compressed_chunk = self._compressor.compress(data)
        if compressed_chunk:
            self.compressed_hasher.update(compressed_chunk)
            self.compressed_bytes += len(compressed_chunk)
            self.raw_handle.write(compressed_chunk)

    def _write_gzip(self, data: bytes) -> None:
        self._gzip_file.write(data)

    def close(self) -> Tuple[int, int, str, str]:
        """Finalize stream and return (record_bytes, uncompressed_bytes, file_sha256, uncompressed_sha256)."""
        if self.codec == CompressionCodec.ZSTD:
            final_chunk = self._compressor.flush()
            if final_chunk:
                self.compressed_hasher.update(final_chunk)
                self.compressed_bytes += len(final_chunk)
                self.raw_handle.write(final_chunk)
            self.raw_handle.flush()
            file_bytes = self.compressed_bytes
            file_sha256 = self.compressed_hasher.hexdigest()
        elif self.codec == CompressionCodec.GZIP:
            self._gzip_file.close()
            self.raw_handle.flush()
            self.raw_handle.seek(0)
            hasher = hashlib.sha256()
            size = 0
            while chunk := self.raw_handle.read(1024 * 1024):
                hasher.update(chunk)
                size += len(chunk)
            file_sha256 = hasher.hexdigest()
            file_bytes = size
        else:
            self.raw_handle.flush()
            file_bytes = self.compressed_bytes
            file_sha256 = self.compressed_hasher.hexdigest()

        uncompressed_sha256 = self.uncompressed_hasher.hexdigest()
        return file_bytes, self.uncompressed_bytes, file_sha256, uncompressed_sha256


# ---------------------------------------------------------------------------
# Core Packaging Functions
# ---------------------------------------------------------------------------

def write_loss_masked_jsonl(
    records: Iterable[Union[Dict[str, Any], List[Dict[str, Any]], MaskedMessage]],
    output_path: Union[str, Path],
    loss_format: Union[LossMaskingFormat, str] = LossMaskingFormat.OPENAI,
    compression: Union[CompressionCodec, str] = CompressionCodec.UNCOMPRESSED,
    split: str = "train",
    compression_level: int = 3,
    weight_key: str = "loss_weight",
    include_reasoning: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> PartitionManifestEntry:
    """
    Write dataset records to a loss-masked JSONL file with streaming compression and SHA-256 CAS digest.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    codec = CompressionCodec.normalize(compression)

    if codec == CompressionCodec.ZSTD and not out_p.name.endswith(".zst"):
        out_p = out_p.with_name(out_p.name + ".zst")
    elif codec == CompressionCodec.GZIP and not out_p.name.endswith(".gz"):
        out_p = out_p.with_name(out_p.name + ".gz")

    record_count = 0
    with out_p.open("w+b") as raw_f:
        stream = StreamingHashingWriter(raw_f, codec, compression_level=compression_level)
        for rec in records:
            formatted = normalize_record_for_format(
                rec,
                loss_format=loss_format,
                weight_key=weight_key,
                include_reasoning=include_reasoning,
            )
            line = (json.dumps(formatted, ensure_ascii=False) + "\n").encode("utf-8")
            stream.write(line)
            record_count += 1

        file_bytes, uncompressed_bytes, file_sha256, uncompressed_sha256 = stream.close()

    schema_summary = {
        "format": "loss_masked_jsonl",
        "loss_masking_standard": _to_loss_format_str(loss_format),
        "weight_key": weight_key,
        "include_reasoning": include_reasoning,
    }

    return PartitionManifestEntry(
        partition_name=out_p.name,
        relative_path=out_p.name,
        split=split,
        format="jsonl",
        compression=str(codec.value),
        record_count=record_count,
        byte_size=file_bytes,
        uncompressed_byte_size=uncompressed_bytes,
        sha256=file_sha256,
        uncompressed_sha256=uncompressed_sha256,
        schema_info=schema_summary,
        metadata=metadata or {},
    )


def write_parquet_dataset(
    records_or_table: Union[List[Any], "pa.Table"],
    output_path: Union[str, Path],
    loss_format: Union[LossMaskingFormat, str] = LossMaskingFormat.OPENAI,
    schema: Optional["pa.Schema"] = None,
    compression: Union[CompressionCodec, str] = CompressionCodec.ZSTD,
    compression_level: Optional[int] = None,
    row_group_size: int = 5000,
    split: str = "train",
    metadata: Optional[Dict[str, Any]] = None,
) -> PartitionManifestEntry:
    """
    Write dataset to high-performance Parquet format with typed schema and CAS SHA-256 digest.
    Compatible with Hugging Face `datasets`, PyTorch `DataLoader`, and Unsloth.
    """
    if not HAS_PYARROW:
        raise RuntimeError("PyArrow is required to write Parquet files. Please install pyarrow.")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if not out_p.name.endswith(".parquet"):
        out_p = out_p.with_name(out_p.name + ".parquet")

    codec = CompressionCodec.normalize(compression)
    pa_compression = "ZSTD" if codec == CompressionCodec.ZSTD else ("GZIP" if codec == CompressionCodec.GZIP else "NONE")

    if isinstance(records_or_table, pa.Table):
        table = records_or_table
    else:
        table = build_arrow_table_from_records(
            records_or_table,
            loss_format=loss_format,
            schema=schema,
            split=split,
        )

    write_kwargs: Dict[str, Any] = {
        "compression": pa_compression,
        "row_group_size": row_group_size,
        "version": "2.6",
        "use_dictionary": True,
    }
    if compression_level is not None and pa_compression != "NONE":
        write_kwargs["compression_level"] = compression_level

    pq.write_table(table, out_p, **write_kwargs)

    file_bytes = out_p.stat().st_size
    file_sha256 = _compute_file_sha256(out_p)
    uncompressed_bytes = table.nbytes

    schema_info = {
        "num_columns": table.num_columns,
        "column_names": table.column_names,
        "schema_str": str(table.schema),
        "row_group_size": row_group_size,
    }

    return PartitionManifestEntry(
        partition_name=out_p.name,
        relative_path=out_p.name,
        split=split,
        format="parquet",
        compression=str(codec.value),
        record_count=table.num_rows,
        byte_size=file_bytes,
        uncompressed_byte_size=uncompressed_bytes,
        sha256=file_sha256,
        uncompressed_sha256=file_sha256,
        schema_info=schema_info,
        metadata=metadata or {},
    )


def write_arrow_dataset(
    records_or_table: Union[List[Any], "pa.Table"],
    output_path: Union[str, Path],
    loss_format: Union[LossMaskingFormat, str] = LossMaskingFormat.OPENAI,
    schema: Optional["pa.Schema"] = None,
    compression: Union[CompressionCodec, str] = CompressionCodec.ZSTD,
    split: str = "train",
    metadata: Optional[Dict[str, Any]] = None,
) -> PartitionManifestEntry:
    """
    Write dataset to Arrow IPC / Feather stream format for ultra-fast memory mapped loading.
    """
    if not HAS_PYARROW:
        raise RuntimeError("PyArrow is required to write Arrow IPC files.")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if not (out_p.name.endswith(".arrow") or out_p.name.endswith(".feather")):
        out_p = out_p.with_name(out_p.name + ".arrow")

    codec = CompressionCodec.normalize(compression)
    pa_codec = "zstd" if codec == CompressionCodec.ZSTD else None

    if isinstance(records_or_table, pa.Table):
        table = records_or_table
    else:
        table = build_arrow_table_from_records(
            records_or_table,
            loss_format=loss_format,
            schema=schema,
            split=split,
        )

    with pa.OSFile(str(out_p), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema, options=pa.ipc.IpcWriteOptions(compression=pa_codec)) as writer:
            writer.write_table(table)

    file_bytes = out_p.stat().st_size
    file_sha256 = _compute_file_sha256(out_p)

    return PartitionManifestEntry(
        partition_name=out_p.name,
        relative_path=out_p.name,
        split=split,
        format="arrow",
        compression=str(codec.value),
        record_count=table.num_rows,
        byte_size=file_bytes,
        uncompressed_byte_size=table.nbytes,
        sha256=file_sha256,
        uncompressed_sha256=file_sha256,
        schema_info={"column_names": table.column_names, "num_columns": table.num_columns},
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# High-Level Dataset Packager
# ---------------------------------------------------------------------------

class DatasetPackager:
    """
    Commercial-grade Dataset Packager for Agentic & Frontier LLM Training.
    
    Manages partition generation, multi-format exports (JSONL / Parquet / Arrow),
    compression pipelines, and Content-Addressable Storage (CAS) manifest binding.
    """

    def __init__(
        self,
        dataset_name: str = "goldtrace_dataset",
        version: str = "v1",
        loss_mask_format: Union[LossMaskingFormat, str] = LossMaskingFormat.OPENAI,
        compression: Union[CompressionCodec, str] = CompressionCodec.ZSTD,
        compression_level: int = 3,
        weight_key: str = "loss_weight",
        include_reasoning: bool = True,
    ):
        self.dataset_name = dataset_name
        self.version = version
        self.loss_mask_format = LossMaskingFormat(_to_loss_format_str(loss_mask_format))
        self.compression = CompressionCodec.normalize(compression)
        self.compression_level = compression_level
        self.weight_key = weight_key
        self.include_reasoning = include_reasoning

    def package_partition(
        self,
        records: List[Any],
        output_file: Union[str, Path],
        target_format: Union[DatasetFormat, str] = DatasetFormat.JSONL,
        split: str = "train",
        schema: Optional["pa.Schema"] = None,
        row_group_size: int = 5000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PartitionManifestEntry:
        """Package a single partition into the specified format."""
        fmt_str = _to_dataset_format_str(target_format)
        out_p = Path(output_file)

        if fmt_str == "jsonl":
            return write_loss_masked_jsonl(
                records=records,
                output_path=out_p,
                loss_format=self.loss_mask_format,
                compression=self.compression,
                split=split,
                compression_level=self.compression_level,
                weight_key=self.weight_key,
                include_reasoning=self.include_reasoning,
                metadata=metadata,
            )
        elif fmt_str == "parquet":
            return write_parquet_dataset(
                records_or_table=records,
                output_path=out_p,
                loss_format=self.loss_mask_format,
                schema=schema,
                compression=self.compression,
                compression_level=self.compression_level,
                row_group_size=row_group_size,
                split=split,
                metadata=metadata,
            )
        elif fmt_str == "arrow":
            return write_arrow_dataset(
                records_or_table=records,
                output_path=out_p,
                loss_format=self.loss_mask_format,
                schema=schema,
                compression=self.compression,
                split=split,
                metadata=metadata,
            )
        else:
            raise ValueError(f"Unsupported dataset format: {target_format}")

    def package_dataset(
        self,
        splits: Dict[str, List[Any]],
        output_dir: Union[str, Path],
        formats: Sequence[Union[DatasetFormat, str]] = (DatasetFormat.JSONL, DatasetFormat.PARQUET),
        shard_size: Optional[int] = None,
        write_manifest: bool = True,
        write_sha256sums: bool = True,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> CASDatasetManifest:
        """
        Package multiple splits (train, validation, eval) across target formats.
        """
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        manifest = CASDatasetManifest(
            dataset_name=self.dataset_name,
            version=self.version,
            loss_mask_format=str(self.loss_mask_format.value),
            target_format=",".join(_to_dataset_format_str(f) for f in formats),
            provenance=provenance or {},
        )

        for split_name, rows in splits.items():
            if not rows:
                continue

            shards: List[List[Any]] = []
            if shard_size and shard_size > 0 and len(rows) > shard_size:
                for i in range(0, len(rows), shard_size):
                    shards.append(rows[i : i + shard_size])
            else:
                shards.append(rows)

            for target_fmt in formats:
                fmt_str = _to_dataset_format_str(target_fmt)

                for shard_idx, shard_rows in enumerate(shards):
                    shard_suffix = f"-{shard_idx:05d}" if len(shards) > 1 else ""
                    base_name = f"{split_name}{shard_suffix}"

                    if fmt_str == "jsonl":
                        ext = ".jsonl"
                        if self.compression == CompressionCodec.ZSTD:
                            ext += ".zst"
                        elif self.compression == CompressionCodec.GZIP:
                            ext += ".gz"
                        part_path = out_root / f"{base_name}{ext}"
                    elif fmt_str == "parquet":
                        part_path = out_root / f"{base_name}.parquet"
                    elif fmt_str == "arrow":
                        part_path = out_root / f"{base_name}.arrow"
                    else:
                        continue

                    entry = self.package_partition(
                        records=shard_rows,
                        output_file=part_path,
                        target_format=fmt_str,
                        split=split_name,
                    )
                    entry.relative_path = part_path.relative_to(out_root).as_posix()
                    manifest.partitions.append(entry)

        manifest.update_totals()

        if write_manifest:
            manifest_file = out_root / "manifest.json"
            manifest.save(manifest_file)

        if write_sha256sums:
            sums_lines = []
            for p in sorted(manifest.partitions, key=lambda x: x.relative_path):
                sums_lines.append(f"{p.sha256}  {p.relative_path}")
            if write_manifest and (out_root / "manifest.json").exists():
                manifest_hash = _compute_file_sha256(out_root / "manifest.json")
                sums_lines.append(f"{manifest_hash}  manifest.json")
            (out_root / "SHA256SUMS").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

        return manifest


# ---------------------------------------------------------------------------
# Partition Readers & Verification Helpers
# ---------------------------------------------------------------------------

def read_partition(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read partition file (JSONL uncompressed/zst/gz, Parquet, or Arrow) into row dicts.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Partition file not found: {p}")

    name = p.name.lower()

    if name.endswith(".parquet"):
        if not HAS_PYARROW:
            raise RuntimeError("PyArrow is required to read Parquet files.")
        table = pq.read_table(p)
        return table.to_pylist()

    if name.endswith(".arrow") or name.endswith(".feather"):
        if not HAS_PYARROW:
            raise RuntimeError("PyArrow is required to read Arrow files.")
        with pa.OSFile(str(p), "rb") as source:
            reader = pa.ipc.open_file(source)
            table = reader.read_all()
            return table.to_pylist()

    if name.endswith(".zst") or name.endswith(".zstandard"):
        if not HAS_ZSTD:
            raise RuntimeError("zstandard package is required to read .zst files.")
        dctx = zstd.ZstdDecompressor()
        with p.open("rb") as f:
            stream_reader = dctx.stream_reader(f)
            text_io = io.TextIOWrapper(stream_reader, encoding="utf-8")
            return [json.loads(line) for line in text_io if line.strip()]

    if name.endswith(".gz") or name.endswith(".gzip"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_cas_dataset(dataset_dir: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Verify all partitions in a dataset directory against SHA-256 digests in SHA256SUMS / manifest.json.
    
    Returns:
        (is_valid, list_of_error_messages)
    """
    root = Path(dataset_dir)
    errors: List[str] = []

    sums_path = root / "SHA256SUMS"
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                errors.append(f"Invalid SHA256SUMS line: {line}")
                continue
            expected_hash, rel_path = parts
            target_path = root / rel_path
            if not target_path.exists():
                errors.append(f"Missing file declared in SHA256SUMS: {rel_path}")
                continue
            actual_hash = _compute_file_sha256(target_path)
            if actual_hash != expected_hash:
                errors.append(f"SHA-256 digest mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
            for part in manifest_dict.get("partitions", []):
                rel = part.get("relative_path")
                if not rel:
                    continue
                target = root / rel
                if not target.exists():
                    errors.append(f"Manifest partition file missing: {rel}")
                    continue
                actual_hash = _compute_file_sha256(target)
                if actual_hash != part.get("sha256"):
                    errors.append(f"Manifest partition hash mismatch for {rel}: expected {part.get('sha256')}, got {actual_hash}")
                actual_size = target.stat().st_size
                if actual_size != part.get("byte_size"):
                    errors.append(f"Manifest partition size mismatch for {rel}: expected {part.get('byte_size')}, got {actual_size}")
        except Exception as exc:
            errors.append(f"Failed to parse manifest.json: {exc}")

    return len(errors) == 0, errors


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in 1MB chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
