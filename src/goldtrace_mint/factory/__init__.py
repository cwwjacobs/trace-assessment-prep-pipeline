"""Factory components for Gold Trace Mint."""

__all__ = []

try:
    from .contamination import (
        ContaminationChecker,
        ContaminationReport,
        FlaggedSpan,
        MinHasher,
    )
    __all__.extend(["ContaminationChecker", "ContaminationReport", "FlaggedSpan", "MinHasher"])
except ImportError:
    pass

try:
    from .deduplicator import (
        CurationSummary,
        DeduplicationRecord,
        SeedCurator,
        extract_seed_canonical_text,
    )
    __all__.extend(["CurationSummary", "DeduplicationRecord", "SeedCurator", "extract_seed_canonical_text"])
except ImportError:
    pass

try:
    from .loss_masking import (
        MaskedMessage,
        export_loss_masked_jsonl,
        format_trace_with_loss_mask,
    )
    __all__.extend(["MaskedMessage", "export_loss_masked_jsonl", "format_trace_with_loss_mask"])
except ImportError:
    pass

try:
    from .sanitizer import (
        ComplianceAuditor,
        LicenseCategory,
        LicenseVerificationResult,
        LicenseVerifier,
        PIICategory,
        PIIFinding,
        PIISanitizer,
        RepositoryLicenseReport,
        SanitizationReceipt,
        SensitivePhraseEntry,
        SensitivePhraseRegistry,
        redact_pii,
        sanitize_file,
        scan_pii,
        verify_license,
        verify_repository_license,
    )
    __all__.extend([
        "ComplianceAuditor",
        "LicenseCategory",
        "LicenseVerificationResult",
        "LicenseVerifier",
        "PIICategory",
        "PIIFinding",
        "PIISanitizer",
        "RepositoryLicenseReport",
        "SanitizationReceipt",
        "SensitivePhraseEntry",
        "SensitivePhraseRegistry",
        "redact_pii",
        "sanitize_file",
        "scan_pii",
        "verify_license",
        "verify_repository_license",
    ])
except ImportError:
    pass

try:
    from .datacard import DataCardGenerator, TokenStatistics
    __all__.extend(["DataCardGenerator", "TokenStatistics"])
except ImportError:
    pass

try:
    from .packager import (
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
    )
    __all__.extend([
        "CASDatasetManifest",
        "CompressionCodec",
        "DatasetFormat",
        "DatasetPackager",
        "LossMaskingFormat",
        "PartitionManifestEntry",
        "build_arrow_table_from_records",
        "format_for_anthropic",
        "format_for_openai_chat",
        "format_for_sharegpt",
        "get_anthropic_arrow_schema",
        "get_openai_chat_arrow_schema",
        "get_sharegpt_arrow_schema",
        "normalize_record_for_format",
        "read_partition",
        "verify_cas_dataset",
        "write_arrow_dataset",
        "write_loss_masked_jsonl",
        "write_parquet_dataset",
    ])
except ImportError:
    pass

try:
    from .assay_report import (
        AssayReport,
        AssayReportGenerator,
        BenchmarkCaseComparison,
        LiftMetric,
        TaskLiftBreakdown,
        export_assay_report,
        generate_assay_report,
    )
    __all__.extend([
        "AssayReport",
        "AssayReportGenerator",
        "BenchmarkCaseComparison",
        "LiftMetric",
        "TaskLiftBreakdown",
        "export_assay_report",
        "generate_assay_report",
    ])
except ImportError:
    pass
