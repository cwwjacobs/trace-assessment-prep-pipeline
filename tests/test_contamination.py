#!/usr/bin/env python3
"""
Comprehensive test suite for ContaminationChecker and Anti-Contamination Engine.
"""

import json
import pytest
from pathlib import Path

from goldtrace_mint.factory.contamination import (
    ContaminationChecker,
    ContaminationReport,
    FlaggedSpan,
    MinHasher,
    DEFAULT_BENCHMARK_SIGNATURES,
)


class TestMinHasher:
    def test_minhasher_deterministic_signatures(self):
        hasher1 = MinHasher(num_perm=64, seed=42)
        hasher2 = MinHasher(num_perm=64, seed=42)

        ngrams = [("def", "solve", "x"), ("solve", "x", "return"), ("x", "return", "42")]
        sig1 = hasher1.compute_signature(ngrams)
        sig2 = hasher2.compute_signature(ngrams)

        assert sig1 == sig2
        assert len(sig1) == 64
        assert hasher1.estimate_similarity(sig1, sig2) == 1.0

    def test_minhasher_disjoint_sets(self):
        hasher = MinHasher(num_perm=128, seed=42)
        ngrams_a = [("alpha", "beta", "gamma", str(i)) for i in range(50)]
        ngrams_b = [("zebra", "yak", "xenon", str(i)) for i in range(50)]

        sig_a = hasher.compute_signature(ngrams_a)
        sig_b = hasher.compute_signature(ngrams_b)

        sim = hasher.estimate_similarity(sig_a, sig_b)
        assert sim < 0.1

    def test_minhasher_empty_signature(self):
        hasher = MinHasher(num_perm=32, seed=42)
        sig = hasher.compute_signature([])
        assert sig == [0] * 32
        assert hasher.estimate_similarity([], sig) == 0.0

    def test_minhasher_near_duplicate_similarity(self):
        hasher = MinHasher(num_perm=128, seed=42)
        base = [("token", str(i), "word", "item") for i in range(100)]
        # 90% overlapping
        near = base[:90] + [("extra", str(i), "variation", "item") for i in range(10)]

        sig_base = hasher.compute_signature(base)
        sig_near = hasher.compute_signature(near)

        sim = hasher.estimate_similarity(sig_base, sig_near)
        assert sim > 0.75


class TestContaminationCheckerBasics:
    @pytest.fixture
    def checker(self):
        return ContaminationChecker(ngram_size=13, overlap_threshold=0.0)

    def test_default_signatures_loaded(self, checker):
        assert "HumanEval" in checker.benchmark_signatures
        assert "MBPP" in checker.benchmark_signatures
        assert "SWE-bench" in checker.benchmark_signatures
        assert "GSM8K" in checker.benchmark_signatures
        assert len(checker.ngram_index) > 0

    def test_empty_and_short_text_is_clean(self, checker):
        report_empty = checker.check_text("")
        assert report_empty.certified_clean is True
        assert report_empty.overlap_percentage == 0.0
        assert report_empty.total_tokens == 0
        assert report_empty.flagged_spans == []

        report_short = checker.check_text("def add(a, b): return a + b")
        assert report_short.certified_clean is True
        assert report_short.overlap_percentage == 0.0
        assert report_short.total_ngrams == 0

    def test_clean_novel_code_and_math(self, checker):
        novel_text = (
            "Here is an entirely novel implementation of an asynchronous high-frequency "
            "order book matching engine with atomic ring buffers and lock-free memory pools "
            "written specifically for latency-sensitive proprietary financial execution systems."
        )
        report = checker.check_text(novel_text)
        assert report.certified_clean is True
        assert report.overlap_percentage == 0.0
        assert len(report.flagged_spans) == 0
        assert report.contaminated_ngrams == 0
        assert report.total_tokens > 20
        assert report.total_ngrams > 0


class TestBenchmarkContaminationDetection:
    @pytest.fixture
    def checker(self):
        return ContaminationChecker(ngram_size=13, overlap_threshold=0.0)

    def test_detects_humaneval_prompt(self, checker):
        humaneval_snippet = (
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n"
            "    given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    \"\"\""
        )
        report = checker.check_text(humaneval_snippet)
        assert report.certified_clean is False
        assert report.overlap_percentage > 0.0
        assert report.contaminated_ngrams > 0
        assert "HumanEval" in report.benchmark_matches
        assert len(report.flagged_spans) >= 1
        assert report.flagged_spans[0].benchmark == "HumanEval"
        assert "has_close_elements" in report.flagged_spans[0].matched_text

        # Verify character offsets match original text
        span = report.flagged_spans[0]
        extracted = humaneval_snippet[span.start_char : span.end_char]
        assert extracted == span.matched_text

    def test_detects_mbpp_prompt(self, checker):
        mbpp_snippet = (
            "Write a function to find the shared elements from the given two lists using set intersection.\n"
            "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))"
        )
        report = checker.check_text(mbpp_snippet)
        assert report.certified_clean is False
        assert report.overlap_percentage > 0.0
        assert "MBPP" in report.benchmark_matches
        assert report.flagged_spans[0].benchmark == "MBPP"

    def test_detects_swebench_prompt(self, checker):
        swe_snippet = (
            "django/django: QuerySet.aggregate() crashes when aggregating over a queryset with extra() "
            "and order_by() clauses in SQL compilation."
        )
        report = checker.check_text(swe_snippet)
        assert report.certified_clean is False
        assert report.overlap_percentage > 0.0
        assert "SWE-bench" in report.benchmark_matches

    def test_detects_gsm8k_prompt(self, checker):
        gsm_snippet = (
            "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
            "muffins for her friends every day with four. She sells the remainder at the farmers' market daily "
            "for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
        )
        report = checker.check_text(gsm_snippet)
        assert report.certified_clean is False
        assert report.overlap_percentage > 0.0
        assert "GSM8K" in report.benchmark_matches


class TestMultiTurnAndDatasetScanning:
    @pytest.fixture
    def checker(self):
        return ContaminationChecker(ngram_size=13, overlap_threshold=0.0)

    def test_check_turn_clean_vs_contaminated(self, checker):
        clean_turn = {
            "role": "user",
            "content": "Create a Python dataclass representing a celestial coordinate in the J2000 epoch.",
        }
        clean_rep = checker.check_turn(clean_turn, turn_index=0)
        assert clean_rep.certified_clean is True

        dirty_turn = {
            "role": "user",
            "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four.",
        }
        dirty_rep = checker.check_turn(dirty_turn, turn_index=1)
        assert dirty_rep.certified_clean is False
        assert dirty_rep.flagged_spans[0].role == "user"

    def test_check_turns_multi_turn_conversation(self, checker):
        turns = [
            {"role": "system", "content": "You are an expert coding assistant."},
            {"role": "user", "content": "Please implement the following function:"},
            {
                "role": "user",
                "content": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing input strings of brackets. Separate into groups of balanced brackets and return list of strings.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"",
            },
            {"role": "assistant", "content": "Here is the code to separate paren groups..."},
        ]
        report = checker.check_turns(turns)
        assert report.certified_clean is False
        assert report.scanned_turns == 4
        assert len(report.flagged_spans) >= 1
        assert any(span.turn_index == 2 for span in report.flagged_spans)

    def test_filter_clean_dataset(self, checker):
        dataset = [
            {"messages": [{"role": "user", "content": "Calculate the trajectory of a rocket in orbital mechanics."}]},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    \"\"\"",
                    }
                ]
            },
            {"messages": [{"role": "user", "content": "Design a custom shader in GLSL for water reflections."}]},
        ]

        clean, dirty = checker.filter_clean_dataset(dataset)
        assert len(clean) == 2
        assert len(dirty) == 1
        assert dirty[0][1].certified_clean is False

    def test_check_dataset_varied_formats(self, checker):
        dataset = [
            "Pure string turn with novel contents for testing dataset scanning.",
            [
                {"role": "user", "content": "Explain Fourier transform."},
                {"role": "assistant", "content": "The Fourier transform decomposes functions into frequency components."},
            ],
            {
                "events": [
                    {"type": "user.input", "payload": {"prompt": "Write a quicksort implementation in Rust."}},
                    {"type": "model.response", "payload": {"content": "fn quicksort<T: Ord>(arr: &mut [T]) {}"}},
                ]
            },
        ]
        reports = checker.check_dataset(dataset)
        assert len(reports) == 3
        assert all(r.certified_clean is True for r in reports)


class TestCustomBenchmarksAndPersistence:
    def test_add_custom_benchmark(self):
        checker = ContaminationChecker(benchmark_signatures={}, ngram_size=5)
        checker.add_benchmark("CustomEval", ["alpha beta gamma delta epsilon zeta eta theta"])

        clean_rep = checker.check_text("one two three four five six seven")
        assert clean_rep.certified_clean is True

        dirty_rep = checker.check_text("Here is alpha beta gamma delta epsilon zeta in text.")
        assert dirty_rep.certified_clean is False
        assert "CustomEval" in dirty_rep.benchmark_matches

    def test_load_benchmark_from_json_and_jsonl(self, tmp_path: Path):
        json_file = tmp_path / "custom_bench.json"
        json_file.write_text(
            json.dumps([
                {"id": "bench_01", "prompt": "custom token sequence for testing file loading integrity in benchmarks"}
            ]),
            encoding="utf-8",
        )

        jsonl_file = tmp_path / "custom_bench.jsonl"
        jsonl_file.write_text(
            json.dumps({"id": "bench_02", "prompt": "another line in jsonl file for testing multi benchmark load"}),
            encoding="utf-8",
        )

        checker = ContaminationChecker(benchmark_signatures={}, ngram_size=6)
        checker.load_benchmark_file("FileBench", json_file)
        checker.load_benchmark_file("JsonlBench", jsonl_file)

        rep1 = checker.check_text("We have custom token sequence for testing file loading integrity in benchmarks right here.")
        assert rep1.certified_clean is False
        assert "FileBench" in rep1.benchmark_matches

        rep2 = checker.check_text("This contains another line in jsonl file for testing multi benchmark load today.")
        assert rep2.certified_clean is False
        assert "JsonlBench" in rep2.benchmark_matches

    def test_tolerance_threshold(self):
        # When overlap_threshold is generous (e.g. 80%), small overlap is allowed
        checker = ContaminationChecker(ngram_size=5, overlap_threshold=80.0)
        checker.add_benchmark("ShortBench", ["alpha beta gamma delta epsilon zeta"])

        text = "This is a long sentence where alpha beta gamma delta epsilon is just a small fragment of the total tokens."
        report = checker.check_text(text)
        assert report.overlap_percentage < 80.0
        # If overlap is below threshold and minhash below threshold, certified clean is True
        assert report.overlap_percentage > 0.0

    def test_report_serialization(self):
        checker = ContaminationChecker(ngram_size=13)
        text = "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold."
        report = checker.check_text(text)

        d = report.to_dict()
        assert isinstance(d, dict)
        assert "certified_clean" in d
        assert "overlap_percentage" in d
        assert "flagged_spans" in d

        j = report.to_json()
        loaded = json.loads(j)
        assert loaded["certified_clean"] == report.certified_clean

        summary = report.summary()
        assert "Contamination Report" in summary
