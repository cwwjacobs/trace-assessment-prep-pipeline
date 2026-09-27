#!/usr/bin/env python3
"""
GTDataworks Anti-Contamination & Benchmark Integrity Engine
==========================================================
Scans training and evaluation datasets against public benchmark signatures:
  - OpenAI HumanEval (Python code synthesis)
  - MBPP (Mostly Basic Python Problems)
  - SWE-bench (Real-world GitHub issue resolution)
  - GSM8K (Grade School Math 8.5K multi-step reasoning)

Detection Algorithms:
  1. Exact 13-gram token matching & sliding window inverted index.
  2. 13-gram MinHash Jaccard similarity estimation (universal hashing).
  3. Contiguous flagged span extraction with character & token offsets.
  4. Multi-turn conversation and loss-masked trace scanning.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class FlaggedSpan:
    """Represents a detected contaminated span in text."""
    benchmark: str
    matched_text: str
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    ngram_count: int
    turn_index: Optional[int] = None
    role: Optional[str] = None
    benchmark_id: Optional[str] = None
    similarity_score: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContaminationReport:
    """Report detailing contamination analysis results."""
    certified_clean: bool
    overlap_percentage: float                   # 0.0 to 100.0%
    overlap_ratio: float                        # 0.0 to 1.0
    flagged_spans: List[FlaggedSpan] = field(default_factory=list)
    total_tokens: int = 0
    total_ngrams: int = 0
    contaminated_ngrams: int = 0
    benchmark_matches: Dict[str, int] = field(default_factory=dict)
    max_minhash_similarity: float = 0.0
    scanned_turns: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["flagged_spans"] = [span.to_dict() if isinstance(span, FlaggedSpan) else span for span in self.flagged_spans]
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        status = "CERTIFIED CLEAN" if self.certified_clean else "CONTAMINATED"
        lines = [
            f"=== Contamination Report: {status} ===",
            f"  Certified Clean     : {self.certified_clean}",
            f"  Overlap Percentage  : {self.overlap_percentage:.2f}% ({self.contaminated_ngrams}/{self.total_ngrams} n-grams)",
            f"  Total Tokens Scanned: {self.total_tokens}",
            f"  Max MinHash Jaccard : {self.max_minhash_similarity:.4f}",
            f"  Flagged Spans Count : {len(self.flagged_spans)}",
        ]
        if self.benchmark_matches:
            lines.append("  Benchmark Hits      :")
            for bm, count in self.benchmark_matches.items():
                lines.append(f"    - {bm}: {count} n-gram matches")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default Curated Public Benchmark Signatures
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARK_SIGNATURES: Dict[str, List[Dict[str, str]]] = {
    "HumanEval": [
        {
            "id": "HumanEval/0",
            "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"",
        },
        {
            "id": "HumanEval/1",
            "prompt": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing input strings of brackets. Separate into groups of balanced brackets and return list of strings.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"",
        },
        {
            "id": "HumanEval/2",
            "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"",
        },
        {
            "id": "HumanEval/3",
            "prompt": "def below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account falls below zero.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"",
        },
        {
            "id": "HumanEval/4",
            "prompt": "def mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case).\n    \"\"\"",
        },
        {
            "id": "HumanEval/5",
            "prompt": "def intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    \"\"\"",
        },
        {
            "id": "HumanEval/6",
            "prompt": "def parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\" Input to this function is a string represented multiple groups for nested parentheses separated by spaces.\n    For each group, output the deepest level of nesting of parentheses.\n    \"\"\"",
        },
        {
            "id": "HumanEval/7",
            "prompt": "def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that start with a given prefix.\n    >>> filter_by_prefix([], 'a')\n    []\n    >>> filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a')\n    ['abc', 'array']\n    \"\"\"",
        },
        {
            "id": "HumanEval/8",
            "prompt": "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n    \"\"\" For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.\n    Empty sum should be equal to 0 and empty product should be equal to 1.\n    \"\"\"",
        },
        {
            "id": "HumanEval/9",
            "prompt": "def rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\" From a given list of integers, generate a list of rolling maximum element found until given moment\n    in the sequence.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    \"\"\"",
        },
        {
            "id": "HumanEval/10",
            "prompt": "def is_palindrome(string: str) -> bool:\n    \"\"\" Test if given string is a palindrome \"\"\"\n    return string == string[::-1]\n\ndef make_palindrome(string: str) -> str:\n    \"\"\" Find the shortest palindrome that begins with a supplied string.\n    \"\"\"",
        },
        {
            "id": "HumanEval/11",
            "prompt": "def string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"",
        },
    ],
    "MBPP": [
        {
            "id": "MBPP/1",
            "prompt": "Write a function to find the shared elements from the given two lists using set intersection.\nassert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))",
        },
        {
            "id": "MBPP/2",
            "prompt": "Write a python function to identify non-prime numbers in a given list of positive integers.\nassert is_not_prime(2) == False\nassert is_not_prime(10) == True",
        },
        {
            "id": "MBPP/3",
            "prompt": "Write a function to find the longest common subsequence of two given strings.\nassert lcs('ABCDGH', 'AEDFHR') == 'ADH'",
        },
        {
            "id": "MBPP/4",
            "prompt": "Write a function to count occurrences of a character in a given string using standard python iteration.",
        },
        {
            "id": "MBPP/5",
            "prompt": "Write a function to check if the given tuple has any none value present inside it.",
        },
        {
            "id": "MBPP/6",
            "prompt": "Write a python function to remove even numbers from a given list of integers and return the odd elements.",
        },
        {
            "id": "MBPP/7",
            "prompt": "Write a function to find the minimum total path sum from top to bottom in a triangular grid.",
        },
        {
            "id": "MBPP/8",
            "prompt": "Write a function to find the surface area of a sphere with radius r using standard math formulas.",
        },
        {
            "id": "MBPP/9",
            "prompt": "Write a function to find the volume of a triangular prism given base, height, and length.",
        },
    ],
    "SWE-bench": [
        {
            "id": "SWE-bench/django-11001",
            "prompt": "django/django: QuerySet.aggregate() crashes when aggregating over a queryset with extra() and order_by() clauses in SQL compilation.",
        },
        {
            "id": "SWE-bench/sympy-13480",
            "prompt": "sympy/sympy: Matrix multiplication with empty matrix of size 0xN fails with IndexError instead of returning valid 0xM matrix.",
        },
        {
            "id": "SWE-bench/scikit-learn-10948",
            "prompt": "scikit-learn/scikit-learn: RandomForestClassifier predict_proba fails when n_classes_ is 1 in multi-output classification settings.",
        },
        {
            "id": "SWE-bench/astropy-14182",
            "prompt": "astropy/astropy: Table column slicing with negative step produces reversed column without updating header and unit metadata.",
        },
        {
            "id": "SWE-bench/pytest-dev-7220",
            "prompt": "pytest-dev/pytest: Fixture parameterization with nested ids causes KeyError in junitxml test suite report generation.",
        },
        {
            "id": "SWE-bench/matplotlib-23964",
            "prompt": "matplotlib/matplotlib: Colorbar ticks disappear when logarithmic scale is applied after calling contourf on non-linear datasets.",
        },
        {
            "id": "SWE-bench/sphinx-doc-8721",
            "prompt": "sphinx-doc/sphinx: Autodoc fails to document inherited class attributes when autodoc_inherit_docstrings configuration is enabled.",
        },
    ],
    "GSM8K": [
        {
            "id": "GSM8K/1",
            "prompt": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        },
        {
            "id": "GSM8K/2",
            "prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total are needed for 45 robes?",
        },
        {
            "id": "GSM8K/3",
            "prompt": "Josh decides to try flipping a house. He buys a house for $80,000 and spends $50,000 on repairs. He then sells it for a 150% profit of total costs. How much profit did he make?",
        },
        {
            "id": "GSM8K/4",
            "prompt": "James decides to run 3 miles every other day for 6 weeks. How many miles does he run in total during that period?",
        },
        {
            "id": "GSM8K/5",
            "prompt": "Weng earns $12 an hour for tutoring. If she tutored for 35 hours this month and spent $100 on groceries, how much money does she have left?",
        },
        {
            "id": "GSM8K/6",
            "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        },
        {
            "id": "GSM8K/7",
            "prompt": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents gave her twice as much as her parents. How much more money does Betty need to buy the wallet?",
        },
    ],
}


# ---------------------------------------------------------------------------
# MinHash Implementation
# ---------------------------------------------------------------------------

class MinHasher:
    """Computes MinHash signatures for n-gram token sets using universal hashing."""

    # Mersenne prime (2^61 - 1)
    MERSENNE_PRIME = (1 << 61) - 1

    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        self.seed = seed
        self._a, self._b = self._generate_hash_parameters(num_perm, seed)

    def _generate_hash_parameters(self, num_perm: int, seed: int) -> Tuple[List[int], List[int]]:
        rnd = random.Random(seed)
        a: List[int] = []
        b: List[int] = []
        for _ in range(num_perm):
            a_val = rnd.randint(1, self.MERSENNE_PRIME - 1)
            b_val = rnd.randint(0, self.MERSENNE_PRIME - 1)
            a.append(a_val)
            b.append(b_val)
        return a, b

    def hash_ngram(self, ngram: Tuple[str, ...]) -> int:
        """Hash an n-gram tuple to a 64-bit unsigned integer."""
        encoded = " ".join(ngram).encode("utf-8")
        return int(hashlib.sha256(encoded).hexdigest()[:16], 16)

    def compute_signature(self, ngrams: Sequence[Tuple[str, ...]]) -> List[int]:
        """Compute the MinHash signature array for a sequence of n-grams."""
        if not ngrams:
            return [0] * self.num_perm

        hashed_ngrams = [self.hash_ngram(ng) for ng in ngrams]
        sig: List[int] = []
        p = self.MERSENNE_PRIME

        for k in range(self.num_perm):
            a_k = self._a[k]
            b_k = self._b[k]
            min_val = min(((a_k * h + b_k) % p) for h in hashed_ngrams)
            sig.append(min_val)

        return sig

    @staticmethod
    def estimate_similarity(sig1: Sequence[int], sig2: Sequence[int]) -> float:
        """Estimate Jaccard similarity between two MinHash signatures."""
        if not sig1 or not sig2 or len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return matches / len(sig1)


# ---------------------------------------------------------------------------
# Contamination Checker
# ---------------------------------------------------------------------------

class ContaminationChecker:
    """
    Scans dataset turns and documents for exact n-gram and MinHash similarity overlap
    against public benchmark signatures (HumanEval, MBPP, SWE-bench, GSM8K).
    """

    def __init__(
        self,
        benchmark_signatures: Optional[Dict[str, List[Union[Dict[str, str], str]]]] = None,
        ngram_size: int = 13,
        overlap_threshold: float = 0.0,
        minhash_num_perm: int = 128,
        minhash_threshold: float = 0.8,
        seed: int = 42,
    ):
        """
        Initialize the Contamination Checker.

        :param benchmark_signatures: Optional dict of benchmark names to lists of items.
                                     If None, uses DEFAULT_BENCHMARK_SIGNATURES.
        :param ngram_size: Length of sliding window token n-grams (default 13).
        :param overlap_threshold: Maximum allowed n-gram overlap percentage (0.0 to 100.0).
        :param minhash_num_perm: Number of MinHash permutations (default 128).
        :param minhash_threshold: MinHash similarity above which a sample is flagged (default 0.8).
        :param seed: Random seed for deterministic MinHash parameters.
        """
        self.ngram_size = ngram_size
        self.overlap_threshold = overlap_threshold
        self.minhash_threshold = minhash_threshold
        self.minhasher = MinHasher(num_perm=minhash_num_perm, seed=seed)

        # Inverted index: {ngram_tuple: [(benchmark_name, item_id, ngram_index)]}
        self.ngram_index: Dict[Tuple[str, ...], List[Tuple[str, str, int]]] = {}

        # MinHash signatures per benchmark item: {benchmark_name: [(item_id, signature, text)]}
        self.benchmark_signatures: Dict[str, List[Tuple[str, List[int], str]]] = {}

        # Raw benchmark texts indexed
        self.benchmark_corpus: Dict[str, List[Dict[str, str]]] = {}

        # Load signatures
        sources = benchmark_signatures if benchmark_signatures is not None else DEFAULT_BENCHMARK_SIGNATURES
        for name, items in sources.items():
            self.add_benchmark(name, items)

    # -----------------------------------------------------------------------
    # Tokenization & N-gram Extraction
    # -----------------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> List[Tuple[str, int, int]]:
        """
        Extract normalized word/symbol tokens along with (start_char, end_char) offsets.
        Preserves character offsets for accurate span location highlighting.
        """
        if not text:
            return []
        tokens: List[Tuple[str, int, int]] = []
        for match in re.finditer(r"[A-Za-z0-9_]+|[^\w\s]", text):
            token_str = match.group(0).lower()
            tokens.append((token_str, match.start(), match.end()))
        return tokens

    def extract_ngrams(
        self, tokens: Sequence[Tuple[str, int, int]], n: Optional[int] = None
    ) -> List[Tuple[Tuple[str, ...], int, int, int, int]]:
        """
        Extract sliding window n-grams.
        Returns tuples of: (ngram_tuple, start_char, end_char, start_token_idx, end_token_idx)
        """
        n_size = n if n is not None else self.ngram_size
        if len(tokens) < n_size:
            return []

        results: List[Tuple[Tuple[str, ...], int, int, int, int]] = []
        for i in range(len(tokens) - n_size + 1):
            window = tokens[i : i + n_size]
            ngram_tuple = tuple(t[0] for t in window)
            start_char = window[0][1]
            end_char = window[-1][2]
            start_token_idx = i
            end_token_idx = i + n_size
            results.append((ngram_tuple, start_char, end_char, start_token_idx, end_token_idx))

        return results

    # -----------------------------------------------------------------------
    # Benchmark Registration & Indexing
    # -----------------------------------------------------------------------

    def add_benchmark(self, benchmark_name: str, items: Sequence[Union[Dict[str, str], str]]) -> int:
        """Register a list of benchmark items (strings or dicts) into the contamination index."""
        count = 0
        if benchmark_name not in self.benchmark_signatures:
            self.benchmark_signatures[benchmark_name] = []
            self.benchmark_corpus[benchmark_name] = []

        for idx, item in enumerate(items):
            if isinstance(item, dict):
                item_id = item.get("id") or f"{benchmark_name}/{idx}"
                text = item.get("prompt") or item.get("text") or item.get("content") or ""
            else:
                item_id = f"{benchmark_name}/{idx}"
                text = str(item)

            if not text:
                continue

            self.add_benchmark_item(benchmark_name, text, item_id=item_id)
            count += 1

        return count

    def add_benchmark_item(self, benchmark_name: str, text: str, item_id: Optional[str] = None) -> None:
        """Index a single benchmark prompt or solution item."""
        b_id = item_id or f"{benchmark_name}/{len(self.benchmark_signatures.get(benchmark_name, []))}"

        tokens = self.tokenize(text)
        ngrams_with_pos = self.extract_ngrams(tokens)
        ngrams = [ng[0] for ng in ngrams_with_pos]

        # Index n-grams
        for ng_idx, (ng_tuple, _, _, _, _) in enumerate(ngrams_with_pos):
            if ng_tuple not in self.ngram_index:
                self.ngram_index[ng_tuple] = []
            self.ngram_index[ng_tuple].append((benchmark_name, b_id, ng_idx))

        # Compute MinHash signature
        sig = self.minhasher.compute_signature(ngrams)
        if benchmark_name not in self.benchmark_signatures:
            self.benchmark_signatures[benchmark_name] = []
            self.benchmark_corpus[benchmark_name] = []

        self.benchmark_signatures[benchmark_name].append((b_id, sig, text))
        self.benchmark_corpus[benchmark_name].append({"id": b_id, "prompt": text})

    def load_benchmark_file(self, benchmark_name: str, file_path: Union[str, Path]) -> int:
        """Load benchmark signatures from a JSON or JSONL file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        items: List[Dict[str, str]] = []
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items = [it if isinstance(it, dict) else {"prompt": str(it)} for it in data]
            elif isinstance(data, dict):
                items = data.get("items") or data.get("benchmarks") or [data]

        return self.add_benchmark(benchmark_name, items)

    # -----------------------------------------------------------------------
    # Scanning & Overlap Detection
    # -----------------------------------------------------------------------

    def check_text(
        self,
        text: str,
        role: Optional[str] = None,
        turn_index: Optional[int] = None,
    ) -> ContaminationReport:
        """
        Scan a single string of text for benchmark contamination.
        Returns a detailed ContaminationReport.
        """
        tokens = self.tokenize(text)
        total_tokens = len(tokens)
        ngrams_with_pos = self.extract_ngrams(tokens)
        total_ngrams = len(ngrams_with_pos)

        if total_ngrams == 0:
            return ContaminationReport(
                certified_clean=True,
                overlap_percentage=0.0,
                overlap_ratio=0.0,
                flagged_spans=[],
                total_tokens=total_tokens,
                total_ngrams=0,
                contaminated_ngrams=0,
                benchmark_matches={},
                max_minhash_similarity=0.0,
                scanned_turns=1,
            )

        # 1. Exact N-Gram Matching
        matched_ngram_indices: Set[int] = set()
        hit_records: List[Dict[str, Any]] = []
        benchmark_counts: Dict[str, int] = {}

        for q_idx, (ng_tuple, s_char, e_char, s_tok, e_tok) in enumerate(ngrams_with_pos):
            if ng_tuple in self.ngram_index:
                matched_ngram_indices.add(q_idx)
                for bm_name, bm_id, bm_ng_idx in self.ngram_index[ng_tuple]:
                    benchmark_counts[bm_name] = benchmark_counts.get(bm_name, 0) + 1
                    hit_records.append({
                        "query_ngram_idx": q_idx,
                        "benchmark": bm_name,
                        "benchmark_id": bm_id,
                        "start_char": s_char,
                        "end_char": e_char,
                        "start_token": s_tok,
                        "end_token": e_tok,
                    })

        # 2. Merge overlapping contiguous n-grams into coherent spans
        flagged_spans: List[FlaggedSpan] = self._merge_hit_spans(
            hit_records=hit_records,
            text=text,
            role=role,
            turn_index=turn_index,
        )

        # 3. MinHash Similarity Calculation
        query_ngrams = [ng[0] for ng in ngrams_with_pos]
        query_sig = self.minhasher.compute_signature(query_ngrams)

        max_minhash_sim = 0.0
        best_match_bm = None
        best_match_id = None

        for bm_name, item_list in self.benchmark_signatures.items():
            for bm_id, bm_sig, bm_text in item_list:
                sim = self.minhasher.estimate_similarity(query_sig, bm_sig)
                if sim > max_minhash_sim:
                    max_minhash_sim = sim
                    best_match_bm = bm_name
                    best_match_id = bm_id

        # Flag full document if MinHash similarity exceeds high threshold
        if max_minhash_sim >= self.minhash_threshold and not flagged_spans:
            flagged_spans.append(
                FlaggedSpan(
                    benchmark=best_match_bm or "Unknown",
                    matched_text=text[:200] + "..." if len(text) > 200 else text,
                    start_char=0,
                    end_char=len(text),
                    start_token=0,
                    end_token=total_tokens,
                    ngram_count=total_ngrams,
                    turn_index=turn_index,
                    role=role,
                    benchmark_id=best_match_id,
                    similarity_score=max_minhash_sim,
                    details={"detection_method": "minhash_jaccard_high_similarity"},
                )
            )

        contaminated_ngrams = len(matched_ngram_indices)
        overlap_ratio = (contaminated_ngrams / total_ngrams) if total_ngrams > 0 else 0.0
        overlap_percentage = round(overlap_ratio * 100.0, 4)

        # Certified Clean determination
        certified_clean = (
            overlap_percentage <= self.overlap_threshold
            and max_minhash_sim < self.minhash_threshold
            and len(flagged_spans) == 0
        )

        return ContaminationReport(
            certified_clean=certified_clean,
            overlap_percentage=overlap_percentage,
            overlap_ratio=overlap_ratio,
            flagged_spans=flagged_spans,
            total_tokens=total_tokens,
            total_ngrams=total_ngrams,
            contaminated_ngrams=contaminated_ngrams,
            benchmark_matches=benchmark_counts,
            max_minhash_similarity=max_minhash_sim,
            scanned_turns=1,
            metadata={
                "ngram_size": self.ngram_size,
                "overlap_threshold": self.overlap_threshold,
                "minhash_threshold": self.minhash_threshold,
            },
        )

    def _merge_hit_spans(
        self,
        hit_records: List[Dict[str, Any]],
        text: str,
        role: Optional[str] = None,
        turn_index: Optional[int] = None,
    ) -> List[FlaggedSpan]:
        """Merge consecutive/adjacent matching n-grams into unified flagged spans."""
        if not hit_records:
            return []

        # Group by benchmark
        by_bm: Dict[str, List[Dict[str, Any]]] = {}
        for hit in hit_records:
            bm = hit["benchmark"]
            if bm not in by_bm:
                by_bm[bm] = []
            by_bm[bm].append(hit)

        spans: List[FlaggedSpan] = []

        for bm, records in by_bm.items():
            # Sort by query_ngram_idx
            records.sort(key=lambda r: r["query_ngram_idx"])
            current_start_char = records[0]["start_char"]
            current_end_char = records[0]["end_char"]
            current_start_tok = records[0]["start_token"]
            current_end_tok = records[0]["end_token"]
            current_bm_id = records[0]["benchmark_id"]
            current_count = 1
            last_ngram_idx = records[0]["query_ngram_idx"]

            for rec in records[1:]:
                ngram_idx = rec["query_ngram_idx"]
                if ngram_idx <= last_ngram_idx + 1:
                    # Adjacent or consecutive n-gram overlap
                    current_end_char = max(current_end_char, rec["end_char"])
                    current_end_tok = max(current_end_tok, rec["end_token"])
                    if ngram_idx > last_ngram_idx:
                        current_count += 1
                    last_ngram_idx = ngram_idx
                else:
                    # Emit previous span
                    matched_str = text[current_start_char:current_end_char]
                    spans.append(
                        FlaggedSpan(
                            benchmark=bm,
                            matched_text=matched_str,
                            start_char=current_start_char,
                            end_char=current_end_char,
                            start_token=current_start_tok,
                            end_token=current_end_tok,
                            ngram_count=current_count,
                            turn_index=turn_index,
                            role=role,
                            benchmark_id=current_bm_id,
                            similarity_score=1.0,
                            details={"detection_method": "exact_13gram_match"},
                        )
                    )
                    # Start new span
                    current_start_char = rec["start_char"]
                    current_end_char = rec["end_char"]
                    current_start_tok = rec["start_token"]
                    current_end_tok = rec["end_token"]
                    current_bm_id = rec["benchmark_id"]
                    current_count = 1
                    last_ngram_idx = ngram_idx

            # Emit final span for benchmark
            matched_str = text[current_start_char:current_end_char]
            spans.append(
                FlaggedSpan(
                    benchmark=bm,
                    matched_text=matched_str,
                    start_char=current_start_char,
                    end_char=current_end_char,
                    start_token=current_start_tok,
                    end_token=current_end_tok,
                    ngram_count=current_count,
                    turn_index=turn_index,
                    role=role,
                    benchmark_id=current_bm_id,
                    similarity_score=1.0,
                    details={"detection_method": "exact_13gram_match"},
                )
            )

        return spans

    # -----------------------------------------------------------------------
    # High-level Dataset & Turn Scanning
    # -----------------------------------------------------------------------

    def check_turn(self, turn: Dict[str, Any], turn_index: int = 0) -> ContaminationReport:
        """Scan a single conversation turn or event payload."""
        role = turn.get("role") or turn.get("speaker") or "user"
        content = turn.get("content") or turn.get("prompt") or turn.get("text") or ""
        reasoning = turn.get("reasoning_content") or turn.get("reasoning") or ""

        full_text = content
        if reasoning:
            full_text = f"{reasoning}\n{content}"

        return self.check_text(full_text, role=role, turn_index=turn_index)

    def check_turns(self, turns: Sequence[Dict[str, Any]]) -> ContaminationReport:
        """
        Scan a sequence of conversation turns (e.g. OpenAI messages format or Labyrinth traces).
        Aggregates token counts, overlap ratios, and flagged spans across all turns.
        """
        all_flagged: List[FlaggedSpan] = []
        total_tokens = 0
        total_ngrams = 0
        contaminated_ngrams = 0
        benchmark_counts: Dict[str, int] = {}
        max_minhash = 0.0

        for idx, turn in enumerate(turns):
            # Extract role and content
            role = turn.get("role") or turn.get("speaker") or "user"
            content = turn.get("content") or turn.get("prompt") or turn.get("text") or ""
            reasoning = turn.get("reasoning_content") or turn.get("reasoning") or ""

            # Check if there are nested tool calls or arguments
            tool_calls = turn.get("tool_calls") or []
            tool_text = ""
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    tool_text += " " + str(fn.get("name") or "") + " " + str(fn.get("arguments") or "")

            combined_text = f"{content} {reasoning} {tool_text}".strip()
            if not combined_text:
                continue

            report = self.check_text(combined_text, role=role, turn_index=idx)
            total_tokens += report.total_tokens
            total_ngrams += report.total_ngrams
            contaminated_ngrams += report.contaminated_ngrams
            if report.max_minhash_similarity > max_minhash:
                max_minhash = report.max_minhash_similarity

            for bm, count in report.benchmark_matches.items():
                benchmark_counts[bm] = benchmark_counts.get(bm, 0) + count

            all_flagged.extend(report.flagged_spans)

        overall_ratio = (contaminated_ngrams / total_ngrams) if total_ngrams > 0 else 0.0
        overall_pct = round(overall_ratio * 100.0, 4)

        certified_clean = (
            overall_pct <= self.overlap_threshold
            and max_minhash < self.minhash_threshold
            and len(all_flagged) == 0
        )

        return ContaminationReport(
            certified_clean=certified_clean,
            overlap_percentage=overall_pct,
            overlap_ratio=overall_ratio,
            flagged_spans=all_flagged,
            total_tokens=total_tokens,
            total_ngrams=total_ngrams,
            contaminated_ngrams=contaminated_ngrams,
            benchmark_matches=benchmark_counts,
            max_minhash_similarity=max_minhash,
            scanned_turns=len(turns),
            metadata={
                "ngram_size": self.ngram_size,
                "overlap_threshold": self.overlap_threshold,
                "minhash_threshold": self.minhash_threshold,
            },
        )

    def check_dataset(self, dataset: Sequence[Any]) -> List[ContaminationReport]:
        """Scan a dataset of items (strings, turn lists, or trace dicts)."""
        reports: List[ContaminationReport] = []
        for item in dataset:
            if isinstance(item, str):
                reports.append(self.check_text(item))
            elif isinstance(item, list):
                reports.append(self.check_turns(item))
            elif isinstance(item, dict):
                if "messages" in item and isinstance(item["messages"], list):
                    reports.append(self.check_turns(item["messages"]))
                elif "events" in item and isinstance(item["events"], list):
                    # Convert event trace
                    reports.append(self.check_turns(item["events"]))
                else:
                    reports.append(self.check_turn(item))
            else:
                reports.append(self.check_text(str(item)))
        return reports

    def filter_clean_dataset(
        self, dataset: Sequence[Any]
    ) -> Tuple[List[Any], List[Tuple[Any, ContaminationReport]]]:
        """
        Partition dataset into clean items and contaminated items with audit reports.
        """
        clean_items: List[Any] = []
        flagged_items: List[Tuple[Any, ContaminationReport]] = []

        for item in dataset:
            if isinstance(item, str):
                rep = self.check_text(item)
            elif isinstance(item, list):
                rep = self.check_turns(item)
            elif isinstance(item, dict) and "messages" in item:
                rep = self.check_turns(item["messages"])
            elif isinstance(item, dict):
                rep = self.check_turn(item)
            else:
                rep = self.check_text(str(item))

            if rep.certified_clean:
                clean_items.append(item)
            else:
                flagged_items.append((item, rep))

        return clean_items, flagged_items
