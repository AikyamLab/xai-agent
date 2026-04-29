"""
XAI RL Dataset

Loads XAI question instances from the dataset directory structure:

    dataset/{mode}/{modality}/{dataset_name}_q{q_type}.json

where:
  mode        = "train" | "test"
  modality    = "tabular" | "vision" | "text"  (auto-inferred from dataset_name)
  dataset_name = e.g. "adult_2layernn", "stl10_resnet", "imdb_cnn"
  q_type      = 1 … 10

Dataset name → modality mapping
  adult_*, cancer_*          → tabular
  stl10_*, cub_*             → vision
  imdb_*, snli_*             → text

Provides batches for GRPO training (same format as xai_pipeline_v2.py).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ── Modality inference ────────────────────────────────────────────────────────

def parse_data_range(s: str) -> Optional[Tuple[int, int]]:
    """
    Parse a per-dataset data-range string.

    "all"      → None         (no limit; use every question in the dataset)
    "0-200"    → (0, 200)     (questions at interleaved indices 0 .. 199)
    "50-150"   → (50, 150)

    The range is applied AFTER round-robin interleaving across Q-types so
    that the slice is always balanced:  "0-100" with 5 Q-types and 40
    questions each gives exactly 20 questions per Q-type (100 total per
    dataset), not a lopsided cut that drops later Q-types entirely.

    Raises:
        ValueError: for unrecognised format or end <= start.
    """
    import re as _re
    s = s.strip()
    if s.lower() == "all":
        return None
    m = _re.match(r"^(\d+)-(\d+)$", s)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if end <= start:
            raise ValueError(
                f"data_range end ({end}) must be greater than start ({start})"
            )
        return start, end
    raise ValueError(
        f"Unrecognised data_range {s!r}. "
        f"Use 'all' or 'START-END' (e.g. '0-100', '0-200')."
    )


_MODALITY_PREFIXES = {
    "tabular": ["adult", "cancer"],
    "vision":  ["stl10", "cub"],
    "text":    ["imdb", "snli"],
}

def infer_modality(dataset_name: str) -> str:
    """
    Infer modality from dataset_name prefix.

    Raises:
        ValueError: if no prefix matches
    """
    name = dataset_name.lower()
    for modality, prefixes in _MODALITY_PREFIXES.items():
        if any(name.startswith(p) for p in prefixes):
            return modality
    raise ValueError(
        f"Cannot infer modality from dataset_name={dataset_name!r}. "
        f"Known prefixes: {_MODALITY_PREFIXES}"
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class XAIRLDataset:
    """
    Dataset of XAI questions for GRPO training.

    Args:
        questions:   List of question dicts (pipeline JSON format)
        shuffle:     Whether to shuffle on each get_batches() call
        seed:        Random seed
        filter_fn:   Optional callable(question) -> bool
    """
    questions: List[Dict[str, Any]]
    shuffle: bool = True
    seed: int = 42
    filter_fn: Optional[Any] = None

    def __post_init__(self):
        if self.filter_fn is not None:
            self.questions = [q for q in self.questions if self.filter_fn(q)]
        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.questions[idx]

    def get_batches(self, batch_size: int) -> List[List[Dict[str, Any]]]:
        """
        Split questions into batches, shuffling at the start of each call (epoch).
        """
        questions = list(self.questions)
        if self.shuffle:
            self._rng.shuffle(questions)
        return [questions[i: i + batch_size] for i in range(0, len(questions), batch_size)]

    def get_stratified_batches(self, batch_size: int) -> List[List[Dict[str, Any]]]:
        """
        Split questions into batches with stratified (modality, Q-type) sampling.

        Each batch contains as even a mix of modality×Q-type combinations as
        possible by interleaving groups in round-robin order before slicing into
        fixed-size batches.  Each group is independently shuffled so question
        order within a group is still random across epochs.

        For single-modality training the modality prefix is constant, so this
        degenerates to the previous Q-type-only stratification with no behaviour
        change.  For mixed-modality training it guarantees every batch sees all
        three modalities.
        """
        from collections import defaultdict

        # Group by (modality, q_type) so mixed-modality batches are balanced
        # across both dimensions.  Single-modality runs have a single modality
        # value, so behaviour is identical to the previous q_type-only grouping.
        groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for q in self.questions:
            key = (q.get("modality", ""), q.get("q_type", 0))
            groups[key].append(q)

        # Shuffle each group independently
        shuffled: Dict[tuple, List[Dict[str, Any]]] = {}
        for key, qs in groups.items():
            qs_copy = list(qs)
            if self.shuffle:
                self._rng.shuffle(qs_copy)
            shuffled[key] = qs_copy

        # Round-robin interleave across (modality, q_type) groups
        sorted_types = sorted(shuffled.keys())
        iters = {qt: iter(shuffled[qt]) for qt in sorted_types}
        interleaved: List[Dict[str, Any]] = []
        while True:
            added = False
            for qt in sorted_types:
                try:
                    interleaved.append(next(iters[qt]))
                    added = True
                except StopIteration:
                    pass
            if not added:
                break

        return [interleaved[i: i + batch_size] for i in range(0, len(interleaved), batch_size)]

    # ── Factory: from dataset_name + q_types + mode ───────────────────────────

    @classmethod
    def from_dataset_name(
        cls,
        dataset_name: str,
        mode: str,
        q_types: Optional[List[int]] = None,
        modality: Optional[str] = None,
        base_dir: Optional[str] = None,
        max_questions: Optional[int] = None,
        data_range: Optional[Tuple[int, int]] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> "XAIRLDataset":
        """
        Load questions by dataset_name, mode, and q_types.

        File path pattern:
            {base_dir}/dataset/{mode}/{modality}/{dataset_name}_q{q_type}.json

        Args:
            dataset_name:  e.g. "adult_2layernn", "stl10_resnet", "imdb_cnn"
            mode:          "train" or "test"
            q_types:       Question types to load. None → load all Q1-Q10.
            modality:      If None, auto-inferred from dataset_name.
            base_dir:      Project root. Defaults to two levels above this file.
            max_questions: Cap total questions loaded (for debugging).
            data_range:    Per-dataset slice as (start, end) int tuple.
                           Applied after round-robin Q-type interleaving so
                           the slice is balanced across Q-types.
                           None → no limit (same as "all").
                           Use parse_data_range() to convert a string flag.
            shuffle:       Shuffle on get_batches().
            seed:          Random seed.

        Returns:
            XAIRLDataset with merged questions from all requested q_types.

        Example::

            ds = XAIRLDataset.from_dataset_name(
                dataset_name="adult_2layernn",
                mode="train",
                q_types=[1, 2, 3],
            )
        """
        if mode not in ("train", "test"):
            raise ValueError(f"mode must be 'train' or 'test', got {mode!r}")

        if modality is None:
            modality = infer_modality(dataset_name)

        if base_dir is None:
            dataset_root = Path(__file__).parents[2] / "dataset" / mode / modality
        else:
            # base_dir is the dataset directory itself (e.g. dataset_unused/)
            dataset_root = Path(base_dir) / mode / modality

        if q_types is None:
            q_types = list(range(1, 11))

        all_questions: List[Dict[str, Any]] = []
        missing: List[str] = []

        for qt in q_types:
            json_path = dataset_root / f"{dataset_name}_q{qt}.json"
            if not json_path.exists():
                missing.append(str(json_path))
                continue
            questions = _load_json_questions(
                json_path, modality=modality, q_type=qt, dataset_name=dataset_name
            )
            all_questions.extend(questions)

        if missing:
            print(f"  [XAIRLDataset] Missing files (skipped): {missing}")

        print(
            f"  [XAIRLDataset] {dataset_name} mode={mode} q_types={q_types}: "
            f"loaded {len(all_questions)} questions from {modality}/"
        )

        if data_range is not None:
            # Interleave across Q-types before slicing so the range is
            # balanced.  Without this, "0-100" on a dataset with 5 Q-types
            # × 40 questions would grab Q1(40)+Q3(40)+Q5(20) and miss Q6/Q7
            # entirely.  With interleaving it gives exactly 20 per Q-type.
            by_qt: Dict[int, List] = defaultdict(list)
            for q in all_questions:
                by_qt[q.get("q_type", 0)].append(q)
            sorted_types = sorted(by_qt.keys())
            iters = {qt: iter(by_qt[qt]) for qt in sorted_types}
            interleaved: List[Dict[str, Any]] = []
            while True:
                added = False
                for qt in sorted_types:
                    try:
                        interleaved.append(next(iters[qt]))
                        added = True
                    except StopIteration:
                        pass
                if not added:
                    break
            start, end = data_range
            all_questions = interleaved[start:end]

        if max_questions is not None:
            all_questions = all_questions[:max_questions]

        return cls(questions=all_questions, shuffle=shuffle, seed=seed)

    @classmethod
    def from_multiple_datasets(
        cls,
        dataset_names: List[str],
        mode: str,
        q_types: Optional[List[int]] = None,
        per_dataset_q_types: Optional[Dict[str, List[int]]] = None,
        base_dir: Optional[str] = None,
        max_questions: Optional[int] = None,
        data_range: Optional[Tuple[int, int]] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> "XAIRLDataset":
        """
        Load and merge questions from multiple dataset_names.

        Useful for training across multiple datasets / modalities simultaneously.
        ``data_range`` is applied independently per dataset before merging, so
        each dataset contributes at most ``end - start`` questions (balanced
        across Q-types).

        Args:
            per_dataset_q_types: Optional dict mapping dataset_name to a list of
                q_types, overriding the global ``q_types`` for that dataset.
                Example: {"cancer_tabnn": [1,3,5,6,7,9], "cancer_2layernn": [1,3,5,6,7,9]}

        Example::

            ds = XAIRLDataset.from_multiple_datasets(
                dataset_names=["adult_2layernn", "adult_tabnn", "cancer_2layernn"],
                mode="train",
                q_types=[1, 2, 3],
                data_range=(0, 100),   # first 100 from each dataset
            )
        """
        all_questions: List[Dict[str, Any]] = []
        for name in dataset_names:
            effective_q_types = (per_dataset_q_types or {}).get(name, q_types)
            sub = cls.from_dataset_name(
                dataset_name=name,
                mode=mode,
                q_types=effective_q_types,
                base_dir=base_dir,
                data_range=data_range,
                shuffle=False,
                seed=seed,
            )
            all_questions.extend(sub.questions)

        if max_questions is not None:
            all_questions = all_questions[:max_questions]

        print(
            f"  [XAIRLDataset] Combined {len(dataset_names)} datasets, "
            f"total {len(all_questions)} questions"
        )
        return cls(questions=all_questions, shuffle=shuffle, seed=seed)

    @classmethod
    def from_json(
        cls,
        json_path: str,
        modality: Optional[str] = None,
        q_types: Optional[List[int]] = None,
        max_questions: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> "XAIRLDataset":
        """
        Load questions from a single JSON file (legacy interface).

        Args:
            json_path: Path to dataset JSON (list of question dicts)
            modality:  Modality tag to inject into questions lacking it
            q_types:   Filter to these question types
            max_questions: Cap total
            shuffle:   Shuffle on get_batches()
            seed:      Random seed
        """
        questions = _load_json_questions(
            Path(json_path), modality=modality, q_type=None
        )
        if q_types:
            questions = [q for q in questions if q.get("q_type") in q_types]
        if max_questions is not None:
            questions = questions[:max_questions]
        print(f"  [XAIRLDataset] Loaded {len(questions)} questions from {json_path}")
        return cls(questions=questions, shuffle=shuffle, seed=seed)

    # ── Train / eval split ────────────────────────────────────────────────────

    def split(self, eval_ratio: float = 0.1, seed: int = 42):
        """
        Return (train_dataset, eval_dataset) with a random split.
        """
        rng = random.Random(seed)
        questions = list(self.questions)
        rng.shuffle(questions)

        n_eval = max(1, int(len(questions) * eval_ratio))
        eval_q  = questions[:n_eval]
        train_q = questions[n_eval:]

        train_ds = XAIRLDataset(questions=train_q, shuffle=self.shuffle, seed=self.seed)
        eval_ds  = XAIRLDataset(questions=eval_q,  shuffle=False,        seed=self.seed)
        print(f"  [XAIRLDataset] Split: {len(train_ds)} train / {len(eval_ds)} eval")
        return train_ds, eval_ds


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_json_questions(
    path: Path,
    modality: Optional[str],
    q_type: Optional[int],
    dataset_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load and normalise question dicts from a JSON file.

    Injects `modality`, `q_type`, and `dataset_name` if absent from individual
    records. `dataset_name` (e.g. "adult_2layernn") is used by setup_question
    in train.py to look up the correct DataModelLoader module.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        questions = data.get("questions", [])
    else:
        return []

    for idx, q in enumerate(questions):
        if modality and not q.get("modality"):
            q["modality"] = modality
        if q_type is not None and not q.get("q_type"):
            q["q_type"] = q_type
        if dataset_name:
            if not q.get("dataset_name"):
                q["dataset_name"] = dataset_name
            # Agents use "dataset_base_name" for output directory naming
            if not q.get("dataset_base_name"):
                q["dataset_base_name"] = dataset_name
        # Store source info so XAIRLEnv can call pipeline.run(source_path, idx)
        q["_source_path"]   = str(path)
        q["_question_idx"]  = idx

    return questions
