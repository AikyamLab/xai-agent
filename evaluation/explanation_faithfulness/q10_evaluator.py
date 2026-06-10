"""
Q10 Evaluator: Similar Instances with Different Predictions (correct vs wrong)

Metric: 1 - Sim(F_correct, F_wrong)
Higher score = better (features for correct vs wrong instance should be distinct)

Similarity measures:
- Vision: Jaccard word similarity on feature phrases
- Text: Word overlap between extracted spans
- Tabular: Jaccard overlap weighted by rank agreement on top_features lists
"""

from typing import Any, Dict, List, Set
import re

from ..base_evaluator import MultiInstanceEvaluator, EvaluationResult

threshold = 0.3

class Q10Evaluator(MultiInstanceEvaluator):
    """Evaluator for Q10: Similar instances with different predictions"""

    @property
    def question_type(self) -> int:
        return 10

    @property
    def metric_name(self) -> str:
        return "Feature Distinctness"

    @property
    def metric_formula(self) -> str:
        return "1-Sim(F_correct, F_wrong)"

    def evaluate_multi(
        self,
        agent_output: Dict[str, Any],
        inputs: List[Any],
        model: Any,
        predictions: List[Dict[str, Any]],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q10: Compare feature similarity between correct and wrong instance.

        Lower similarity = better explanation (features are distinct).

        Args:
            agent_output: Agent output with correct/wrong instance features
            inputs: List of input data (strings, dicts, or tensors)
            model: Target model
            predictions: List of prediction dicts
            **kwargs: Additional arguments

        Returns:
            EvaluationResult with negative similarity score
        """
        output_data = agent_output.get('output', {})

        # Get features for correct and wrong instances
        correct_features = output_data.get('correct_instance_features')
        wrong_features = output_data.get('wrong_instance_features')

        # Normalize: baseline agents may output a plain list instead of {"feature_keys": [...]}
        if isinstance(correct_features, list):
            correct_features = {"feature_keys": correct_features}
        if isinstance(wrong_features, list):
            wrong_features = {"feature_keys": wrong_features}

        if correct_features is None or wrong_features is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                errors=["Features not provided for both instances"]
            )

        # Validate agent output: bad responses → score 0.0
        if self.modality == "vision":
            # Q10 vision output is a concise feature phrase (string), not a bounding box
            for label, feat in [("correct_instance", correct_features),
                                 ("wrong_instance", wrong_features)]:
                text = str(feat).lower().strip()
                if not text or text in ('none', 'null', '{}', '[]'):
                    return EvaluationResult(score=0.0, passed=False,
                                            metric_name=self.metric_name,
                                            metric_formula=self.metric_formula,
                                            errors=[f"Empty vision phrase for {label}"])
                if any(m in text for m in self._REFUSAL_MARKERS):
                    return EvaluationResult(score=0.0, passed=False,
                                            metric_name=self.metric_name,
                                            metric_formula=self.metric_formula,
                                            errors=[f"Refusal marker in {label} vision phrase"])
        else:
            # text/tabular: features are dicts in standard region format.
            # For multi-instance Q10, the agent may pick spans from ANY of the N
            # inputs (not just inputs[0]/inputs[1]), so validate against the full
            # combined text of all inputs.
            if self.modality == "text":
                def _to_text(inp) -> str:
                    if isinstance(inp, str):
                        return inp
                    if isinstance(inp, dict):
                        return (inp.get('text', '')
                                + ' ' + inp.get('premise', '')
                                + ' ' + inp.get('hypothesis', ''))
                    return str(inp) if inp is not None else ''
                combined_input = " ".join(_to_text(inp) for inp in inputs)
                for label, feat in [("correct_instance", correct_features),
                                     ("wrong_instance", wrong_features)]:
                    err = self.validate_region(feat, combined_input, **kwargs)
                    if err:
                        return EvaluationResult(score=0.0, passed=False,
                                                metric_name=self.metric_name,
                                                metric_formula=self.metric_formula,
                                                errors=[f"{label}: {err}"])
            else:
                inp_a = inputs[0] if inputs else None
                inp_b = inputs[1] if len(inputs) > 1 else None
                for label, feat, inp in [("correct_instance", correct_features, inp_a),
                                          ("wrong_instance", wrong_features, inp_b)]:
                    err = self.validate_region(feat, inp, **kwargs)
                    if err:
                        return EvaluationResult(score=0.0, passed=False,
                                                metric_name=self.metric_name,
                                                metric_formula=self.metric_formula,
                                                errors=[f"{label}: {err}"])

        # Calculate similarity based on modality
        if self.modality == "vision":
            similarity = self._compute_text_similarity(
                str(correct_features),
                str(wrong_features)
            )
        elif self.modality == "text":
            similarity = self._compute_span_overlap(
                correct_features,
                wrong_features,
                inputs
            )
        else:  # tabular
            similarity = self._compute_feature_similarity(
                correct_features,
                wrong_features
            )

        # Soft score: 1 - similarity, range [0, 1], higher = more distinct
        score = 1.0 - similarity

        # Passed logic unchanged: similarity below threshold
        passed = similarity < threshold

        return EvaluationResult(
            score=score,
            passed=passed,
            metric_name=self.metric_name,
            metric_formula=self.metric_formula,
            details={
                "soft_score": score,
                "size_score": score,
                "correct_features": correct_features,
                "wrong_features": wrong_features,
                "similarity": similarity,
                "threshold": threshold,
                "interpretation": "Soft score: 1 - similarity, range [0,1], higher = more distinct"
            }
        )

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute word-level similarity between two text descriptions.

        Uses Jaccard similarity on word sets.
        """
        # Tokenize and normalize
        words1 = self._tokenize(text1)
        words2 = self._tokenize(text2)

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into word set"""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'and', 'but', 'or',
                      'that', 'this', 'these', 'those', 'it', 'its'}
        return set(w for w in words if w not in stop_words)

    def _extract_text(self, inp: Any) -> str:
        """Extract plain text from an input that may be a string, dict, or other type."""
        if isinstance(inp, str):
            return inp
        if isinstance(inp, dict):
            # NLI format: combine premise + hypothesis
            if 'premise' in inp:
                return f"{inp.get('premise', '')} {inp.get('hypothesis', '')}"
            if 'text' in inp:
                return inp['text']
            if 'review_text' in inp:
                return inp['review_text']
        return str(inp) if inp else ""

    def _compute_span_overlap(
        self,
        feat1: Dict,
        feat2: Dict,
        inputs: List
    ) -> float:
        """
        Compute overlap between text features from correct/wrong instances.

        Expects text_spans format: {"text_spans": ["phrase1", ...]}.
        Joins phrase strings and computes word similarity directly.
        """
        text_spans1 = feat1.get('text_spans')
        text_spans2 = feat2.get('text_spans')
        if not text_spans1 or not text_spans2:
            return 1.0  # max similarity → score 0.0

        text1 = " ".join(s for s in text_spans1 if isinstance(s, str))
        text2 = " ".join(s for s in text_spans2 if isinstance(s, str))
        return self._compute_text_similarity(text1, text2)

    @staticmethod
    def _strip_value_annotation(feat: str) -> str:
        """Strip parenthesized value annotations from feature strings.

        e.g. 'mean_compactness (0.03834-0.08468)' -> 'mean_compactness'
        """
        return re.sub(r'\s*\(.*?\)\s*$', '', str(feat)).strip()

    def _compute_feature_similarity(
        self,
        features1: Dict,
        features2: Dict
    ) -> float:
        """
        Compute similarity between tabular feature rankings.

        Uses Jaccard overlap weighted by rank agreement on shared features.
        - Jaccard measures how many features overlap between two top-k lists
        - Rank agreement measures whether shared features have similar ranks
        - Final similarity = Jaccard * rank_agreement (0 if no overlap)

        Feature strings may include value annotations like 'mean_compactness (0.123)'
        which are stripped before comparison so the same feature name matches.
        """
        if not isinstance(features1, dict) or not isinstance(features2, dict):
            return 1.0  # max similarity → score 0.0

        list1 = features1.get('feature_keys') or features1.get('top_features', [])
        list2 = features2.get('feature_keys') or features2.get('top_features', [])

        if not list1 or not list2:
            return 1.0  # max similarity → score 0.0

        # Normalize: strip value annotations before comparison
        norm1 = [self._strip_value_annotation(f) for f in list1]
        norm2 = [self._strip_value_annotation(f) for f in list2]

        set1, set2 = set(norm1), set(norm2)
        intersection = set1 & set2
        union = set1 | set2

        jaccard = len(intersection) / len(union) if union else 0.0

        if not intersection:
            return 0.0

        # Rank agreement: average of 1/(1+|rank_diff|) for shared features
        rank_scores = []
        for feat in intersection:
            r1 = norm1.index(feat) + 1
            r2 = norm2.index(feat) + 1
            rank_scores.append(1.0 / (1.0 + abs(r1 - r2)))
        rank_agreement = sum(rank_scores) / len(rank_scores)

        return jaccard * rank_agreement
