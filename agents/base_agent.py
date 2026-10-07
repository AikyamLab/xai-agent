"""
Base Agent class for XAI Agent Framework

Provides common functionality shared across all agents.
"""

from __future__ import annotations
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _fix_bare_array_objects(text: str) -> str:
    """
    Fix arrays whose elements are bare key-value pairs (VLM forgot opening/closing {}).

    VLMs occasionally emit:
        "findings": [
            "bounding_box": [1, 0, 224, 224],
            "description": "..."
        ]
    instead of:
        "findings": [
            {"bounding_box": [1, 0, 224, 224],
             "description": "..."}
        ]

    Detects `[ <whitespace> "word":` (bare KV after array open) and inserts the
    missing `{` at the start and `}` right before the matching `]`.
    """
    # Match [ <whitespace> "any-key": — the key can be any non-empty quoted string,
    # including bbox-style keys like "[30, 110, 190, 220]".
    # In valid JSON arrays elements are never followed by ":", so this pattern is safe.
    pattern = re.compile(r'\[(\s*)("[^"\n]+")\s*:')
    insertions: list[tuple[int, str]] = []

    for m in pattern.finditer(text):
        open_bracket = m.start()
        # Position to insert `{`: right after `[` and any leading whitespace
        insert_open = open_bracket + 1 + len(m.group(1))

        # Walk forward to find the `]` that closes this `[`
        depth = 1
        i = open_bracket + 1
        in_str = False
        esc = False
        while i < len(text) and depth > 0:
            c = text[i]
            if esc:
                esc = False
                i += 1
                continue
            if in_str:
                if c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
                i += 1
                continue
            if c == '"':
                in_str = True
                i += 1
                continue
            if c in '[{':
                depth += 1
            elif c in ']}':
                depth -= 1
            i += 1

        if depth == 0:
            close_bracket = i - 1  # position of matching `]`
            insertions.append((close_bracket, '}'))
            insertions.append((insert_open, '{'))

    if not insertions:
        return text

    # Apply right-to-left so earlier positions stay valid
    insertions.sort(key=lambda x: x[0], reverse=True)
    result = list(text)
    for pos, char in insertions:
        result.insert(pos, char)
    return ''.join(result)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the XAI framework.

    Provides common functionality:
    - VLM integration
    - Output directory management
    - JSON parsing utilities
    - Error handling
    """

    def __init__(
        self,
        vlm: Any,
        output_dir: Optional[str] = None,
        agent_name: str = "BaseAgent"
    ):
        """
        Initialize base agent.

        Args:
            vlm: VisionLanguageModel instance
            output_dir: Output directory path
            agent_name: Name for logging
        """
        self.vlm = vlm
        self.agent_name = agent_name

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs")

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"{self.agent_name} initialized")
        print(f"  Output directory: {self.output_dir}")

    def invoke_vlm(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Invoke the VLM with a prompt and optional images.

        Args:
            prompt: Text prompt
            images: Optional list of image paths
            max_new_tokens: Optional per-call override of the generation
                token budget. Every invoke/invoke_with_images/
                invoke_multimodal implementation across vlm_wrapper.py,
                training/rl/rl_vlm.py, and training/rl_local/local_vlm.py
                already declares **kwargs, so passing this through is safe
                everywhere -- only LocalSamplingVLM (the local RL rollout
                path) actually acts on it; other VLM wrappers silently
                ignore it via their **kwargs, i.e. unchanged behavior for
                callers that don't pass this.

        Returns:
            VLM response string
        """
        vlm_kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}
        try:
            if images:
                if hasattr(self.vlm, 'invoke_with_images'):
                    return self.vlm.invoke_with_images(prompt, images, **vlm_kwargs)
                elif hasattr(self.vlm, 'invoke_multimodal'):
                    return self.vlm.invoke_multimodal(prompt, images, **vlm_kwargs)
                else:
                    raise RuntimeError(f"VLM does not support image input but {len(images)} images were provided")
            else:
                return self.vlm.invoke(prompt, **vlm_kwargs)
        except Exception as e:
            error_msg = str(e)
            print(f"  Warning: VLM call failed: {error_msg}")

            # Handle CUDA/NVML errors
            if "nvml" in error_msg.lower() or "CUDA" in error_msg:
                print("  Detected CUDA/NVML error. Attempting to recover...")
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass

            raise

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from VLM response.

        Args:
            response: VLM response string

        Returns:
            Parsed dictionary, or empty dict on failure
        """
        # Pre-processing: strip <think>...</think> blocks (Qwen3 reasoning chains).
        # Must be done FIRST — thinking blocks often contain { } chars that cause
        # Strategy 1's greedy \{.*\} to capture thinking content + JSON as one blob.
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # Handle truncated thinking (max_new_tokens hit inside <think>, no </think> emitted).
        if '<think>' in response and '</think>' not in response:
            response = re.sub(r'<think>.*', '', response, flags=re.DOTALL).strip()

        # Pre-processing: remove invalid JSON number prefixes (+0.12 → 0.12).
        # JSON spec forbids a leading '+' on numbers; some VLMs emit it anyway.
        response = re.sub(r'(?<!["\w])\+(\d)', r'\1', response)

        # Pre-processing: quote bare percentage values (100% → "100%", 0% → "0%").
        # VLMs sometimes emit: "frequency_of_opposition": 100%
        response = re.sub(r'(?<=:\s)(\d+(?:\.\d+)?)%', r'"\1%"', response)

        # Pre-processing: strip markdown code fences (```json ... ``` or ``` ... ```).
        # Handles both closed fences and VLM responses truncated before the closing fence.
        code_fence = re.search(r'```(?:json)?\s*\n?(.*?)(?:\n?```|$)', response, re.DOTALL)
        if code_fence:
            inner = code_fence.group(1).strip()
            if inner and '{' in inner:
                response = inner

        # Pre-processing: strip inline // comments (VLMs add them after array elements).
        # Matches: <value-end char> <optional spaces> // <rest of line>
        # where value-end chars are }, ], ", comma, or a digit.
        # This pattern cannot match // inside string values because those are preceded
        # by string content (e.g. "https://…" — the char before // is :, not in the set).
        response = re.sub(r'([}\]",\d])[ \t]*//[^\n]*', r'\1', response)

        # Fast path: try to parse immediately after code fence extraction, before running
        # the aggressive heuristic fixes below.  When the VLM returns well-formed JSON the
        # preprocessing regexes can corrupt it (e.g. values with leading spaces, CUB species
        # names, emoji-containing strings).  Parsing here avoids that entirely.
        _start = response.find('{')
        if _start != -1:
            _json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if _json_match:
                try:
                    return json.loads(_json_match.group())
                except json.JSONDecodeError:
                    pass
            try:
                _obj, _ = json.JSONDecoder().raw_decode(response, _start)
                return _obj
            except json.JSONDecodeError:
                pass

        # Pre-processing: replace <placeholder> values with null.
        # Some VLMs emit template tokens like "new_value": <value> instead of a real value.
        response = re.sub(r'(?<=:)\s*<\w+>', ' null', response)

        # Pre-processing: fix unclosed strings that end at a newline.
        # VLMs sometimes forget the closing " before the newline, e.g.:
        #   "Both have a similar overall dark grayish coloration.   ← no closing "
        #   "Both are seabirds with medium-sized bodies."
        # In JSON, strings cannot contain literal newlines, so a line that starts
        # with " and ends without one is always an unclosed string.
        # Strategy: match lines starting with " whose last char is not ", \, :, or ,
        # (those indicate a properly closed/key/trailing-comma line) and append ",
        # to close the string and separate it from the next array element.
        response = re.sub(
            r'^(\s*"(?:[^"\\\n]|\\.)*[^"\\\n:,])$',
            lambda m: m.group(1) + '",',
            response,
            flags=re.MULTILINE,
        )

        # Pre-processing: fix "X" vs "Y" comparison expressions inside JSON arrays/strings.
        # VLMs (especially on NLI tasks) write: "differing_tokens": ["The" vs "Girls", ...]
        # Merge into a single string: "The vs Girls"
        response = re.sub(
            r'"([^"\n]+)"\s+vs\s+"([^"\n]+)"',
            lambda m: '"' + m.group(1) + ' vs ' + m.group(2) + '"',
            response,
        )

        # Pre-processing: fix embedded unescaped double quotes inside JSON string values.
        # VLMs (especially on CUB/vision tasks) write things like:
        #   "has a "grooved" beak"  or  ["item with "quoted term" in it"]
        #   "giving a rounded, "bulging" facial appearance"
        #   "dismissal ("not good", "poorly edited")"  ← preceded by (
        # Patterns handled:
        #   (a) letter/digit + space before the embedded phrase — most common case.
        #   (b) comma + space before the embedded phrase followed by prose.
        #   (c) open-paren before the embedded phrase (e.g. ("term", "term")).
        # Max phrase length 60 chars limits false positives.
        response = re.sub(
            r'([a-zA-Z0-9]) "([^"\n:{}\[\]]{1,60})"(?=[ ,\.!?;])',
            lambda m: m.group(1) + " '" + m.group(2) + "'",
            response,
        )
        response = re.sub(
            r'(,\s)"([^"\n:{}\[\]]{1,60})"(?=\s[a-zA-Z])',
            lambda m: m.group(1) + "'" + m.group(2) + "'",
            response,
        )
        response = re.sub(
            r'(\()"([^"\n:{}\[\]]{1,60})"(?=[,\)])',
            lambda m: m.group(1) + "'" + m.group(2) + "'",
            response,
        )

        # Pre-processing: fix array elements where a quoted string is followed by unquoted prose.
        # VLMs sometimes emit:  "age" favors >50K (explanation),
        # instead of:           "age favors >50K (explanation)",
        # The fix merges them into a single properly-quoted string.
        # Guard: first char after whitespace must not be :, ", comma, or bracket (would be valid JSON).
        response = re.sub(
            r'"([^"\\]*)"\s+([^",:\[\]{},\n\s][^",:\[\]{},\n]*)',
            lambda m: '"' + m.group(1) + ' ' + m.group(2).rstrip() + '"',
            response
        )

        # Pre-processing: fix unquoted string values after JSON keys.
        # VLMs sometimes emit: "key": Some prose text here
        # instead of:          "key": "Some prose text here"
        # Heuristic: after `"key": ` (no opening quote), if the value starts with a letter
        # (not a JSON-valid token starter: ", {, [, digit, -, t/f/n for true/false/null),
        # wrap the entire rest-of-line in quotes (converting embedded " to ').
        # Only matches single-line values; multi-line truncation is handled by Strategy 3.
        response = re.sub(
            r'("[\w]+"\s*:\s+)([^"\d{\[\-tfn\r\n][^\r\n]*)',
            lambda m: m.group(1) + '"' + re.sub(r'(?<!\\)"', "'", m.group(2).rstrip()) + '"',
            response
        )

        # Pre-processing: fix arrays whose elements are bare key-value pairs.
        # VLMs sometimes emit:  "findings": ["bounding_box": [...], "description": "..."]
        # instead of:           "findings": [{"bounding_box": [...], "description": "..."}]
        # _fix_bare_array_objects inserts the missing { and } around the object.
        response = _fix_bare_array_objects(response)

        # Strategy 1: existing regex approach (fast path, works for most responses)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 2: raw_decode from first '{' (handles code fences and trailing content)
        start = response.find('{')
        if start != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(response, start)
                return obj
            except json.JSONDecodeError:
                pass

        # Strategy 3: truncated JSON recovery — balance unmatched brackets and retry.
        # Handles VLM output that was cut off mid-stream (e.g. a long "findings" array).
        if start != -1:
            partial = response[start:].rstrip()

            def _try_close(p: str) -> dict | None:
                """Try to close p by balancing braces/brackets; return parsed obj or None."""
                p2 = re.sub(r',\s*$', '', p)  # strip trailing comma before closing
                # Use a stack to determine correct closing order (respects nesting).
                # e.g. {[{ needs }]} not ]}} which _count_unbalanced + reversed-flat would produce.
                stack = []
                in_str = False
                esc = False
                for ch in p2:
                    if esc:
                        esc = False
                        continue
                    if ch == '\\' and in_str:
                        esc = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == '{':
                        stack.append('}')
                    elif ch == '[':
                        stack.append(']')
                    elif ch in ']}' and stack and stack[-1] == ch:
                        stack.pop()
                if not stack:
                    try:
                        return json.loads(p2)
                    except json.JSONDecodeError:
                        return None
                suffix2 = ''.join(reversed(stack))
                try:
                    return json.loads(p2 + suffix2)
                except json.JSONDecodeError:
                    return None

            # 3a: trailing comma
            partial = re.sub(r',\s*$', '', partial)
            # 3b: dangling open string (e.g. "key": "truncated mid-word)
            # Guard: only apply when strings are unbalanced (odd " count).  When the
            # JSON is structurally complete but merely missing close-braces, the last "
            # is a proper closing quote; blindly stripping it (and the trailing }) would
            # destroy valid content and prevent _try_close from recovering the JSON.
            _quote_count = 0
            _esc = False
            for _ch in partial:
                if _esc:
                    _esc = False
                    continue
                if _ch == '\\':
                    _esc = True
                    continue
                if _ch == '"':
                    _quote_count += 1
            if _quote_count % 2 == 1:  # unclosed string → safe to strip
                partial = re.sub(r'"[^"]*$', '', partial)
            # 3c: dangling "key": with no value — replace with null so the key is preserved
            #     (removing it entirely can cascade and wipe the whole parent object)
            partial = re.sub(r'(,?\s*"[^"]+"\s*:)\s*$', r'\1 null', partial)
            # Try to close here — covers {"explanation": null} → {"explanation": null}
            obj = _try_close(partial)
            if obj is not None:
                return obj
            # 3d: dangling incomplete object start (no nested braces inside)
            partial = re.sub(r',?\s*\{[^{}]*$', '', partial)
            obj = _try_close(partial)
            if obj is not None:
                return obj

        raise RuntimeError(f"JSON parse error in VLM response. Response preview: {response[:300]}")

    def invoke_vlm_for_json(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call VLM and parse JSON response, retrying on parse errors.

        Args:
            prompt: Prompt string
            images: Optional list of image paths
            max_retries: Maximum number of attempts (default 3)
            retry_delay: Seconds to wait between retries (default 2)
            max_new_tokens: Optional per-call generation token budget
                override (see invoke_vlm). Confirmed real cause of a class
                of guaranteed-to-fail JSON parse retries: for hard vision/
                multi-instance prompts, the (still lightly-trained) local
                policy often writes long free-form reasoning before ever
                reaching the JSON object, routinely exhausting the default
                --max_new_tokens=4096 budget and getting cut off mid-JSON --
                observed failed responses of 14000-17000+ chars (~4000+
                tokens), all truncated, none malformed for any other
                reason. Retrying with the SAME budget on the SAME prompt
                fails again for the identical structural reason (not
                random bad luck), burning up to max_retries full
                generations for a guaranteed loss. Callers generating a
                strategy JSON (the failure mode observed) should pass a
                larger budget here; other JSON calls (e.g. short critic
                verdicts) can leave this None.

        Returns:
            Parsed JSON dictionary

        Raises:
            RuntimeError: If parsing fails on all attempts
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            response = self.invoke_vlm(prompt, images, max_new_tokens=max_new_tokens)
            try:
                return self.parse_json_response(response)
            except RuntimeError as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  JSON parse failed (attempt {attempt}/{max_retries}), retrying in {retry_delay}s... Error: {e}")
                    print(f"  Full response ({len(response)} chars): {response[:2000]}")
                    time.sleep(retry_delay)
        raise last_error

    def save_json(self, data: Dict[str, Any], filename: str, subdir: str = "") -> Path:
        """
        Save dictionary as JSON file.

        Args:
            data: Dictionary to save
            filename: File name (with or without .json extension)
            subdir: Optional subdirectory within output_dir

        Returns:
            Path to saved file
        """
        if not filename.endswith('.json'):
            filename += '.json'

        if subdir:
            save_dir = self.output_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir

        filepath = save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def _save_prompt(self, prompt: str, question: Dict[str, Any], label: str) -> Path:
        """Save a prompt string to outputs/prompts/{modality}/{dataset}/{q_type}/{row_no}/{label}.txt"""
        import re
        dataset_base_name = question.get('dataset_base_name', 'unknown')
        row_no = question.get('row_no', question.get('question_id', 0))
        modality = question.get('modality', 'vision')
        match = re.match(r'(.+?)_(q\d+)(?:_.*)?$', dataset_base_name)
        if match:
            dataset_name = match.group(1)
            q_type_str = match.group(2)
        else:
            dataset_name = dataset_base_name
            q_type_str = f"q{question.get('q_type', 1)}"
        save_dir = self.output_dir / f"prompts/{modality}/{dataset_name}/{q_type_str}/{row_no}"
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"{label}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt)
        return filepath

    def log(self, message: str, level: str = "INFO"):
        """Log a message with agent name prefix"""
        print(f"[{self.agent_name}] {level}: {message}")

    @staticmethod
    def _extract_question_context_fields(question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Q5/Q6/Q7 specific fields from the question's 'example' text.

        Returns a dict with zero or more of:
          - target_class  (Q6): the flip-target class label
          - queried_part  (Q5): the feature/region being masked
          - part_to_change (Q7): the feature/region being removed/changed
        """
        q_type = question.get('q_type')
        example = question.get('example', '') or question.get('question', '') or ''
        _q_raw = question.get('q', '')
        q_template = str(_q_raw) if isinstance(_q_raw, str) else ''
        result = {}

        if q_type == 6:
            # Standard: "flip the model into X" or "flip the model's prediction to X"
            m = re.search(
                r"flip the (?:model(?:'s prediction)?|prediction)(?:\s+into\s+|\s+to\s+)(.+?)(?:\?|$)",
                example, re.IGNORECASE
            )
            if not m:
                # Paraphrase text: "cause the model to predict X instead"
                m = re.search(
                    r"model to predict\s+([^?.\n,]+?)(?:\s+instead|\?|$)",
                    example, re.IGNORECASE
                )
            if m:
                result['target_class'] = m.group(1).strip().rstrip('?').strip()

        elif q_type == 5:
            # Standard text: "mask the word 'X' from this input"
            m = re.search(r"mask the word ['\"](.+?)['\"]", example, re.IGNORECASE)
            if not m:
                # Paraphrase text: "word 'X' were hidden/masked/removed"
                m = re.search(r"\bword\s+['\"](.+?)['\"]", example, re.IGNORECASE)
            if m:
                result['queried_part'] = m.group(1).strip()
            else:
                # Standard tabular: "mask the X feature of this"
                m = re.search(r'mask the (.+?)(?:\s+feature\b|\s+(?:of|from)\s+this)', example, re.IGNORECASE)
                if not m:
                    # OOD tabular: "remove or alter the feature 'X' from this input"
                    m = re.search(r"remove or alter(?:\s+the)?\s+feature\s+['\"](.+?)['\"]", example, re.IGNORECASE)
                if not m:
                    # Paraphrase tabular: "if the X feature were removed/masked/hidden"
                    m = re.search(
                        r"if\s+the\s+(.+?)\s+feature\s+(?:were|was|is)\s+(?:removed|masked|hidden|eliminated)",
                        example, re.IGNORECASE
                    )
                if m:
                    result['queried_part'] = m.group(1).strip()
                elif q_template:
                    # Fallback: {placeholder} in q template (skip generic placeholders)
                    m2 = re.search(r'\{(.+?)\}', q_template)
                    if m2 and m2.group(1) not in ('certain', 'certain_part'):
                        result['queried_part'] = m2.group(1)

        elif q_type == 7:
            # Standard text: "remove/change the word 'X' from this input"
            m = re.search(r"remove/change (?:the )?word ['\"](.+?)['\"]", example, re.IGNORECASE)
            if not m:
                # Paraphrase text: "word 'X' were modified or removed"
                m = re.search(r"\bword\s+['\"](.+?)['\"]", example, re.IGNORECASE)
            if m:
                result['part_to_change'] = m.group(1).strip()
            else:
                # Standard tabular/general: "remove/change X,"
                m = re.search(r'remove/change\s+(.+?)(?:,|\?{1,2}|$)', example, re.IGNORECASE)
                if not m:
                    # OOD tabular: "remove or change the feature 'X'"
                    m = re.search(r"remove or change(?:\s+the)?\s+feature\s+['\"](.+?)['\"]", example, re.IGNORECASE)
                if not m:
                    # Paraphrase tabular: "if X were altered/removed/changed"
                    m = re.search(
                        r"if\s+(?:the\s+)?(.+?)\s+(?:were|was)\s+(?:altered|removed|changed|modified)",
                        example, re.IGNORECASE
                    )
                if m:
                    result['part_to_change'] = m.group(1).strip().rstrip('?,').strip()
                elif q_template:
                    # Fallback: {placeholder} in q template
                    m2 = re.search(r'\{(.+?)\}', q_template)
                    if m2:
                        result['part_to_change'] = m2.group(1)

        return result

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main execution method. Must be implemented by subclasses.
        """
        pass
