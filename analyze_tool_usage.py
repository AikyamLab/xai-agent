"""
Analyze which XAI tool's bounding box was chosen as the final output.

Parses actor_prompt.txt to extract ALL candidate bounding boxes that were
shown to the agent (EXPANDED bbox, attention bbox, LIME segments, grounding
location, comparison region, object detection bbox, shap, layer_cam, etc.)
and compares them to the output bounding box in result.json.

Usage:
  python analyze_tool_usage.py [results_dir] [--tolerance N] [--verbose]
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Bbox utilities
# ---------------------------------------------------------------------------

def parse_bbox(text):
    """Extract [x1, y1, x2, y2] from a string like '[39, 36, 54, 51]'."""
    m = re.search(r'\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]', text)
    if m:
        return [int(m.group(i)) for i in range(1, 5)]
    return None


def boxes_match(a, b, tol=0):
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return False
    try:
        return all(abs(int(av) - int(bv)) <= tol for av, bv in zip(a, b))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Prompt parser — extract every candidate bbox visible to the agent
# ---------------------------------------------------------------------------

# Tools that produce heatmap/saliency bboxes
HEATMAP_TOOLS = {
    'gradcam', 'guided_backprop', 'layer_cam', 'integrated_gradients',
    'smooth_grad', 'shap', 'guided_gradcam', 'scorecam', 'xrai',
}


def _extract_section_bboxes(tool_name, section_text, candidates):
    """Parse bbox lines inside one '### toolname:' section of the prompt."""

    if tool_name in HEATMAP_TOOLS:
        # EXPANDED attention bbox: [x1, y1, x2, y2]**
        for m in re.finditer(
            r'EXPANDED attention bbox[^:]*:\s*(\[\d+[^\]]+\])',
            section_text
        ):
            bb = parse_bbox(m.group(1))
            if bb:
                candidates.setdefault(f'{tool_name}_expanded', []).append(bb)

        # Attention bbox: [x1, y1, x2, y2] (... pixels)
        # HIGH-attribution bbox ...: [x1, y1, x2, y2]
        # Must NOT be preceded by EXPANDED (that's already captured above)
        for m in re.finditer(
            r'(?<!EXPANDED )(?:Attention bbox|HIGH-attribution bbox[^:]*)\s*[^:]*:\s*(\[\d+[^\]]+\])',
            section_text
        ):
            bb = parse_bbox(m.group(1))
            if bb:
                candidates.setdefault(f'{tool_name}_attention', []).append(bb)

    elif tool_name == 'lime':
        # Positive segments: line ends with '**' (marked by prompt code)
        for line in section_text.splitlines():
            if 'bbox=' in line:
                m = re.search(r'Segment\s+(\d+).*?weight=([-\d.]+).*?bbox=(\[\d+[^\]]+\])', line)
                if m:
                    bb = parse_bbox(m.group(3))
                    if bb:
                        positive = line.rstrip().endswith('**')
                        label = 'lime_positive' if positive else 'lime_negative'
                        candidates.setdefault(label, []).append(bb)

    elif tool_name == 'object_detection':
        # Format in prompt: "bird (conf=0.81): bbox=[163, 124, 373, 281]**"
        for m in re.finditer(r'bbox\s*=\s*(\[\d+[^\]]+\])', section_text):
            bb = parse_bbox(m.group(1))
            if bb:
                candidates.setdefault('object_detection', []).append(bb)
        # Also handle dict format: {x1: N, y1: N, x2: N, y2: N}
        for m in re.finditer(
            r'"?x1"?\s*:\s*(\d+)[^}]*"?y1"?\s*:\s*(\d+)[^}]*"?x2"?\s*:\s*(\d+)[^}]*"?y2"?\s*:\s*(\d+)',
            section_text
        ):
            bb = [int(m.group(i)) for i in range(1, 5)]
            candidates.setdefault('object_detection', []).append(bb)


def extract_bboxes_from_prompt(prompt_text):
    """
    Parse actor_prompt.txt and return dict:
      { source_label: [list of [x1,y1,x2,y2]] }

    Source labels:
      gradcam_expanded, gradcam_attention
      guided_backprop_expanded, guided_backprop_attention
      layer_cam_expanded, ...
      integrated_gradients_expanded, integrated_gradients_attention
      smooth_grad_expanded, ...
      shap_expanded, ...
      lime_positive, lime_negative
      object_detection
      grounding_location, grounding_region
      comparison_region
    """
    candidates = {}

    # Split on section headers: "### toolname:" or "### TASK Task"
    # Keep delimiter so we know the section name
    parts = re.split(r'\n(###\s+[^\n]+)', prompt_text)
    # parts[0] = preamble, then alternating [header, body]

    current_tool = None
    for i, part in enumerate(parts):
        if part.startswith('###'):
            header = part.lstrip('#').strip().rstrip(':').lower()

            # Map header to canonical tool name
            if header in HEATMAP_TOOLS:
                current_tool = header
            elif header == 'lime':
                current_tool = 'lime'
            elif header == 'object_detection':
                current_tool = 'object_detection'
            elif 'grounding' in header:
                current_tool = 'grounding'
            elif 'comparison' in header:
                current_tool = 'comparison'
            elif 'reasoning' in header:
                current_tool = 'reasoning'
            else:
                current_tool = None
        else:
            # This is a section body
            if current_tool is None:
                continue
            body = part

            if current_tool in HEATMAP_TOOLS or current_tool in ('lime', 'object_detection'):
                _extract_section_bboxes(current_tool, body, candidates)

            elif current_tool == 'grounding':
                # location: "[x1, y1, x2, y2]"  (JSON-encoded string)
                for m in re.finditer(
                    r'location\s*:\s*["\']?\s*(\[\d+[^\]]+\])\s*["\']?',
                    body
                ):
                    bb = parse_bbox(m.group(1))
                    if bb:
                        candidates.setdefault('grounding_location', []).append(bb)

                # region: [x1, y1, x2, y2]  (inside findings list)
                for m in re.finditer(r'"region"\s*:\s*(\[\d+[^\]]+\])', body):
                    bb = parse_bbox(m.group(1))
                    if bb:
                        candidates.setdefault('grounding_region', []).append(bb)

            elif current_tool == 'comparison':
                # most_responsible_region: [x1, y1, x2, y2]
                for m in re.finditer(
                    r'"most_responsible_region"\s*:\s*(\[\d+[^\]]+\])',
                    body
                ):
                    bb = parse_bbox(m.group(1))
                    if bb:
                        candidates.setdefault('comparison_region', []).append(bb)

                # contrastive_instance region
                for m in re.finditer(r'"region"\s*:\s*(\[\d+[^\]]+\])', body):
                    bb = parse_bbox(m.group(1))
                    if bb:
                        candidates.setdefault('comparison_region', []).append(bb)

    return candidates


# ---------------------------------------------------------------------------
# Match output bbox to a tool source
# ---------------------------------------------------------------------------

# Priority order when multiple sources match (most specific first)
MATCH_PRIORITY = [
    'object_detection',
    'grounding_location',
    'grounding_region',
    'comparison_region',
    'lime_positive',
    'gradcam_expanded',
    'gradcam_attention',
    'guided_backprop_expanded',
    'guided_backprop_attention',
    'layer_cam_expanded',
    'layer_cam_attention',
    'integrated_gradients_expanded',
    'integrated_gradients_attention',
    'smooth_grad_expanded',
    'smooth_grad_attention',
    'shap_expanded',
    'shap_attention',
    'lime_negative',
    'comparison_region',
]


def classify_bbox(output_bbox, candidates, tol=0):
    """
    Return list of matched source labels (may be multiple if tie).
    Returns ['no_match'] if nothing matches.
    """
    if output_bbox is None:
        return ['no_bbox']

    matched = []
    for label, boxes in candidates.items():
        for box in boxes:
            if boxes_match(output_bbox, box, tol):
                matched.append(label)
                break

    return matched if matched else ['no_match']


# ---------------------------------------------------------------------------
# Fallback: extract candidates from result.json when no prompt file exists
# ---------------------------------------------------------------------------

def extract_bboxes_from_result(tool_results):
    """Fallback extractor using result.json data (less accurate for gradcam etc.)"""
    candidates = {}

    for tool in HEATMAP_TOOLS:
        if tool in tool_results:
            bb = tool_results[tool].get('suggested_bounding_box')
            if bb:
                candidates.setdefault(f'{tool}_attention', []).append(bb)

    if 'lime' in tool_results:
        lime = tool_results['lime']
        for seg in lime.get('statistics', {}).get('top_positive_segments', []):
            bb = seg.get('bbox')
            if bb:
                candidates.setdefault('lime_positive', []).append(bb)
        for seg in lime.get('statistics', {}).get('top_negative_segments', []):
            bb = seg.get('bbox')
            if bb:
                candidates.setdefault('lime_negative', []).append(bb)

    if 'object_detection' in tool_results:
        for det in tool_results['object_detection'].get('detections', []):
            bbox = det.get('bbox')
            if bbox and isinstance(bbox, dict):
                bb = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
                candidates.setdefault('object_detection', []).append(bb)

    # Autonomous grounding location
    autonomous = (
        tool_results.get('autonomous_tasks') or
        tool_results.get('autonomous_results')
    )
    if autonomous:
        gr = autonomous.get('grounding', {})
        loc = gr.get('result', {}).get('location')
        if loc:
            bb = parse_bbox(str(loc))
            if bb:
                candidates.setdefault('grounding_location', []).append(bb)

        comp = autonomous.get('comparison', {})
        mrr = comp.get('result', {}).get('findings', {})
        if isinstance(mrr, dict):
            region = mrr.get('most_responsible_region')
            if region:
                candidates.setdefault('comparison_region', []).append(region)

    return candidates


# ---------------------------------------------------------------------------
# Output bbox extraction — handles all question types
# ---------------------------------------------------------------------------

def _extract_output_bboxes(output):
    """
    Extract all bounding boxes from an output dict, handling all Q-type formats:
      Q1/Q2/Q3/Q8:  output.bounding_box
      Q5/Q7:        output.masked_region.bounding_box
      Q6:           output.change_plan.bounding_box
      Q9:           output.instances[i].bounding_box  (multiple)
    Returns list of [x1,y1,x2,y2] lists (may be empty).
    """
    if not isinstance(output, dict):
        return []

    bboxes = []

    # Direct bbox (Q1, Q2, Q3, Q8)
    bb = output.get('bounding_box')
    if bb and len(bb) == 4:
        bboxes.append(bb)

    # masked_region (Q5, Q7)
    mr = output.get('masked_region')
    if isinstance(mr, dict):
        bb = mr.get('bounding_box')
        if bb and len(bb) == 4:
            bboxes.append(bb)

    # change_plan (Q6)
    cp = output.get('change_plan')
    if isinstance(cp, dict):
        bb = cp.get('bounding_box')
        if bb and len(bb) == 4:
            bboxes.append(bb)

    # instances list (Q9)
    for inst in output.get('instances', []):
        if isinstance(inst, dict):
            bb = inst.get('bounding_box')
            if bb and len(bb) == 4:
                bboxes.append(bb)

    return bboxes


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def find_prompt_file(result_file, results_dir, prompts_dir):
    """Find corresponding actor_prompt.txt for a result.json."""
    try:
        rel = result_file.relative_to(results_dir)
        # rel = vision/cub_resnet/q1/115/result.json
        # prompt = prompts/vision/cub_resnet/q1/115/actor_prompt.txt
        prompt_file = prompts_dir / rel.parent / 'actor_prompt.txt'
        if prompt_file.exists():
            return prompt_file
    except Exception:
        pass
    return None


def analyze_results_dir(results_dir, prompts_dir, tol=0, verbose=False):
    results_dir = Path(results_dir)
    prompts_dir = Path(prompts_dir) if prompts_dir else None

    global_counts = defaultdict(int)
    per_qtype    = defaultdict(lambda: defaultdict(int))
    per_dataset  = defaultdict(lambda: defaultdict(int))

    total        = 0
    skipped      = 0
    no_prompt    = 0
    multi_match  = 0

    for result_file in sorted(results_dir.rglob('result.json')):
        with open(result_file) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                skipped += 1
                continue

        output       = data.get('output', {})
        tool_results = data.get('tool_results', {})

        # Collect all output bboxes from this result (may be multiple for Q9)
        output_bboxes = _extract_output_bboxes(output)

        # If no bbox and no tool_results, it's tabular/text — skip entirely
        if not output_bboxes and not tool_results:
            skipped += 1
            continue

        # Parse path: results/<modality>/<dataset>/<qtype>/<sample>/result.json
        parts   = result_file.relative_to(results_dir).parts
        qtype   = parts[2] if len(parts) >= 4 else 'unknown'
        dataset = parts[1] if len(parts) >= 3 else 'unknown'

        # Try to load prompt file for accurate bbox extraction
        prompt_file = find_prompt_file(result_file, results_dir, prompts_dir) if prompts_dir else None
        if prompt_file:
            prompt_text = prompt_file.read_text(errors='replace')
            candidates  = extract_bboxes_from_prompt(prompt_text)
        else:
            no_prompt += 1
            candidates = extract_bboxes_from_result(tool_results)

        # Count each output bbox (Q9 contributes multiple)
        for output_bbox in output_bboxes:
            matched = classify_bbox(output_bbox, candidates, tol)
            total  += 1

            if len(matched) > 1:
                multi_match += 1

            for m in matched:
                global_counts[m]         += 1
                per_qtype[qtype][m]      += 1
                per_dataset[dataset][m]  += 1

            if verbose:
                src = result_file.relative_to(results_dir)
                print(f'{src}: {output_bbox} -> {matched}')

        # If there were truly no bboxes (tabular Q types in vision), count as no_bbox
        if not output_bboxes and tool_results:
            total += 1
            global_counts['no_bbox']        += 1
            per_qtype[qtype]['no_bbox']      += 1
            per_dataset[dataset]['no_bbox']  += 1

    return global_counts, per_qtype, per_dataset, total, skipped, no_prompt, multi_match


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_counts(counts, title, total):
    print(f"\n{'='*65}")
    print(f'{title}  (total={total})')
    print(f"{'='*65}")
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    for tool, cnt in sorted_counts:
        pct = cnt / total * 100 if total > 0 else 0
        bar = '#' * int(pct / 2)
        print(f'  {tool:<35} {cnt:>5}  ({pct:5.1f}%)  {bar}')


def main():
    parser = argparse.ArgumentParser(description='Analyze XAI tool bounding box usage')
    parser.add_argument(
        'results_dir', nargs='?',
        default='/standard/AikyamLab/yuyang/xai_agent/framework/trial_2/baseline_outputs/Qwen30B_VL_multi_trained_30step/results',
    )
    parser.add_argument(
        '--prompts-dir', default=None,
        help='Path to prompts directory (default: auto-detect sibling of results_dir)'
    )
    parser.add_argument('--tolerance', '-t', type=int, default=0,
                        help='Pixel tolerance for bbox matching (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--by-qtype',   action='store_true', default=True)
    parser.add_argument('--by-dataset', action='store_true', default=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Auto-detect prompts dir: sibling 'prompts' next to 'results'
    if args.prompts_dir:
        prompts_dir = Path(args.prompts_dir)
    else:
        candidate = results_dir.parent / 'prompts'
        prompts_dir = candidate if candidate.exists() else None

    print(f'Results dir: {results_dir}')
    print(f'Prompts dir: {prompts_dir or "(not found — fallback to result.json)"}')
    print(f'Tolerance:   {args.tolerance} pixels')

    global_counts, per_qtype, per_dataset, total, skipped, no_prompt, multi_match = analyze_results_dir(
        results_dir, prompts_dir, tol=args.tolerance, verbose=args.verbose
    )

    print_counts(global_counts, 'GLOBAL TOOL USAGE', total)
    print(f'\n  Skipped (no bbox / tabular):   {skipped}')
    print(f'  No prompt file (used fallback): {no_prompt}')
    print(f'  Multi-match (tied):             {multi_match}')

    # Summary excluding no_bbox
    bbox_total = total - global_counts.get('no_bbox', 0)
    if bbox_total > 0:
        bbox_counts = {k: v for k, v in global_counts.items() if k != 'no_bbox'}
        print_counts(bbox_counts, 'GLOBAL TOOL USAGE (bbox outputs only)', bbox_total)

    if args.by_dataset:
        for dataset, counts in sorted(per_dataset.items()):
            ds_total = sum(counts.values())
            ds_bbox_total = ds_total - counts.get('no_bbox', 0)
            if ds_bbox_total > 0:
                bbox_counts = {k: v for k, v in counts.items() if k != 'no_bbox'}
                print_counts(bbox_counts, f'DATASET: {dataset} (bbox only)', ds_bbox_total)

    if args.by_qtype:
        for qtype in sorted(per_qtype.keys()):
            counts = per_qtype[qtype]
            qt_bbox_total = sum(v for k, v in counts.items() if k != 'no_bbox')
            if qt_bbox_total > 0:
                bbox_counts = {k: v for k, v in counts.items() if k != 'no_bbox'}
                print_counts(bbox_counts, f'QUESTION TYPE: {qtype} (bbox only)', qt_bbox_total)


if __name__ == '__main__':
    main()
