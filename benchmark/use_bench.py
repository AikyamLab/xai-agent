import json

def load_data_(file_path="q1_snli.json"):
    """Loads the data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_entry_by_row_idx(row_idx, q_data):
    """Retrieves the full data for a specific original row_idx from json data."""
    if q_data:
        for entry in q_data:
            if entry.get('row_idx') == row_idx:
                return entry
    return None

def get_features(entry):
    """Extracts premise and hypothesis from an entry."""
    if entry and 'features' in entry:
        return entry['features']
    return None

def get_target(entry):
    """Extracts target value and label from entry."""
    if entry and 'target' in entry:
        return entry['target']
    return None

def get_predicted(entry):
    """Extracts predicted value and label from entry."""
    if entry and 'predicted' in entry:
        return entry['predicted']
    return None

def get_question(entry):
    """Extracts the 'example' and 'question' strings from entry."""
    if entry:
        return {
            'question': entry.get('example'),
            'question_type': entry.get('question')
        }
    return None

