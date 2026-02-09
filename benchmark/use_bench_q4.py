import json

def load_q4_data(file_path="q4_snli.json"):
    """Loads the Q4 SNLI data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_q4_pair_by_id(pair_id, q4_data):
    """Retrieves the full data for a specific pair_id from the loaded Q4 data."""
    for entry in q4_data:
        if entry.get('pair_id') == pair_id:
            return entry
    return None

def get_row_ids_for_pair(pair_data):
    """Extracts row IDs for instances A and B from a pair's data."""
    if pair_data:
        return {
            'instance_A_row_id': pair_data['instance_A']['row_no'],
            'instance_B_row_id': pair_data['instance_B']['row_no']
        }
    return None

def get_predictions_for_pair(pair_data):
    """Extracts predicted labels for instances A and B from a pair's data."""
    if pair_data:
        return {
            'instance_A_prediction_label': pair_data['instance_A']['prediction']['label'],
            'instance_B_prediction_label': pair_data['instance_B']['prediction']['label']
        }
    return None

def get_targets_for_pair(pair_data):
    """Extracts target labels for instances A and B from a pair's data."""
    if pair_data:
        return {
            'instance_A_target_label': pair_data['instance_A']['target']['label'],
            'instance_B_target_label': pair_data['instance_B']['target']['label']
        }
    return None

def get_question_for_pair(pair_data):
    """Extracts the example question for a pair."""
    if pair_data:
        return pair_data['example']
    return None
