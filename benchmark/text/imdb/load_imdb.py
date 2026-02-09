from datasets import load_dataset

def get_test_example_by_idx(idx):
    """Yields the example at the given index from the global test_set."""
    dataset = load_dataset("imdb")
    dataset = dataset.filter(lambda x: x["label"] != -1)
    test_set = dataset["test"]
    if 0 <= idx < len(test_set):
        return test_set[idx]
    else:
        raise IndexError(f"Index {idx} is out of bounds for test_set of size {len(test_set)}")
