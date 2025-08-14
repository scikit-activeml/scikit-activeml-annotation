

def is_sorted(seq):
    """
    Returns True if the sequence is sorted in non-decreasing order.
    Works with any sequence supporting indexing and comparisons.
    """
    return all(seq[i] <= seq[i+1] for i in range(len(seq)-1))

