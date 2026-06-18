import math
from typing import List
from ngram_model import NGramModel

def calculate_perplexity(model: NGramModel, tokens: List[str]) -> float:
    """
    Computes the perplexity of a sequence of tokens given a trained N-gram model.
    """
    n = model.n
    if len(tokens) < n:
        return float('inf') # Sequence too short to evaluate

    log_prob_sum = 0.0
    # N is the total number of n-grams in the test sequence
    N_count = len(tokens) - n + 1

    for i in range(N_count):
        context = tuple(tokens[i : i + n - 1])
        target = tokens[i + n - 1]
        
        log_prob_sum += model.get_log_probability(context, target)

    # Calculate average negative log probability
    avg_negative_log_prob = - (log_prob_sum / N_count)
    
    # Exponentiate to return to the perplexity scale
    return math.exp(avg_negative_log_prob)