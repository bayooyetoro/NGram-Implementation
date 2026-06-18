from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import Dict, Tuple, List
import math


@dataclass
class NGramModel:
    n: int
    vocab_size: int
    alpha: float = 1.0

    # Using default_factory for mutable types
    counts: Dict[Tuple[str, ...], Counter] = field(default_factory=lambda: defaultdict(Counter))
    context_totals: Counter = field(default_factory=Counter)


    def __post_init__(self):
        if self.n < 2:
            raise ValueError("This implementation requires n >= 2 (Bigram or higher).")
        
    
    def train_n_gram(self, tokenised_texts: List[List[str]]) -> None:
        for tokens in tokenised_texts:
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                target_word = tokens[i + self.n - 1]

                self.counts[context][target_word] += 1
                self.context_totals[context] += 1


    def get_probability(self, context: Tuple[str, ...], word: str) -> float:
        """
        Calculates the smoothed conditional probability P(word | context).
        """
        # Numerator: Count of the specific N-gram + Alpha
        numerator = self.counts[context][word] + self.alpha
        
        # Denominator: Count of the context + (Alpha * Vocabulary Size)
        denominator = self.context_totals[context] + (self.alpha * self.vocab_size)
        
        return numerator / denominator


    def get_log_probability(self, context: Tuple[str, ...], word: str) -> float:
        """
        Returns the natural logarithm of the probability to prevent arithmetic underflow 
        when multiplying many small probabilities during model evaluation.
        """
        prob = self.get_probability(context, word)
        return math.log(prob)
                
    