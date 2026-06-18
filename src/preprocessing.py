import re
from collections import Counter
from typing import List, Set
from dataclasses import dataclass


@dataclass
class Tokenizer:
    vocab_size: int = 10000
    vocab: Set[str] = None
    UNK_TOKEN: str = "<UNK>"
    START_TOKEN: str = "<s>"
    END_TOKEN: str = "</s>" 


    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"([?.!,;])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    

    def build_vocab(self, corpus: List[str]) -> None:
        word_counts = Counter()
        for sentence in corpus:
            tokens = self.clean_text(sentence).split()
            word_counts.update(tokens)
        
        most_common = word_counts.most_common(self.vocab_size)
        self.vocab = {word for word, count in most_common}
        self.vocab.add(self.UNK_TOKEN)


    def tokenize(self, text: str) -> List[str]:
        if not self.vocab:
            raise ValueError("Build Vocabulary first with build_vocab()")
        
        raw_tokens = self.clean_text(text).split()
        processed_tokens = [token if token in self.vocab else self.UNK_TOKEN for token in raw_tokens]

        return [self.START_TOKEN] + processed_tokens + [self.END_TOKEN]