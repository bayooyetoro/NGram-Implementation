import pandas as pd
from preprocessing import Tokenizer
from ngram_model import NGramModel
from evaluator import calculate_perplexity


def execute_attribution_pipeline(csv_path: str, n: int = 3, vocab_limit: int = 10000) -> None:
    """Orchestrates training and extrinsic evaluation of N-gram models."""
    
    # Ingestion & Partitioning
    df = pd.read_csv(csv_path)
    train_df = df.sample(frac=0.9, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Global Vocabulary Construction
    tokenizer = Tokenizer(vocab_size=vocab_limit)
    tokenizer.build_vocab(train_df['text'].tolist())
    
    # Model Compilation per Suspect
    models = {}
    authors = train_df['author'].unique()
    
    for author in authors:
        author_corpus = train_df[train_df['author'] == author]['text'].tolist()
        tokenized_corpus = [tokenizer.tokenize(text) for text in author_corpus]
        
        model = NGramModel(n=n, vocab_size=len(tokenizer.vocab), alpha=1.0)
        model.train_n_gram(tokenized_corpus)
        models[author] = model
        print(f"Compiled {n}-gram matrix for: {author}")

    # Extrinsic Evaluation
    # Sample a single 'anonymous' text from the held-out test set
    test_record = test_df.sample(1, random_state=42).iloc[0]
    anonymous_text = test_record['text']
    true_author = test_record['author']
    
    print(f"\nEvaluating Anonymous Text: '{anonymous_text[:80]}...'")
    anon_tokens = tokenizer.tokenize(anonymous_text)
    
    results = {}
    for author, model in models.items():
        ppl = calculate_perplexity(model, anon_tokens)
        results[author] = ppl
        print(f"Perplexity ({author}): {ppl:.2f}")

    # Verdict
    predicted_author = min(results, key=results.get)
    print(f"\nPrediction: {predicted_author}")
    print(f"Ground Truth: {true_author}")

if __name__ == "__main__":
    # Execute with Trigrams (N=3)
    execute_attribution_pipeline('data/train.csv', n=3)