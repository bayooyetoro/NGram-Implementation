import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing import Tokenizer
from ngram_model import NGramModel
from evaluator import calculate_perplexity

# Configuration
AUTHOR_NAMES = {
    "EAP": "Edgar Allan Poe",
    "HPL": "H.P. Lovecraft",
    "MWS": "Mary Wollstonecraft Shelley",
}
NGRAM_SIZE = 3
VOCAB_SIZE = 10000
TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test.csv"


@st.cache_resource
def load_models() -> tuple[Tokenizer, dict[str, NGramModel], list[str]]:
    """
    Load and train N-gram models on training data.
    Executed once per Streamlit session using cached resources.
    """
    # Load training data
    df = pd.read_csv(TRAIN_CSV)

    # Build global tokenizer and vocabulary from training corpus
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(df["text"].tolist())

    # Train per-author N-gram models
    models = {}
    authors = sorted(df["author"].unique().tolist())

    for author in authors:
        author_texts = df[df["author"] == author]["text"].tolist()
        author_tokens = [tokenizer.tokenize(text) for text in author_texts]

        model = NGramModel(n=NGRAM_SIZE, vocab_size=len(tokenizer.vocab), alpha=1.0)
        model.train_n_gram(author_tokens)
        models[author] = model

    return tokenizer, models, authors


@st.cache_data
def load_test_sample() -> str:
    """Load a sample text from the test set as placeholder."""
    df = pd.read_csv(TEST_CSV)
    sample = df.sample(1, random_state=42).iloc[0]["text"]
    return sample[:500] + "..." if len(sample) > 500 else sample


def run_attribution(text: str, tokenizer: Tokenizer, models: dict) -> dict:
    """
    Evaluate text against all author models and return perplexity results.
    """
    tokens = tokenizer.tokenize(text)

    results = {}
    for author, model in models.items():
        ppl = calculate_perplexity(model, tokens)
        results[author] = ppl

    return results


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Forensic Stylometry Engine",
        page_icon="🔍",
        layout="wide",
    )

    # Header
    st.title("🔍 Forensic Stylometry Engine")
    st.markdown(
        """
    **N-gram Perplexity Attribution:** This tool analyzes an anonymous manuscript 
    by measuring how well it matches the linguistic patterns of three suspect authors 
    using trigram language models. Lower perplexity scores indicate a closer stylistic match.
    """
    )

    st.divider()

    # Load models once at startup
    tokenizer, models, authors = load_models()
    test_sample = load_test_sample()

    # Input section
    st.subheader("📝 Anonymous Manuscript")
    manuscript = st.text_area(
        "Paste or type the text to analyze:",
        value=test_sample,
        height=200,
        label_visibility="collapsed",
    )

    # Action button
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button(
            "🚀 Run Forensic Attribution",
            use_container_width=True,
            type="primary",
        )

    st.divider()

    # Execute analysis on button click
    if analyze_button:
        if not manuscript.strip():
            st.error("❌ Please enter some text to analyze.")
            return

        with st.spinner("⏳ Analyzing manuscript..."):
            results = run_attribution(manuscript, tokenizer, models)

        # Determine predicted author
        predicted_author_code = min(results, key=results.get)
        predicted_author_name = AUTHOR_NAMES[predicted_author_code]
        min_perplexity = results[predicted_author_code]

        # Results section
        st.subheader("🎯 Attribution Results")

        # Display predicted author prominently
        col1, col2, col3 = st.columns(3)
        with col2:
            st.success(f"**Predicted Author:** {predicted_author_name}")
            st.metric(
                "Minimum Perplexity",
                f"{min_perplexity:.2f}",
                delta=None,
            )

        st.divider()

        # Display all scores in a table
        st.subheader("📊 Detailed Perplexity Scores")
        results_df = pd.DataFrame(
            [
                {
                    "Author": AUTHOR_NAMES[code],
                    "Perplexity": f"{ppl:.2f}",
                    "Score": ppl,
                }
                for code, ppl in sorted(results.items(), key=lambda x: x[1])
            ]
        )
        st.dataframe(results_df[["Author", "Perplexity"]], use_container_width=True)

        # Bar chart comparison
        st.subheader("📈 Perplexity Comparison")
        chart_data = pd.DataFrame(
            {
                "Author": [AUTHOR_NAMES[code] for code in authors],
                "Perplexity": [results[code] for code in authors],
            }
        )
        st.bar_chart(chart_data.set_index("Author"))

        # Interpretation
        st.info(
            f"✅ **Interpretation:** The manuscript exhibits stylistic patterns most "
            f"consistent with **{predicted_author_name}** (perplexity: {min_perplexity:.2f})."
        )


if __name__ == "__main__":
    main()
