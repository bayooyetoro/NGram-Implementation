# 🔍 Forensic Stylometry Engine

> **Authorship Attribution via N-gram Perplexity Analysis**

Identify the author of anonymous manuscripts by analyzing linguistic patterns using trigram language models. This system compares stylometric signatures against three classic authors and renders a verdict with statistical confidence.

---

## 🎯 What It Does

```
Anonymous Manuscript
        ↓
   [Tokenization]
        ↓
┌───────────────────────────────┐
│ Calculate Perplexity Against: │
│  • Edgar Allan Poe (EAP)      │
│  • H.P. Lovecraft (HPL)       │
│  • Mary Wollstonecraft Shelley│
└───────────────────────────────┘
        ↓
   [Compare Scores]
        ↓
   PREDICTED AUTHOR
```

**How it works:** N-gram models learn the likelihood of word sequences from each author's corpus. When evaluating a new text, perplexity measures how "surprised" each model is by the sequence. Lower perplexity = stylistic match.

---

## 📋 Prerequisites

- **Python:** 3.11+
- **OS:** macOS, Linux, or Windows
- **Git:** For cloning and version control

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/bayooyetoro/NGram-Implementation.git
cd NGram-Implementation
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📊 Usage

### Via Streamlit Web Interface (Recommended)

1. **Launch the app** (see Quick Start, step 4)
2. **Paste manuscript text** into the text area or use the provided sample
3. **Click "Run Forensic Attribution"**
4. **Review results:**
   - 🎯 **Predicted author** (highlighted)
   - 📈 **Perplexity scores** for all three suspects
   - 📊 **Visual comparison chart**

### Via Command Line (Direct Model API)

```bash
cd src
python main.py
```

This runs the pipeline once on a random test sample.

---

## 📁 Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── app.py                       # Streamlit web interface (NEW)
├── data/
│   ├── train.csv               # Training corpus (3 authors)
│   ├── test.csv                # Evaluation samples
│   └── sample_submission.csv    # Submission template
└── src/
    ├── preprocessing.py         # Tokenizer class
    ├── ngram_model.py           # N-gram model implementation
    ├── evaluator.py             # Perplexity calculator
    └── main.py                  # CLI pipeline
```

---

## 👥 The Three Suspects

| Author | Code | Era | Style Notes |
|--------|------|-----|-------------|
| **Edgar Allan Poe** | `EAP` | 1809–1849 | Gothic, melancholic, ornate vocabulary |
| **H.P. Lovecraft** | `HPL` | 1890–1937 | Cosmic horror, philosophical, dense prose |
| **Mary W. Shelley** | `MWS` | 1797–1851 | Romantic, introspective, scientific wonder |

---

## 🔧 How Perplexity Works

Perplexity is the inverse probability of a sequence normalized by length:

$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | \text{context})\right)$$

**In plain English:**
- Train a trigram model on each author's works
- For each word in the test text, predict it from the two preceding words
- A *low* perplexity score means the model predicted well → stylistic match
- Compare perplexity across all three models → lowest score wins

---

## 📊 Understanding Results

### Example Output

```
┌─────────────────────────────────────┐
│ Predicted Author: Edgar Allan Poe   │
├─────────────────────────────────────┤
│ EAP Perplexity:  45.23  ← WINNER    │
│ HPL Perplexity:  58.67              │
│ MWS Perplexity:  62.41              │
└─────────────────────────────────────┘
```

✅ **Strong confidence:** Gap of 10+ between winner and second place  
⚠️ **Weak confidence:** Gap of 5 or less (models less certain)

---

## 🏗️ Core Components

### `Tokenizer` (`preprocessing.py`)
- Lowercases and normalizes text
- Splits on word boundaries and punctuation
- Builds vocabulary from training corpus (top 10k words)
- Replaces OOV words with `<UNK>` token
- Wraps sequences with `<s>` (start) and `</s>` (end) markers

### `NGramModel` (`ngram_model.py`)
- Stores n-gram counts and context totals
- Implements Laplace smoothing (alpha=1.0)
- Computes conditional probabilities: P(word | context)
- Converts to log-space to prevent underflow

### `calculate_perplexity()` (`evaluator.py`)
- Iterates through n-grams in a sequence
- Accumulates log-probabilities
- Averages and exponentiates to perplexity scale

---

## ⚡ Performance Notes

**Model Loading (~5–10 seconds on first run):**
- Trains trigram matrices on 3 author corpuses
- Builds vocabulary from ~10k most common words
- Streamlit caches this with `@st.cache_resource` → no re-training on UI interaction

**Evaluation (~50–100ms per submission):**
- Fast token lookup and probability computation
- No network calls or external API dependencies

---

## 🚀 Deployment

### Local Development

```bash
streamlit run app.py
```

### Streamlit Community Cloud (Free)

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your GitHub repo and `app.py` as the entry point
4. Deploy—auto-updates on each push

### Hugging Face Spaces

1. Create a new Space with Streamlit runtime
2. Connect your GitHub repo
3. Set entrypoint to `app.py`
4. Auto-deploys on push

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t forensic-stylometry .
docker run -p 8501:8501 forensic-stylometry
```

---

## 🧪 Testing Your Setup

After installation, verify everything works:

```bash
# Test imports
python -c "from src.preprocessing import Tokenizer; print('✅ Tokenizer OK')"
python -c "from src.ngram_model import NGramModel; print('✅ NGramModel OK')"
python -c "from src.evaluator import calculate_perplexity; print('✅ Evaluator OK')"

# Test CLI pipeline
cd src && python main.py

# Test Streamlit app
streamlit run app.py
```

---

## 📈 Key Files Explained

| File | Purpose | Read-Only? |
|------|---------|-----------|
| `preprocessing.py` | Tokenization & vocabulary | ✅ Yes |
| `ngram_model.py` | Language model core | ✅ Yes |
| `evaluator.py` | Perplexity metric | ✅ Yes |
| `main.py` | CLI orchestration | ✅ Yes |
| `app.py` | **Web UI (NEW)** | — |

The web interface (`app.py`) is built entirely *on top* of the existing core modules—no refactoring of core logic.

---

## 🔮 Future Enhancements

- [ ] Support for arbitrary author corpuses (user-uploaded)
- [ ] Confidence intervals via bootstrap resampling
- [ ] N-gram size comparison (bigrams vs trigrams vs 4-grams)
- [ ] Text preprocessing options (stopword removal, stemming)
- [ ] Batch submission (CSV of multiple manuscripts)
- [ ] Model interpretability (top predictive n-grams per author)

---

## ⚙️ Configuration

Edit these constants in `app.py` to customize:

```python
AUTHOR_NAMES = {
    "EAP": "Edgar Allan Poe",
    "HPL": "H.P. Lovecraft",
    "MWS": "Mary Wollstonecraft Shelley",
}
NGRAM_SIZE = 3                    # Trigrams
VOCAB_SIZE = 10000                # Top 10k words
TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test.csv"
```

---

## 📝 Data Format

### Training Data (`data/train.csv`)

```csv
id,text,author
id26305,"This process, however, afforded me no means...",EAP
id17569,"It never once occurred to me...",HPL
```

**Columns:**
- `id` — Unique identifier
- `text` — Full manuscript excerpt
- `author` — Author code (EAP, HPL, or MWS)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` |
| `ValueError: Build Vocabulary first` | Ensure training data is loaded before tokenization |
| `Perplexity = inf` | Test text too short (< 3 tokens). Increase text length. |
| `Port 8501 already in use` | Run `streamlit run app.py --server.port 8502` |

---

## 📚 References

- **N-gram Language Models:** Jurafsky & Martin, *Speech and Language Processing* (Chapter 3)
- **Stylometry:** Mosteller & Wallace, "Inference in an Authorship Problem" (1964)
- **Laplace Smoothing:** Chen & Goodman, "An Empirical Study of Smoothing Techniques for Language Modeling" (1999)

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

Found a bug or have an idea? Submit an issue or pull request.

---

**Built with ❤️ using Python, Streamlit, and classic literature**
