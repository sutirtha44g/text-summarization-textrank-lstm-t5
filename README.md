# Text Summarization: TextRank vs LSTM vs T5

A comparison of three text summarization approaches on the CNN/DailyMail dataset:

1. **TextRank** — unsupervised, extractive (graph-based, uses Word2Vec + PageRank)
2. **LSTM Encoder-Decoder** — abstractive, trained from scratch
3. **T5 (t5-small)** — abstractive, pretrained transformer used zero-shot

Each method is scored with **ROUGE-1 / ROUGE-L** on a held-out test set, and the
best-performing model is then applied to a real PDF document as a practical demo.

## Results

| Model    | Type                          | Avg ROUGE-1/L F1 |
|----------|-------------------------------|-------------------|
| TextRank | Extractive, unsupervised      | ~0.24             |
| LSTM     | Abstractive, trained from scratch | ~0.07         |
| T5-small | Abstractive, pretrained       | ~0.27             |

**Why T5 wins:** it comes pretrained on huge amounts of text, so it produces fluent
summaries with zero fine-tuning.

**Why the LSTM underperforms:** it's trained from scratch on only 300 articles —
not nearly enough data for a seq2seq model to learn general language patterns.
With more training data and an attention mechanism, this gap would likely shrink.

**Why TextRank is a strong lightweight baseline:** no training data or GPU required,
yet it lands close to T5 on ROUGE by extracting genuinely relevant sentences.

## How to run

This notebook is built for **Google Colab**.

1. Open `text_summarization_textrank_lstm_t5.ipynb` in Colab
2. Run all cells top to bottom (`Runtime → Run all`)
3. In the "USER TESTING" section, upload any PDF when prompted to generate a
   summary of your own document

> Note: the PDF upload cell uses `google.colab.files` and only works inside
> Google Colab. To run locally, replace that cell with a direct file path,
> e.g. `pdf_path = "your_file.pdf"`.

## Requirements

```
nltk
tensorflow
pandas
numpy
matplotlib
scikit-learn
transformers
datasets
networkx
gensim
rouge-score
pypdf
```

(All installed automatically by the first cell in the notebook.)

## Dataset

[CNN/DailyMail 3.0.0](https://huggingface.co/datasets/abisee/cnn_dailymail) via
Hugging Face `datasets`. A small subset (300 train / 50 test examples) is used
to keep runtime short on free-tier Colab GPUs.
