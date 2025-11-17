import os
import numpy as np
from transformers import T5TokenizerFast

DATA_DIR = "data"

# ------------------------
# Helpers
# ------------------------

def load_lines(path):
    with open(path, "r") as f:
        return [l.strip() for l in f.readlines()]


def preprocess_nl(text):
    """
    Matches the exact behavior in T5Dataset:
      - strip
      - lower
      - prefix with 'translate to SQL: '
      - single-space normalize
    """
    text = text.strip().lower()
    text = " ".join(text.split())
    return "translate to SQL: " + text


def preprocess_sql(text):
    """
    Matches T5Dataset:
      - strip
      - remove trailing ';'
      - collapse whitespace
      - lower
    """
    text = text.strip().lower()
    if text.endswith(";"):
        text = text[:-1].strip()
    text = " ".join(text.split())
    return text


def compute_length_stats(tokenized):
    arr = np.array([len(x) for x in tokenized])
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": int(np.median(arr)),
        "std": float(arr.std())
    }


def vocab_size(lines):
    vocab = set()
    for x in lines:
        for tok in x.split():
            vocab.add(tok)
    return len(vocab)


# ------------------------
# Table Printers
# ------------------------

def print_table_1(train_nl_stats, train_sql_stats, dev_nl_stats, dev_sql_stats,
                  train_nl_vocab, train_sql_vocab, dev_nl_vocab, dev_sql_vocab,
                  n_train, n_dev):
    print("==============================================================")
    print("Table 1: Before Pre-processing")
    print("==============================================================")
    print(f"{'Statistics Name':<35}{'Train':>12}{'Dev':>12}")
    print(f"{'Number of examples':<35}{n_train:>12}{n_dev:>12}")
    print(f"{'Mean sentence length (NL)':<35}{train_nl_stats['mean']:>12.2f}{dev_nl_stats['mean']:>12.2f}")
    print(f"{'Mean SQL query length':<35}{train_sql_stats['mean']:>12.2f}{dev_sql_stats['mean']:>12.2f}")
    print(f"{'Vocabulary size (NL)':<35}{train_nl_vocab:>12}{dev_nl_vocab:>12}")
    print(f"{'Vocabulary size (SQL)':<35}{train_sql_vocab:>12}{dev_sql_vocab:>12}")
    print()


def print_table_2(train_nl_stats, train_sql_stats, dev_nl_stats, dev_sql_stats,
                  train_nl_vocab, train_sql_vocab, dev_nl_vocab, dev_sql_vocab):
    print("==============================================================")
    print("Table 2: After Pre-processing")
    print("==============================================================")
    print(f"{'Statistics Name':<35}{'Train':>12}{'Dev':>12}")
    print(f"{'Model name':<35}{'t5-small':>12}{'t5-small':>12}")
    print(f"{'Mean sentence length (NL)':<35}{train_nl_stats['mean']:>12.2f}{dev_nl_stats['mean']:>12.2f}")
    print(f"{'Mean SQL query length':<35}{train_sql_stats['mean']:>12.2f}{dev_sql_stats['mean']:>12.2f}")
    print(f"{'Vocabulary size (NL)':<35}{train_nl_vocab:>12}{dev_nl_vocab:>12}")
    print(f"{'Vocabulary size (SQL)':<35}{train_sql_vocab:>12}{dev_sql_vocab:>12}")
    print()


# ------------------------
# Main
# ------------------------

if __name__ == "__main__":
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    # raw data
    train_x = load_lines(os.path.join(DATA_DIR, "train.nl"))
    train_y = load_lines(os.path.join(DATA_DIR, "train.sql"))
    dev_x = load_lines(os.path.join(DATA_DIR, "dev.nl"))
    dev_y = load_lines(os.path.join(DATA_DIR, "dev.sql"))

    # tokenize BEFORE preprocessing
    raw_train_nl_ids = [tokenizer(x).input_ids for x in train_x]
    raw_train_sql_ids = [tokenizer(y).input_ids for y in train_y]
    raw_dev_nl_ids = [tokenizer(x).input_ids for x in dev_x]
    raw_dev_sql_ids = [tokenizer(y).input_ids for y in dev_y]

    # preprocess NL + SQL with new logic
    proc_train_x = [preprocess_nl(x) for x in train_x]
    proc_train_y = [preprocess_sql(y) for y in train_y]
    proc_dev_x   = [preprocess_nl(x) for x in dev_x]
    proc_dev_y   = [preprocess_sql(y) for y in dev_y]

    # tokenize AFTER preprocessing
    proc_train_nl_ids = [tokenizer(x).input_ids for x in proc_train_x]
    proc_train_sql_ids = [tokenizer(y).input_ids for y in proc_train_y]
    proc_dev_nl_ids    = [tokenizer(x).input_ids for x in proc_dev_x]
    proc_dev_sql_ids   = [tokenizer(y).input_ids for y in proc_dev_y]

    # print tables
    print_table_1(
        compute_length_stats(raw_train_nl_ids),
        compute_length_stats(raw_train_sql_ids),
        compute_length_stats(raw_dev_nl_ids),
        compute_length_stats(raw_dev_sql_ids),
        vocab_size(train_x),
        vocab_size(train_y),
        vocab_size(dev_x),
        vocab_size(dev_y),
        len(train_x),
        len(dev_x)
    )

    print_table_2(
        compute_length_stats(proc_train_nl_ids),
        compute_length_stats(proc_train_sql_ids),
        compute_length_stats(proc_dev_nl_ids),
        compute_length_stats(proc_dev_sql_ids),
        vocab_size(proc_train_x),
        vocab_size(proc_train_y),
        vocab_size(proc_dev_x),
        vocab_size(proc_dev_y)
    )

    print("Finished computing Q4 statistics.")