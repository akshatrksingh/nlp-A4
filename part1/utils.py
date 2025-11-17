import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

        # -----------------------
    # 1. US → UK spelling map
    # -----------------------
    us_to_uk = {
        "favorite": "favourite",
        "favorites": "favourites",
        "color": "colour",
        "colors": "colours",
        "colorful": "colourful",
        "theater": "theatre",
        "theaters": "theatres",
        "humor": "humour",
        "humorous": "humourous",
        "center": "centre",
        "centered": "centred",
        "criticize": "criticise",
        "criticized": "criticised",
        "criticizing": "criticising",
        "realize": "realise",
        "realized": "realised",
        "realizing": "realising",
        "dialog": "dialogue",
        "catalog": "catalogue",
        "cozy": "cosy",
        "gray": "grey",
        "behavior": "behaviour",
        "behaviors": "behaviours",
        "organize": "organise",
        "organized": "organised",
        "organizing": "organising",
        "apologize": "apologise",
        "apologized": "apologised"
    }

    # -----------------------
    # 2. Gender pronoun swap
    # -----------------------
    pronoun_map = {
        "he": "she", "she": "he",
        "him": "her", "her": "him",
        "his": "hers", "hers": "his",
        "himself": "herself", "herself": "himself"
    }

    # -----------------------
    # 3. Keyboard-neighbor typos
    # -----------------------
    neighbor_keys = {
        "a": ["q", "w", "s", "z"],
        "b": ["v", "g", "h", "n"],
        "c": ["x", "d", "f", "v"],
        "d": ["s", "e", "r", "f", "c", "x"],
        "e": ["w", "s", "d", "r"],
        "f": ["d", "r", "t", "g", "v", "c"],
        "g": ["f", "t", "y", "h", "b", "v"],
        "h": ["g", "y", "u", "j", "n", "b"],
        "i": ["u", "j", "k", "o"],
        "j": ["h", "u", "i", "k", "m", "n"],
        "k": ["j", "i", "o", "l", "m"],
        "l": ["k", "o", "p"],
        "m": ["n", "j", "k"],
        "n": ["b", "h", "j", "m"],
        "o": ["i", "k", "l", "p"],
        "p": ["o", "l"],
        "q": ["w", "a"],
        "r": ["e", "d", "f", "t"],
        "s": ["a", "w", "e", "d", "x", "z"],
        "t": ["r", "f", "g", "y"],
        "u": ["y", "h", "j", "i"],
        "v": ["c", "f", "g", "b"],
        "w": ["q", "a", "s", "e"],
        "x": ["z", "s", "d", "c"],
        "y": ["t", "g", "h", "u"],
        "z": ["a", "s", "x"]
    }


    # -----------------------
    # 4. Light punctuation jitter
    # -----------------------
    punct_insert = ["..", "...", ",", ";", "-", "!", "!!", "!!!", "?", "??", "???"]

    # -----------------------
    # Start transforming
    # -----------------------
    text = example["text"]
    tokens = word_tokenize(text)
    new_tokens = []

    for tok in tokens:
        lower_tok = tok.lower()

        # (A) US → UK spelling (high probability)
        if lower_tok in us_to_uk and random.random() < 0.4:
            new_tokens.append(us_to_uk[lower_tok])
            continue

        # (B) Gender pronoun flip (moderate probability)
        if lower_tok in pronoun_map and random.random() < 0.40:
            new_tokens.append(pronoun_map[lower_tok])
            continue

        # (C) Keyboard-neighbor typo (low probability)
        if len(tok) > 4 and random.random() < 0.2:
            chars = list(tok)
            idx = random.randint(0, len(chars) - 1)
            c = chars[idx].lower()
            if c in neighbor_keys:
                chars[idx] = random.choice(neighbor_keys[c])
                new_tokens.append("".join(chars))
                continue

        # (D) Random synonym substitution (very low probability)
        if random.random() < 0.3:
            syns = wordnet.synsets(lower_tok)
            if syns:
                lemmas = syns[0].lemmas()
                if lemmas:
                    syn_word = lemmas[0].name().replace("_", " ")
                    if syn_word.lower() != lower_tok:
                        new_tokens.append(syn_word)
                        continue

        # (E) Default: keep token
        new_tokens.append(tok)

        # (F) Random punctuation injection AFTER token
        if random.random() < 0.05:
            new_tokens.append(random.choice(punct_insert))

    # Re-detokenize
    detok = TreebankWordDetokenizer().detokenize(new_tokens)
    example["text"] = detok
    return example