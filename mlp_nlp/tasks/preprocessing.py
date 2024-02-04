from tqdm.auto import tqdm
from more_itertools import windowed
import conllu
import numpy as np
import pandas as pd

import itertools


def preprocess(df: pd.DataFrame, window_size: int, max_window_count: int, embedding_model, pad_token: str = "<PAD>") \
        -> tuple[np.ndarray, list[str]]:
    print("Calculating windows...")
    windows = windowed_sentence(df, window_size, pad_token)

    print("Calculating targets...")
    targets = windowed_tags(df, window_size, pad_token)

    print("Computing window embeddings...")
    window_lim = min(max_window_count, len(windows))
    embeddings = compute_embeddings(embedding_model, windows[:window_lim], pad_token)
    embeddings = np.array(embeddings).T

    return embeddings, targets[:window_lim]


def windowed_sentence(df, window_size: int, pad_token: str):
    windows = []

    for sentence_id in tqdm(set(df.sent_id)):
        words_df = df[df.sent_id == sentence_id]
        sentence = list(words_df.words)
        padding = [pad_token] * (window_size // 2)

        for window in windowed(itertools.chain(padding, sentence, padding), window_size, fillvalue=pad_token):
            windows.append(window)

    return windows


def windowed_tags(df,  window_size: int, pad_token: str):
    targets = []

    for sentence_id in tqdm(set(df.sent_id)):
        words_df = df[df.sent_id == sentence_id]
        tags = list(words_df.pos)
        padding = [pad_token] * (window_size // 2)

        for window in windowed(itertools.chain(padding, tags, padding), window_size, fillvalue=pad_token):
            targets.append(window[window_size // 2])

    return targets


def compute_embeddings(model, windows, pad_token: str) -> list[np.ndarray]:
    """
    Compute context-aware window embeddings for a list of windowed strings.
    :param pad_token: the string with which padding at the start and end of the sentence will be represented
    :param model: a spacy model used to compute individual word embeddings
    :param windows: a list of windows, each being a list of strings

    :return: A list of window embeddings.
    :rtype: list[str]
    """
    embeddings = []
    embedding_size = 300

    for window in tqdm(windows):
        word_embeddings = []

        for word in window:
            if word == pad_token:
                word_embeddings.append(np.zeros(embedding_size))
            else:
                word_embeddings.append(model(word).vector)

        embeddings.append(np.hstack(word_embeddings))

    return embeddings


def conllu_to_pd(file_path: str) -> pd.DataFrame:
    print("\tReading data...")
    with open(file_path, "r") as file:
        data = file.read()

    print("\tParsing data...")
    conllu_sentences = conllu.parse(data)

    print("\tGetting words...")
    words = [[word["form"].lower() for word in sentence] for sentence in tqdm(conllu_sentences)]

    print("\tGetting POS tags...")
    pos = [[word["upos"] for word in sentence] for sentence in tqdm(conllu_sentences)]

    print("\tGetting Sentence ids...")
    sent_ids = [[sent.metadata["sent_id"]] * len(sent) for sent in tqdm(conllu_sentences)]

    return pd.DataFrame({"words": itertools.chain.from_iterable(words),
                         "pos": itertools.chain.from_iterable(pos),
                         "sent_id": itertools.chain.from_iterable(sent_ids)})
