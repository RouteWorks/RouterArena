# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import random
import nltk
from nltk.corpus import wordnet, stopwords

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)

_STOPWORDS = set(stopwords.words("english"))


def _get_wordnet_pos(treebank_tag: str) -> str | None:
    """Map a Penn Treebank POS tag to a WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def _synonym_swap(text: str, swap_rate: float = 0.15, rng: random.Random = None) -> str:
    """
    Replace swap_rate fraction of eligible tokens with a WordNet synonym.
    Eligible = not a stopword, has at least one synonym in WordNet.
    Returns the original text unchanged if no eligible tokens are found.
    """
    if rng is None:
        rng = random.Random()

    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    n_to_swap = max(1, int(len(tokens) * swap_rate))
    eligible_indices = [
        i
        for i, (tok, tag) in enumerate(pos_tags)
        if tok.lower() not in _STOPWORDS
        and tok.isalpha()
        and _get_wordnet_pos(tag) is not None
    ]

    if not eligible_indices:
        return text

    swap_indices = set(
        rng.sample(eligible_indices, min(n_to_swap, len(eligible_indices)))
    )

    result = []
    for i, (tok, tag) in enumerate(pos_tags):
        if i not in swap_indices:
            result.append(tok)
            continue

        wn_pos = _get_wordnet_pos(tag)
        synsets = wordnet.synsets(tok, pos=wn_pos)
        synonyms = [
            lemma.name().replace("_", " ")
            for syn in synsets
            for lemma in syn.lemmas()
            if lemma.name().lower() != tok.lower()
        ]

        if synonyms:
            result.append(rng.choice(synonyms))
        else:
            result.append(tok)

    return " ".join(result)


def augment_records(
    records: list[dict], n_paraphrases: int = 1, seed: int = 42
) -> list[dict]:
    """
    For each record with a numeric budget, generate n_paraphrases variants by
    synonym substitution. Augmented records carry the same model, budget,
    accuracy, and cost as the original — only the prompt text changes.

    Returns only the augmented records (not the originals), so the caller
    can concatenate: all_records = records + augment_records(records)
    """
    rng = random.Random(seed)
    eligible = [r for r in records if r["budget"] is not None]
    augmented = []

    for record in eligible:
        for _ in range(n_paraphrases):
            new_prompt = _synonym_swap(record["prompt"], rng=rng)
            augmented.append(
                {
                    "global_index": record["global_index"],
                    "prompt": new_prompt,
                    "model": record["model"],
                    "budget": record["budget"],
                    "accuracy": record["accuracy"],
                    "cost": record["cost"],
                }
            )

    return augmented
