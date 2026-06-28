# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Match simple emoticons, dataset emojis, and lowercase words.
TOKEN_PATTERN = re.compile(
    r":-?\)|:-?\(|💀|🥲|😂|[a-z']+"
)

NEGATORS = {"not", "never"}

POSITIVE_SIGNALS = {":)", ":-)", "😂"}
NEGATIVE_SIGNALS = {":(", ":-(", "💀"}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Lowercases the text, then extracts words, simple emoticons, and
        a small set of emojis used in the dataset.
        """
        cleaned = text.strip().lower()
        return TOKEN_PATTERN.findall(cleaned)

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _sentiment_delta(self, token: str) -> Optional[int]:
        """Return +1, -1, or None for a single token."""
        if token in self.positive_words or token in POSITIVE_SIGNALS:
            return 1
        if token in self.negative_words or token in NEGATIVE_SIGNALS:
            return -1
        return None

    def _walk_tokens(
        self, tokens: List[str]
    ) -> Tuple[int, List[str], List[str]]:
        """
        Loop over tokens, adjust a running score, and collect evidence.

        Starts at 0. Positive matches add points; negative matches subtract.
        Negation flips the next sentiment token. "no cap" is not treated as
        negation.
        """
        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negate_next = False

        for i, token in enumerate(tokens):
            # "no cap" is slang, not negation.
            if token == "no" and i + 1 < len(tokens) and tokens[i + 1] == "cap":
                continue

            if token in NEGATORS:
                negate_next = True
                continue

            base = self._sentiment_delta(token)
            if base is not None:
                delta = -base if negate_next else base
                score += delta
                if delta > 0:
                    positive_hits.append(token)
                else:
                    negative_hits.append(token)
                negate_next = False
            else:
                negate_next = False

        return score, positive_hits, negative_hits

    def score_text(self, text: str) -> int:
        """
        Compute a numeric mood score for the given text.

        Starts at 0 and loops over preprocessed tokens. Each positive match
        adds 1; each negative match subtracts 1. Negation flips the next
        sentiment word (e.g. "not happy" subtracts instead of adding).
        """
        tokens = self.preprocess(text)
        score, _, _ = self._walk_tokens(tokens)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Returns "mixed" when both positive and negative evidence appear.
        Otherwise uses the score sign, with zero mapping to "neutral".
        """
        tokens = self.preprocess(text)
        score, positive_hits, negative_hits = self._walk_tokens(tokens)

        if positive_hits and negative_hits and score == 0:
            return "mixed"

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.
        """
        tokens = self.preprocess(text)
        score, positive_hits, negative_hits = self._walk_tokens(tokens)

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
