"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "hopeful",
    "proud",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "Lowkey stressed but proud I finished it",
    "No cap this was actually a great day :)",
    "I just love when my code breaks five minutes before class 💀",
    "Feeling kinda meh about everything today",
    "Highkey tired but excited for what comes next 🥲",
    "I am so happy today",
    "Feeling happy and relaxed",
    "Honestly just a good chill day",
    "This exam was awful and I'm so upset",
    "Lowkey bored but whatever",
    "That meme had me crying from laughter 😂",
    "I absolutely love getting stuck in traffic for an hour",
    "Not gonna lie I'm kinda sad and tired today",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "mixed",     # "Lowkey stressed but proud I finished it"
    "positive",  # "No cap this was actually a great day :)"
    "negative",  # "I just love when my code breaks five minutes before class 💀"
    "neutral",   # "Feeling kinda meh about everything today"
    "mixed",     # "Highkey tired but excited for what comes next 🥲"
    "positive",  # "I am so happy today"
    "positive",  # "Feeling happy and relaxed"
    "positive",  # "Honestly just a good chill day"
    "negative",  # "This exam was awful and I'm so upset"
    "neutral",   # "Lowkey bored but whatever"
    "positive",  # "That meme had me crying from laughter 😂"
    "negative",  # "I absolutely love getting stuck in traffic for an hour"
    "negative",  # "Not gonna lie I'm kinda sad and tired today"
]

# TODO: Add more posts and labels as you experiment.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
