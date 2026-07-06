"""Category registry shared by the data generator, prompts and the eval harness.

Categories loosely follow MCC groupings. Keep this list in sync with the
merchant pool in ``merchant_llm.data.generate`` (there is a test for that).
"""

CATEGORIES: tuple[str, ...] = (
    "grocery",
    "fuel",
    "restaurants",
    "fast_food",
    "coffee",
    "streaming",
    "airlines",
    "hotels",
    "rideshare",
    "transit",
    "pharmacy",
    "electronics",
    "clothing",
    "home_improvement",
    "telecom",
    "utilities",
    "fitness",
    "entertainment",
    "online_retail",
    "insurance",
)

CATEGORY_SET = frozenset(CATEGORIES)


def is_valid_category(value: str) -> bool:
    return value in CATEGORY_SET
