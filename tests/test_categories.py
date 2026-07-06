import re

from merchant_llm.categories import CATEGORIES, CATEGORY_SET, is_valid_category
from merchant_llm.data.generate import MERCHANTS
from merchant_llm.parsing import normalize_name

SNAKE_CASE = re.compile(r"^[a-z]+(_[a-z]+)*$")


def test_category_registry_shape():
    assert len(CATEGORIES) == 20
    assert len(CATEGORY_SET) == 20
    for category in CATEGORIES:
        assert SNAKE_CASE.match(category)
    assert is_valid_category("grocery")
    assert not is_valid_category("Groceries")


def test_generator_pool_matches_registry():
    # every category in the registry has merchants, and no stray categories exist
    assert set(MERCHANTS) == CATEGORY_SET


def test_pool_size_and_uniqueness():
    all_names = [name for names in MERCHANTS.values() for name in names]
    assert len(all_names) >= 200
    assert len(set(all_names)) == len(all_names)
    # normalized forms stay unique too, so name matching is unambiguous
    normalized = [normalize_name(name) for name in all_names]
    assert len(set(normalized)) == len(normalized)
    for names in MERCHANTS.values():
        assert len(names) >= 10
