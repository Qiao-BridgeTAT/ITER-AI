"""Identity namespaces are evidence boundaries, not interchangeable labels."""

import re

from backend.contracts.city_registry import CITY_ID_PATTERN


def is_city_entity_ref(value: str) -> bool:
    return re.fullmatch(CITY_ID_PATTERN, value) is not None
