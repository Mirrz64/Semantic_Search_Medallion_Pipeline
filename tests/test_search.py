import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.search.semantic_search import truncate_snippet  # noqa: E402


def test_short_text_is_untouched():
    assert truncate_snippet("Great product") == "Great product"


def test_long_text_is_truncated_with_ellipsis():
    long_text = "a" * 200
    result = truncate_snippet(long_text, max_len=160)
    assert result.endswith("...")
    assert len(result) == 163  # 160 chars + "..."


def test_newlines_are_flattened():
    text = "Line one\nLine two"
    assert "\n" not in truncate_snippet(text)
