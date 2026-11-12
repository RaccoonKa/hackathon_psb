from src.indexing.tokenizer_utils import count_tokens


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_returns_int():
    n = count_tokens("привет мир")
    assert isinstance(n, int)


def test_count_tokens_positive_for_text():
    assert count_tokens("привет мир") > 0


def test_count_tokens_monotone():
    assert count_tokens("привет") <= count_tokens("привет мир")


def test_count_tokens_longer_than_shorter():
    short = count_tokens("а")
    long = count_tokens("а" * 1000)
    assert long > short


def test_count_tokens_punctuation():
    assert count_tokens("...") >= 0


def test_count_tokens_cyrillic():
    n = count_tokens("Съешь же ещё этих мягких французских булок")
    assert n > 0