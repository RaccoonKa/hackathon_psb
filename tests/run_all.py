import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from tests import _common

MODULES = [
    "tests.test_config",
    "tests.test_types",
    "tests.test_tokenizer",
    "tests.test_parser",
    "tests.test_parser_deep",
    "tests.test_parser_dates",
    "tests.test_chunker",
    "tests.test_chunker_deep",
    "tests.test_chunker_overlap",
    "tests.test_keyword_store",
    "tests.test_keyword_store_deep",
    "tests.test_vector_store",
    "tests.test_vector_store_deep",
    "tests.test_vector_store_local",
    "tests.test_hybrid",
    "tests.test_hybrid_deep",
    "tests.test_reranker",
    "tests.test_prompts",
    "tests.test_intent",
    "tests.test_context_builder",
    "tests.test_generator",
    "tests.test_refusal",
    "tests.test_llm_client",
    "tests.test_retrieval_bridge",
]


def run_module(name):
    mod = __import__(name, fromlist=["_"])
    passed = failed = skipped = 0
    for attr in sorted(dir(mod)):
        if not attr.startswith("test_"):
            continue
        fn = getattr(mod, attr)
        if not callable(fn):
            continue
        try:
            fn()
            passed += 1
            print(f"  PASS {name}.{attr}")
        except Exception as e:
            if type(e).__name__ == "SkipTest":
                skipped += 1
                print(f"  SKIP {name}.{attr}: {e}")
                continue
            failed += 1
            print(f"  FAIL {name}.{attr}: {e}")
            traceback.print_exc()
    return passed, failed, skipped


def main():
    tp = tf = ts = 0
    for name in MODULES:
        print(f"\n=== {name} ===")
        try:
            p, f, s = run_module(name)
        except Exception as e:
            print(f"  ERROR importing {name}: {e}")
            traceback.print_exc()
            tf += 1
            continue
        tp += p
        tf += f
        ts += s
        print(f"  {p} passed, {f} failed, {s} skipped")
    print(f"\n=== TOTAL: {tp} passed, {tf} failed, {ts} skipped ===")
    return 0 if tf == 0 else 1


if __name__ == "__main__":
    sys.exit(main())