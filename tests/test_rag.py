"""GPU 없이 법령 검색과 입력 예산·근거 분리를 검사합니다."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import unicodedata

SPEC = importlib.util.spec_from_file_location(
    "submission_script", Path(__file__).resolve().parents[1] / "open/baseline/script.py")
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


def record(text):
    return {"id": "check", "docs": [{"doc_id": "d1", "type": "공고문", "text": text}], "meta": {}}


class RagTests(unittest.TestCase):
    def test_retrieval_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "법령패키지/법령"
            folder.mkdir(parents=True)
            (folder / "예시법.txt").write_text(
                "표제\n제1조(지역제한) 지역제한 경쟁입찰\n"
                "제2조(소프트웨어) 소프트웨어 사업자\n"
                "제3조(보증금) 보증금 반환\n"
                "제4조(계약기간) 계약기간 연장\n", encoding="utf-8")
            index = script.LawIndex(tmp)
            query = record(unicodedata.normalize("NFD", "지역제한 경쟁입찰"))
            result = index.search(query, topk=1, max_chars=100)
            self.assertIn("제1조(지역제한)", result)
            self.assertNotIn("제2조", result)
            self.assertLessEqual(len(index.search(query, 8, 24)), 24)
            self.assertEqual(index.search(record("☃"), 8, 100), "")
            self.assertEqual(index.search(record("zzzzz"), 8, 100), "")

    def test_missing_laws_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                script.LawIndex(tmp)

    def test_rag_is_in_prompt_but_never_an_evidence_source(self):
        rec = record("이 문장은 공고 원문입니다.")
        law = "법령만의 고유 문장입니다."
        messages = script.build_messages(rec, "지시", 4000, law)
        self.assertIn(law, messages[1]["content"])
        self.assertIn(rec["docs"][0]["text"], messages[1]["content"])
        prediction = {"v1": {"위반여부": 1, "근거문구": law}}
        self.assertEqual(script.postprocess(prediction, rec)["v1"]["근거문구"], "")

    def test_over_budget_rag_shrinks_before_notice(self):
        rec = record("공고 내용 " * 100)
        runner = script.MockRunner({})
        messages, count, chars = script.fit_to_budget(
            rec, "지시", runner, 4000, budget=800, law_context="법령 조문 " * 4000)
        self.assertLessEqual(count, 800)
        self.assertEqual(chars, 4000)
        self.assertIn(rec["docs"][0]["text"], messages[1]["content"])

    def test_impossible_budget_does_not_reach_model(self):
        with self.assertRaises(ValueError):
            script.fit_to_budget(record("공고"), "지시" * 1000,
                                 script.MockRunner({}), 4000, budget=100)


if __name__ == "__main__":
    unittest.main()
