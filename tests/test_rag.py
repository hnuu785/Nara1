"""GPU 없이 hybrid 제출 파이프라인의 핵심 계약을 검사한다."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "open/data"
SPEC = importlib.util.spec_from_file_location(
    "submission_script", ROOT / "open/baseline/script.py")
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def make_record(text="공고 원문", *, meta=None, extra_docs=None, rec_id="check"):
    docs = [{"doc_id": "D0", "type": "공고문", "text": text}]
    docs.extend(extra_docs or [])
    return {"id": rec_id, "docs": docs, "meta": meta or {},
            "input_completeness": {"완전관측": True}}


def compact_output(value=0):
    return json.dumps(
        {item: {"y": value, "s": -1} for item in script.ITEMS},
        ensure_ascii=False, separators=(",", ":"))


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = script.ProductCatalog(str(DATA))
        cls.system = script.build_system_prompt(script.item_table(str(DATA)))

    def test_all_document_types_and_model_candidates_survive(self):
        notice = ("일반 안내 문장입니다.\n" * 1400) + "마지막 공고 문장"
        specification = (
            "규격 안내\n" + ("일반 기능 조건\n" * 300) +
            "제조사: ACME / 모델명: ZX-9000을 납품하여야 한다.\n")
        rec = make_record(
            notice,
            extra_docs=[{"doc_id": "D1", "type": "규격서", "text": specification}])
        spans = script.select_spans(rec, 3000)
        self.assertTrue(any(span.doc_id == "D0" for span in spans))
        self.assertTrue(any(span.doc_id == "D1" for span in spans))
        self.assertTrue(any("ZX-9000" in span.text for span in spans))
        self.assertTrue(all(len(span.text) <= 470 for span in spans))
        for span in spans:
            source = next(doc["text"] for doc in rec["docs"] if doc["doc_id"] == span.doc_id)
            self.assertEqual(source[span.start:span.end], span.text)

    def test_catalog_applies_price_notes(self):
        self.assertEqual(
            script.ProductCatalog._applicability("추정가격 3억원 미만에 한함", 299_999_999),
            "적용조건 원문확인")
        self.assertTrue(
            script.ProductCatalog._applicability(
                "추정가격 3억원 미만에 한함", 300_000_000).startswith("금액기준 제외"))

    def test_high_confidence_synthetic_rules(self):
        rec = make_record(
            "최근 3년간 2억원 이상의 납품 실적이 있는 업체이어야 합니다.\n"
            "공동이행 구성원별 최소 지분율 3% 이상이어야 합니다.\n"
            "사업설명회에 참석한 자에 한하여 제안서를 제출할 수 있습니다.",
            meta={"적용계약법": "지방계약법", "업무구분": "일반용역",
                  "계약방법": "제한경쟁", "낙찰방법": "협상에의한계약",
                  "배정예산금액": 150_000_000, "입찰추정가격": 180_000_000})
        facts = script.analyze_record(rec, self.catalog)
        self.assertTrue(facts["performance_violation"])
        self.assertTrue(facts["joint_violation"])
        self.assertTrue(facts["briefing_violation"])

        unrelated = make_record(
            "입찰참가자격\n가. 자본금 2억원 이상인 업체\n"
            "나. 최근 3년 유사사업 수행실적을 보유한 업체",
            meta={"배정예산금액": 100_000_000, "입찰추정가격": 90_000_000})
        unrelated_facts = script.analyze_record(unrelated, self.catalog)
        self.assertFalse(unrelated_facts["performance_violation"])

        evaluation_only = make_record(
            "입찰참가자격: 관련 업종 등록 업체\n정량평가 수행실적 평가기준\n"
            "2억원 이상: 5점\n1억원 이상: 3점",
            meta={"배정예산금액": 100_000_000, "입찰추정가격": 90_000_000})
        evaluation_facts = script.analyze_record(evaluation_only, self.catalog)
        self.assertFalse(evaluation_facts["performance_violation"])

    def test_v23_same_day_and_short_range_dates(self):
        parsed = [day.isoformat() for day, _pos in script._dates(
            "접수기간 2026.1.10.(토) ~ 1.20.(화)")]
        self.assertEqual(parsed, ["2026-01-10", "2026-01-20"])
        rollover = [day.isoformat() for day, _pos in script._dates(
            "접수기간 2025.12.29. ~ 1.2.")]
        self.assertEqual(rollover, ["2025-12-29", "2026-01-02"])

        rec = make_record(
            "현장설명회 일시: 2026.1.8. 10:00\n"
            "제안서 제출 접수마감 일시: 2026.1.8. 17:00",
            meta={"적용계약법": "지방계약법", "낙찰방법": "협상에의한계약",
                  "입찰추정가격": 50_000_000, "공고게시일자": "20260101"})
        decision, _location, _detail = script._briefing_timing_violation(rec, 50_000_000)
        self.assertTrue(decision)

    def test_hard_gates_for_small_local_quote(self):
        rec = make_record(
            "지역과 실적을 모두 요구합니다.",
            meta={"적용계약법": "지방계약법", "업무구분": "일반용역",
                  "계약방법": "수의계약", "낙찰방법": "소액수의견적",
                  "입찰추정가격": 80_000_000, "배정예산금액": 88_000_000})
        facts = script.analyze_record(rec, self.catalog)
        self.assertTrue({"v2", "v6", "v7", "v8", "v22", "v23"} <= facts["hard_zero"])

    def test_compact_parse_and_exact_evidence(self):
        rec = make_record(
            "기본 공고",
            extra_docs=[{"doc_id": "D1", "type": "규격서",
                         "text": "제조사: ACME, 모델명: ZX-9000 지정"}])
        facts = script.analyze_record(rec, self.catalog)
        spans = script.select_spans(rec, 4000, facts)
        model_span = next(span.index for span in spans if "ZX-9000" in span.text)
        judgment, missing = script.parse_judgment(compact_output())
        self.assertFalse(missing)
        judgment["v9"] = {"y": 1, "s": model_span}
        judgment["v10"] = {"y": 1, "s": model_span}
        result = script.postprocess(judgment, rec, spans, facts)
        self.assertEqual(result["v9"]["위반여부"], 1)
        self.assertIn(result["v9"]["근거문구"], script.full_text(rec))
        self.assertEqual(result["v10"]["근거문구"], "")

    def test_malformed_cells_are_not_silently_coerced(self):
        obj = json.loads(compact_output())
        obj["v1"] = {}
        obj["v2"] = {"y": 7, "s": "junk"}
        _parsed, missing = script.parse_judgment(json.dumps(obj))
        self.assertIn("v1", missing)
        self.assertIn("v2", missing)
        obj = json.loads(compact_output())
        obj["unexpected"] = {"y": 0, "s": -1}
        _parsed, missing = script.parse_judgment(json.dumps(obj))
        self.assertFalse(missing)

    def test_bad_evidence_does_not_discard_valid_positive(self):
        rec = make_record("모델명: ZX-9000 지정")
        facts = script.analyze_record(rec, self.catalog)
        spans = script.select_spans(rec, 4000, facts)
        for span_no in (-2, 513, 999999, None, "S0", True):
            with self.subTest(span_no=span_no):
                obj = json.loads(compact_output())
                obj["v9"] = {"y": 1, "s": span_no}
                parsed, missing = script.parse_judgment(json.dumps(obj))
                self.assertFalse(missing)
                final = script.postprocess(parsed, rec, spans, facts)
                self.assertEqual(final["v9"]["위반여부"], 1)
                self.assertTrue(final["v9"]["근거문구"])
                self.assertIn(final["v9"]["근거문구"], script.full_text(rec))
        obj["v9"] = {"y": 1}
        parsed, missing = script.parse_judgment(json.dumps(obj))
        self.assertFalse(missing)
        self.assertEqual(parsed["v9"], {"y": 1, "s": -1})

    def test_invalid_labels_remain_missing(self):
        for value in (True, False, "1", 1.0, None, -1, 2):
            obj = json.loads(compact_output())
            obj["v9"] = {"y": value, "s": 0}
            _parsed, missing = script.parse_judgment(json.dumps(obj))
            self.assertEqual(missing, ["v9"])

    def test_fit_to_budget_reduces_only_selected_context(self):
        rec = make_record("참가자격 및 실적 조건입니다.\n" * 2500)
        runner = script.MockRunner({})
        messages, spans, _facts, tokens, used = script.fit_to_budget(
            rec, self.system, runner, 18_000, self.catalog, budget=6_000)
        self.assertLessEqual(tokens, 6_000)
        self.assertLess(used, 18_000)
        self.assertTrue(spans)
        self.assertIn("판정 카드", messages[0]["content"])

    def test_batch_failure_is_split_without_losing_rows(self):
        class SingleOnly:
            def chat(self, batch):
                if len(batch) > 1:
                    raise RuntimeError("too large")
                return ["ok"]

        self.assertEqual(script.run_chunk(SingleOnly(), [[1], [2], [3]]), ["ok"] * 3)

    def run_responses(self, responses, count=1, chunk=8):
        calls = []
        queued = iter(responses)

        class ScriptedRunner:
            load_seconds = 0.0

            def __init__(self, _schema, **_kwargs):
                pass

            def count_tokens(self, _messages):
                return 100

            def chat(self, batch):
                calls.append(batch)
                return next(queued)

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl.gz"
            with gzip.open(input_path, "wt", encoding="utf-8") as handle:
                for index in range(count):
                    handle.write(json.dumps(make_record(rec_id=f"case-{index}"),
                                            ensure_ascii=False) + "\n")
            output = Path(tmp) / "submission.csv"
            report = script.run(str(input_path), str(output), ScriptedRunner,
                                None, chunk, 4000, str(DATA))
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        return report, rows, calls

    def test_partial_labels_survive_retry_and_remaining_cells_are_reported(self):
        obj = json.loads(compact_output())
        obj["v9"] = {"y": 1, "s": -1}
        del obj["v24"]
        text = json.dumps(obj)
        report, rows, calls = self.run_responses([[text], [text]])
        self.assertEqual(report["자가검증"], "PASS")
        self.assertEqual(report["누락셀"], 1)
        self.assertEqual(report["보완공고수"], 1)
        self.assertEqual(rows[0]["v9"], "1")
        self.assertEqual(rows[0]["v24"], "0")
        self.assertEqual(len(rows[0]), 49)
        self.assertIn("v24", calls[1][0][0]["content"])
        self.assertNotEqual(calls[0], calls[1])

    def test_retry_merges_only_missing_labels(self):
        first = json.loads(compact_output())
        first["v9"] = {"y": 1, "s": -1}
        del first["v24"]
        second = {"v24": {"y": 1, "s": -1}}
        report, rows, _calls = self.run_responses(
            [[json.dumps(first)], [json.dumps(second)]])
        self.assertEqual(report["누락셀"], 0)
        self.assertEqual(rows[0]["v9"], "1")
        self.assertEqual(rows[0]["v24"], "1")

    def test_retry_batches_respect_chunk_and_preserve_record_order(self):
        first = json.loads(compact_output())
        del first["v24"]
        partial = json.dumps(first)
        report, rows, calls = self.run_responses(
            [[partial] * 2, [partial], [compact_output()] * 2, [compact_output()]],
            count=3, chunk=2)
        self.assertEqual([len(batch) for batch in calls], [2, 1, 2, 1])
        self.assertEqual(report["재시도공고수"], 3)
        self.assertEqual(report["누락셀"], 0)
        self.assertEqual([row["id"] for row in rows], [f"case-{i}" for i in range(3)])

    def test_unusable_model_output_still_fails(self):
        for text in ("", "not json", "{}", '{"v1":{"y":7,"s":-1}}'):
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, "유효한 모델 판정 없음"):
                self.run_responses([[text], [text]])

    def test_failed_retry_keeps_first_valid_labels(self):
        obj = json.loads(compact_output())
        obj["v9"] = {"y": 1, "s": -1}
        del obj["v24"]
        report, rows, _calls = self.run_responses([[json.dumps(obj)], ["not json"]])
        self.assertEqual(report["누락셀"], 1)
        self.assertEqual(rows[0]["v9"], "1")

    def test_mock_end_to_end_csv_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "submission.csv"
            report = script.run(
                str(DATA / "test.jsonl.gz"), str(output), script.MockRunner,
                3, 8, 18_000, str(DATA))
            self.assertEqual(report["자가검증"], "PASS")
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0], script.COLUMNS)
            self.assertTrue(all(len(row) == 49 for row in rows))

    def test_dev_guards_and_evidence_recall(self):
        labels_path = ROOT / "open/dev_labels.csv"
        dev_path = ROOT / "open/dev.jsonl.gz"
        if not labels_path.exists() or not dev_path.exists():
            self.skipTest("dev 자료 없음")
        with labels_path.open(encoding="utf-8", newline="") as handle:
            labels = {row["id"]: row for row in csv.DictReader(handle)}
        records = list(script.iter_records(str(dev_path)))

        forced = {item: set() for item in ("v2", "v3", "v21", "v22", "v23")}
        evidence_total = evidence_kept = 0
        for rec in records:
            facts = script.analyze_record(rec, self.catalog)
            gold = labels[rec["id"]]
            for item in script.ITEMS:
                self.assertFalse(
                    gold[item] == "1" and item in facts["hard_zero"],
                    f"gold positive suppressed: {rec['id']} {item}")
            if facts["performance_violation"]:
                forced["v3"].add(rec["id"])
                if "v2" not in facts["hard_zero"]:
                    forced["v2"].add(rec["id"])
            if facts["joint_violation"]:
                forced["v21"].add(rec["id"])
            if facts["briefing_violation"]:
                forced["v22"].add(rec["id"])
            if facts["briefing_timing_violation"]:
                forced["v23"].add(rec["id"])

            spans = script.select_spans(rec, script.DEFAULT_CONTEXT_CHARS, facts)
            selected = "\n".join(span.text for span in spans)
            for index in range(1, 25):
                evidence = gold[f"e{index}"]
                if evidence:
                    evidence_total += 1
                    evidence_kept += int(evidence in selected)

        expected_counts = {"v2": 3, "v3": 8, "v21": 6, "v22": 5, "v23": 5}
        self.assertEqual({item: len(ids) for item, ids in forced.items()}, expected_counts)
        for item, ids in forced.items():
            self.assertTrue(all(labels[rec_id][item] == "1" for rec_id in ids))
        self.assertEqual((evidence_kept, evidence_total), (54, 54))


if __name__ == "__main__":
    unittest.main()
