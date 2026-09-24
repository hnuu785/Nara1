#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""나라장터 24개 법령 위반 판정 제출 파이프라인.

공고마다 고정 Gemma를 한 번 호출하되, 모든 문서에서 법률 단서가 있는 원문
span을 골고루 수록하고 모델은 근거문장 대신 span 번호만 반환한다. 확정 가능한
금액/계약유형 규칙만 후처리하고 평가 공고끼리 정보나 통계를 공유하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = os.environ.get("PPS_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("PPS_OUTPUT_DIR", "./output")
MODEL_DIR = os.environ.get("PPS_MODEL_DIR", "/opt/models/gemma-4-26B-A4B-it")
ITEMS = [f"v{i}" for i in range(1, 25)]
EVID = [f"e{i}" for i in range(1, 25)]
COLUMNS = ["id"] + ITEMS + EVID
ABSENCE = {"v10", "v11", "v16", "v18", "v20"}
META_FIELDS = [
    "적용계약법", "업무구분", "계약방법", "낙찰방법", "낙찰하한율",
    "배정예산금액", "입찰추정가격", "소관구분", "공동도급구성방식", "정보화사업여부",
    "세부품명번호목록", "제한지역코드목록", "지역제한여부", "면허업종제한목록", "업종제한여부",
    "조항호내용", "공고게시일자", "개찰예정일자", "긴급공고여부", "입찰방법", "조달방식",
]
DOC_ORDER = {name: i for i, name in enumerate(
    ["공고문", "규격서", "과업지시서", "제안요청서", "예외공표서", "기타"]
)}
SEED = 20260826
MAX_MODEL_LEN = 32768                   # 대회 공식 최대 컨텍스트 상한 (기존 16384 -> 32768 확장)
MAX_TOKENS = 768
TOKEN_SAFETY_MARGIN = 96
PROMPT_BUDGET = MAX_MODEL_LEN - MAX_TOKENS - TOKEN_SAFETY_MARGIN
DEFAULT_CONTEXT_CHARS = 18000
MIN_CONTEXT_CHARS = 1800
EVIDENCE_MAX = 500
QUANT = "int8_per_channel_weight_only"
NOTICE_AMOUNT = 230_000_000
SMALL_AMOUNT = 100_000_000
MICRO_AMOUNT = 20_000_000


def log(message: str) -> None:
    print(f"[nara-v2] {message}", file=sys.stderr, flush=True)


def _open(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, "r", encoding="utf-8")


def validate_record(rec: Any) -> None:
    if not isinstance(rec, dict):
        raise ValueError("레코드는 object여야 합니다")
    for key in ("id", "docs", "meta"):
        if key not in rec:
            raise ValueError(f"필수 키 없음: {key}")
    if not isinstance(rec["id"], str) or not rec["id"]:
        raise ValueError("id가 비어 있습니다")
    if not isinstance(rec["docs"], list) or not rec["docs"]:
        raise ValueError(f"docs가 비어 있습니다: {rec['id']}")
    for doc in rec["docs"]:
        if not isinstance(doc, dict) or not all(k in doc for k in ("doc_id", "type", "text")):
            raise ValueError(f"docs 형식 오류: {rec['id']}")
        if not isinstance(doc["text"], str):
            raise ValueError(f"docs.text 형식 오류: {rec['id']}")
    if not any(doc["type"] == "공고문" for doc in rec["docs"]):
        raise ValueError(f"공고문 없음: {rec['id']}")
    if not isinstance(rec["meta"], dict):
        raise ValueError(f"meta 형식 오류: {rec['id']}")


def normalize(rec: Dict[str, Any]) -> Dict[str, Any]:
    for doc in rec.get("docs", []):
        for key in ("text", "type", "doc_id"):
            doc[key] = unicodedata.normalize("NFC", str(doc[key]))
    return rec


def iter_records(path: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    count = 0
    with _open(path) as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 오류: {exc}") from exc
            validate_record(rec)
            yield normalize(rec)
            count += 1
            if limit is not None and count >= limit:
                return


def full_text(rec: Dict[str, Any]) -> str:
    return "\n".join(doc["text"] for doc in rec["docs"])


def _display(value: Any) -> str:
    if value is None:
        return "미기재(null)"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def format_meta(rec: Dict[str, Any]) -> str:
    meta = rec.get("meta", {})
    return "\n".join(f"- {key}: {_display(meta[key])}" for key in META_FIELDS if key in meta)


def item_table(data_dir: str = DATA_DIR) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(data_dir, "항목표.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with io.open(path, encoding="utf-8") as handle:
        table = json.load(handle)["항목"]
    if set(table) != set(ITEMS):
        raise ValueError("항목표 v1~v24가 완전하지 않습니다")
    return table


class ProductCatalog:
    """배포된 중기부 고시 CSV의 세부품명 코드/특이사항 사전."""

    def __init__(self, data_dir: str):
        self.rows: Dict[str, Dict[str, str]] = {}
        for root, _dirs, files in os.walk(data_dir):
            for filename in files:
                if not filename.lower().endswith(".csv"):
                    continue
                path = os.path.join(root, filename)
                try:
                    with io.open(path, encoding="utf-8-sig", newline="") as handle:
                        reader = csv.DictReader(handle)
                        if "세부품명번호" not in (reader.fieldnames or []):
                            continue
                        for row in reader:
                            code = re.sub(r"\D", "", row.get("세부품명번호") or "")
                            if len(code) == 10:
                                self.rows[code] = row
                except (OSError, UnicodeError, csv.Error):
                    continue
        if not self.rows:
            raise FileNotFoundError("중기부 경쟁제품 CSV를 찾지 못했습니다")
        log(f"경쟁제품 세부품명 {len(self.rows):,}개 로드")

    @staticmethod
    def _applicability(note: str, price: Optional[int]) -> str:
        """고시 특이사항 중 금액으로 확정 가능한 적용 제외만 계산한다."""
        match = re.search(
            r"(?:추정가격|공공입찰\s*금액)\s*([\d.]+)\s*억\s*원?\s*미만", note)
        if match and price is not None:
            ceiling = int(float(match.group(1)) * 100_000_000)
            if price >= ceiling:
                return f"금액기준 제외({ceiling}원 미만만 적용)"
        return "적용" if not note else "적용조건 원문확인"

    def matches(self, rec: Dict[str, Any], price: Optional[int] = None
                ) -> List[Tuple[str, str, str, str, str]]:
        meta_text = str(rec.get("meta", {}).get("세부품명번호목록") or "")
        meta_codes = set(re.findall(r"(?<!\d)\d{10}(?!\d)", meta_text))
        doc_codes = set(re.findall(r"(?<!\d)\d{10}(?!\d)", full_text(rec)))
        catalog_codes = set(self.rows)
        ordered = sorted(meta_codes & catalog_codes) + sorted((doc_codes & catalog_codes) - meta_codes)
        result = []
        for code in ordered[:16]:
            row = self.rows[code]
            source = "meta" if code in meta_codes else "문서"
            note = row.get("특이사항", "")
            result.append((code, row.get("세부품명", ""), note, source,
                           self._applicability(note, price)))
        return result


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text)).lower()


def _has_any(text: str, words: Iterable[str]) -> bool:
    return any(word in text for word in words)


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = re.sub(r"[^0-9.-]", "", value)
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
    return None


def _price_band(price: Optional[int]) -> str:
    if price is None:
        return "미기재"
    if price >= NOTICE_AMOUNT:
        return "2.3억원 이상"
    if price >= SMALL_AMOUNT:
        return "1억원 이상~2.3억원 미만"
    if price > MICRO_AMOUNT:
        return "2천만원 초과~1억원 미만"
    return "2천만원 이하"


PERFORMANCE_MONEY = re.compile(
    r"(?<![\d.])((?:\d{1,3}(?:,\d{3})+)|(?:\d+(?:\.\d+)?))"
    r"\s*(억|천만|백만|만)?\s*원")


def _money_value(match: re.Match[str]) -> int:
    value = float(match.group(1).replace(",", ""))
    multiplier = {"억": 100_000_000, "천만": 10_000_000,
                  "백만": 1_000_000, "만": 10_000, None: 1}[match.group(2)]
    return int(value * multiplier)


def _performance_budget_violation(
        rec: Dict[str, Any], budget: Optional[int]) -> Tuple[bool, Optional[Tuple[str, int]]]:
    """참가요건의 최소 실적금액이 배정예산 이상인 명시적 사례만 잡는다."""
    if budget is None or budget <= 0:
        return False, None
    strong_requirement = re.compile(
        r"참가자격|있는\s*업체|업체이어야|업체만|보유한\s*(?:자|업체)|"
        r"실적이\s*있어야|실적을\s*보유|실적증명.{0,60}가능한\s*업체", re.S)
    for doc in rec["docs"]:
        text = doc["text"]
        for match in PERFORMANCE_MONEY.finditer(text):
            if _money_value(match) < budget:
                continue
            after = text[match.end():match.end() + 55]
            if not re.match(r"\s*(?:\([^\n)]{0,40}\)\s*)?이상", after):
                continue
            amount_neighborhood = text[max(0, match.start() - 100):match.end() + 100]
            if re.search(
                    r"자본금|매출액?|신용|보증금|보험금?|적격심사|실적평가|정량평가|"
                    r"평가기준|평가항목|배점|점수", amount_neighborhood):
                continue
            start, end = max(0, match.start() - 220), min(len(text), match.end() + 220)
            window = text[start:end]
            positions = [hit.start() for hit in re.finditer("실적", window)]
            relative = match.start() - start
            if not positions or min(abs(pos - relative) for pos in positions) > 180:
                continue
            if not strong_requirement.search(window):
                continue
            return True, (doc["doc_id"], match.start())
    return False, None


def _joint_share_violation(rec: Dict[str, Any]) -> Tuple[bool, Optional[Tuple[str, int]]]:
    meta = rec.get("meta", {})
    threshold = 5.0 if meta.get("적용계약법") == "지방계약법" else 10.0
    if "공사" in str(meta.get("업무구분") or ""):
        amount = _integer(meta.get("입찰추정가격")) or 0
        if meta.get("적용계약법") != "지방계약법" and amount >= 100_000_000_000:
            threshold = 5.0
    pattern = re.compile(
        r"(?:최소\s*)?(?:참여\s*)?(?:지분율|지분|출자비율|참여비율)"
        r"[^\n%]{0,45}?(\d+(?:\.\d+)?)\s*%|"
        r"(\d+(?:\.\d+)?)\s*%[^\n]{0,35}?(?:최소\s*)?(?:지분율|지분)", re.I)
    for doc in rec["docs"]:
        for match in pattern.finditer(doc["text"]):
            value = float(match.group(1) or match.group(2))
            around = _compact(doc["text"][max(0, match.start() - 130):match.end() + 150])
            near = _compact(doc["text"][max(0, match.start() - 70):match.end() + 80])
            if not _has_any(around, ("최소", "이상", "구성원별", "각구성원")):
                continue
            if "분담이행" in around and "공동이행" not in around:
                continue
            if "입찰보증" in near:
                continue
            if _has_any(near, ("지역업체", "지역의무", "대표사", "대표업체")) and not \
                    _has_any(near, ("구성원별", "각구성원", "구성원은", "구성원의")):
                continue
            if value < threshold:
                return True, (doc["doc_id"], match.start())
    return False, None


def _mandatory_briefing(rec: Dict[str, Any]) -> Tuple[bool, Optional[Tuple[str, int]]]:
    if "협상" not in str(rec.get("meta", {}).get("낙찰방법") or ""):
        return False, None
    patterns = [
        re.compile(r"(?:현장|사업|제안요청서?)?\s*설명회.{0,120}?참석한\s*자", re.S),
        re.compile(r"(?:현장|사업|제안요청서?)?\s*설명회.{0,120}?(?:참석하지\s*아니한|미참석|불참).{0,120}?(?:허용되지|대상에서\s*제외|접수하지)", re.S),
        re.compile(r"(?:현장|사업|제안요청서?)?\s*설명회.{0,120}?참석업체.{0,80}?한하여.{0,80}?제안서", re.S),
    ]
    for doc in rec["docs"]:
        for pattern in patterns:
            match = pattern.search(doc["text"])
            if match:
                return True, (doc["doc_id"], match.start())
    return False, None


FULL_DATE = re.compile(
    r"(?<!\d)(20\d{2})\s*(?:[.\-/년])\s*(\d{1,2})\s*(?:[.\-/월])\s*"
    r"(\d{1,2})\s*(?:일|\.)?")
SHORT_RANGE_END = re.compile(
    r"^[^\n~∼～]{0,25}(?:~|∼|～|–|—|-)\s*(\d{1,2})\s*[./월]\s*"
    r"(\d{1,2})\s*(?:일|\.)?")
BRIEFING_EVENT = re.compile(
    r"(?:제안\s*요청(?:서)?\s*설명(?:회)?|사업\s*(?:\(\s*현장\s*\))?\s*설명회|"
    r"현장\s*설명회|과업\s*설명회)")
BRIEFING_NEGATIVE = re.compile(
    r"(?:없음|생략|갈음|미실시|개최하지|상관없이|무관|"
    r"(?:일시|일정|날짜).{0,18}(?:개별|별도)\s*(?:통보|안내|공지)|"
    r"추후.{0,18}(?:통보|안내|공지))")
PROPOSAL_DEADLINE = re.compile(
    r"(?:제안서|기술제안서|입찰등록)[^\n]{0,30}(?:제출|접수)|"
    r"(?:제출|접수)\s*(?:마감|기간|일시)|입찰서\s*제출\s*마감\s*일시")
DEADLINE_NEGATIVE = re.compile(
    r"(?:전\s*일까지|자격|확인서|증명서|보증금|평가|발표|개찰|서류\s*보완)")


def _dates(text: str) -> List[Tuple[date, int]]:
    result = []
    for match in FULL_DATE.finditer(text):
        try:
            start_date = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            result.append((start_date, match.start()))
            # `2026.1.10. ~ 1.20.`처럼 종료 연도를 생략한 공고 관행을 복원한다.
            suffix = text[match.end():match.end() + 55]
            short = SHORT_RANGE_END.search(suffix)
            if short:
                month, day = int(short.group(1)), int(short.group(2))
                year = start_date.year + int(month < start_date.month)
                result.append((date(year, month, day), match.end() + short.start(1)))
        except ValueError:
            continue
    return sorted(set(result), key=lambda pair: pair[1])


def _nonempty_lines(text: str) -> List[Tuple[int, str]]:
    result, offset = [], 0
    for line in text.splitlines(keepends=True):
        cleaned = line.strip()
        if cleaned:
            leading = len(line) - len(line.lstrip())
            result.append((offset + leading, cleaned))
        offset += len(line)
    return result


def _briefing_timing_violation(
        rec: Dict[str, Any], price: Optional[int]
        ) -> Tuple[Optional[bool], Optional[Tuple[str, int]], str]:
    """날짜가 모두 명시된 지방 협상계약만 v23을 산술 판정한다."""
    meta = rec.get("meta", {})
    if meta.get("적용계약법") != "지방계약법" or "협상" not in str(meta.get("낙찰방법") or ""):
        return None, None, "비적용"
    if price is None:
        return None, None, "추정가격 미기재"
    try:
        posted = datetime.strptime(str(meta.get("공고게시일자") or ""), "%Y%m%d").date()
    except ValueError:
        return None, None, "게시일 파싱 실패"

    briefing_candidates: List[Tuple[date, str, int]] = []
    notice_docs = [doc for doc in rec["docs"] if "공고" in doc["type"]]
    for doc in notice_docs:
        lines = _nonempty_lines(doc["text"])
        for index, (offset, line) in enumerate(lines):
            event = BRIEFING_EVENT.search(line)
            if not event:
                continue
            block = " ".join(value for _line_offset, value in lines[index:index + 3])
            current_dates = _dates(line)
            negative_scope = line if current_dates else block
            if BRIEFING_NEGATIVE.search(negative_scope):
                continue
            candidates = current_dates or _dates(block)
            if not candidates:
                continue
            # 같은 일정표 한 줄에 공고기간이 앞서면 설명회 키워드 뒤의 첫 날짜를 쓴다.
            following = [(day, pos) for day, pos in candidates if pos >= event.end()]
            selected = min(following or candidates, key=lambda pair: pair[1])[0]
            if selected >= posted:
                briefing_candidates.append((selected, doc["doc_id"], offset + event.start()))
    if not briefing_candidates:
        return None, None, "설명회 날짜 불확실"
    distinct_briefing_dates = {row[0] for row in briefing_candidates}
    if len(distinct_briefing_dates) > 1:
        return None, None, "서로 다른 설명회 날짜 복수"
    brief_date, brief_doc, brief_offset = min(briefing_candidates, key=lambda row: row[0])

    deadlines: List[Tuple[int, date]] = []
    for doc in notice_docs:
        lines = _nonempty_lines(doc["text"])
        for index, (_offset, line) in enumerate(lines):
            if not PROPOSAL_DEADLINE.search(line):
                continue
            block = " ".join(value for _line_offset, value in lines[index:index + 3])
            candidates = [(day, pos) for day, pos in _dates(block) if day >= brief_date]
            candidates = [(day, pos) for day, pos in candidates
                          if not DEADLINE_NEGATIVE.search(block[:pos + 12])]
            if not candidates:
                continue
            deadline = max(day for day, _pos in candidates)
            score = 0
            score += 6 if re.search(r"제안서|기술제안서", block) else 0
            score += 4 if (re.search(r"접수|제출", block) and
                           re.search(r"마감|기간|일시", block)) else 0
            score += 3 if "입찰등록" in block else 0
            score += 2 if "까지" in block else 0
            deadlines.append((score, deadline))
    if not deadlines:
        return None, None, "제안서 마감 불확실"
    best_score = max(score for score, _deadline in deadlines)
    deadline = min(day for score, day in deadlines if score == best_score)

    notice_gap = (brief_date - posted).days
    submit_gap = (deadline - brief_date).days
    urgent = str(meta.get("긴급공고여부") or "").upper() == "Y"
    required = 7 if urgent else (10 if price < 100_000_000 else
                                 20 if price < 1_000_000_000 else 40)
    violation = notice_gap < 7 or submit_gap < required
    detail = (f"게시={posted.isoformat()},설명={brief_date.isoformat()},"
              f"마감={deadline.isoformat()},간격={notice_gap}/{submit_gap},요건=7/{required}")
    return violation, (brief_doc, brief_offset), detail


def analyze_record(rec: Dict[str, Any], catalog: ProductCatalog) -> Dict[str, Any]:
    meta, text = rec.get("meta", {}), full_text(rec)
    compact = _compact(text)
    price = _integer(meta.get("입찰추정가격"))
    budget = _integer(meta.get("배정예산금액"))
    law, award = str(meta.get("적용계약법") or ""), str(meta.get("낙찰방법") or "")
    matches = catalog.matches(rec, price)
    competition = any(not status.startswith("금액기준 제외")
                      for _code, _name, _note, _source, status in matches)
    competition_text = _has_any(compact, (
        "중소기업자간경쟁제품", "중기간경쟁제품", "중소벤처기업부장관이지정", "중소기업청장이지정"))
    general = _has_any(compact, ("일반제품으로", "일반물품으로", "직접생산확인품목에서제외"))
    direct = bool(re.search(r"직접생산(?:확인)?(?:증명서|확인서)", compact))
    eligibility = "\n".join(line for line in text.splitlines() if _has_any(
        _compact(line), ("입찰참가", "참가자격", "소지한", "확인서", "업체이어야", "자로서")))
    ec = _compact(eligibility)
    broad_sme = bool(re.search(r"(?:중[·ㆍ・]?소기업|중소기업자|중기업).{0,100}(?:확인서|소지|업체|자로서)", ec))
    small = bool(re.search(r"(?:소기업|소상공인).{0,100}(?:확인서|소지|업체|자로서)",
                           re.sub(r"중[·ㆍ・]?소기업", "", ec)))
    sw = bool(re.search(
        r"소프트웨어사업자|소프트웨어\s*사업|정보시스템\s*(?:구축|개발|유지)|"
        r"시스템\s*(?:구축|개발)\s*사업|업종코드\s*[:：]?\s*1468|컴퓨터관련서비스사업", text, re.I))
    negotiation = "협상" in award or "협상에의한계약" in compact

    hard_zero: Set[str] = set()
    if price is not None:
        if price >= NOTICE_AMOUNT:
            hard_zero.add("v2")
        # v5 지방 일반 물품·용역은 dev/시행규칙상 5억원, 국가는 2.3억원.
        v5_threshold = 500_000_000 if law == "지방계약법" else NOTICE_AMOUNT
        if price < v5_threshold:
            hard_zero.add("v5")
        if price < NOTICE_AMOUNT:
            hard_zero.add("v14")
        if not (SMALL_AMOUNT <= price < NOTICE_AMOUNT):
            hard_zero.update(("v15", "v16"))
        if not (MICRO_AMOUNT < price < SMALL_AMOUNT):
            hard_zero.update(("v17", "v18"))
        local_small_quote = (
            law == "지방계약법" and price <= SMALL_AMOUNT and
            ("수의" in str(meta.get("계약방법") or "") or "소액수의" in award)
        )
        if local_small_quote:
            hard_zero.update(("v2", "v6", "v7", "v8"))
    if not negotiation:
        hard_zero.update(("v22", "v23"))
    if law != "지방계약법":
        hard_zero.add("v23")

    joint, joint_location = _joint_share_violation(rec)
    briefing, briefing_location = _mandatory_briefing(rec)
    performance, performance_location = _performance_budget_violation(rec, budget)
    timing, timing_location, timing_detail = _briefing_timing_violation(rec, price)
    if timing is False:
        hard_zero.add("v23")
    dropped = rec.get("dropped_doc_counts") or {}
    fully_observed = not dropped
    return {
        "law": law or "미기재", "work": str(meta.get("업무구분") or "미기재"),
        "contract": str(meta.get("계약방법") or "미기재"), "award": award or "미기재",
        "price": price, "budget": budget, "price_band": _price_band(price),
        "catalog_matches": matches, "competition_signal": competition,
        "competition_text_signal": competition_text, "general_signal": general,
        "direct_signal": direct, "broad_sme_signal": broad_sme, "small_signal": small,
        "sw_signal": sw, "negotiation": negotiation, "fully_observed": fully_observed,
        "dropped": dropped, "hard_zero": hard_zero, "joint_violation": joint,
        "joint_location": joint_location, "briefing_violation": briefing,
        "briefing_location": briefing_location, "performance_violation": performance,
        "performance_location": performance_location,
        "briefing_timing_violation": timing is True,
        "briefing_timing_decision": timing, "briefing_timing_location": timing_location,
        "briefing_timing_detail": timing_detail,
    }


def facts_prompt(facts: Dict[str, Any]) -> str:
    products = ", ".join(
        f"{code}({name},특이사항={note or '없음'},출처={source},판정={status})"
        for code, name, note, source, status in facts["catalog_matches"]
    ) or "정확일치 없음"
    hard = ", ".join(sorted(facts["hard_zero"], key=lambda x: int(x[1:]))) or "없음"
    return "\n".join([
        "[코드 계산 보조사실 — 검색 신호는 결론이 아니므로 원문 span으로 확인]",
        f"- 계약법={facts['law']} / 업무={facts['work']} / 계약={facts['contract']} / 낙찰={facts['award']}",
        f"- 배정예산={facts['budget']} / 추정가격={facts['price']} / 금액구간={facts['price_band']}",
        f"- 경쟁제품 고시 일치={products}",
        f"- 고시상 적용가능 경쟁제품={facts['competition_signal']} / 문서의 경쟁제품표현={facts['competition_text_signal']} / 일반제품명시={facts['general_signal']} / 직접생산문구={facts['direct_signal']}",
        f"- 중소기업자격신호={facts['broad_sme_signal']} / 소기업자격신호={facts['small_signal']} / SW사업신호={facts['sw_signal']}",
        f"- 실적최소금액>=배정예산 명시규칙={facts['performance_violation']}",
        f"- v23 날짜산술={facts['briefing_timing_detail']} / 확정판정={facts['briefing_timing_decision']}",
        f"- 완전관측추정={facts['fully_observed']} / 누락문서={_display(facts['dropped'])}",
        f"- 적용조건상 확정 0 항목={hard}",
    ])


# ---------------------------------------------------------------------------
# 모든 문서를 500자 이하 exact span으로 분할하고 중요 구간을 우선 선택
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceSpan:
    index: int
    doc_id: str
    doc_type: str
    start: int
    end: int
    text: str
    score: float


ANCHORS: Sequence[Tuple[re.Pattern[str], float]] = [
    (re.compile(r"입찰\s*참가\s*자격|참가\s*자격|자격\s*요건", re.I), 10.0),
    (re.compile(r"실적|이행실적|납품실적|수행실적", re.I), 8.0),
    (re.compile(r"본점|주된\s*영업소|소재지|지역\s*제한|인접", re.I), 7.0),
    (re.compile(r"직접\s*생산|경쟁\s*제품|세부품명|중소기업|소기업|소상공인", re.I), 8.0),
    (re.compile(r"물품\s*공급|기술\s*지원|확약서|공급\s*증명", re.I), 9.0),
    (re.compile(r"소프트웨어|정보시스템|대기업\s*참여|중견기업|1468", re.I), 8.0),
    (re.compile(r"공동\s*(?:수급|도급|계약|이행)|지분율|출자비율|참여비율", re.I), 9.0),
    (re.compile(r"현장\s*설명|사업\s*설명|제안요청서?\s*설명|설명회", re.I), 10.0),
    (re.compile(r"모델명|모델\s*[:：]|제조사|브랜드|상표|동등\s*이상|동급\s*이상", re.I), 9.0),
    (re.compile(r"추정가격|기초금액|사업비|사업예산|배정예산|계약방법|낙찰방법", re.I), 6.0),
    (re.compile(r"제안서\s*제출|입찰서\s*제출|접수\s*마감|제출\s*마감", re.I), 7.0),
    (re.compile(r"제2조의3|비영리법인|특별법인|유찰|재공고", re.I), 6.0),
]
MODEL_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,12}[-_/]?[A-Za-z0-9]{2,}(?![A-Za-z0-9])")
MODEL_CORE = re.compile(
    r"모델명|제조사\s*[·:/：]|브랜드\s*[·:/：]|상표\s*[·:/：]|"
    r"(?:Chipset|Model)\s*[:：]|(?<![A-Za-z0-9])[A-Za-z]{2,12}[-_/][A-Za-z0-9-_/]{2,}", re.I)
EQUIVALENCE = re.compile(r"동등\s*이상|동급\s*이상")
BRIEFING_WINDOW = re.compile(r"현장\s*설명|사업\s*설명|제안요청서?\s*설명|설명회", re.I)
DATE_TOKEN = re.compile(r"20\d{2}\s*[.년/-]\s*\d{1,2}\s*[.월/-]\s*\d{1,2}")
MONEY_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:억\s*)?(?:원|만원|억원)")


def _split_exact(text: str, max_len: int = 470, min_len: int = 180) -> List[Tuple[int, int, str]]:
    """원문을 변경하지 않고 가까운 줄/문장 경계에서 나눈다."""
    result: List[Tuple[int, int, str]] = []
    pos, size = 0, len(text)
    while pos < size:
        while pos < size and text[pos].isspace():
            pos += 1
        if pos >= size:
            break
        hard, end = min(size, pos + max_len), min(size, pos + max_len)
        if hard < size:
            window = text[pos + min_len:hard]
            candidates = []
            for token in ("\n\n", "\n", "다. ", "요. ", ". ", " | "):
                found = window.rfind(token)
                if found >= 0:
                    token_pos = pos + min_len + found
                    if token == ". " and re.search(
                            r"(?:^|\n)\s*(?:[가-하]|\d{1,3})$",
                            text[max(pos, token_pos - 12):token_pos]):
                        continue
                    candidates.append(token_pos + len(token))
            if candidates:
                end = max(candidates)
        raw_end = end
        while raw_end > pos and text[raw_end - 1].isspace():
            raw_end -= 1
        if raw_end > pos:
            result.append((pos, raw_end, text[pos:raw_end]))
        pos = max(end, pos + 1)
    return result


def _span_score(text: str, doc_type: str, first: bool, last: bool) -> float:
    score = 4.0 if first else 0.0
    score += 1.0 if last else 0.0
    for pattern, weight in ANCHORS:
        score += min(2, len(pattern.findall(text))) * weight
    if doc_type in ("규격서", "과업지시서", "제안요청서"):
        score += 2.0 + min(3, len(MODEL_TOKEN.findall(text))) * 1.5
    score += min(2, len(DATE_TOKEN.findall(text))) * 1.5
    score += min(2, len(MONEY_TOKEN.findall(text)))
    return score


def select_spans(rec: Dict[str, Any], max_chars: int,
                 facts: Optional[Dict[str, Any]] = None) -> List[EvidenceSpan]:
    docs = sorted(rec["docs"], key=lambda d: (DOC_ORDER.get(d["type"], 99), d["doc_id"]))
    raw: List[Dict[str, Any]] = []
    per_doc: Dict[str, List[int]] = {}
    rescue_indices: Set[int] = set()
    for doc_order, doc in enumerate(docs):
        parts = _split_exact(doc["text"])
        doc_key = f"{doc_order}:{doc['doc_id']}"
        per_doc[doc_key] = []
        for part_i, (start, end, text) in enumerate(parts):
            raw.append({
                "doc_key": doc_key, "doc_order": doc_order, "doc_id": doc["doc_id"],
                "doc_type": doc["type"], "start": start, "end": end, "text": text,
                "score": _span_score(text, doc["type"], part_i == 0, part_i == len(parts) - 1),
            })
            per_doc[doc_key].append(len(raw) - 1)

    # 중요 문장이 경계에서 잘린 경우 이웃 문맥도 순위가 올라가게 한다.
    for indices in per_doc.values():
        base = [raw[i]["score"] for i in indices]
        for local, index in enumerate(indices):
            if local:
                raw[index]["score"] += base[local - 1] * 0.18
            if local + 1 < len(indices):
                raw[index]["score"] += base[local + 1] * 0.18

    # 설명회 일정은 공고기간·설명일·접수마감이 한 덩어리여야 날짜 산술이 쉽다.
    # anchor가 기본 span 끝에 걸린 경우에만 좁은 centered window를 하나 보충한다.
    for doc_order, doc in enumerate(docs):
        doc_key = f"{doc_order}:{doc['doc_id']}"
        existing_ranges = {(raw[i]["start"], raw[i]["end"]) for i in per_doc[doc_key]}
        for base_index in list(per_doc[doc_key]):
            row = raw[base_index]
            match = BRIEFING_WINDOW.search(row["text"])
            if not match:
                continue
            absolute = row["start"] + match.start()
            if row["end"] - absolute > 170:
                continue
            start = max(0, absolute - 170)
            end = min(len(doc["text"]), start + 470)
            start = max(0, end - 470)
            if (start, end) in existing_ranges:
                continue
            text = doc["text"][start:end]
            raw.append({
                "doc_key": doc_key, "doc_order": doc_order, "doc_id": doc["doc_id"],
                "doc_type": doc["type"], "start": start, "end": end, "text": text,
                "score": _span_score(text, doc["type"], start == 0, end == len(doc["text"])) + 8.0,
            })
            new_index = len(raw) - 1
            per_doc[doc_key].append(new_index)
            rescue_indices.add(new_index)
            existing_ranges.add((start, end))

    def cost(indices: Iterable[int]) -> int:
        return sum(len(raw[i]["text"]) + 55 for i in indices)

    if cost(range(len(raw))) <= max_chars:
        chosen: Set[int] = set(range(len(raw)))
    else:
        chosen = set()
        heads: Set[int] = set()
        mandatory_spans: Set[int] = set(rescue_indices)
        category_spans: Set[int] = set()
        forced_spans: Set[int] = set()
        # 문서별 head와 최고점 span을 최소 보장한다.
        for indices in per_doc.values():
            if indices:
                heads.add(indices[0])
                chosen.add(indices[0])
                best = max(indices, key=lambda i: (raw[i]["score"], -raw[i]["start"]))
                chosen.add(best)
                mandatory_spans.update((indices[0], best))
        # v9의 모델 지정과 '동등 이상' 예외를 첨부문서별로 최소 1개씩 보장한다.
        for doc_key, indices in per_doc.items():
            if not indices or raw[indices[0]]["doc_type"] not in ("규격서", "과업지시서", "제안요청서"):
                continue
            for pattern in (MODEL_CORE, EQUIVALENCE):
                candidates = [i for i in indices if pattern.search(raw[i]["text"])]
                if candidates:
                    mandatory_spans.add(max(candidates, key=lambda i: raw[i]["score"]))
        chosen.update(mandatory_spans)
        # 실적·지역·SME·설명회 등 서로 다른 판정 축이 한 종류의 반복 문구에
        # 밀리지 않도록 anchor 종류마다 상위 2개 구간을 먼저 확보한다.
        for pattern, _weight in ANCHORS:
            candidates = [i for i, row in enumerate(raw) if pattern.search(row["text"])]
            category_spans.update(sorted(
                candidates, key=lambda i: (-raw[i]["score"], raw[i]["doc_order"], raw[i]["start"]))[:2])
        chosen.update(category_spans)
        # 확정 규칙 근거는 반드시 남긴다.
        if facts:
            for location_key in ("performance_location", "joint_location", "briefing_location",
                                 "briefing_timing_location"):
                location = facts.get(location_key)
                if not location:
                    continue
                target_doc, target_pos = location
                for i, row in enumerate(raw):
                    if row["doc_id"] == target_doc and row["start"] <= target_pos < row["end"]:
                        forced_spans.add(i)
                        mandatory_spans.add(i)
                        chosen.add(i)
                        break
        if cost(chosen) > max_chars:
            ranked_keep = sorted(
                chosen,
                key=lambda i: (i in forced_spans, i in mandatory_spans, i in category_spans, i in heads,
                               raw[i]["score"], -raw[i]["start"]),
                reverse=True)
            chosen, used = set(), 0
            for i in ranked_keep:
                item_cost = len(raw[i]["text"]) + 55
                if used + item_cost <= max_chars or not chosen:
                    chosen.add(i)
                    used += item_cost
        used = cost(chosen)
        ranked = sorted((i for i in range(len(raw)) if i not in chosen),
                        key=lambda i: (-raw[i]["score"], raw[i]["doc_order"], raw[i]["start"]))
        for i in ranked:
            item_cost = len(raw[i]["text"]) + 55
            if used + item_cost <= max_chars:
                chosen.add(i)
                used += item_cost

    ordered = sorted((raw[i] for i in chosen), key=lambda row: (row["doc_order"], row["start"]))
    return [EvidenceSpan(i + 1, row["doc_id"], row["doc_type"], row["start"], row["end"],
                         row["text"], row["score"]) for i, row in enumerate(ordered)]


def render_spans(spans: Sequence[EvidenceSpan]) -> str:
    return "\n\n".join(
        f"[S{span.index}|{span.doc_type}|{span.doc_id}|원문위치={span.start}]\n{span.text}"
        for span in spans)


def build_context(rec: Dict[str, Any], max_chars: int = DEFAULT_CONTEXT_CHARS) -> str:
    return render_spans(select_spans(rec, max_chars))


# 법령 패키지와 dev 사례에서 정리한 판정 카드. 원문 법령이 아니라 추론 체크리스트다.
ITEM_CARDS = r"""
v1 특정기관 제한: 법정 면허·등록이 아닌 대학/산학협력단/협회/특정 기관형태만 참가시키면 1. 단순 업종·면허·SME·지역·실적은 별도 항목이다.
v2 저가 실적제한: 추정가격<2.3억인데 과거 납품·수행실적을 참가자격으로 요구하면 1. 단 지방계약 소액수의(<=1억)는 0. 평가배점 실적은 제외한다.
v3 1배수 실적: 요구 최소실적금액>=배정예산이면 1. 추정가격이 아닌 사업예산과 비교한다.
v4 특정 실적: 실적을 국가기관/공공기관/대학병원 등 특정 발주처나 지나치게 동일한 대상에 한정하면 1. 금액 hard gate가 없고 v2/v3과 동시 가능하다.
v5 고액 지역제한: 참가자격의 본점·주된 영업소 지역제한이 국가 추정가격>=2.3억 또는 지방 일반 물품·용역>=5억이면 1. 납품장소 주소는 제외한다.
v6 기초지역 제한: 경쟁입찰 지역을 시·군·구 한 곳으로 부당하게 좁히면 1. 지방 소액수의<=1억은 0. 익명 토큰 단위=기초를 읽는다.
v7 인접 확대: 둘 이상의 시·도/인접 지역으로 참가범위를 임의 확대하면 1. 적법 사유와 지방 소액수의<=1억은 0.
v8 중복제한: 참가자격에 의무 실적+지역을 함께 걸면 1. 지방 소액수의<=1억은 0.
v9 특정 모델: 규격서/과업/RFP가 특정 제조사·브랜드·모델·카탈로그를 사실상 지정하면 1. 기능규격·기존장비 설명·명확한 동등이상 허용은 제외한다.
v10 경쟁제품 직생 없음[부재]: 실제 조달품목이 고시 경쟁제품인데 그 품목의 직접생산확인 자격이 없으면 1. 일반 제재문구/다른 품목 코드는 충족이 아니다.
v11 경쟁제품 중소 없음[부재]: 실제 경쟁제품인데 참가자격에 중소기업자/확인서 제한이 없으면 1. 메타에만 있는 제한은 충족이 아니다.
v12 일반제품 직생: 비경쟁 일반제품 또는 실제 사업과 무관한 품목의 직접생산확인을 강제하면 1. 실제 조달대상 코드와 결합해 본다.
v13 경쟁제품 소기업만: 경쟁제품인데 중기업을 배제하고 소기업·소상공인만 허용하면 1.
v14 고액 일반제품 SME: 비경쟁 일반제품, 추정가격>=2.3억인데 중소/소기업으로 참가를 제한하면 1.
v15 중간금액 소기업만: 일반제품, 1억<=가격<2.3억인데 소기업·소상공인만 허용하면 1. 제2조의3 실제 예외면 0.
v16 중간금액 SME 없음[부재]: 같은 구간 일반제품인데 중소기업 참가요건이 없고 실제 법정예외도 없으면 1.
v17 저가 중기업 허용: 일반제품, 2천만원<가격<1억인데 넓은 중소기업 제한으로 중기업까지 허용하면 1.
v18 저가 소기업 없음[부재]: 같은 구간인데 적정 소기업·소상공인 제한과 실제 예외가 모두 없으면 1. <=2천만원은 0.
v19 공급확약 입찰단계: 제조/공급/기술지원 확약을 입찰·제안 마감 전 보유·제출시키거나 미제출자를 배제하면 1. 낙찰/계약 뒤 제출만이면 0. 일반 확약서는 무관하다.
v20 SW 참가제한 없음[부재]: 실질 SW 구축·개발·유지 사업이면 SW진흥법48조와 20/40/80억 구간별 대기업 참여제한 적용여부·근거가 있어야 한다. 업종코드1468만으로 충족되지 않는다.
v21 공동 최소지분: 공동이행 최소지분이 지방 5%/국가 10%보다 낮으면 1. 분담이행, 국가 1천억 이상 공사, 명시적 조정 예외를 구별한다.
v22 설명회 강제참석[협상만]: 설명회 참석자만 입찰/제안 가능하거나 미참석자를 제외하면 1. 단순 개최·권장·불참 가능은 0. 비협상은 항상 0.
v23 지방협상 설명회 기간: 지방+협상+입찰 전 설명회만. 공고→설명회>=7일, 설명회→제안마감은 <1억 10일/1~10억 20일/>=10억 40일(긴급은 7일). 부족하면 1. 개찰일과 비교하지 않는다.
v24 메타 불일치: 문서와 메타의 같은 개념인 예산/추정가격, 계약·낙찰방법, 지역·업종, 공동도급, 입찰방식이 명백히 다르면 1. 총액과 추정가격은 서로 비교하지 않는다.
""".strip()

SYSTEM_HEAD = """당신은 배포 법령 스냅샷 기준 공공입찰 점검자다.
1) 카드의 적용조건→위반조건→예외 순서로 v1~v24를 독립 판정한다. 여러 항목이 동시에 1일 수 있다.
2) 공고문, 규격서, 과업지시서, RFP의 모든 제시 span과 메타를 함께 본다.
3) 법령명·공공구매론·일반 유의사항의 키워드만으로 위반이라 하지 말고 실제 참가자격 문장을 찾는다.
4) 적용조건과 위반문구가 합리적으로 확인되면 지나치게 보수적으로 0을 택하지 않되 추측만으로 1을 만들지 않는다.
5) 부재형 v10/v11/v16/v18/v20은 먼저 제품·금액·SW 적용성을 확정하고 필수문구 부재를 본다. 문서 누락 자체를 0이나 1로 자동 결정하지 않는다.
6) 출력 y는 0/1. s는 가장 직접적인 원문 S번호의 정수다. 비위반 및 부재형은 s=-1. 적합한 span이 없어도 y는 독립적으로 정확히 낸다.
7) JSON 외 설명은 출력하지 않는다.
"""


def build_system_prompt(table: Dict[str, Dict[str, Any]]) -> str:
    names = "\n".join(f"- {v}: {table[v]['항목명']}" for v in ITEMS)
    return (f"{SYSTEM_HEAD}\n[판정 카드]\n{ITEM_CARDS}\n\n[배포 항목표 이름]\n{names}\n\n"
            '출력 예: {"v1":{"y":0,"s":-1},...,"v24":{"y":1,"s":12}}')


def build_user_prompt(rec: Dict[str, Any], spans: Sequence[EvidenceSpan],
                      facts: Dict[str, Any]) -> str:
    return (
        f"[공고 ID] {rec['id']}\n\n{facts_prompt(facts)}\n"
        f"- input_completeness 원문값={_display(rec.get('input_completeness'))}\n\n"
        f"[나라장터 메타]\n{format_meta(rec)}\n\n"
        "[문서 원문 span]\nS번호 아래 원문만 근거로 선택한다.\n"
        f"{render_spans(spans)}\n"
    )


def build_messages(rec: Dict[str, Any], system_prompt: str, max_chars: int,
                   catalog: ProductCatalog, facts: Optional[Dict[str, Any]] = None
                   ) -> Tuple[List[Dict[str, str]], List[EvidenceSpan], Dict[str, Any]]:
    facts = facts or analyze_record(rec, catalog)
    spans = select_spans(rec, max_chars, facts)
    return ([{"role": "system", "content": system_prompt},
             {"role": "user", "content": build_user_prompt(rec, spans, facts)}], spans, facts)


# ---------------------------------------------------------------------------
# compact structured output, vLLM, budget fitting
# ---------------------------------------------------------------------------
def compact_schema() -> Dict[str, Any]:
    cell = {
        "type": "object", "additionalProperties": False, "required": ["y", "s"],
        "properties": {
            "y": {"type": "integer", "enum": [0, 1]},
            "s": {"type": "integer", "minimum": -1},
        },
    }
    return {"type": "object", "additionalProperties": False, "required": list(ITEMS),
            "properties": {item: cell for item in ITEMS}}


class VLLMRunner:
    def __init__(self, schema: Dict[str, Any], model_dir: str = MODEL_DIR,
                 quant: Optional[str] = QUANT, max_tokens: int = MAX_TOKENS,
                 max_model_len: int = MAX_MODEL_LEN, seed: int = SEED,
                 gpu_mem: float = 0.92, tp: int = 1):
        started = time.time()
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        kwargs = dict(model=model_dir, tokenizer=model_dir, max_model_len=max_model_len,
                      gpu_memory_utilization=gpu_mem, seed=seed, tensor_parallel_size=tp,
                      dtype="auto")
        if quant:
            kwargs["quantization"] = quant
        log(f"vllm {vllm.__version__} · max_model_len={max_model_len} · quant={quant}")
        self.llm = LLM(**kwargs)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, seed=seed,
            structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True))
        self.load_seconds = time.time() - started

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            return len(ids)
        except Exception:
            return len(self.tok.encode("\n".join(message["content"] for message in messages)))

    def chat(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        outputs = self.llm.chat(batch, sampling_params=self.sp, use_tqdm=False)
        texts = []
        for output in outputs:
            if not output.outputs:
                raise RuntimeError("vLLM이 completion을 반환하지 않았습니다")
            completion = output.outputs[0]
            reason = getattr(completion, "finish_reason", None)
            if reason != "stop":
                raise RuntimeError(f"구조화 출력 비정상 종료: {reason}")
            texts.append(completion.text)
        return texts


class MockRunner:
    load_seconds = 0.0

    def __init__(self, _schema: Dict[str, Any], **_kwargs):
        pass

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        return max(1, sum(len(message["content"]) for message in messages) // 2)

    def chat(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        obj = {item: {"y": 0, "s": -1} for item in ITEMS}
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        return [text for _ in batch]


class APIRunner:
    """RunPod 등의 vLLM OpenAI 호환 API 서버(HTTP)로 추론합니다."""
    load_seconds = 0.0

    def __init__(self, schema: Dict[str, Any], api_url: str = "http://localhost:8000/v1",
                 model_dir: str = "/workspace/models/gemma-4-26B-A4B-it",
                 max_tokens: int = MAX_TOKENS, seed: int = SEED, **_kwargs):
        self.schema = schema
        self.api_url = api_url.rstrip("/")
        if not self.api_url.endswith("/v1"):
            self.api_url += "/v1"
        self.max_tokens = max_tokens
        self.seed = seed

        # 서버에서 사용 가능한 모델명 자동 감지 및 연결 확인
        self.model = model_dir
        try:
            req = urllib.request.Request(f"{self.api_url}/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("data") and len(data["data"]) > 0:
                    self.model = data["data"][0]["id"]
        except Exception as e:
            err_str = str(e)
            if any(token in err_str for token in ("Connection refused", "Errno 61", "Operation not permitted", "Errno 1")):
                raise ConnectionError(
                    f"\n[오류] vLLM API 서버({self.api_url})에 연결할 수 없습니다 ({err_str}).\n"
                    f"다음 사항을 확인해 주세요:\n"
                    f"  1) 맥북의 별도 터미널에서 SSH 포트포워딩 터널이 실행 중인가요?\n"
                    f"     명령어: ssh -N -L 8000:localhost:8000 root@<RUNPOD_IP> -p <PORT>\n"
                    f"  2) RunPod 컨테이너 내부에서 vLLM 서버가 8000번 포트로 실행 중인가요?\n"
                    f"     명령어: vllm serve /workspace/models/gemma-4-26B-A4B-it --port 8000 ...\n"
                ) from e
            log(f"모델 목록 조회 생략 ({e}), 지정 모델명 사용: {self.model}")

        log(f"APIRunner 연결: {self.api_url} · 모델 {self.model} · max_tokens={max_tokens}")

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        return max(1, sum(len(message["content"]) for message in messages) // 2)

    def _one(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.api_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        if self.schema:
            payload["guided_json"] = self.schema

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "")
                return ""
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            log(f"API HTTP {e.code} 에러: {err_msg[:200]}")
            return ""
        except Exception as e:
            log(f"API 요청 실패: {type(e).__name__}: {str(e)[:160]}")
            return ""

    def chat(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        if len(batch) == 1:
            return [self._one(batch[0])]
        with ThreadPoolExecutor(max_workers=min(len(batch), 8)) as ex:
            return list(ex.map(self._one, batch))


def fit_to_budget(rec: Dict[str, Any], system_prompt: str, runner: Any,
                  max_chars: int, catalog: ProductCatalog, budget: int = PROMPT_BUDGET
                  ) -> Tuple[List[Dict[str, str]], List[EvidenceSpan], Dict[str, Any], int, int]:
    facts, chars = analyze_record(rec, catalog), max_chars
    while True:
        messages, spans, _facts = build_messages(rec, system_prompt, chars, catalog, facts)
        tokens = runner.count_tokens(messages)
        if tokens <= budget:
            return messages, spans, facts, tokens, chars
        if chars <= MIN_CONTEXT_CHARS:
            raise ValueError(f"{rec['id']}: 최소 문맥도 토큰 예산 초과 ({tokens}>{budget})")
        ratio = max(0.45, min(0.82, budget / max(tokens, 1) * 0.92))
        chars = max(MIN_CONTEXT_CHARS, int(chars * ratio))


def run_chunk(runner: Any, batch: List[List[Dict[str, str]]], depth: int = 0,
              max_depth: int = 4) -> List[str]:
    """큰 배치가 실패하면 전체 순차 재시도 대신 이분한다."""
    try:
        outputs = runner.chat(batch)
        if len(outputs) != len(batch):
            raise RuntimeError(f"응답 수 {len(outputs)} != 요청 수 {len(batch)}")
        return outputs
    except Exception as exc:
        message = str(exc).lower()
        fatal = any(token in message for token in (
            "engine dead", "enginedead", "engine died", "enginecore failed",
            "engine core failed", "worker died",
            "illegal memory access", "xgrammar", "structured output", "max_tokens"))
        fatal = fatal or "구조화 출력" in str(exc)
        if len(batch) <= 1 or depth >= max_depth or fatal:
            log(f"LLM 호출 실패: {type(exc).__name__}: {str(exc)[:240]}")
            raise
        middle = len(batch) // 2
        log(f"청크 {len(batch)} 실패 → {middle}+{len(batch)-middle} 이분 재시도")
        return (run_chunk(runner, batch[:middle], depth + 1, max_depth) +
                run_chunk(runner, batch[middle:], depth + 1, max_depth))


# ---------------------------------------------------------------------------
# 파싱, hard gate/고신뢰 규칙, 원문 근거 복원
# ---------------------------------------------------------------------------
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None
    for candidate in [text] + [match.group(1) for match in FENCE.finditer(text)]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except json.JSONDecodeError:
            pass
    return None


def parse_judgment(text: str) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """판정 누락만 재시도 대상으로 삼고 근거 번호 오류는 독립 처리한다."""
    obj = extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("판정"), dict):
        obj = obj["판정"]
    output: Dict[str, Dict[str, int]] = {}
    missing: List[str] = []
    for item in ITEMS:
        raw = obj.get(item) if isinstance(obj, dict) else None
        valid = (
            isinstance(raw, dict) and
            type(raw.get("y")) is int and raw["y"] in (0, 1)
        )
        if not valid:
            missing.append(item)
            output[item] = {"y": 0, "s": -1}
        else:
            # 실제 span 존재 여부는 postprocess에서 확인한다. 근거 오류로
            # 유효한 양성 판정을 버리거나 공고 전체를 실패시키지 않는다.
            span_no = raw.get("s")
            if type(span_no) is not int or span_no < -1:
                span_no = -1
            output[item] = {"y": raw["y"], "s": span_no}
    return output, missing


ITEM_TERMS: Dict[str, Sequence[str]] = {
    "v1": ("참가자격", "대학", "산학협력단", "협회"),
    "v2": ("실적", "이행", "납품"), "v3": ("실적", "이상", "기초금액"),
    "v4": ("실적", "국가기관", "공공기관", "대학병원", "단일"),
    "v5": ("소재지", "본점", "영업소"), "v6": ("소재지", "단위=기초"),
    "v7": ("인접", "지역", "소재지"), "v8": ("실적", "소재지", "지역"),
    "v9": ("모델", "제조사", "브랜드", "상표", "규격"),
    "v10": ("직접생산",), "v11": ("중소기업", "확인서"), "v12": ("직접생산",),
    "v13": ("소기업", "소상공인"), "v14": ("중소기업", "소기업"),
    "v15": ("소기업", "소상공인"), "v16": ("중소기업",),
    "v17": ("중소기업",), "v18": ("소기업", "소상공인"),
    "v19": ("공급", "기술지원", "확약서", "입찰"),
    "v20": ("소프트웨어", "대기업", "사업금액", "참여제한"),
    "v21": ("지분", "출자비율", "참여비율", "공동수급"),
    "v22": ("설명회", "미참석", "불참", "참석업체"),
    "v23": ("설명회", "제안서", "일시"),
    "v24": ("예산", "추정가격", "계약방법", "지역", "업종"),
}


def _best_span(item: str, spans: Sequence[EvidenceSpan]) -> int:
    best, best_score = -1, 0
    for span in spans:
        compact = _compact(span.text)
        score = sum(1 for term in ITEM_TERMS.get(item, ()) if _compact(term) in compact)
        if score > best_score:
            best, best_score = span.index, score
    return best


def _span_for_location(location: Optional[Tuple[str, int]],
                       spans: Sequence[EvidenceSpan]) -> int:
    if not location:
        return -1
    doc_id, offset = location
    for span in spans:
        if span.doc_id == doc_id and span.start <= offset < span.end:
            return span.index
    return -1


def _safe_evidence(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r", "").strip()
    while text.startswith(("=", "+", "@")):
        text = text[1:].lstrip()
    return text[:EVIDENCE_MAX]


def postprocess(judgment: Dict[str, Dict[str, int]], rec: Dict[str, Any],
                spans: Sequence[EvidenceSpan], facts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    span_map = {span.index: span for span in spans}
    source_text = full_text(rec)
    output: Dict[str, Dict[str, Any]] = {}
    for item in ITEMS:
        cell = judgment.get(item, {"y": 0, "s": -1})
        hit, span_no = int(cell.get("y") == 1), int(cell.get("s", -1))
        if item in facts["hard_zero"]:
            hit, span_no = 0, -1
        if item == "v2" and item not in facts["hard_zero"] and facts.get("performance_violation"):
            hit, span_no = 1, _span_for_location(facts.get("performance_location"), spans)
        if item == "v3" and item not in facts["hard_zero"] and facts.get("performance_violation"):
            hit, span_no = 1, _span_for_location(facts.get("performance_location"), spans)
        # dev에서 각각 6/6 FP 0, 5/5 FP 0인 고신뢰 규칙.
        if item == "v21" and item not in facts["hard_zero"] and facts.get("joint_violation"):
            hit, span_no = 1, _span_for_location(facts.get("joint_location"), spans)
        if item == "v22" and item not in facts["hard_zero"] and facts.get("briefing_violation"):
            hit, span_no = 1, _span_for_location(facts.get("briefing_location"), spans)
        if item == "v23" and item not in facts["hard_zero"] and facts.get("briefing_timing_violation"):
            hit, span_no = 1, _span_for_location(facts.get("briefing_timing_location"), spans)
        if hit and span_no not in span_map:
            span_no = _best_span(item, spans)
        if not hit or item in ABSENCE:
            evidence = ""
        else:
            evidence = _safe_evidence(span_map[span_no].text) if span_no in span_map else ""
            if evidence and evidence not in source_text:
                evidence = ""
        output[item] = {"위반여부": hit, "근거문구": evidence}
    return output


def to_row(rec_id: str, judgment: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"id": rec_id}
    for index, item in enumerate(ITEMS, 1):
        row[item] = int(judgment[item]["위반여부"])
        row[f"e{index}"] = judgment[item]["근거문구"]
    return row


# ---------------------------------------------------------------------------
# CSV 검증, dev 평가, 실행
# ---------------------------------------------------------------------------
def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: unicodedata.normalize("NFC", str(row.get(key, ""))) for key in COLUMNS})


def validate_csv(path: str, expected_ids: Sequence[str],
                 source_by_id: Optional[Dict[str, str]] = None) -> List[str]:
    errors: List[str] = []
    with io.open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header, rows = next(reader, None), list(reader)
    if header != COLUMNS:
        return ["헤더 불일치"]
    if len(rows) != len(expected_ids):
        errors.append(f"행 수 {len(rows)} != {len(expected_ids)}")
    ids = [row[0] for row in rows if row]
    if len(ids) != len(set(ids)):
        errors.append("id 중복")
    if set(ids) != set(expected_ids):
        errors.append("id 집합 불일치")
    absence_columns = {COLUMNS.index("e" + item[1:]) for item in ABSENCE}
    for row in rows:
        if len(row) != len(COLUMNS):
            errors.append(f"{row[0] if row else '?'}: 열 수 오류")
            continue
        if any(value not in ("0", "1") for value in row[1:25]):
            errors.append(f"{row[0]}: v 값 오류")
        if any(len(value) > EVIDENCE_MAX for value in row[25:]):
            errors.append(f"{row[0]}: 근거 길이 오류")
        if any(row[index] for index in absence_columns):
            errors.append(f"{row[0]}: 부재탐지 근거 존재")
        for item_index, evidence in enumerate(row[25:]):
            if row[1 + item_index] == "0" and evidence:
                errors.append(f"{row[0]}: 비위반 항목 e{item_index + 1} 존재")
            if evidence and source_by_id is not None and evidence not in source_by_id.get(row[0], ""):
                errors.append(f"{row[0]}: e{item_index + 1} 원문 불일치")
        if any(value.startswith(("=", "+", "@")) for value in row[25:]):
            errors.append(f"{row[0]}: 수식 접두 근거")
    return errors


def evaluate_rows(rows: Sequence[Dict[str, Any]], labels_path: str) -> Dict[str, Any]:
    with io.open(labels_path, encoding="utf-8", newline="") as handle:
        truth = {row["id"]: row for row in csv.DictReader(handle)}
    scores, details = [], {}
    for item in ITEMS:
        tp = fp = fn = 0
        for row in rows:
            if row["id"] not in truth:
                continue
            pred, gold = int(row[item]), int(truth[row["id"]][item])
            tp += int(pred == 1 and gold == 1)
            fp += int(pred == 1 and gold == 0)
            fn += int(pred == 0 and gold == 1)
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        scores.append(f1)
        details[item] = {"TP": tp, "FP": fp, "FN": fn, "F1": round(f1, 4)}
    return {"MacroF1": round(sum(scores) / len(scores), 6), "items": details}


def run(input_path: str, out_path: str, runner_cls: Any, limit: Optional[int],
        chunk: int, max_chars: int, data_dir: str, labels_path: Optional[str] = None,
        max_model_len: int = MAX_MODEL_LEN, **runner_kwargs: Any) -> Dict[str, Any]:
    started = time.time()
    records = list(iter_records(input_path, limit))
    log(f"입력 {len(records)}건 ← {input_path}")
    if not records:
        write_csv([], out_path)
        return {"건수": 0, "자가검증": "PASS"}

    table, catalog = item_table(data_dir), ProductCatalog(data_dir)
    system_prompt, schema = build_system_prompt(table), compact_schema()
    runner = runner_cls(schema, max_model_len=max_model_len, **runner_kwargs) if runner_cls is VLLMRunner else runner_cls(schema, **runner_kwargs)
    budget = max_model_len - runner_kwargs.get("max_tokens", MAX_TOKENS) - TOKEN_SAFETY_MARGIN
    if budget <= 0:
        raise ValueError("출력 토큰 수가 모델 컨텍스트보다 큽니다")
    log(f"모델 로드 {runner.load_seconds:.1f}s · max_model_len={max_model_len}")

    messages_all: List[List[Dict[str, str]]] = []
    spans_all: List[List[EvidenceSpan]] = []
    facts_all: List[Dict[str, Any]] = []
    token_counts: List[int] = []
    shrunk = 0
    for rec in records:
        messages, spans, facts, tokens, used_chars = fit_to_budget(
            rec, system_prompt, runner, max_chars, catalog, budget)
        messages_all.append(messages)
        spans_all.append(spans)
        facts_all.append(facts)
        token_counts.append(tokens)
        shrunk += int(used_chars < max_chars)
    ordered_tokens = sorted(token_counts)
    p50 = ordered_tokens[len(ordered_tokens) // 2]
    p90 = ordered_tokens[min(len(ordered_tokens) - 1, int(len(ordered_tokens) * 0.9))]
    span_counts = sorted(map(len, spans_all))
    log(f"prompt token p50={p50:,} p90={p90:,} max={max(token_counts):,} · 축소={shrunk}")
    log(f"span p50={span_counts[len(span_counts)//2]} max={max(span_counts)}")

    inference_started = time.time()
    texts: List[str] = []
    for start in range(0, len(messages_all), chunk):
        texts.extend(run_chunk(runner, messages_all[start:start + chunk]))
        log(f"추론 {min(start + chunk, len(messages_all))}/{len(messages_all)} · {time.time()-inference_started:.0f}s")

    parsed_all = [parse_judgment(text) for text in texts]
    invalid_indices = [i for i, (_parsed, missing) in enumerate(parsed_all) if missing]
    if invalid_indices:
        log(f"판정 누락 {len(invalid_indices)}건 재호출")
        for start in range(0, len(invalid_indices), chunk):
            indices = invalid_indices[start:start + chunk]
            retry_batch = []
            for index in indices:
                missing = parsed_all[index][1]
                messages = [dict(message) for message in messages_all[index]]
                messages[0]["content"] += (
                    "\n이전 응답에서 다음 항목의 y가 누락되거나 잘못되었다: "
                    + ", ".join(missing)
                    + ". 해당 항목을 재검토하고 v1~v24 전부를 출력한다. "
                    "y는 정수 0 또는 1, 근거를 선택할 수 없으면 s=-1이다.")
                # 재시도 안내는 예약된 안전 여유 안에서만 추가한다.
                if runner.count_tokens(messages) > budget + TOKEN_SAFETY_MARGIN:
                    messages = messages_all[index]
                retry_batch.append(messages)
            retried = run_chunk(runner, retry_batch)
            for index, text in zip(indices, retried):
                previous, missing = parsed_all[index]
                recovered, retry_missing = parse_judgment(text)
                for item in missing:
                    if item not in retry_missing:
                        previous[item] = recovered[item]
                parsed_all[index] = (previous, [item for item in missing if item in retry_missing])

    # 정상 판정을 하나도 얻지 못한 공고는 규칙만으로 대체하지 않는다.
    unusable = [records[i]["id"] for i, (_parsed, missing) in enumerate(parsed_all)
                if len(missing) == len(ITEMS)]
    if unusable:
        raise RuntimeError("2회 호출 뒤 유효한 모델 판정 없음: " + ", ".join(unusable[:8]))
    still_invalid = [(records[i]["id"], missing) for i, (_parsed, missing)
                     in enumerate(parsed_all) if missing]
    if still_invalid:
        sample = ", ".join(f"{rec_id}({','.join(missing)})"
                           for rec_id, missing in still_invalid[:8])
        log(f"재시도 후 남은 판정은 0 보완 후 규칙 적용: {sample}")

    rows, missing_cells, invalid, positives = [], 0, 0, 0
    for rec, (parsed, missing), spans, facts in zip(records, parsed_all, spans_all, facts_all):
        missing_cells += len(missing)
        invalid += int(len(missing) == len(ITEMS))
        final = postprocess(parsed, rec, spans, facts)
        positives += sum(final[item]["위반여부"] for item in ITEMS)
        rows.append(to_row(rec["id"], final))

    write_csv(rows, out_path)
    errors = validate_csv(
        out_path, [rec["id"] for rec in records],
        {rec["id"]: full_text(rec) for rec in records})
    report: Dict[str, Any] = {
        "건수": len(records), "모델로드_s": round(runner.load_seconds, 1),
        "추론_s": round(time.time() - inference_started, 1),
        "전체_s": round(time.time() - started, 1), "양성예측수": positives,
        "비정상JSON": invalid, "누락셀": missing_cells, "출력": out_path,
        "재시도공고수": len(invalid_indices), "보완공고수": len(still_invalid),
        "자가검증": "PASS" if not errors else errors,
    }
    if labels_path:
        eval_res = evaluate_rows(rows, labels_path)
        report["dev평가"] = eval_res
        log("\n==================== [dev 채점 결과] ====================")
        log(f"★ Macro F1 (대회 공식 지표): {eval_res['MacroF1']:.6f}")
        log("-------------------------------------------------------")
        log(f"{'항목':<6} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'F1':<6}")
        log("-------------------------------------------------------")
        for v in ITEMS:
            d = eval_res["items"][v]
            if d["TP"] or d["FP"] or d["FN"]:
                log(f"{v:<6} | {d['TP']:<4} | {d['FP']:<4} | {d['FN']:<4} | {d['F1']:<6.4f}")
        log("=======================================================\n")
    log(json.dumps(report, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="나라장터 법령 위반 hybrid 추론")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--input", default=None, help="기본: <data-dir>/test.jsonl.gz")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--quantization", default=os.environ.get("PPS_QUANT", QUANT))
    parser.add_argument("--gpu-mem", type=float, default=0.92)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_CONTEXT_CHARS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--labels", default=None, help="로컬 dev_labels.csv 평가")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN,
                        help=f"모델 컨텍스트 상한 (대회 규정상 최대 32768, 기본값 {MAX_MODEL_LEN})")
    parser.add_argument("--api-url", default=None, help="RunPod 등의 vLLM API 서버 주소 (예: http://localhost:8000/v1)")
    args = parser.parse_args()
    if min(args.chunk, args.max_chars, args.max_tokens, args.max_model_len) <= 0:
        parser.error("chunk/max-chars/max-tokens/max-model-len은 양수여야 합니다")
    if args.max_tokens >= args.max_model_len:
        parser.error("max-tokens는 max-model-len보다 작아야 합니다")

    input_path = args.input or os.path.join(args.data_dir, "test.jsonl.gz")
    out_path = os.path.join(args.output_dir, "submission.csv")
    quant = None if str(args.quantization).lower() in ("", "none") else args.quantization
    if args.mock:
        runner_cls = MockRunner
        runner_kwargs = {}
    elif args.api_url:
        runner_cls = APIRunner
        runner_kwargs = {
            "api_url": args.api_url, "model_dir": args.model_dir,
            "max_tokens": args.max_tokens, "seed": SEED,
        }
    else:
        runner_cls = VLLMRunner
        runner_kwargs = {
            "model_dir": args.model_dir, "quant": quant, "max_tokens": args.max_tokens,
            "seed": SEED, "gpu_mem": args.gpu_mem, "tp": args.tp,
        }
    report = run(input_path, out_path, runner_cls,
                 args.limit, args.chunk, args.max_chars, args.data_dir, args.labels,
                 max_model_len=args.max_model_len, **runner_kwargs)
    return 0 if report.get("자가검증") in (None, "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
