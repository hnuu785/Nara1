#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""나라장터 자체입찰 공고 법령 위반사항 모니터링 AI 경진대회 베이스라인.

평가 서버는 이 파일을 `python script.py`로 그대로 실행합니다.
  입력   ./data/test.jsonl.gz (+ 항목표.json · 정답스키마_디코딩.json)
  출력   ./output/submission.csv  (열 = id, v1..v24, e1..e24)
         v = 위반 여부 0/1, e = 근거 문구(원문 부분문자열, 비위반은 빈칸)
  경로   PPS_DATA_DIR · PPS_OUTPUT_DIR · PPS_MODEL_DIR 환경변수 우선

전체 흐름
  데이터 로드 → 프롬프트 구성 → vLLM 배치 추론 → JSON 파싱
  → 근거 문구 검증 → submission.csv 저장 → 형식 검증

로컬 실행
  python script.py --mock          # 모델 없이 입력·출력 흐름 확인
  python script.py --limit 10      # 앞 10건 실행
"""
from __future__ import annotations

# ===== 1. 상수·경로 =====
import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

def _resolve_default_dir(env_var: str, default_name: str) -> str:
    val = os.environ.get(env_var)
    if val:
        return val
    if os.path.exists(default_name):
        return default_name
    parent_rel = os.path.join("..", default_name)
    if os.path.exists(parent_rel):
        return parent_rel
    return f"./{default_name}"

DATA_DIR = _resolve_default_dir("PPS_DATA_DIR", "data")
OUTPUT_DIR = os.environ.get("PPS_OUTPUT_DIR", "./output")
MODEL_DIR = os.environ.get("PPS_MODEL_DIR", "/opt/models/gemma-4-26B-A4B-it")

ITEMS = [f"v{i}" for i in range(1, 25)]
EVID = [f"e{i}" for i in range(1, 25)]
COLUMNS = ["id"] + ITEMS + EVID
ABSENCE = ["v10", "v11", "v16", "v18", "v20"]          # 부재탐지 항목: 근거 문구 빈칸

DOC_ORDER = ["공고문", "규격서", "과업지시서", "제안요청서", "예외공표서", "기타"]
DOCUMENT_SIGNALS = (
    (re.compile(r"입찰\s*참가\s*자격|참가\s*자격|입찰\s*방법|계약\s*방법"), 9),
    (re.compile(r"직접\s*생산|세부\s*품명|중소기업|중기업|소기업|소상공인|중소기업자간\s*경쟁|경쟁제품"), 8),
    (re.compile(r"공동\s*(?:수급|계약|이행|도급)|지분율|출자\s*비율"), 8),
    (re.compile(r"소프트웨어|정보\s*시스템|정보화\s*사업|SW\s*사업|사업금액별\s*참여|대기업|상호출자"), 9),
    (re.compile(r"설명회|현장\s*설명|제안서.*제출|입찰서.*제출"), 7),
    (re.compile(r"추정\s*가격|기초\s*금액|사업\s*금액|예산|부가가치세|낙찰하한율"), 6),
    (re.compile(r"본점|본사|주된\s*영업소|지역\s*제한|소재지|업종코드|면허업종|실적\s*제한"), 7),
    (re.compile(r"실적"), 10),
    (re.compile(r"제조사|모델명|시리즈|호환|배터리"), 8),
    (re.compile(r"제출\s*서류|규격|품명|과업\s*내용|특수\s*조건|공급|기술\s*지원"), 3),
)
META_FIELDS = [
    "적용계약법", "업무구분", "계약방법", "낙찰방법", "낙찰하한율",
    "배정예산금액", "입찰추정가격", "소관구분", "공동도급구성방식", "정보화사업여부",
    "세부품명번호목록", "제한지역코드목록", "지역제한여부", "면허업종제한목록", "업종제한여부",
    "조항호내용", "공고게시일자", "개찰예정일자", "긴급공고여부", "입찰방법", "조달방식",
]

SEED = 20260826
MAX_MODEL_LEN = 32768                   # 대회 허용 최대 컨텍스트 길이
MAX_TOKENS = 1536                       # 구조화 출력 토큰 예산
PROMPT_BUDGET = MAX_MODEL_LEN - MAX_TOKENS
EVIDENCE_MAX = 500                      # 근거 문구 셀 글자 수 상한
QUANT = "int8_per_channel_weight_only"  # 평가 서버 양자화 설정

# 대회 운영진의 법령패키지 보완 공지(2026-09-22)에 포함된 판정 기준.
# https://dacon.io/en/competitions/official/236754/talkboard/417908
# 평가 서버는 오프라인이므로 제출 시 이 대회 제공 자료의 금액을 함께 싣는다.
NATIONAL_NOTICE_AMOUNT = 230_000_000
LOCAL_METRO_AMOUNT = 350_000_000
LOCAL_OTHER_AMOUNT = 500_000_000
LOCAL_TECHNICAL_AMOUNT = 330_000_000
LOCAL_SAFETY_AMOUNT = 150_000_000


def log(msg: str) -> None:
    print(f"[v2] {msg}", file=sys.stderr, flush=True)


# ===== 2. 데이터 로더 =====
def _open(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, "r", encoding="utf-8")


def validate_record(rec: Any) -> None:
    """레코드 1건의 최소 스키마 검사 (id · docs(공고문 1개 이상) · meta)"""
    if not isinstance(rec, dict):
        raise ValueError(f"레코드가 object가 아니다: {type(rec).__name__}")
    for k in ("id", "docs", "meta"):
        if k not in rec:
            raise ValueError(f"필수 키 없음: {k}")
    if not isinstance(rec["id"], str) or not rec["id"]:
        raise ValueError("id가 비어 있다")
    docs = rec["docs"]
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"docs가 비어 있다 (id={rec['id']})")
    for d in docs:
        if not isinstance(d, dict) or not all(k in d for k in ("doc_id", "type", "text")):
            raise ValueError(f"docs 원소 형식 오류 (id={rec['id']})")
        if not isinstance(d["text"], str):
            raise ValueError(f"docs.text가 문자열이 아니다 (id={rec['id']})")
    if not any(d["type"] == "공고문" for d in docs):
        raise ValueError(f"공고문이 없다 (id={rec['id']})")
    if not isinstance(rec["meta"], dict):
        raise ValueError(f"meta가 object가 아니다 (id={rec['id']})")


def normalize(rec: Dict[str, Any]) -> Dict[str, Any]:
    """NFC 정규화 — macOS에서 만든 파일은 한글이 NFD로 저장될 수 있어 문자열 비교가 어긋날 수 있습니다."""
    for d in rec.get("docs", []):
        d["text"] = unicodedata.normalize("NFC", d["text"])
        if isinstance(d.get("type"), str):
            d["type"] = unicodedata.normalize("NFC", d["type"])
    return rec


def iter_records(path: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    n = 0
    with _open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {e}") from e
            validate_record(rec)
            yield normalize(rec)
            n += 1
            if limit and n >= limit:
                return


def full_text(rec: Dict[str, Any]) -> str:
    """근거문구 대조용 원문 (프롬프트에 넣은 것과 같은 텍스트 · NFC)"""
    return "\n".join(d["text"] for d in rec["docs"])


def _excerpt(body: str, limit: int) -> str:
    """긴 문서는 앞부분과 법령 판정에 필요한 원문 행을 위치순으로 발췌합니다."""
    if len(body) <= limit:
        return body
    if limit < 500:
        return body[:limit]

    lead = min(500, limit // 5)
    candidates = []
    offset = 0
    for line in body.splitlines(keepends=True):
        raw = line.strip()
        if not raw:
            offset += len(line)
            continue
        if len(raw) > 1000:
            # 일부 OCR 문서는 줄바꿈 없이 여러 쪽이 한 줄로 합쳐져 있다.
            # 예전에는 그 줄에서 가장 앞선 키워드 주변 900자만 남겨 뒤쪽의
            # 참가자격·기업규모·지역·SW 조항이 통째로 빠질 수 있었다. 각 신호
            # 주변 창을 따로 후보로 만들어 관련 문장이 여러 곳에 있어도 보존한다.
            windows = []
            for pattern, weight in DOCUMENT_SIGNALS:
                for match in pattern.finditer(raw):
                    start = max(0, match.start() - 180)
                    end = min(len(raw), max(start + 900, match.end() + 360))
                    snippet = raw[start:end]
                    score = sum(w for p, w in DOCUMENT_SIGNALS if p.search(snippet))
                    windows.append((score, start, end, snippet))
            windows.sort(key=lambda item: (-item[0], item[1]))
            selected = []
            for score, start, end, snippet in windows:
                if any(start < used_end and end > used_start for used_start, used_end, _ in selected):
                    continue
                selected.append((start, end, (score, offset + start, snippet)))
            candidates.extend(item for _, _, item in selected)
        else:
            score = sum(weight for pattern, weight in DOCUMENT_SIGNALS if pattern.search(raw))
            if score:
                candidates.append((score, offset, raw))
        offset += len(line)

    chosen = []
    used = lead
    for _, pos, raw in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if pos < lead or used + len(raw) + 5 > limit:
            continue
        chosen.append((pos, raw))
        used += len(raw) + 5
    chosen.sort()
    result = body[:lead]
    for _, raw in chosen:
        result += "\n[…]\n" + raw
    return result[:limit]


def build_context(rec: Dict[str, Any], max_chars: int = 10000) -> str:
    """공고문과 첨부 모두에서 판정 관련 내용을 한 번의 모델 입력에 싣습니다."""
    order = {t: i for i, t in enumerate(DOC_ORDER)}
    pool = sorted(rec["docs"], key=lambda d: (order.get(d["type"], len(DOC_ORDER)), d["doc_id"]))
    overhead = sum(len(d["type"]) + len(d["doc_id"]) + 16 for d in pool) + 120
    body_budget = max(200, max_chars - overhead)
    weights = [3.0 if d["type"] == "공고문" else 1.5 if d["type"] == "제안요청서" else 1.0 for d in pool]
    total_weight = sum(weights)
    quotas = [min(len(d["text"]), max(200, int(body_budget * w / total_weight)))
              for d, w in zip(pool, weights)]
    # 짧은 문서에서 남은 예산을 아직 긴 문서로 돌립니다.
    spare = max(0, body_budget - sum(quotas))
    for i in sorted(range(len(pool)), key=lambda j: (-weights[j], j)):
        extra = min(spare, len(pool[i]["text"]) - quotas[i])
        quotas[i] += extra
        spare -= extra

    chunks, partial = [], []
    for d, quota in zip(pool, quotas):
        body = _excerpt(d["text"], quota)
        chunks.append(f"[{d['type']}:{d['doc_id']}]\n{body}")
        if len(body) < len(d["text"]):
            partial.append(d["doc_id"])
    text = "\n\n".join(chunks)
    if partial:
        text += "\n[발췌 문서] " + ", ".join(partial) + " — 일부 문장은 길이 제한으로 빠졌다."
    dropped = rec.get("dropped_doc_counts") or {}
    if dropped:
        text += "\n[미제공 문서] " + ", ".join(f"{t} {n}건" for t, n in sorted(dropped.items()))
    return text


def format_meta(rec: Dict[str, Any]) -> str:
    """나라장터 메타를 한 줄짜리 목록으로. 값이 없는 필드는 '미기재'로 표시합니다."""
    m = rec.get("meta", {})
    lines = []
    for k in META_FIELDS:
        if k in m:
            v = m[k]
            lines.append(f"- {k}: {'미기재' if v is None else v}")
    return "\n".join(lines)


def v24_comparison_note(rec: Dict[str, Any], max_chars: int = 3500) -> str:
    """v24에서 같은 뜻의 입력값과 공고 문구를 나란히 읽도록 발췌한다.

    발췌만으로 불일치를 단정하지 않는다. VAT 포함 기초금액과 VAT 제외 추정가격,
    납품장소와 업체 소재지처럼 의미가 다른 값을 모델이 혼동하지 않게 돕는다.
    """
    meta = rec.get("meta", {})
    fields = (
        "입찰추정가격", "배정예산금액", "계약방법", "낙찰방법", "낙찰하한율",
        "공동도급구성방식", "정보화사업여부", "세부품명번호목록",
        "지역제한여부", "제한지역코드목록", "업종제한여부", "면허업종제한목록",
    )
    meta_lines = []
    for field in fields:
        if field not in meta:
            continue
        value = meta[field]
        meta_lines.append(f"- 메타 {field}: {'미기재' if value is None else value}")

    specs = (
        ("업체 지역", re.compile(r"주된\s*영업소|본점\s*소재지|본점소재지|지역\s*제한"), 3),
        ("계약·낙찰 방식", re.compile(r"입찰\s*및\s*계약\s*방법|계약\s*방법|입찰\s*방법|낙찰\s*방법|낙찰하한율"), 2),
        ("추정가격", re.compile(r"추정\s*가격|입찰추정가격"), 2),
        ("업종", re.compile(r"업종\s*제한|업종코드|면허업종|입찰참가자격등록"), 2),
        ("예산·기초금액", re.compile(r"기초\s*금액|기초금액|사업\s*금액|예산액"), 1),
        ("공동도급", re.compile(r"공동\s*(?:도급|수급|계약|이행)|공동이행|분담이행"), 1),
        ("SW·세부품명", re.compile(r"정보화\s*사업|소프트웨어\s*사업|SW\s*사업|세부\s*품명(?:번호)?"), 1),
    )
    doc_lines = []
    for label, pattern, cap in specs:
        used = set()
        for doc in sorted(rec.get("docs", []), key=lambda d: (0 if d.get("type") == "공고문" else 1, d.get("doc_id", ""))):
            text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(doc.get("text") or "")))
            for match in pattern.finditer(text):
                start = max(0, match.start() - 95)
                end = min(len(text), match.end() + 190)
                snippet = text[start:end].strip()
                key = snippet[:100]
                if key in used:
                    continue
                used.add(key)
                doc_lines.append(f"- 문서 {label} ({doc.get('type', '')}): {snippet}")
                if sum(1 for line in doc_lines if line.startswith(f"- 문서 {label} ")) >= cap:
                    break
            if sum(1 for line in doc_lines if line.startswith(f"- 문서 {label} ")) >= cap:
                break
    if not meta_lines and not doc_lines:
        return ""
    header = "[v24 동일 의미 필드 대조 발췌 — 불일치 후보가 아니라 원문 확인용]\n"
    guidance = (
        "\n같은 종류의 필드끼리만 비교한다. 기초금액·예산(부가세 포함)과 추정가격(부가세 제외), "
        "단가계약의 단가와 총액, 1원 반올림 차이, 납품·행사 장소와 입찰업체의 본점 소재지, "
        "업종 명칭 언급과 실제 참가자격 등록요건을 혼동하지 않는다. "
        "문서끼리 내용이 다르면 공고의 실제 적용 조건을 확인한다.\n\n"
    )
    note = header + "\n".join(meta_lines)
    for line in doc_lines:
        if len(note) + len(line) + len(guidance) + 1 > max_chars:
            continue
        note += "\n" + line
    return note + guidance


V24_ESTIMATE = re.compile(
    r"추정\s*가격\s*[:：]?\s*(?:금\s*)?(\d[\d,]*)\s*원"
)
V24_METHOD = re.compile(
    r"(?:입찰\s*방법|입찰\s*및\s*계약\s*방법|계약\s*방법|입찰방식)"
    r"\s*[:：]\s*(?:전자입찰\s*[,·]?\s*)?(일반경쟁|제한경쟁|수의계약)"
)


def v24_explicit_mismatch_evidence(rec: Dict[str, Any]) -> Optional[str]:
    """같은 의미임이 분명한 추정가격·계약방법의 불일치만 규칙으로 보정한다."""
    meta = rec.get("meta", {})
    expected_price = meta.get("입찰추정가격")
    # 단가계약의 단가와 총 추정가격, 원 단위 반올림은 직접 비교하지 않는다.
    unit_contract = bool(re.search(
        r"단가\s*(?:계약|입찰)|(?:계약|입찰)\s*단가", full_text(rec)
    ))
    if (isinstance(expected_price, (int, float)) and not isinstance(expected_price, bool)
            and not unit_contract):
        for doc in rec.get("docs", []):
            if doc.get("type") != "공고문":
                continue
            text = str(doc.get("text") or "")
            for match in V24_ESTIMATE.finditer(text):
                documented_price = int(match.group(1).replace(",", ""))
                if abs(documented_price - int(expected_price)) >= 1000:
                    return text[match.start():match.end()].strip()[:EVIDENCE_MAX]

    method = str(meta.get("계약방법") or "")
    expected_method = next((key for key in ("일반경쟁", "제한경쟁", "수의계약") if key in method), None)
    if expected_method == "일반경쟁":
        for doc in rec.get("docs", []):
            if doc.get("type") != "공고문":
                continue
            text = str(doc.get("text") or "")
            for match in V24_METHOD.finditer(text):
                if match.group(1) == "제한경쟁":
                    return text[match.start():match.end()].strip()[:EVIDENCE_MAX]
    return None


SOFTWARE_SCOPE_CUE = re.compile(
    r"정보화\s*사업|정보\s*시스템.{0,45}(?:구축|개발|유지\s*보수|고도화|전환)|"
    r"소프트웨어.{0,45}(?:사업|개발|구축|유지\s*보수|참여\s*제한)|"
    r"사업금액별\s*참여\s*제한|SW\s*사업"
)


def software_scope_candidate(rec: Dict[str, Any]) -> bool:
    """메타 플래그가 비어 있어도 문서의 실질 SW 업무 표현을 검토 대상으로 삼는다."""
    meta = rec.get("meta", {})
    if str(meta.get("정보화사업여부") or "").strip().upper() == "Y":
        return True
    return bool(SOFTWARE_SCOPE_CUE.search(full_text(rec)))


# ===== 3. 항목표·디코딩 스키마 =====
# data/에 항목표.json·정답스키마_디코딩.json이 동봉됩니다.
# 항목명·근거조문·비고는 항목표.json 에 있으니 여기에 사본을 두지 않습니다.


def item_table(data_dir: str = DATA_DIR) -> Dict[str, Dict[str, Any]]:
    p = os.path.join(data_dir, "항목표.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} 가 없습니다 — data/ 를 그대로 둔 채 실행하세요.")
    return json.load(io.open(p, encoding="utf-8"))["항목"]


def decode_schema(data_dir: str = DATA_DIR) -> Dict[str, Any]:
    """베이스라인의 구조화 출력에 사용할 JSON Schema를 불러옵니다."""
    p = os.path.join(data_dir, "정답스키마_디코딩.json")
    if os.path.exists(p):
        s = json.load(io.open(p, encoding="utf-8"))
        return s["properties"]["판정"] if "판정" in s.get("properties", {}) else s
    props = {}
    for v in ITEMS:
        props[v] = {
            "type": "object", "additionalProperties": False,
            "required": ["위반여부", "근거문구"],
            "properties": {
                "위반여부": {"type": "integer", "enum": [0, 1]},
                "근거문구": {"type": "null"} if v in ABSENCE else {"type": ["string", "null"]},
            },
        }
    return {"type": "object", "additionalProperties": False, "required": list(ITEMS), "properties": props}


# 항목표의 법령 매핑과 함께 보여줄 핵심 조문. 파일은 대회 법령패키지에서 읽는다.
LAW_ARTICLES = {
    "지방계약법": [
        ("지방자치단체를 당사자로 하는 계약에 관한 법률 시행령", "20"),
        ("지방자치단체를 당사자로 하는 계약에 관한 법률 시행규칙", "24"),
        ("지방자치단체를 당사자로 하는 계약에 관한 법률 시행규칙", "25"),
    ],
    "국가계약법": [
        ("국가를 당사자로 하는 계약에 관한 법률 시행령", "21"),
        ("국가를 당사자로 하는 계약에 관한 법률 시행규칙", "25"),
    ],
    "물품": [
        ("중소기업제품 구매촉진 및 판로지원에 관한 법률", "7"),
        ("중소기업제품 구매촉진 및 판로지원에 관한 법률", "9"),
        ("중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령", "2조의2"),
        ("중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령", "2조의3"),
    ],
    "소프트웨어": [("소프트웨어 진흥법", "48")],
}


def _article_text(source: str, number: str, max_chars: int = 2600) -> str:
    marker = "제" + (number if "조" in number else number + "조")
    match = re.search(r"(?m)^" + re.escape(marker) + r"(?=\(|\s)", source)
    if not match:
        return ""
    next_article = re.search(r"(?m)^제\d+조(?:의\d+)?(?=\(|\s)", source[match.end():])
    end = match.end() + next_article.start() if next_article else len(source)
    body = source[match.start():end].replace("<![CDATA[", "").replace("]]>", "").strip()
    if len(body) > max_chars:
        cut = body.rfind("\n", 0, max_chars)
        body = body[:cut if cut > max_chars // 2 else max_chars].rstrip() + "\n[이하 생략]"
    return body


def load_law_notes(tbl: Dict[str, Dict[str, Any]], data_dir: str) -> Dict[str, str]:
    """평가 서버가 제공하는 법령패키지의 고정 스냅샷으로 법령별 프롬프트를 만든다."""
    law_dir = os.path.join(data_dir, "법령패키지", "법령")
    if not os.path.isdir(law_dir):
        raise FileNotFoundError(f"대회 제공 법령패키지 디렉토리를 찾을 수 없습니다: {law_dir}")

    files = {}
    for name in os.listdir(law_dir):
        if name.endswith(".txt"):
            files[unicodedata.normalize("NFC", name[:-4])] = os.path.join(law_dir, name)

    needed = {stem for specs in LAW_ARTICLES.values() for stem, _ in specs}
    missing_files = sorted(needed - files.keys())
    if missing_files:
        raise FileNotFoundError(
            "대회 제공 법령패키지에서 필요한 파일을 찾지 못했습니다: "
            + ", ".join(missing_files)
        )

    sources = {stem: io.open(files[stem], encoding="utf-8").read() for stem in needed}
    notes = {}
    for law in ("지방계약법", "국가계약법"):
        lines = ["[대회 제공 항목표의 적용 조문]",
                 "아래 조문은 판정 기준이다. 출력 근거문구에는 반드시 공고 문서의 원문만 사용한다."]
        lines.extend(f"- {v}: {tbl[v][law]}" for v in ITEMS)
        notes[law] = "\n".join(lines) + "\n\n"

    missing_articles = []
    for group, specs in LAW_ARTICLES.items():
        parts = []
        for stem, number in specs:
            excerpt = _article_text(sources[stem], number)
            if excerpt:
                parts.append(f"[{stem} 제{number if '조' in number else number + '조'}]\n{excerpt}")
            else:
                missing_articles.append(f"{stem} 제{number if '조' in number else number + '조'}")
        notes[group] = (notes.get(group, "") +
                        ("[대회 제공 법령 원문 발췌]\n" + "\n\n".join(parts) + "\n\n" if parts else ""))

    if missing_articles:
        raise ValueError(
            "대회 제공 법령패키지에서 필요한 조문을 찾지 못했습니다: "
            + ", ".join(missing_articles)
        )
    return notes


# ===== 대회 제공 자료만 사용하는 경량 검색 RAG =====
# 임베딩 모델을 추가로 로드하지 않는다. 법령패키지와 해당 공고의 원문을
# 글자 단위 BM25로 검색하므로 평가 서버의 네트워크·GPU 메모리에 의존하지 않는다.
RAG_WORD = re.compile(r"[가-힣]+|[a-z][a-z0-9]+|\d+(?:\.\d+)?", re.I)
RAG_ARTICLE = re.compile(r"(?m)^제\d+조(?:의\d+)?(?=\(|\s)")
RAG_DOC_TOPICS = (
    ("기업규모·경쟁제품", re.compile(r"중소기업자(?:간)?|중소기업\s*확인서|(?<!중)소기업|소상공인|직접\s*생산|경쟁제품"),
     "입찰 참가 자격 중소기업자 소기업 소상공인 직접생산확인 경쟁제품 세부품명", 1),
    ("업체 소재지", re.compile(r"본점|주된\s*영업소|사업장\s*소재지|지역\s*제한"),
     "입찰 참가 자격 본점 소재지 주된 영업소 시 군 구 지역 제한", 1),
    ("소프트웨어 과업", SOFTWARE_SCOPE_CUE,
     "정보시스템 개발 구축 유지보수 소프트웨어 사업 사업금액별 참여 제한", 1),
)
RAG_LAW_QUERIES = {
    "기업규모·경쟁제품": "중소기업자간 경쟁제품 직접생산 소기업 소상공인 제한입찰 추정가격 예외",
    "업체 소재지": "입찰 참가 자격 본점 주된 영업소 소재지 지역제한 시 군 구 인접 지역 예외",
    "소프트웨어 과업": "사업금액별 중소 소프트웨어사업자 참여 지원 대기업 참여 제한",
}
RAG_LAW_ARTICLES = {
    "중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령": {
        "제9조": "중소기업자간 경쟁입찰의 참여자격",
    },
    "중소기업제품 구매촉진 및 판로지원에 관한 법률": {
        "제8조": "경쟁입찰 참여자격",
    },
    "중소 소프트웨어사업자의 사업 참여 지원에 관한 지침": {
        "제3조": "사업금액의 하한 적용",
    },
    "소프트웨어 진흥법 시행령": {
        "제41조": "중소 소프트웨어사업자의 기준",
    },
}


def _rag_terms(text: str) -> List[str]:
    """한글 띄어쓰기 차이에 견디도록 단어와 음절 2그램을 함께 사용한다."""
    terms = []
    for match in RAG_WORD.finditer(unicodedata.normalize("NFKC", text).lower()):
        word = match.group()
        if len(word) < 2:
            continue
        terms.append(word)
        if "가" <= word[0] <= "힣" and len(word) >= 3:
            terms.extend(word[i:i + 2] for i in range(len(word) - 1))
    return terms


class LexicalRAGIndex:
    """작은 법령·공고 청크 집합을 위한 표준 라이브러리 BM25 검색기."""

    def __init__(self, passages: List[Dict[str, str]]):
        self.passages = passages
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.lengths: List[int] = []
        for index, passage in enumerate(passages):
            counts = Counter(_rag_terms(passage["text"]))
            self.lengths.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings[term].append((index, freq))
        self.average = sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query: str, allowed: Optional[set] = None,
               allowed_keys: Optional[set] = None,
               excluded: Optional[set] = None, limit: int = 8) -> List[Tuple[float, int]]:
        n = len(self.passages)
        if not n:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for term in set(_rag_terms(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            inverse = math.log1p((n - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, freq in posting:
                passage = self.passages[index]
                if allowed is not None and passage["source"] not in allowed:
                    continue
                if allowed_keys is not None and (passage["source"], passage.get("article")) not in allowed_keys:
                    continue
                if excluded is not None and (passage["source"], passage.get("article")) in excluded:
                    continue
                norm = 1.2 * (0.25 + 0.75 * self.lengths[index] / self.average)
                scores[index] += inverse * (freq * 2.2) / (freq + norm)
        return sorted(((score, index) for index, score in scores.items() if score > 0),
                      key=lambda pair: (-pair[0], pair[1]))[:limit]


def build_law_rag_index(data_dir: str) -> LexicalRAGIndex:
    """대회 법령패키지에서 판정에 직접 연결되는 보충 조문만 인덱싱한다."""
    law_dir = os.path.join(data_dir, "법령패키지", "법령")
    passages = []
    for filename in sorted(os.listdir(law_dir)):
        if not filename.endswith(".txt"):
            continue
        stem = unicodedata.normalize("NFC", filename[:-4])
        titles = RAG_LAW_ARTICLES.get(stem)
        if not titles:
            continue
        source = io.open(os.path.join(law_dir, filename), encoding="utf-8").read()
        matches = list(RAG_ARTICLE.finditer(source))
        for index, match in enumerate(matches):
            article = match.group()
            if article not in titles:
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
            body = source[match.start():end].replace("<![CDATA[", "").replace("]]>", "")
            # 같은 파일의 부칙·개정문에 반복되는 제n조는 현행 본문이 아니다.
            if titles[article] not in body[:100]:
                continue
            passages.append({"source": stem, "article": article,
                             "text": body[:1000].strip()})
    return LexicalRAGIndex(passages)


def _rag_allowed_laws(topic: str, rec: Dict[str, Any]) -> set:
    if topic == "기업규모·경쟁제품":
        enactment = "중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령"
        statute = "중소기업제품 구매촉진 및 판로지원에 관한 법률"
        return {(enactment, "제9조"), (statute, "제8조")}
    if topic == "소프트웨어 과업":
        return {("중소 소프트웨어사업자의 사업 참여 지원에 관한 지침", "제3조"),
                ("소프트웨어 진흥법 시행령", "제41조")}
    return set()


def law_rag_note(rec: Dict[str, Any], index: LexicalRAGIndex, existing: str,
                 max_chars: int = 2200) -> str:
    """이미 넣은 핵심 조문을 제외하고 기록별 관련 조문만 추가한다."""
    full = full_text(rec)
    fixed = {(stem, "제" + (number if "조" in number else number + "조"))
             for specs in LAW_ARTICLES.values() for stem, number in specs}
    lines = []
    for topic, cue, _, _ in RAG_DOC_TOPICS:
        if not cue.search(full):
            continue
        if topic == "기업규모·경쟁제품":
            catalog_note = str(rec.get("_competition_note") or "")
            if not re.search(r"(?m)^- (?:\d{10}: .+ 지정(?:;|$)|명칭·과업 후보)", catalog_note):
                continue
        allowed = _rag_allowed_laws(topic, rec)
        if not allowed:
            continue
        for _, passage_id in index.search(RAG_LAW_QUERIES[topic], allowed_keys=allowed,
                                           excluded=fixed, limit=6):
            passage = index.passages[passage_id]
            body = passage["text"][:650]
            if body in existing or any(body in line for line in lines):
                continue
            line = f"- {topic} [{passage['source']} {passage['article']}]: {body}"
            if sum(len(x) for x in lines) + len(line) > max_chars:
                continue
            lines.append(line)
            break
        if len(lines) >= 3:
            break
    if not lines:
        return ""
    return ("[대회 법령패키지 검색 발췌 — 조문 적용 여부는 공고 조건과 대조]\n"
            + "\n".join(lines)
            + "\n출력 근거문구에는 법령이 아닌 공고·첨부의 원문만 인용한다.\n\n")


def document_rag_note(rec: Dict[str, Any], max_chars: int = 2600) -> str:
    """긴 공고에서 항목별 참가자격·지역·SW 과업 문구를 찾아 강조한다."""
    if sum(len(d["text"]) for d in rec["docs"]) <= 10000:
        return ""
    full = full_text(rec)
    lines = []
    for topic, cue, query, take in RAG_DOC_TOPICS:
        if not cue.search(full):
            continue
        passages = []
        for doc in rec["docs"]:
            seen = set()
            for match in cue.finditer(doc["text"]):
                start = max(0, match.start() - 250)
                bucket = start // 350
                if bucket in seen:
                    continue
                seen.add(bucket)
                snippet = doc["text"][start:min(len(doc["text"]), match.end() + 430)].strip()
                if len(snippet) >= 60:
                    passages.append({"source": doc["type"], "article": doc["doc_id"],
                                     "text": snippet})
                if len(seen) >= 120:
                    break
        index = LexicalRAGIndex(passages)
        ranked = []
        for score, passage_id in index.search(query, limit=20):
            passage = passages[passage_id]
            snippet = passage["text"]
            if topic in ("기업규모·경쟁제품", "업체 소재지"):
                score += 5 if re.search(r"입찰\s*참가\s*자격|참가\s*자격|업체이어야|입찰에\s*참가", snippet) else 0
            if passage["source"] == "공고문":
                score += 2
            if re.search(r"붙임|제출\s*서류\s*목록|작성\s*서식", snippet[:170]):
                score -= 3
            ranked.append((score, passage_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        count = 0
        for _, passage_id in ranked:
            passage = passages[passage_id]
            snippet = passage["text"][:640]
            line = f"- {topic} ({passage['source']}): {snippet}"
            if sum(len(x) for x in lines) + len(line) > max_chars:
                continue
            lines.append(line)
            count += 1
            if count >= take:
                break
        if len(lines) >= 4:
            break
    if not lines:
        return ""
    return ("[공고·첨부 검색 발췌 — 원문 일부이므로 앞뒤 조건을 확인]\n"
            + "\n".join(lines) + "\n\n")


# ===== 4. 프롬프트 구성 =====
# 베이스라인 프롬프트는 출력 형식과 항목 목록을 구성합니다.
SYSTEM_HEAD = """당신은 공공 입찰공고의 법령 위반 여부를 점검한다.
공고문과 첨부 문서, 그리고 나라장터 입력 메타를 함께 읽고 아래 24개 항목 각각에 대해
위반 여부(1/0)와 근거 문구를 판정한다.

지켜야 할 것
1. 24개 항목 전부에 답한다. 불확실해도 적용 조건과 제공된 근거를 비교해 0 또는 1로 판정한다.
2. 근거 문구는 반드시 **주어진 문서에 그대로 있는 문장**을 옮긴다. 요약하거나 고쳐 쓰지 않는다.
   원문에 없는 문구는 근거로 인정되지 않는다. 500자를 넘기지 않는다.
3. 아래 '근거 없음' 표시가 붙은 항목은 **있어야 할 문구가 없는 것**이 위반이다.
   인용할 원문이 존재하지 않으므로 근거 문구를 null로 둔다.
4. 항목마다 적용 조건부터 확인한다. 금액 구간은 단가나 부가세 포함 예산이 아니라
   입찰추정가격으로 나눈다. 일반 물품·용역의 1억원 미만과 1억원 이상 구간을 혼동하지 않는다.
5. 경쟁제품은 품명코드뿐 아니라 지정표의 세부 조건까지 확인한다. 기업범위는
   참가자격의 실제 허용 집합으로 판단한다. 중기업도 허용하면 소기업만 허용한 것이 아니다.
6. 직접생산·중소기업·소기업의 '부재'는 참가자격에 해당 보유/규모 조건이 있는지
   확인한다. 메타, 공고 제목, 제출서류, 일반 법령 인용만으로 참가자격을 대신하지 않는다.
7. 공동수급을 불허하면 공동수급체 구성원 최소지분율 항목은 적용되지 않는다.
   물품공급 협약은 입찰 전 제출 요구인지 낙찰 후 절차인지 구분한다.
8. SW 참여제한은 일반 중소기업 제한과 구별한다. 공고와 제공 제안요청서에
   소프트웨어 진흥법 제48조에 따른 참여제한/적용 안내가 있는지 확인한다.
9. v24는 메타와 문서의 동일한 의미를 가진 필드를 비교한다. null과 N을 구분하며
    익명화된 지역 토큰에서 실제 지명을 추측하지 않는다.
10. 필수 문서가 누락되어 항목을 판단할 수 없으면 제출 규칙에 따라 0을 선택한다.
    누락을 위반 근거로 추정하지 않는다.
11. v3은 참가자격에서 요구하는 최소 실적액을 사업예산과 비교한다. 실적 금액이
    사업예산보다 작으면 v3 위반이 아니다. 단순한 실적 제출·평가·가점은 제한이 아니다.
12. v13은 지정표의 품목과 실제 구매 대상이 일치할 때만 판단한다. 경쟁제품을
    소기업·소상공인만으로 제한하면 위반 후보이며, 직접생산 요건만으로는 규모 제한이 아니다.
13. v17은 추정가격 1억원 미만에서 참가자격이 중기업까지 허용되는지 본다.
    소기업·소상공인만 허용하면 v17 위반이 아니다. 법령 이름에 '중소기업기본법'이
    나온다는 이유만으로 중소기업까지 허용했다고 판단하지 않는다.
14. v19는 제조사·공급사의 물품공급·기술지원 확약서를 입찰 단계에서 보유·제출하도록
    요구했는지 본다. 입찰자 자신의 납품·A/S 약속, 정품·제조자·대리점 증명서,
    판매업자의 의료기기 공급물량 확인은 구분한다. 계약·낙찰 후 제출 문구만으로는
    위반이 아니지만, 입찰마감 전 보유를 요구하면 계약 때 제출하더라도 위반 후보이다.
15. v21은 국가계약의 최소 지분율 10%, 지방계약의 5%를 기준으로 비교한다.
    그 기준 이상을 요구하거나 최소 지분율 문구가 없으면 v21 위반이 아니다.
16. v11·v14·v16·v18은 나라장터 메타의 조항호내용이나 문서의 일반 법령 인용만으로
    참가자격 요건이 있다고 보지 않는다. 공고의 실제 입찰참가자격에서 허용 기업 범위를
    확인한다. 직접생산확인·업종등록·제출서류 목록은 기업규모 제한을 대신하지 않는다.
17. v6은 메타의 지역제한여부보다 공고의 본점·주된 영업소 참가조건을 확인한다.
    메타가 N이어도 문서에 업체 소재지 제한이 있으면 문서 문구를 기준으로 판단한다.
    납품장소·행사장·시설 위치는 업체 소재지 제한이 아니다.
18. v20은 정보화사업여부가 미입력이어도 과업 내용과 제안요청서에서 실제 SW 사업인지
    확인한다. 소프트웨어사업자 업종코드만으로는 적용을 확정하지 않으며, 일반 중소기업
    제한 문구만으로 사업금액별 SW 참여 기준을 알렸다고 보지 않는다.
19. v24는 대조 발췌가 있으면 같은 의미의 메타 필드와 공고 문구만 직접 비교한다.
    추정가격끼리, 계약방법끼리, 업체 지역제한끼리, 참가자격 업종끼리 비교하고
    부가세 포함 기초금액·단가계약의 단가·장소·일반 안내문을 다른 의미의 메타값과 비교하지 않는다.
    1원 반올림 차이나 문서 간 표현 차이만으로 불일치를 확정하지 않는다.

판정할 24개 항목"""

SYSTEM_TAIL = """
출력은 JSON 하나로만 낸다. 키는 v1~v24, 각 값은 {"위반여부": 0 또는 1, "근거문구": 문자열 또는 null}이다.
설명이나 머리말을 덧붙이지 않는다."""


# 항목별 판정 카드: 적용조건·위반조건·주요 예외를 분리해 판단한다.
# 경쟁제품 품목과 지역별 금액은 대회 제공 지정표·법령 메모를 우선한다.
ITEM_CARDS = {
    "v1": "참가자격을 특정 기관 유형으로만 제한했는지 본다. 특정 기관 실적의 평가·가점, 법정 면허·등록은 별도다.",
    "v2": "추정가격 2억3천만원 미만에서 과거 납품·수행실적을 참가자격으로 요구하면 위반 후보다. 평가·가점 실적은 제외하고 지방계약 소액수의(1억원 이하)는 제외한다.",
    "v3": "참가자격의 최소 실적금액이 배정예산 이상인지 본다. 추정가격이 아니라 사업예산과 비교하고, 실적 평가·가점은 제외한다.",
    "v4": "고시금액 이상에서 특정 발주기관·공공기관·대학병원 또는 지나치게 동일한 대상의 실적만 참가자격으로 인정하는지 본다. 일반 유사실적 요건과 구분한다.",
    "v5": "참가자격의 본점·주된 영업소 지역제한을 확인하고 계약법·업무유형별 허용 기준과 비교한다. 납품장소 주소나 단가만으로 지역제한을 단정하지 않는다.",
    "v6": "고시금액 미만에서 업체의 본점·주된 영업소를 시·군·구 단위로 제한했는지 본다. 메타 지역제한여부=N만 믿지 말고 공고의 참가자격을 읽는다. 지방 소액수의 등 예외와 납품장소 주소는 구분한다.",
    "v7": "지방계약의 참가 지역을 인접 지역까지 임의로 확대했는지 본다. 허용되는 지역 범위와 법정 예외, 소액수의를 구분한다.",
    "v8": "참가자격으로 실적과 지역을 동시에 제한했는지 본다. 평가요소·납품장소는 제외하고 법정 예외와 소액수의를 확인한다.",
    "v9": "규격·과업·제안요청서가 특정 제조사·브랜드·모델을 사실상 강제하는지 본다. 성능 규격, 기존 장비 설명, 명확한 동등 이상 허용은 구분한다.",
    "v10": "실제 구매품목이 지정표상 경쟁제품이면 해당 품목의 직접생산확인을 참가자격으로 요구했는지 본다. 다른 품목 코드나 일반 제재문구는 대신하지 않는다.",
    "v11": "지정표에 있는 실제 경쟁제품을 구매하면서 공고의 참가자격에 중소기업자 제한이 빠졌는지 본다. 문서 안 품명코드·정식 품명도 대조한다. 메타 조항호내용, 직접생산확인, 제출서류 목록은 중소기업자 제한을 대신하지 않는다.",
    "v12": "실제 경쟁제품이 아닌 품목에 직접생산확인을 참가자격으로 강제했는지 본다. 구매 대상과 무관한 품목 코드는 근거가 아니다.",
    "v13": "실제 경쟁제품 입찰을 소기업·소상공인만으로 제한했는지 본다. 지정표의 품목·세부 조건과 법정 예외를 먼저 확인한다.",
    "v14": "추정가격이 2억3천만원 이상인 적용 대상 물품·용역에서 참가자격을 중소기업으로 제한했는지 본다. 문서의 실제 허용 기업 범위와 경쟁제품·법정 예외를 확인하며, 메타의 제한 표시는 증거가 아니다.",
    "v15": "일반물품의 추정가격이 1억원 이상 2억3천만원 미만인데 소기업·소상공인만 참가하도록 제한했는지 본다. 실제 법정 예외를 확인한다.",
    "v16": "적용 대상 물품·용역의 추정가격이 1억원 이상 2억3천만원 미만인데 실제 입찰참가자격에 중소기업자 제한이 빠졌는지 본다. 메타의 조항호내용이나 확인서 제출목록만으로 제한이 있다고 판단하지 말고, 법정 예외를 확인한다.",
    "v17": "일반물품의 추정가격이 2천만원 초과 1억원 미만인데 중기업까지 참가할 수 있게 허용했는지 본다. 소기업·소상공인 한정과 법정 예외를 구분한다.",
    "v18": "추정가격이 2천만원 초과 1억원 미만인 적용 대상 물품·용역에서 실제 참가자격이 소기업·소상공인으로 제한됐는지 본다. 중기업까지 허용하는 중소기업자 조건은 소기업 한정과 다르다. 확인서 제출목록만으로 참가범위를 단정하지 말고 예외를 확인한다.",
    "v19": "제조사·공급사의 물품공급·기술지원 확약서를 입찰 전에 보유·제출시키는지 본다. 입찰자 자체 납품/A/S 약속, 정품·제조자·대리점 증명서, 판매업자의 의료기기 공급물량 확인과 구분한다. 계약·낙찰 후 제출만 요구하면 0이지만, 입찰마감 전 보유 의무가 있으면 계약 때 제출해도 1 후보이다.",
    "v20": "메타 정보화사업여부가 미입력이어도 실제 과업에 정보시스템 구축·개발·유지보수 등 SW 사업이 있는지 확인한다. 사업금액 구간별 중소 SW사업자 참여 기준을 공고에 밝혔는지 본다. 소프트웨어사업자 업종코드나 일반 중소기업 제한만으로 충족하지 않는다.",
    "v21": "공동수급 허용 여부와 공동이행인지 분담이행인지 확인한 뒤 최소 지분율을 본다. 지방 5%·국가 10% 기준과 법정 예외를 적용한다.",
    "v22": "협상계약에서 설명회 참석자만 입찰·제안할 수 있게 하거나 불참자를 배제하는지 본다. 단순 개최·권장·불참 가능은 위반이 아니며 비협상계약은 0이다.",
    "v23": "지방계약 협상계약에서 입찰 전 설명회와 제안서 마감 사이의 날짜를 계산한다. 공고일부터 설명회까지 7일, 설명회부터 마감까지는 일반 10·20·40일, 긴급 7일 기준을 확인한다. 날짜가 불명확하면 제출 기본값 0을 쓴다. 개찰일은 마감일로 보지 않는다.",
    "v24": "문서와 메타에서 추정가격, 계약방법, 업체 지역제한, 참가자격 업종 등 같은 의미 필드를 직접 비교한다. 부가세 포함 예산과 추정가격, 장소와 업체 소재지, null과 N은 의미를 확인한 뒤 비교한다.",
}


def build_system_prompt(tbl: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for v in ITEMS:
        it = tbl[v]
        tag = "  [근거 없음 — null]" if it["부재탐지"] else ""
        note = f" ({it['비고']})" if it.get("비고") else ""
        lines.append(f"- {v}: {it['항목명']}{note}{tag}\n  판정 카드: {ITEM_CARDS[v]}")
    return SYSTEM_HEAD + "\n" + "\n".join(lines) + "\n" + SYSTEM_TAIL


def build_user_prompt(rec: Dict[str, Any], max_chars: int) -> str:
    catalog_note = rec.get("_competition_note", "")
    amount_note = rec.get("_amount_note", "")
    law_note = rec.get("_law_note", "")
    rag_note = rec.get("_rag_note", "")
    comparison_note = rec.get("_v24_comparison_note", "")
    return (
        f"[공고 ID] {rec['id']}\n\n"
        f"[나라장터 입력 메타]\n{format_meta(rec)}\n\n"
        f"{amount_note}"
        f"{law_note}"
        f"{rag_note}"
        f"{catalog_note}"
        f"{comparison_note}"
        f"[문서]\n{build_context(rec, max_chars=max_chars)}\n"
    )


def build_messages(rec: Dict[str, Any], system_prompt: str, max_chars: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(rec, max_chars)},
    ]


# ===== 5. 모델 러너 (vLLM offline / mock) =====
class VLLMRunner:
    """평가 서버의 모델을 vLLM offline API로 실행합니다."""

    def __init__(self, schema: Dict[str, Any], model_dir: str = MODEL_DIR, quant: Optional[str] = QUANT,
                 max_tokens: int = MAX_TOKENS, seed: int = SEED, gpu_mem: float = 0.92, tp: int = 1):
        t0 = time.time()
        import vllm                                    # --mock 실행 시 vllm이 없어도 되도록 지연 import
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        log(f"vllm {vllm.__version__} · 모델 {model_dir} · quant={quant} · max_model_len={MAX_MODEL_LEN}")
        kw = dict(model=model_dir, tokenizer=model_dir, max_model_len=MAX_MODEL_LEN,
                  gpu_memory_utilization=gpu_mem, seed=seed, tensor_parallel_size=tp, dtype="auto")
        if quant:
            kw["quantization"] = quant
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, seed=seed,
            structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True),
        )
        self.load_seconds = time.time() - t0

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys") and "input_ids" in ids:   # transformers 버전에 따라 dict가 반환되는 경우
                ids = ids["input_ids"]
            return len(ids)
        except Exception:
            return len(self.tok.encode("\n".join(m["content"] for m in messages)))

    def chat(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        outs = self.llm.chat(batch, sampling_params=self.sp, use_tqdm=False)
        return [o.outputs[0].text if o.outputs else "" for o in outs]


class MockRunner:
    """모델 없이 입력·출력 및 제출 형식을 확인합니다."""
    load_seconds = 0.0

    def __init__(self, schema: Dict[str, Any], **_):
        pass

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        return sum(len(m["content"]) for m in messages) // 2     # Mock 실행용 간이 추정치

    def _one(self, _messages: List[Dict[str, str]]) -> str:
        out = {v: {"위반여부": 0, "근거문구": None} for v in ITEMS}
        return json.dumps(out, ensure_ascii=False)

    def chat(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        return [self._one(m) for m in batch]


def fit_to_budget(rec: Dict[str, Any], system_prompt: str, runner, max_chars: int,
                  budget: int = PROMPT_BUDGET) -> Tuple[List[Dict[str, str]], int, int]:
    """설정된 토큰 예산에 맞게 문서 글자 수를 조정합니다."""
    while True:
        msgs = build_messages(rec, system_prompt, max_chars)
        n = runner.count_tokens(msgs)
        if n <= budget or max_chars <= 2000:
            return msgs, n, max_chars
        max_chars = int(max_chars * min(0.85, budget / n * 0.95))


def run_chunk(runner, batch: List[List[Dict[str, str]]]) -> List[str]:
    """배치 실패 시 건별로 재시도하고, 처리하지 못한 건은 빈 출력으로 반환합니다."""
    try:
        return runner.chat(batch)
    except Exception as e:
        log(f"  ! 청크({len(batch)}건) 실패 → 건 단위 재시도: {type(e).__name__}: {str(e)[:160]}")
    outs = []
    for m in batch:
        try:
            outs.append(runner.chat([m])[0])
        except Exception as e:
            log(f"  ! 건 단위 실패 → 빈 출력: {type(e).__name__}: {str(e)[:160]}")
            outs.append("")
    return outs


# ===== 6. 파싱·후처리 =====
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None
    for cand in (text, *(m.group(1) for m in FENCE.finditer(text))):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_judgment(text: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """모델 출력을 24항목 판정으로 정리합니다. 빠진 항목은 0/None으로 채우고 결손 목록을 함께 반환합니다."""
    obj = extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("판정"), dict):
        obj = obj["판정"]
    out, missing = {}, []
    for v in ITEMS:
        raw = obj.get(v) if isinstance(obj, dict) else None
        if not isinstance(raw, dict):
            missing.append(v)
            out[v] = {"위반여부": 0, "근거문구": None}
            continue
        hit = raw.get("위반여부", raw.get("violation", 0))
        if isinstance(hit, bool):
            hit = int(hit)
        if isinstance(hit, str):
            hit = 1 if hit.strip() in ("1", "위반", "true", "True") else 0
        if hit not in (0, 1):
            hit = 1 if hit else 0
        ev = raw.get("근거문구", raw.get("evidence"))
        if ev is not None and not isinstance(ev, str):
            ev = str(ev)
        out[v] = {"위반여부": int(hit), "근거문구": ev}
    return out, missing


def clean_evidence(ev: Optional[str], src: str) -> str:
    """근거문구 셀 규약: NFC · 앞뒤 공백 제거 · 500자 상한 · 수식 접두(=,+,@)면 빈칸 ·
    원문 부분문자열이 아니면 빈칸(원문에 없는 근거는 채점에서 인정되지 않습니다)."""
    if not ev:
        return ""
    ev = unicodedata.normalize("NFC", ev).replace("\r", "").strip()
    if not ev or ev[0] in "=+@":
        return ""
    ev = ev[:EVIDENCE_MAX]
    return ev if ev in src else ""


def load_competition_catalog(data_dir: str) -> Dict[str, Dict[str, str]]:
    """제공된 중기간 경쟁제품 지정표에서 품명과 적용 예외를 읽습니다."""
    wanted = "중기부고시_경쟁제품_세부품명.csv"
    for base, _, names in os.walk(data_dir):
        for name in names:
            if unicodedata.normalize("NFC", name) == wanted:
                with io.open(os.path.join(base, name), encoding="utf-8-sig", newline="") as f:
                    return {row["세부품명번호"].strip(): row for row in csv.DictReader(f)
                            if row.get("세부품명번호", "").strip()}
    log("[주의] 중기간 경쟁제품 세부품명표를 찾지 못했습니다. v13 품목 필터를 생략합니다.")
    return {}


TEN_DIGITS = re.compile(r"(?<!\d)\d{10}(?!\d)")
CATALOG_NAME_SPACE = re.compile(r"[\s·,()]+")
PRICE_LIMIT = re.compile(r"추정가격\s*(\d+(?:\.\d+)?)\s*억원\s*미만")


def catalog_name_candidates(rec: Dict[str, Any], catalog: Dict[str, Dict[str, str]]) -> List[Tuple[str, str]]:
    """번호가 없을 때 구매 대상의 명칭·과업으로 지정표 후보를 찾는다.

    입찰자격에 적힌 증명서 품명은 구매 대상과 다를 수 있으므로, 공고 첫머리와
    과업·제안 문서의 실제 업무 설명만 검색한다. 반환값은 확정 분류가 아닌 후보이다.
    """
    notices = "\n".join(d["text"] for d in rec["docs"] if d["type"] == "공고문")
    notice = notices[:1800]
    task = "\n".join(d["text"] for d in rec["docs"] if d["type"] in ("과업지시서", "제안요청서", "규격서"))
    target = notice + "\n" + task
    compact = CATALOG_NAME_SPACE.sub("", target)
    found: Dict[str, str] = {}
    for code, row in catalog.items():
        name = row.get("세부품명", "").strip()
        key = CATALOG_NAME_SPACE.sub("", name)
        if len(key) >= 6 and key in compact:
            found[code] = f"세부품명 '{name}' 명시"

    # 행사 대행 용역은 공고에서 지정표의 정식 명칭보다 과업 표현으로 나타나는 경우가 많다.
    # '행사'가 부수 업무에만 나오는 물품 입찰을 막기 위해 용역 + 실제 과업 표현을 함께 확인한다.
    business = str(rec.get("meta", {}).get("업무구분") or "")
    event_target = re.search(r"공연|버스커|축제|전시회|성과공유회|행사", notices)
    # Some notices include a precise event-agency title but no separate task attachment.
    event_work = re.search(
        r"행사\s*기획|행사\s*대행|행사\s*운영|공연.{0,25}운영\s*대행|거리\s*공연.{0,25}운영",
        task or notices,
    )
    if "용역" in business and event_target and event_work and "8014199001" in catalog:
        found.setdefault("8014199001", f"과업 표현 '{event_work.group(0)[:45]}'")
    return sorted(found.items(), key=lambda item: (0 if item[1].startswith("세부품명") else 1, item[0]))[:6]


def competition_note(rec: Dict[str, Any], catalog: Dict[str, Dict[str, str]]) -> str:
    """품목번호 또는 과업 명칭으로 지정표를 대조하고, 근거의 강도를 구분한다."""
    if not catalog:
        return ""
    meta_codes = set(TEN_DIGITS.findall(str(rec["meta"].get("세부품명번호목록") or "")))
    doc_codes = set(TEN_DIGITS.findall(full_text(rec)))
    codes = sorted(meta_codes | (doc_codes & catalog.keys()))
    name_candidates = catalog_name_candidates(rec, catalog) if not codes else []
    if not codes and not name_candidates:
        return ""
    lines = ["[제공된 중기간 경쟁제품 지정표 대조]"]
    for code in codes[:8]:
        row = catalog.get(code)
        if row is None:
            lines.append(f"- {code}: 지정표에 없음")
        else:
            name = row.get("세부품명", "").strip()
            special = row.get("특이사항", "").strip()
            detail = f"; 특이사항: {special[:250]}" if special else ""
            lines.append(f"- {code}: {name} 지정{detail}")
    if len(codes) > 8:
        lines.append(f"- 그 밖의 품목코드 {len(codes) - 8}개 생략")
    for code, reason in name_candidates:
        row = catalog[code]
        special = row.get("특이사항", "").strip()
        detail = f"; 특이사항: {special[:250]}" if special else ""
        limit = PRICE_LIMIT.search(special)
        price = rec.get("meta", {}).get("입찰추정가격")
        if limit and isinstance(price, (int, float)) and not isinstance(price, bool):
            ceiling = float(limit.group(1)) * 100_000_000
            detail += f"; 금액 조건 {'충족' if price < ceiling else '미충족'} (입찰추정가격 {price:,.0f}원)"
        lines.append(f"- 명칭·과업 후보 {code}: {row.get('세부품명', '').strip()} ({reason}){detail}")
    lines.append("명칭·과업 일치는 후보일 뿐 확정 분류가 아니다. 실제 구매 대상과 지정표의 세부품명이 같은지, 특이사항의 조건을 충족하는지 확인한 뒤 판정한다. 참가자격에 적힌 증명서 품명만으로 구매 대상을 정하지 않는다.\n")
    return "\n".join(lines) + "\n"


def local_general_category(rec: Dict[str, Any]) -> str:
    """메타와 공고 첫머리의 익명화 기관 토큰에서 확인되는 발주기관 유형."""
    meta = rec.get("meta", {})
    issuer = " ".join(d["text"][:250] for d in rec["docs"] if d["type"] == "공고문")[:500]
    match = re.search(r"\[(?:수요기관|기관)\(([^)]{1,50})\)\]", issuer)
    agency = str(meta.get("소관구분") or "") + " " + (match.group(1) if match else "")
    if re.search(r"세종특별자치시|기초자치단체|시[·/]군[·/]구", agency):
        return "basic"
    if re.search(r"교육청|학교|교육기관|지방공기업", agency):
        return "education_or_corporation"
    if "광역자치단체" in agency:
        return "metro"
    return "unknown"


def local_general_amount(rec: Dict[str, Any]) -> Optional[int]:
    category = local_general_category(rec)
    if category in ("basic", "education_or_corporation"):
        return LOCAL_OTHER_AMOUNT
    if category == "metro":
        return LOCAL_METRO_AMOUNT
    return None


def notice_amount_note(rec: Dict[str, Any]) -> str:
    """대회가 제공한 고시금액을 기관 유형과 함께 프롬프트에 명시한다."""
    meta = rec.get("meta", {})
    law = str(meta.get("적용계약법") or "")
    if "지방" in law:
        category = local_general_category(rec)
        if category == "basic":
            general = f"{LOCAL_OTHER_AMOUNT:,}원 (세종시 또는 시·군·구)"
        elif category == "education_or_corporation":
            general = f"{LOCAL_OTHER_AMOUNT:,}원 (교육청·학교 또는 지방공기업)"
        elif category == "metro":
            general = f"{LOCAL_METRO_AMOUNT:,}원 (시·도, 세종시 제외)"
        else:
            general = (f"기관 유형에 따라 {LOCAL_METRO_AMOUNT:,}원(시·도, 세종시 제외) 또는 "
                       f"{LOCAL_OTHER_AMOUNT:,}원(교육청·학교·지방공기업·세종시·시군구); "
                       "익명화된 기관의 유형이 불명확하면 단정하지 말 것")
        return (
            "[대회 제공 고시금액: 지방계약 v5·v6·v7]\n"
            f"- 이 기관의 일반 물품·용역 기준: {general}.\n"
            f"- 건설기술·설계/감리·엔지니어링기술 용역은 {LOCAL_TECHNICAL_AMOUNT:,}원, "
            f"안전점검·정밀안전진단은 {LOCAL_SAFETY_AMOUNT:,}원. 업무구분만으로 세부 용역을 확정하지 말 것.\n"
            "- 입찰추정가격(부가세 제외)과 비교한다. 단가계약은 총 추정가격을 확인한다. "
            "공고일별 과거 고시를 조회하지 않고 대회 제공 기준을 적용한다. "
            f"v2·v14~v16에 쓰는 국가 고시금액 {NATIONAL_NOTICE_AMOUNT:,}원과 혼동하지 말 것.\n\n"
        )
    if "국가" in law:
        return (
            "[대회 제공 국가 고시금액]\n"
            f"- 국가계약의 물품·용역 고시금액: {NATIONAL_NOTICE_AMOUNT:,}원. "
            "입찰추정가격(부가세 제외)과 비교하고, 단가계약은 총 추정가격을 확인한다.\n\n"
        )
    return ""


NO_JOINT = re.compile(
    r"(?:공동수급|공동계약|공동도급)(?:(?![.。\n]).){0,45}"
    r"(?:불허|허용하지\s*않|불가|금지)|"
    r"공동제안이\s*아닌\s*단독|단독으로\s*신청한\s*사업자"
)

MONEY_IN_TEXT = re.compile(
    r"(?<!\d)(\d[\d,]*(?:\.\d+)?)\s*"
    r"(억원|억\s*원|억|천만원|천만\s*원|천만|백만원|백만\s*원|백만|만원|만\s*원|만|천원|원)"
)


def money_values(text: str) -> List[int]:
    """본문의 숫자 금액을 원 단위로 바꾼다."""
    factors = {
        "억원": 100_000_000, "억 원": 100_000_000, "억": 100_000_000,
        "천만원": 10_000_000, "천만 원": 10_000_000, "천만": 10_000_000,
        "백만원": 1_000_000, "백만 원": 1_000_000, "백만": 1_000_000,
        "만원": 10_000, "만 원": 10_000, "만": 10_000,
        "천원": 1_000, "원": 1,
    }
    tokens = []
    for match in MONEY_IN_TEXT.finditer(text or ""):
        number, unit = match.groups()
        key = re.sub(r"\s+", "", unit)
        tokens.append((match.start(), match.end(), round(float(number.replace(",", "")) * factors[key]), factors[key]))

    # Korean amounts often split a single value into descending units, e.g. "1억 5천만원".
    values: List[int] = []
    index = 0
    while index < len(tokens):
        _, end, amount, factor = tokens[index]
        index += 1
        while index < len(tokens):
            start, next_end, next_amount, next_factor = tokens[index]
            gap = (text or "")[end:start]
            if gap.strip() or next_factor >= factor:
                break
            amount += next_amount
            factor = next_factor
            end = next_end
            index += 1
        values.append(amount)
    return values


PERFORMANCE_MONEY = re.compile(
    r"(?<![\d.])((?:\d{1,3}(?:,\d{3})+)|(?:\d+(?:\.\d+)?))"
    r"\s*(억|천만|백만|만)?\s*원")


def _performance_money_value(match: re.Match[str]) -> int:
    value = float(match.group(1).replace(",", ""))
    multiplier = {"억": 100_000_000, "천만": 10_000_000,
                  "백만": 1_000_000, "만": 10_000, None: 1}[match.group(2)]
    return int(value * multiplier)


def _performance_budget_violation(
        rec: Dict[str, Any], budget: Optional[int]) -> Tuple[bool, Optional[Tuple[str, int]]]:
    """참가요건의 최소 실적액이 배정예산 이상인 명시적 문구만 찾는다."""
    if budget is None or budget <= 0:
        return False, None
    strong_requirement = re.compile(
        r"참가자격|있는\s*업체|업체이어야|업체만|보유한\s*(?:자|업체)|"
        r"실적이\s*있어야|실적을\s*보유|실적증명.{0,60}가능한\s*업체", re.S)
    for doc in rec["docs"]:
        text = doc["text"]
        for match in PERFORMANCE_MONEY.finditer(text):
            if _performance_money_value(match) < budget:
                continue
            if not re.match(r"\s*(?:\([^\n)]{0,40}\)\s*)?이상", text[match.end():match.end() + 55]):
                continue
            neighborhood = text[max(0, match.start() - 100):match.end() + 100]
            if re.search(
                    r"자본금|매출액?|신용|보증금|보험금?|적격심사|실적평가|정량평가|"
                    r"평가기준|평가항목|배점|점수", neighborhood):
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


def _mandatory_briefing(rec: Dict[str, Any]) -> Tuple[bool, Optional[Tuple[str, int]]]:
    """협상계약에서 설명회 불참자를 배제한다는 문구만 찾는다."""
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
    """날짜가 분명한 지방 협상계약만 v23 기간을 계산한다."""
    meta = rec.get("meta", {})
    if "지방" not in str(meta.get("적용계약법") or "") or "협상" not in str(meta.get("낙찰방법") or ""):
        return None, None, "비적용"
    if price is None:
        return None, None, "추정가격 미기재"
    try:
        posted = datetime.strptime(str(meta.get("공고게시일자") or ""), "%Y%m%d").date()
    except ValueError:
        return None, None, "게시일 파싱 실패"

    briefing_candidates: List[Tuple[date, str, int]] = []
    notice_docs = [doc for doc in rec["docs"] if "공고" in str(doc.get("type") or "")]
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
            following = [(day, pos) for day, pos in candidates if pos >= event.end()]
            selected = min(following or candidates, key=lambda pair: pair[1])[0]
            if selected >= posted:
                briefing_candidates.append((selected, doc["doc_id"], offset + event.start()))
    if not briefing_candidates:
        return None, None, "설명회 날짜 불확실"
    if len({row[0] for row in briefing_candidates}) > 1:
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


def _evidence_at_location(rec: Dict[str, Any], location: Optional[Tuple[str, int]]) -> str:
    """규칙 판정 위치 주변에서 원문 그대로의 근거를 잘라낸다."""
    if not location:
        return ""
    doc_id, position = location
    for doc in rec.get("docs", []):
        if doc.get("doc_id") != doc_id:
            continue
        text = doc.get("text") or ""
        start = max(0, position - 200)
        end = min(len(text), position + 300)
        return text[start:end].strip()[:EVIDENCE_MAX]
    return ""


def minimum_share_clauses(text: str) -> List[Tuple[float, str]]:
    """최소 지분율을 직접 정한 원문 구절과 퍼센트를 찾는다."""
    results: List[Tuple[float, str]] = []
    for line in (text or "").splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        for match in re.finditer(
            r"(?:최소.{0,30}(?:지분율|지분|참여비율)|"
            r"(?:지분율|지분|참여비율).{0,20}최소)", normalized
        ):
            fragment = normalized[max(0, match.start() - 12):match.end() + 100]
            for pct in re.finditer(r"(\d+(?:\.\d+)?)\s*%", fragment):
                results.append((float(pct.group(1)), normalized[:1000]))
                break
    return results


def v17_allows_medium(evidence: str) -> bool:
    """True only for explicit medium/SME membership language, not statute titles."""
    return bool(re.search(
        r"중기업(?:자)?|중[·ㆍ･-]\s*소기업|"
        r"중소기업(?:자(?!간)|\s*(?:또는|및|으로서|만|확인서))",
        evidence or "",
    ))


def v19_evidence_contexts(rec: Dict[str, Any], evidence: str) -> List[str]:
    """Return source windows around each exact evidence occurrence."""
    needle = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", evidence or "")).strip()
    if not needle:
        return []
    contexts: List[str] = []
    for doc in rec.get("docs", []):
        text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(doc.get("text") or "")))
        start = 0
        while len(contexts) < 12:
            pos = text.find(needle, start)
            if pos < 0:
                break
            contexts.append(text[max(0, pos - 500):min(len(text), pos + len(needle) + 500)])
            start = pos + max(1, len(needle))
        if len(contexts) >= 12:
            break
    if re.search(r"첨부\s*3", needle):
        for doc in rec.get("docs", []):
            text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(doc.get("text") or "")))
            form = re.search(r"장비별\s*공급\s*(?:및|·)\s*기술지원\s*확약서", text)
            if form:
                contexts.append(text[max(0, form.start() - 180):min(len(text), form.end() + 420)])
    return contexts or [needle]


def v19_explicit_prebid_obligation(text: str) -> bool:
    """A deadline to hold/submit the letter before bidding takes precedence."""
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or ""))
    letter = r"(?:물품\s*공급|공급|기술지원|제조사.{0,12}확약|확약)\s*(?:및\s*기술지원\s*)?확약서?"
    patterns = (
        rf"(?:전자)?입찰서?\s*(?:제출\s*)?마감(?:일시|일)?\s*(?:전|까지).{{0,90}}{letter}.{{0,60}}(?:보유|발급|제출)",
        rf"{letter}.{{0,100}}(?:입찰|전자입찰서)\s*(?:제출\s*)?마감(?:일시|일)?\s*(?:전|까지).{{0,50}}(?:보유|발급|제출)",
        rf"입찰\s*참가자는.{{0,100}}{letter}.{{0,70}}(?:보유|발급|제출)",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def v19_non_target_document(evidence: str, context: str) -> bool:
    """Exclude evidence types that the v19 labels distinguish from supplier pledges."""
    evidence = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", evidence or ""))
    context = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", context or ""))
    explicit_pledge = bool(re.search(
        r"물품\s*공급.{0,20}(?:기술지원\s*)?확약서|기술지원\s*확약서|"
        r"제조사.{0,30}공급.{0,15}확약서|공급자.{0,20}공급.{0,15}확약서",
        evidence,
    ))
    if re.search(r"(?:납품\s*및\s*)?사후\s*A\s*/?\s*S\s*확약서", evidence, re.I) and not explicit_pledge:
        return True
    if re.search(r"(?:제조자증명서|공급자증명서|판매대리점\s*계약서)", evidence) and not explicit_pledge:
        return True
    if re.search(r"정품\s*공급\s*확약서", evidence) and not explicit_pledge:
        return True
    self_performance_form = (
        re.search(r"장비별\s*공급\s*(?:및|·)\s*기술지원\s*확약서", context)
        and re.search(
            r"당사는.{0,100}(?:의료원|발주기관).{0,100}공급\s*(?:및|·)\s*기술지원.{0,60}제공",
            context,
        )
        and not re.search(r"(?:제조사|제조업체|공급사|공급업체)로부터", context)
    )
    if self_performance_form:
        return True
    quantity_capacity = re.search(
        r"(?:공급물량|공고물량|예정수량)\s*이상.{0,35}공급\s*확약서|"
        r"공급\s*확약서.{0,35}(?:공급물량|공고물량|예정수량)\s*이상",
        evidence,
    )
    if (quantity_capacity and "의료기기" in context and "판매업자" in context
            and not re.search(r"기술지원\s*확약서", evidence)):
        return True
    return False


def after_bid_supply_requirement(evidence: str, context: str = "") -> bool:
    """Detect post-selection/contract-stage requirements without erasing pre-bid possession duties."""
    evidence = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", evidence or "")).strip()
    context = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", context or ""))
    pos = context.find(evidence) if evidence else -1
    if pos >= 0:
        text = context[max(0, pos - 400):min(len(context), pos + len(evidence) + 400)]
    else:
        text = f"{evidence} {context[:800]}"
    if not text or v19_explicit_prebid_obligation(text):
        return False
    provisional_winner_review = re.search(r"개찰\s*후.{0,160}낙찰예정\s*업체", text)
    if provisional_winner_review:
        return True
    later = re.search(
        r"계약\s*시|계약체결\s*(?:시|후|이후)|낙찰\s*후|계약상대자\s*결정\s*후|"
        r"낙찰일.{0,35}이내.{0,100}(?:확약서|제출서류|서류를\s*제출)|"
        r"계약\s*전\s*제출.{0,100}(?:확인|인정).{0,50}후\s*계약",
        text,
    )
    return bool(later)


def apply_guards(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any],
                 competition_catalog: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """적용 조건이 명백히 성립하지 않는 예측만 0으로 교정한다."""
    out = {v: dict(cell) for v, cell in judgment.items()}
    meta = rec.get("meta", {})
    price = meta.get("입찰추정가격")
    law = str(meta.get("적용계약법") or "")
    business = str(meta.get("업무구분") or "")
    full = full_text(rec)
    budget = meta.get("배정예산금액")
    award = str(meta.get("낙찰방법") or "")
    contract_method = str(meta.get("계약방법") or "")
    local_small_quote = (
        "지방" in law and isinstance(price, (int, float)) and price <= 100_000_000
        and ("수의" in contract_method or "소액수의" in award)
    )
    if (isinstance(price, (int, float)) and price >= NATIONAL_NOTICE_AMOUNT
            and ("물품" in business or "용역" in business)):
        # 운영진 보완 공지: v2의 고시금액은 지방 지역제한액이 아닌 2억 3천만원.
        out["v2"]["위반여부"] = 0
    if local_small_quote:
        out["v2"]["위반여부"] = 0
    elif (isinstance(price, (int, float)) and price < NATIONAL_NOTICE_AMOUNT
          and isinstance(budget, (int, float))):
        performance_violation, performance_location = _performance_budget_violation(rec, int(budget))
        if performance_violation:
            out["v2"]["위반여부"] = 1
            current_evidence = str(out["v2"].get("근거문구") or "")
            if not current_evidence or current_evidence not in full:
                out["v2"]["근거문구"] = _evidence_at_location(rec, performance_location)
    if isinstance(price, (int, float)) and 1_000_000 <= price:
        # 단가계약은 메타의 작은 금액이 단가일 수 있고, 기술용역에는 별도
        # 기준액이 있으므로 두 경우는 금액만으로 v5를 교정하지 않는다.
        doc_text = "\n".join(d["text"] for d in rec["docs"])
        uncertain = re.search(
            r"단가\s*(?:계약|입찰)|(?:계약|입찰)\s*단가|안전점검|정밀안전진단|"
            r"건설기술|엔지니어링|설계|감리", doc_text
        )
        if not uncertain:
            if "국가" in law and price < NATIONAL_NOTICE_AMOUNT:
                out["v5"]["위반여부"] = 0
            elif "지방" in law:
                amount = local_general_amount(rec)
                if amount is not None and price < amount:
                    out["v5"]["위반여부"] = 0
                elif amount is None and price < LOCAL_METRO_AMOUNT:
                    out["v5"]["위반여부"] = 0
    if isinstance(price, (int, float)) and price >= 100_000_000:
        out["v17"]["위반여부"] = 0  # v17은 1억원 미만에만 적용

    # v3 compares the required experience threshold against the allocated project budget.
    budget = meta.get("배정예산금액")
    v3_evidence = str(out["v3"].get("근거문구") or "")
    if (out["v3"]["위반여부"] == 1 and isinstance(budget, (int, float))
            and v3_evidence and v3_evidence in full):
        amounts = money_values(v3_evidence)
        if amounts and max(amounts) < budget:
            out["v3"]["위반여부"] = 0

    # v17 is violated by an explicit SME/mid-business scope, not a small-only scope.
    v17_evidence = str(out["v17"].get("근거문구") or "")
    if (out["v17"]["위반여부"] == 1 and isinstance(price, (int, float))
            and price < 100_000_000 and v17_evidence and v17_evidence in full
            and not v17_allows_medium(v17_evidence)):
        out["v17"]["위반여부"] = 0

    # A designated item (or an explicit catalogue-name candidate) is required for v13.
    competition_codes = set(competition_catalog)
    codes = set(TEN_DIGITS.findall(str(meta.get("세부품명번호목록") or "")))
    doc_codes = set(TEN_DIGITS.findall(full))
    matched_codes = (codes | doc_codes) & competition_codes
    candidate_codes = {code for code, _ in catalog_name_candidates(rec, competition_catalog)} \
        if competition_catalog and not matched_codes else set()
    applicable_codes = set(matched_codes)
    for code in list(applicable_codes | candidate_codes):
        row = competition_catalog.get(code, {})
        limit = PRICE_LIMIT.search(str(row.get("특이사항") or ""))
        if limit and isinstance(price, (int, float)) and price >= float(limit.group(1)) * 100_000_000:
            applicable_codes.discard(code)
            candidate_codes.discard(code)
    if out["v13"]["위반여부"] == 1 and competition_catalog and not (applicable_codes | candidate_codes):
        out["v13"]["위반여부"] = 0

    v19_evidence = str(out["v19"].get("근거문구") or "")
    if (out["v19"]["위반여부"] == 1 and v19_evidence and v19_evidence in full
            and (v19_contexts := v19_evidence_contexts(rec, v19_evidence))):
        if v19_non_target_document(v19_evidence, " ".join(v19_contexts)):
            out["v19"]["위반여부"] = 0
        elif (not any(v19_explicit_prebid_obligation(context) for context in v19_contexts)
              and any(after_bid_supply_requirement(v19_evidence, context) for context in v19_contexts)):
            out["v19"]["위반여부"] = 0

    # v21 is a percentage comparison. The legal minimum differs by contract law.
    notice = full
    if NO_JOINT.search(notice):
        out["v21"]["위반여부"] = 0
    elif "국가" in law or "지방" in law:
        share_clauses = minimum_share_clauses(notice)
        if "분담이행" in str(meta.get("공동도급구성방식") or "") \
                and "공동이행" not in str(meta.get("공동도급구성방식") or ""):
            out["v21"]["위반여부"] = 0
        elif share_clauses:
            threshold = 10.0 if "국가" in law else 5.0
            below = [(pct, clause) for pct, clause in share_clauses if pct < threshold]
            if below:
                out["v21"]["위반여부"] = 1
                share_evidence = str(out["v21"].get("근거문구") or "")
                if not share_evidence or share_evidence not in full:
                    out["v21"]["근거문구"] = below[0][1]
            else:
                out["v21"]["위반여부"] = 0
        elif out["v21"]["위반여부"] == 1:
            share_evidence = str(out["v21"].get("근거문구") or "")
            if share_evidence and share_evidence in full and not re.search(r"\d+(?:\.\d+)?\s*%", share_evidence):
                out["v21"]["위반여부"] = 0

    # v22는 협상계약에서 불참자를 배제하는 명시 문구가 있을 때만 1이다.
    negotiation = "협상" in award
    if not negotiation:
        out["v22"]["위반여부"] = 0
    else:
        mandatory, briefing_location = _mandatory_briefing(rec)
        out["v22"]["위반여부"] = int(mandatory)
        if mandatory:
            current_evidence = str(out["v22"].get("근거문구") or "")
            if not current_evidence or current_evidence not in full:
                out["v22"]["근거문구"] = _evidence_at_location(rec, briefing_location)

    # v23은 지방 협상계약에서 설명회·제안서 마감일이 확인되어 기간이 부족할 때만 1이다.
    # 날짜가 불분명한 경우도 제출 규칙에 따라 0을 선택한다.
    if not negotiation or "지방" not in law:
        out["v23"]["위반여부"] = 0
    else:
        timing_violation, timing_location, _timing_detail = _briefing_timing_violation(
            rec, int(price) if isinstance(price, (int, float)) else None)
        out["v23"]["위반여부"] = int(timing_violation is True)
        if timing_violation is True:
            current_evidence = str(out["v23"].get("근거문구") or "")
            if not current_evidence or current_evidence not in full:
                out["v23"]["근거문구"] = _evidence_at_location(rec, timing_location)

    # v24는 같은 의미 필드가 문서와 메타데이터에서 명시적으로 다를 때만 자동 보정한다.
    if (v24_evidence := v24_explicit_mismatch_evidence(rec)):
        out["v24"]["위반여부"] = 1
        out["v24"]["근거문구"] = v24_evidence
    return out


def postprocess(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """부재·비위반 근거 빈칸 고정 및 근거문구 원문 대조(NFC)."""
    src = unicodedata.normalize("NFC", full_text(rec))
    out = {}
    for v in ITEMS:
        cell = dict(judgment.get(v, {"위반여부": 0, "근거문구": None}))
        hit = 1 if cell.get("위반여부") == 1 else 0
        ev = "" if (hit == 0 or v in ABSENCE) else clean_evidence(cell.get("근거문구"), src)
        out[v] = {"위반여부": hit, "근거문구": ev}
    return out


def to_row(rec_id: str, judgment: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row = {"id": rec_id}
    for i, v in enumerate(ITEMS, 1):
        row[v] = judgment[v]["위반여부"]
        row[f"e{i}"] = judgment[v]["근거문구"]
    return row


def empty_row(rec_id: str) -> Dict[str, Any]:
    return to_row(rec_id, {v: {"위반여부": 0, "근거문구": ""} for v in ITEMS})


# ===== 7. submission.csv 저장·자가검증 =====
def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as f:   # UTF-8(BOM 없음) · RFC4180 quoting
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: unicodedata.normalize("NFC", str(r[k])) for k in COLUMNS})


def validate_csv(path: str, expected_ids: List[str]) -> List[str]:
    """자가검증: 열 49 · 행 수 = 입력 건수 · id 유일·일치 · v 0/1 · e 500자 이하 · 부재탐지 e 빈칸"""
    errs: List[str] = []
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        rows = list(rd)
    if header != COLUMNS:
        errs.append(f"헤더 불일치: {len(header or [])}열 (기대 {len(COLUMNS)})")
        return errs
    if len(rows) != len(expected_ids):
        errs.append(f"행 수 {len(rows)} ≠ 입력 {len(expected_ids)}")
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        errs.append("id 중복")
    if set(ids) != set(expected_ids):
        errs.append(f"id 집합 불일치 (누락 {len(set(expected_ids) - set(ids))})")
    absence_idx = {COLUMNS.index("e" + v[1:]) for v in ABSENCE}
    for r in rows:
        if len(r) != len(COLUMNS):
            errs.append(f"{r[0]}: 열 수 {len(r)}")
            continue
        if any(x not in ("0", "1") for x in r[1:25]):
            errs.append(f"{r[0]}: 위반여부에 0/1 아닌 값")
        if any(len(x) > EVIDENCE_MAX for x in r[25:]):
            errs.append(f"{r[0]}: 근거문구 {EVIDENCE_MAX}자 초과")
        if any(r[j] for j in absence_idx):
            errs.append(f"{r[0]}: 부재탐지 항목에 근거문구")
        if any(x.startswith(("=", "+", "@")) for x in r[25:]):
            errs.append(f"{r[0]}: 수식 접두 근거문구")
    return errs


# ===== 8. 실행 =====
def run(input_path: str, out_path: str, runner_cls, limit: Optional[int], chunk: int,
        max_chars: int, data_dir: str, **runner_kw) -> Dict[str, Any]:
    t_all = time.time()
    recs = list(iter_records(input_path, limit=limit))
    log(f"입력 {len(recs)}건 ← {input_path}")
    if not recs:
        write_csv([], out_path)
        return {"건수": 0}

    tbl, schema = item_table(data_dir), decode_schema(data_dir)
    law_notes = load_law_notes(tbl, data_dir)
    law_rag_index = build_law_rag_index(data_dir)
    log(f"대회 법령 검색 인덱스 {len(law_rag_index.passages)}조각")
    competition_catalog = load_competition_catalog(data_dir)
    for rec in recs:
        rec["_competition_note"] = competition_note(rec, competition_catalog)
        rec["_amount_note"] = notice_amount_note(rec)
        rec["_v24_comparison_note"] = v24_comparison_note(rec)
        law = str(rec["meta"].get("적용계약법") or "")
        rec["_law_note"] = law_notes.get(law, "")
        if "물품" in str(rec["meta"].get("업무구분") or ""):
            rec["_law_note"] += law_notes.get("물품", "")
        if software_scope_candidate(rec):
            rec["_law_note"] += law_notes.get("소프트웨어", "")
            rec["_law_note"] += (
                "[SW 적용 판단 유의]\n"
                "메타의 정보화사업여부가 미입력이어도 과업·제안요청서에 실제 SW 사업이 있을 수 있다. "
                "반대로 SW 업종코드나 일반적인 법령 인용만으로는 v20 적용을 확정하지 않는다. "
                "사업의 실제 범위, 사업금액, 입찰참가자격의 참여제한 문구를 확인한다.\n\n"
            )
        rec["_rag_note"] = (law_rag_note(rec, law_rag_index, rec["_law_note"])
                            + document_rag_note(rec))
    system_prompt = build_system_prompt(tbl)
    runner = runner_cls(schema, **runner_kw)
    log(f"모델 로드 {runner.load_seconds:.1f}s")

    # 전건 메시지 구성(길이 예산 맞춤)
    msgs_all, shrunk, ntok = [], 0, []
    for rec in recs:
        m, n, mc = fit_to_budget(rec, system_prompt, runner, max_chars)
        msgs_all.append(m)
        ntok.append(n)
        shrunk += int(mc < max_chars)
    log(f"프롬프트 토큰 중앙값 {sorted(ntok)[len(ntok) // 2]:,} · 최대 {max(ntok):,} · 예산 축소 {shrunk}건")

    # 배치 추론
    t_inf = time.time()
    texts: List[str] = []
    for s in range(0, len(msgs_all), chunk):
        texts.extend(run_chunk(runner, msgs_all[s:s + chunk]))
        log(f"  {min(s + chunk, len(msgs_all))}/{len(msgs_all)}건 … {time.time() - t_inf:.0f}s")
    inf_seconds = time.time() - t_inf

    # 파싱·후처리 → 행
    rows, invalid, filled, ev_kept, ev_dropped = [], 0, 0, 0, 0
    guard_changes = Counter()
    for rec, text in zip(recs, texts):
        try:
            parsed, missing = parse_judgment(text)
            invalid += int(len(missing) == 24)
            filled += len(missing)
            before = sum(1 for v in ITEMS if parsed[v]["근거문구"] and parsed[v]["위반여부"] == 1 and v not in ABSENCE)
            guarded = apply_guards(parsed, rec, competition_catalog)
            guard_changes.update(v for v in ITEMS
                                 if guarded[v]["위반여부"] != parsed[v]["위반여부"])
            final = postprocess(guarded, rec)
            kept = sum(1 for v in ITEMS if final[v]["근거문구"])
            ev_kept += kept
            ev_dropped += max(0, before - kept)
            rows.append(to_row(rec["id"], final))
        except Exception as e:                           # 한 건의 실패가 전체 실행을 막지 않도록
            log(f"  ! {rec['id']} 후처리 실패 → 전항목 0: {type(e).__name__}: {e}")
            rows.append(empty_row(rec["id"]))
    assert len(rows) == len(recs)

    write_csv(rows, out_path)
    errs = validate_csv(out_path, [r["id"] for r in recs])
    report = {
        "건수": len(recs), "모델로드_s": round(runner.load_seconds, 1), "추론_s": round(inf_seconds, 1),
        "건당_s": round(inf_seconds / len(recs), 2), "전체_s": round(time.time() - t_all, 1),
        "유효JSON": len(recs) - invalid, "메운_항목수": filled,
        "근거_유지": ev_kept, "근거_원문불일치_폐기": ev_dropped,
        "규칙_교정": dict(guard_changes),
        "출력": out_path, "자가검증": "PASS" if not errs else errs,
    }
    log(json.dumps(report, ensure_ascii=False))
    if invalid:
        log("[주의] 유효 JSON이 아닌 출력이 있습니다. 구조화 출력 설정과 JSON Schema를 확인하세요.")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="24개 항목의 법령 위반 여부 판정 베이스라인")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--input", default=None, help="기본 = <data-dir>/test.jsonl.gz")
    ap.add_argument("--model-dir", default=MODEL_DIR, help="로컬에서는 HF ID(google/gemma-4-26B-A4B-it)도 가능")
    ap.add_argument("--quantization", default=os.environ.get("PPS_QUANT", QUANT),
                    help="채점 서버 = int8_per_channel_weight_only · 'none'이면 미양자화")
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=128, help="LLM.chat 한 번에 넘길 건수")
    ap.add_argument("--max-chars", type=int, default=10000, help="문서 글자 수의 초기 상한(토큰 예산에 맞춰 자동 조정)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="모델 없이 흐름만 확인")
    a = ap.parse_args()

    input_path = a.input or os.path.join(a.data_dir, "test.jsonl.gz")
    out_path = os.path.join(a.output_dir, "submission.csv")
    quant = None if str(a.quantization).lower() in ("none", "") else a.quantization
    runner_kw = {} if a.mock else dict(model_dir=a.model_dir, quant=quant, max_tokens=a.max_tokens,
                                       seed=SEED, gpu_mem=a.gpu_mem, tp=a.tp)
    report = run(input_path, out_path, MockRunner if a.mock else VLLMRunner,
                 limit=a.limit, chunk=a.chunk, max_chars=a.max_chars, data_dir=a.data_dir, **runner_kw)
    return 0 if report.get("자가검증") in ("PASS", None) else 1


if __name__ == "__main__":
    sys.exit(main())
