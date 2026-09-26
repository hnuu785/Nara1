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
import os
import re
import sys
import time
import unicodedata
from collections import Counter
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
    (re.compile(r"직접\s*생산|세부\s*품명|중소기업|중기업|소기업|소상공인"), 8),
    (re.compile(r"공동\s*(?:수급|계약|이행|도급)|지분율|출자\s*비율"), 8),
    (re.compile(r"소프트웨어|정보\s*시스템|SW\s*사업|대기업|상호출자"), 8),
    (re.compile(r"설명회|현장\s*설명|제안서.*제출|입찰서.*제출"), 7),
    (re.compile(r"추정\s*가격|기초\s*금액|사업\s*금액|예산|부가가치세"), 6),
    (re.compile(r"본점|본사|주된\s*영업소|지역\s*제한|실적\s*제한"), 6),
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
MAX_MODEL_LEN = 16384                   # 베이스라인 모델 컨텍스트 길이
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
        score = sum(weight for pattern, weight in DOCUMENT_SIGNALS if pattern.search(raw))
        if score:
            # 긴 표/문단에서도 실제로 일치한 곳을 남깁니다.
            if len(raw) > 1000:
                hits = [m.start() for pattern, _ in DOCUMENT_SIGNALS for m in pattern.finditer(raw)]
                pos = min(hits) if hits else 0
                start = max(0, pos - 150)
                raw = raw[start:start + 900]
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
    """대회 항목표·법령패키지의 고정 스냅샷만 사용해 법령별 프롬프트를 만든다."""
    law_dir = os.path.join(data_dir, "법령패키지", "법령")
    files = {}
    if os.path.isdir(law_dir):
        for name in os.listdir(law_dir):
            if name.endswith(".txt"):
                files[unicodedata.normalize("NFC", name[:-4])] = os.path.join(law_dir, name)
    needed = {stem for specs in LAW_ARTICLES.values() for stem, _ in specs}
    sources = {stem: io.open(files[stem], encoding="utf-8").read() for stem in needed if stem in files}
    bundled_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "law_articles.json")
    bundled = {}
    if os.path.isfile(bundled_path):
        with io.open(bundled_path, encoding="utf-8") as f:
            bundled = json.load(f)
    notes = {}
    for law in ("지방계약법", "국가계약법"):
        lines = ["[대회 제공 항목표의 적용 조문]",
                 "아래 조문은 판정 기준이다. 출력 근거문구에는 반드시 공고 문서의 원문만 사용한다."]
        lines.extend(f"- {v}: {tbl[v][law]}" for v in ITEMS)
        notes[law] = "\n".join(lines) + "\n\n"
    for group, specs in LAW_ARTICLES.items():
        parts = []
        for stem, number in specs:
            excerpt = (_article_text(sources[stem], number) if stem in sources else
                       bundled.get(f"{stem}|{number}", ""))
            if excerpt:
                parts.append(f"[{stem} 제{number if '조' in number else number + '조'}]\n{excerpt}")
        notes[group] = (notes.get(group, "") +
                        ("[대회 제공 법령 원문 발췌]\n" + "\n\n".join(parts) + "\n\n" if parts else ""))
    if not sources and not bundled:
        log("[주의] 법령패키지 원문을 찾지 못해 항목표의 조문 매핑만 사용합니다.")
    return notes


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
10. [발췌 문서]와 [미제공 문서] 표시는 입력의 한계다. 보이지 않은 부분에
    어떤 조건이 반드시 없다고 단정하지 말고 제공된 자료로 판단한다.

판정할 24개 항목"""

SYSTEM_TAIL = """
출력은 JSON 하나로만 낸다. 키는 v1~v24, 각 값은 {"위반여부": 0 또는 1, "근거문구": 문자열 또는 null}이다.
설명이나 머리말을 덧붙이지 않는다."""


def build_system_prompt(tbl: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for v in ITEMS:
        it = tbl[v]
        tag = "  [근거 없음 — null]" if it["부재탐지"] else ""
        note = f" ({it['비고']})" if it.get("비고") else ""
        lines.append(f"- {v}: {it['항목명']}{note}{tag}")
    return SYSTEM_HEAD + "\n" + "\n".join(lines) + "\n" + SYSTEM_TAIL


def build_user_prompt(rec: Dict[str, Any], max_chars: int) -> str:
    catalog_note = rec.get("_competition_note", "")
    amount_note = rec.get("_amount_note", "")
    law_note = rec.get("_law_note", "")
    return (
        f"[공고 ID] {rec['id']}\n\n"
        f"[나라장터 입력 메타]\n{format_meta(rec)}\n\n"
        f"{amount_note}"
        f"{law_note}"
        f"{catalog_note}"
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
    notice = "\n".join(d["text"][:1800] for d in rec["docs"] if d["type"] == "공고문")
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
    title = " ".join(d["text"][:250] for d in rec["docs"])
    event_target = re.search(r"공연|버스커|축제|전시회|성과공유회|행사", title)
    event_work = re.search(r"행사\s*기획|행사\s*대행|행사\s*운영|공연.{0,25}운영\s*대행|거리\s*공연.{0,25}운영", task)
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
    r"(?:불허|허용하지\s*않|불가|금지)"
)


def apply_guards(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any],
                 competition_codes: set[str]) -> Dict[str, Dict[str, Any]]:
    """적용 조건이 명백히 성립하지 않는 예측만 0으로 교정한다."""
    out = {v: dict(cell) for v, cell in judgment.items()}
    meta = rec.get("meta", {})
    price = meta.get("입찰추정가격")
    law = str(meta.get("적용계약법") or "")
    business = str(meta.get("업무구분") or "")
    if (isinstance(price, (int, float)) and price >= NATIONAL_NOTICE_AMOUNT
            and ("물품" in business or "용역" in business)):
        # 운영진 보완 공지: v2의 고시금액은 지방 지역제한액이 아닌 2억 3천만원.
        out["v2"]["위반여부"] = 0
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

    codes = set(TEN_DIGITS.findall(str(meta.get("세부품명번호목록") or "")))
    doc_codes = set(TEN_DIGITS.findall("\n".join(d["text"] for d in rec["docs"])))
    if (competition_codes and codes and not codes.intersection(competition_codes)
            and not doc_codes.intersection(competition_codes)):
        out["v13"]["위반여부"] = 0  # 명시된 품목이 모두 비경쟁제품

    notice = "\n".join(d["text"] for d in rec["docs"] if d["type"] == "공고문")
    if not meta.get("공동도급구성방식") and NO_JOINT.search(notice):
        out["v21"]["위반여부"] = 0  # 공동수급 불허 시 구성원 지분율 미적용
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
    competition_catalog = load_competition_catalog(data_dir)
    competition_codes = set(competition_catalog)
    for rec in recs:
        rec["_competition_note"] = competition_note(rec, competition_catalog)
        rec["_amount_note"] = notice_amount_note(rec)
        law = str(rec["meta"].get("적용계약법") or "")
        rec["_law_note"] = law_notes.get(law, "")
        if "물품" in str(rec["meta"].get("업무구분") or ""):
            rec["_law_note"] += law_notes.get("물품", "")
        if str(rec["meta"].get("정보화사업여부") or "").upper() == "Y":
            rec["_law_note"] += law_notes.get("소프트웨어", "")
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
            guarded = apply_guards(parsed, rec, competition_codes)
            guard_changes.update(v for v in ITEMS
                                 if guarded[v]["위반여부"] != parsed[v]["위반여부"])
            final = postprocess(guarded, rec)
            kept = sum(1 for v in ITEMS if final[v]["근거문구"])
            ev_kept += kept
            ev_dropped += before - kept
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
