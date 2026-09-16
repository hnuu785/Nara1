# 법령 RAG 제출본

공식 예제 https://dacon.io/competitions/official/236754/codeshare/14155 의
조문 분할·BM25 검색·프롬프트 추가 방식을 `open/baseline/script.py`에 통합했습니다.

- 검색 자료: 평가 서버가 제공하는 `PPS_DATA_DIR/법령패키지/법령/*.txt`만 사용합니다.
- 실행 시 법령 색인을 한 번 생성하며, 평가 공고를 색인에 추가하지 않습니다.
- 공고문 앞 3,000자로 최대 8개 조각을 검색하고 총 8,000자 이내로 제공합니다.
- 기존 공고 문서 4,000자 설정과 제약 디코딩을 유지했습니다.
- 검색 법령을 포함하여 토큰을 계수합니다. 한도 초과 시 법령부터 줄이고,
  그래도 초과하면 공고 분량을 줄입니다. 최소 입력도 넘치면 명시적으로 실패합니다.
- 근거 문구 검증은 기존 공고·첨부 원문만 대상으로 합니다.
- `python script.py` 실행 시 RAG와 실제 모델 추론이 기본으로 켜집니다.
- 비교 실행은 `--no-rag`, 모델 없는 흐름 확인은 `--mock`을 사용합니다.

## 로컬 점검

로컬 Python 환경에 `rank-bm25==0.2.2`가 필요합니다. 평가 서버에는 기본 설치되어
있으므로 제출용 requirements.txt에는 추가하지 않았습니다.

```sh
python -m unittest discover -s tests -v
python open/baseline/script.py --mock --data-dir open/data --output-dir /tmp/nara-rag-sample
python open/baseline/script.py --mock --data-dir open/data --input open/dev.jsonl.gz --output-dir /tmp/nara-rag-dev
```

`--mock`은 실제 법령 검색과 프롬프트 생성·CSV 검증을 수행하되 모델 응답만 전부
정상(0)으로 대체합니다. 이 결과로 예측 성능이나 GPU 실행 시간을 판단할 수 없습니다.

## 제출

새 제출본은 `python3 tools/submissions.py save <실험이름> --note "변경 내용"`으로
만든 뒤 출력된 `submissions/<실험이름>/submit.zip`을 데이콘 제출 페이지에 업로드합니다.
버전별 코드·점수 기록 방법은 [제출 기록 안내](submissions/README.md)를 참고하세요.
압축파일 최상위에는 `script.py`와 `requirements.txt`만 포함합니다.
데이터와 모델은 평가 서버에서 제공되므로 압축파일에 포함하지 않습니다.
기존 루트 압축파일 두 개는 `submissions/legacy-before-rag/`,
`submissions/legacy-rag/`에도 원본 그대로 보관했습니다.

## 검증 결과 (2026-09-14)

- 검색 관련 자동 검사 5개 통과: 관련 조문 검색, 없는 법령 처리, 근거 출처 분리,
  법령 입력 축소, 처리 불가능한 토큰 예산 검출.
- 배포 법령 23개에서 검색 조각 2,541개 생성.
- dev 200건 모두 법령 검색 성공 및 모의 추론 CSV 49열 형식 검사 통과.
- 완성된 ZIP을 임시 폴더에 풀어 서버 경로 환경변수로 실행한 샘플 10건 검사 통과.
- `--no-rag` 모의 실행도 통과.
- ZIP 내부 파일과 작업 코드의 바이트 일치 및 압축 무결성 확인.
- 로컬 검증 환경: Python 3.14, rank-bm25 0.2.2, NumPy 2.5.3.
  평가 서버와 Python·NumPy 버전이 다르며, vLLM·GPU 실제 추론과 점수·2시간 제한은
  이 컴퓨터에서 검증하지 못했습니다. 데이콘 업로드는 아직 수행하지 않았습니다.
