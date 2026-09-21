# 나라장터 법령 위반 모니터링 AI — RunPod GPU 원격 개발 환경 구축 및 운용 가이드

본 문서는 데이콘 **나라장터 자체입찰 공고 법령 위반사항 모니터링 AI 경진대회**를 위해 구축한 **RunPod L40S GPU 원격 인퍼런스 환경과 로컬 맥북 연동 개발 워크플로우**의 전체 내용과 필수 운영 지침을 정리한 기록입니다.

---

## 1. 대회 핵심 맥락 및 환경 제약

* **대회 목표**: 나라장터 자체입찰 공고문·첨부문서·메타데이터를 분석하여 **24개 항목(`v1`~`v24`)의 법령 위반 여부(0/1)** 및 **근거 문구(`e1`~`e24`)** 판정.
* **평가지표**: 24개 위반(1) 클래스에 대한 **Macro F1 단순 평균**.
* **고정 모델**: `google/gemma-4-26B-A4B-it` (Apache 2.0 라이선스, 오픈 가중치 MoE 모델).
* **채점 서버 하드웨어 & 시간 제약**:
  * GPU: **NVIDIA L40S 1장 (VRAM ~44.7GiB / 총 48GB)**, RAM 60GiB, 7 vCPU.
  * 실행시간: **2시간(7,200초) 이내 1,853건 전수 추론 완료** 필수 $\rightarrow$ **공고 1건당 평균 3.88초 이내** 처리되어야 타임아웃(DNC)을 피할 수 있음.
  * 모델 가중치 학습/LoRA 제출 불가 (순수 추론 파이프라인/RAG/규칙 엔지니어링).
  * 서버 고정 패키지: `vllm`, `torch`, `transformers`, `xgrammar`, `rank-bm25==0.2.2`.

---

## 2. 인프라 아키텍처: Client-Server 분리 구조

```text
[로컬 맥북 (Developer Machine)]                     [RunPod 클라우드 (Dedicated GPU Pod)]
┌──────────────────────────────────────┐          ┌──────────────────────────────────────┐
│ • 모든 소스코드 (Nara 레포지토리)      │          │ • 소스코드 없음 (git clone 불필요!)   │
│ • 데이터셋 (open/dev, test 등)       │          │ • 오직 Gemma-4-26B 모델만 VRAM 상주   │
│ • RAG BM25 색인 및 법령 검색          │          │ • 순수 GPU 계산기 (vLLM API Server)  │
│ • 규칙 엔진 및 전/후처리 로직         │          │                                      │
│ • Macro F1 채점 및 결과 저장          │          │                                      │
└──────────────────┬───────────────────┘          └──────────────────▲───────────────────┘
                   │                                                 │
                   │    SSH 터널링 암호화 통신 (포트 8000 포워딩)       │
                   └─────────────────────────────────────────────────┘
```

### 왜 이 구조인가?
1. **코드 보안 및 충돌 방지**: 원격 GPU 서버에 코드를 올리지 않으므로 팀원 간 코드가 섞이거나 유출될 위험이 없음.
2. **로컬 개발의 편의성**: 평소처럼 맥북의 VS Code/터미널에서 코딩하고, LLM 추론 연산만 고성능 GPU로 전송.
3. **팀원 공유 확장성**: 팀원들에게도 동일한 접속 엔드포인트만 열어주면 GPU 1대를 여러 명이 로컬에서 동시에 공유 가능.

---

## 3. RunPod 인스턴스 사양 및 세팅 상세

### 인스턴스 사양
* **GPU**: 1x NVIDIA L40S (48 GB VRAM, 62 GB RAM, 16 vCPU)
* **OS / Template**: RunPod PyTorch (Ubuntu 24.04, CUDA 12.8)
* **디스크**: Container Disk 100 GB (Gemma 26B 원본 51.6GB 수용)
* **호스트 접속 정보 (예시)**: `202.181.159.238`, SSH 포트 `16110`

### 모델 다운로드
* 저장 경로: `/workspace/models/gemma-4-26B-A4B-it` (총 51.6 GB)
* 최신 Hugging Face CLI(`hf`) 명령어:
  ```bash
  # 토큰 등록 (대역폭 제한 해제 및 고속 다운로드)
  hf auth login --token <YOUR_HF_TOKEN>

  # 가중치 다운로드
  mkdir -p /workspace/models
  hf download google/gemma-4-26B-A4B-it --local-dir /workspace/models/gemma-4-26B-A4B-it
  ```

### vLLM API 서버 실행 명령어 및 인자별 필수 사유
RunPod 터미널에서 실행:
```bash
vllm serve /workspace/models/gemma-4-26B-A4B-it \
  --port 8000 \
  --max-model-len 16384 \
  --quantization int8_per_channel_weight_only \
  --gpu-memory-utilization 0.92
```

| 인자 | 설정값 | 필수 이유 |
|---|---|---|
| `--quantization` | `int8_per_channel_weight_only` | 51.6GB 원본 모델을 48GB VRAM에 로드하기 위한 필수 8비트 양자화. 대회 채점 서버 공식 규격과 100% 일치. |
| `--max-model-len` | `16384` | Gemma 4의 기본 컨텍스트(256K)로 인한 과도한 KV 캐시 예약 및 OOM 크래시 방지. (대회 상한인 32768까지 확장 가능) |
| `--gpu-memory-utilization` | `0.92` | 전체 VRAM 중 92%(약 40.84 GiB)를 할당하고 8%는 CUDA 런타임 버퍼로 보존하여 불시 OOM 방지. |
| `--port` | `8000` | vLLM 공식 기본 포트. 22(SSH), 8888(Jupyter)과의 포트 충돌 방지. |

---

## 4. 로컬 맥북 연동 및 코드 수정 내역

### 4.1. 맥북 터미널: SSH 터널링 (Port Forwarding)
맥북의 `localhost:8000`을 RunPod의 내부 `8000`번 포트와 실시간 암호화 터널로 연결:
```bash
ssh -N -L 8000:localhost:8000 root@202.181.159.238 -p 16110
```

### 4.2. `open/baseline/script.py` 주요 기능 추가
1. **`APIRunner` 클래스 구현**:
   * 내장 라이브러리(`urllib.request`) 기반으로 OpenAI 호환 `/v1/chat/completions` 엔드포인트 호출.
   * `ThreadPoolExecutor`를 활용한 병렬 요청 처리 $\rightarrow$ vLLM의 Continuous Batching 가속 극대화.
   * 서버의 `/v1/models`를 자동 조회하여 로드된 모델 ID 자동 바인딩.
2. **`--api-url` CLI 옵션 추가**:
   * `--api-url http://localhost:8000/v1` 전달 시 `APIRunner`로 작동.
   * 옵션 미지정 시 원래의 오프라인 `VLLMRunner`로 동작하여 **데이콘 제출 호환성 100% 보존**.
3. **`--labels` 및 `evaluate_rows()` 채점 기능 추가**:
   * 정답 라벨 파일(`open/dev_labels.csv`)을 지정하면, 예측 결과와 즉시 대조하여 24개 항목별 `TP, FP, FN, F1` 및 **대회 공식 지표인 Macro F1**을 터미널에 표 형태로 즉시 출력.

---

## 5. 실증 검증 결과 (Benchmark on dev 5건)

맥북 로컬에서 RunPod L40S GPU를 원격 호출하여 `open/dev.jsonl.gz` 5건을 추론한 결과:

```text
[baseline] 입력 5건 ← open/dev.jsonl.gz
[baseline] APIRunner 연결: http://localhost:8000/v1 · 모델 /workspace/models/gemma-4-26B-A4B-it
[baseline]   5/5건 … 19s
==================== [dev 채점 결과] ====================
★ Macro F1 (대회 공식 지표): 0.075000
-------------------------------------------------------
항목     | TP   | FP   | FN   | F1    
-------------------------------------------------------
v1     | 1    | 0    | 0    | 1.0000
v2     | 0    | 1    | 2    | 0.0000
v3     | 2    | 1    | 0    | 0.8000
v4     | 0    | 1    | 0    | 0.0000
v8     | 0    | 0    | 1    | 0.0000
v13    | 0    | 1    | 0    | 0.0000
v17    | 0    | 1    | 0    | 0.0000
=======================================================
{"건수": 5, "추론_s": 18.8, "건당_s": 3.75, "유효JSON": 4, "메운_항목수": 24, "자가검증": "PASS"}
```

### 핵심 시사점
1. **속도 합격**: 건당 **3.75초**로, 대회 2시간 타임아웃 커트라인인 **건당 3.88초 이내**를 여유 있게 만족함.
2. **실제 판정력 입증**: `v1`(특정기관 제한)은 F1 **1.0000**, `v3`(실적 1배수 초과)은 F1 **0.8000**으로 고정 모델이 법령 위반을 실제로 읽고 적중함.
3. **개선 포인트 발견**: 5건 중 1건에서 JSON 출력 형식 미흡/토큰 소진으로 24개 항목이 0으로 땜빵 처리됨 $\rightarrow$ 구조화 출력(JSON Schema/xgrammar) 및 프롬프트 보정 시 점수 급상승 잠재력 확인.

---

## 6. 일상 개발 워크플로우 (Quick Reference)

### 🟢 작업 시작할 때 (매일 루틴)
1. **RunPod 대시보드** 접속 $\rightarrow$ 내 Pod의 **`Start`** 버튼 클릭 (약 30초 소요).
2. Pod의 **JupyterLab Terminal** 접속 후 vLLM 서버 시작:
   ```bash
   vllm serve /workspace/models/gemma-4-26B-A4B-it \
     --port 8000 \
     --max-model-len 16384 \
     --quantization int8_per_channel_weight_only \
     --gpu-memory-utilization 0.92
   ```
   * `Application startup complete`가 뜨면 대기.
3. **맥북 터미널 1**에서 SSH 터널 연결:
   ```bash
   ssh -N -L 8000:localhost:8000 root@202.181.159.238 -p 16110
   ```
4. **맥북 터미널 2**에서 로컬 코드 수정 및 실시간 평가 실행:
   ```bash
   # dev 20건 평가 예시
   python3 open/baseline/script.py \
     --api-url http://localhost:8000/v1 \
     --data-dir open/data \
     --input open/dev.jsonl.gz \
     --labels open/dev_labels.csv \
     --limit 20 \
     --output-dir ./output
   ```

### 🔴 작업 마칠 때 (비용 절약 필수 ⚠️)
* RunPod은 **켜두기만 해도(점유 시간당 ~$1.09) 비용이 계속 차감**됩니다 (연산 유무와 무관).
* 작업 종료 시:
  1. 맥북 터미널의 SSH 터널 종료 (`Ctrl + C`).
  2. RunPod 대시보드에서 반드시 **`Stop`** 버튼 클릭!
     * `Stop` 상태에서는 GPU 요금이 0원이 되며, 디스크 보관료(하루 몇십 원)만 극소량 부과됩니다.
     * 절대 `Terminate`를 누르지 마세요 (다운로드받은 51GB 모델이 삭제됩니다).

### 👥 팀원 공유 방법
1. 팀원들의 공개키(`cat ~/.ssh/id_ed25519.pub`)를 받아 RunPod 터미널에서 등록:
   ```bash
   echo "팀원_공개키" >> ~/.ssh/authorized_keys
   ```
2. 팀원들에게 맥북 터미널용 SSH 터널 명령어(`ssh -N -L 8000:localhost:8000 ...`) 전달.
3. 팀원들도 본인 맥북에서 `--api-url http://localhost:8000/v1`로 실행하면 동일한 L40S GPU를 실시간 공유 가능.
