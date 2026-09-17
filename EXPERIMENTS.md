# Git으로 실험과 제출 버전 관리

실험마다 브랜치를 만들고, 제출할 코드는 커밋과 태그로 고정합니다.
실험별 코드 사본이나 ZIP을 저장소에 추가하지 않습니다.

## 1. 실험 브랜치

시작할 기준 브랜치 또는 커밋으로 이동한 뒤 새 브랜치를 만듭니다.
아래 이름과 설명은 예시이며 실험마다 바꿉니다.

```sh
git switch -c codex/exp-001
```

코드를 수정하고 검증한 뒤, 관련 파일만 선택하여 커밋합니다.
실험 가설과 변경 이유는 커밋 설명에 기록합니다.

```sh
git add open/baseline/script.py open/baseline/requirements.txt
git commit -m "exp-001: 공고 입력 길이를 늘려 절단 영향 비교"
```

추가한 모듈이나 정적 자산이 있으면 함께 커밋해야 합니다.

## 2. 제출 버전 고정과 ZIP 생성

제출할 커밋에서 주석 태그를 만들고, **태그의 코드로** ZIP을 만듭니다.
작업 폴더에 커밋하지 않은 변경이 있어도 ZIP에는 들어가지 않습니다.

```sh
git tag -a submit/exp-001 -m "exp-001 제출용: 공고 입력 길이 확대"
mkdir -p .artifacts
git archive --format=zip --output=.artifacts/submit.zip submit/exp-001:open/baseline
```

`.artifacts/submit.zip`을 제출 페이지에 업로드합니다. `.artifacts/`는 Git에서
제외되며, 다음 제출 때 같은 경로에 ZIP을 다시 만들면 됩니다.
`open/baseline/`의 커밋된 파일 전체가 ZIP 최상위에 들어가므로 이 폴더에는
제출에 필요한 코드·의존성 목록·정적 자산만 둡니다.
이 명령은 패키징만 수행하므로 모델 실행 검증은 별도로 해야 합니다.

## 3. 점수 기록

점수가 나오면 **제출 태그와 같은 커밋**에 결과 태그를 추가합니다.
브랜치에서 실험을 계속했더라도 점수는 실제 제출 버전에 연결됩니다.
기존 제출 태그를 이동하거나 덮어쓰지 않습니다.

```sh
git tag -a result/exp-001-public submit/exp-001^{}
```

열리는 편집기에 실제 Public 점수, 제출일, 제출번호, 관찰 내용을 적습니다.
dev 결과는 `result/exp-001-dev`로 구분하고 평가 데이터·실행 옵션도 적습니다.
잘못된 결과는 `result/exp-001-public-correction-01`처럼 새 태그에 정정 사유와
함께 남깁니다. 재제출은 `submit/exp-001-r2`처럼 새 이름을 사용합니다.

```sh
git tag -l 'result/*' --format='%(refname:short) %(contents)'
git diff submit/exp-001 submit/exp-002 -- open/baseline
```

과거 버전의 ZIP도 2번 명령의 태그만 바꾸면 다시 만들 수 있습니다.
코드 내용은 복원되지만 ZIP 자체의 바이트 일치까지 보장하는 방식은 아닙니다.
원격 저장소에 백업할 때는 실험 브랜치와 제출·결과 태그도 함께 푸시합니다.
일반 브랜치 푸시만으로 태그가 모두 전송되지는 않습니다.

## 기존 자료

기존 `submit.zip`, `submit_before_rag.zip`은 원본을 유지합니다.
이전에 만든 `submissions/`의 중복 사본과 전용 기록 도구는 제거했습니다.
해당 사본은 필요하면 Git의 과거 커밋에서 확인할 수 있습니다.

사용자 보고 점수 **0.1979200359**(2026-09-16)는 실제 업로드 ZIP과의 연결이
확인되지 않아 특정 커밋의 결과 태그로 등록하지 않았습니다.
