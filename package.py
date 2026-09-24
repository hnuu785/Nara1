#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DACON 제출용 ZIP 패키징 스크립트.

`submit/` 디렉토리 내부의 파일들을 최상위(Root) 구조로 압축하여
`submit_YYYYMMDD_HHMMSS.zip` (또는 submit_YYYYMMDD.zip) 파일을 생성합니다.

주요 기능:
1. `submit/` 폴더 내부 내용물만 최상위로 압축 (불필요한 상위 폴더 제거)
2. `__pycache__`, `.DS_Store`, `*.pyc` 등 불필요한 파일 자동 제외
3. 필수 파일(`script.py`) 최상위 존재 여부 및 용량 제한(2GB / 8GB) 자동 자가검증
4. 편의를 위한 `submit.zip` 동시 복사 옵션 기본 지원

사용법:
  python3 package.py                   # submit_YYYYMMDD_HHMMSS.zip 및 submit.zip 생성
  python3 package.py --date-only       # submit_YYYYMMDD.zip 생성
  python3 package.py -o my_sub.zip     # 지정한 이름으로 생성
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import os
import shutil
import sys
import zipfile
from typing import List, Tuple

# 제외할 파일 및 디렉토리 패턴
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    ".git*",
    "*.swp",
    "*~",
    ".ipynb_checkpoints",
    "output",
    "output/*",
    "submission.csv",
]

MAX_ZIP_BYTES = 2 * 1024 * 1024 * 1024       # 2GB
MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024 # 8GB


def is_excluded(rel_path: str) -> bool:
    """제외 패턴에 일치하는지 확인."""
    parts = rel_path.split(os.sep)
    for part in parts:
        for pat in EXCLUDE_PATTERNS:
            if fnmatch.fnmatch(part, pat):
                return True
    return False


def collect_files(source_dir: str) -> List[Tuple[str, str]]:
    """source_dir 내의 유효한 파일 목록을 (절대경로, zip내 상대경로) 튜플 리스트로 반환."""
    files_to_pack = []
    for root, dirs, files in os.walk(source_dir):
        # 제외 대상 디렉토리는 탐색에서 배제
        dirs[:] = [d for d in dirs if not is_excluded(d)]
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, source_dir)
            if not is_excluded(rel_path):
                files_to_pack.append((full_path, rel_path))
    return sorted(files_to_pack, key=lambda x: x[1])


def create_submission_zip(source_dir: str, output_path: str) -> Tuple[bool, str]:
    """source_dir의 파일들을 최상위 루트 구조로 output_path에 압축."""
    if not os.path.exists(source_dir):
        return False, f"소스 디렉토리가 존재하지 않습니다: {source_dir}"

    files_to_pack = collect_files(source_dir)
    if not files_to_pack:
        return False, f"압축할 파일이 없습니다: {source_dir}"

    # 필수 파일 검증: 최상위 script.py가 있어야 함
    has_script = any(rel == "script.py" for _, rel in files_to_pack)
    if not has_script:
        return False, "제출 필수 파일인 'script.py'가 소스 디렉토리 루트에 없습니다!"

    # 대상 디렉토리 생성
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_uncompressed = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for full_path, rel_path in files_to_pack:
            file_size = os.path.getsize(full_path)
            total_uncompressed += file_size
            zf.write(full_path, arcname=rel_path)

    zip_size = os.path.getsize(output_path)

    # 규격 검증
    if zip_size > MAX_ZIP_BYTES:
        return False, f"ZIP 파일 크기 초과: {zip_size / (1024**2):.2f}MB (최대 2GB)"
    if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
        return False, f"압축 해제 예상 크기 초과: {total_uncompressed / (1024**3):.2f}GB (최대 8GB)"

    return True, ""


def print_zip_info(zip_path: str) -> None:
    """생성된 zip 파일의 내부 구조와 크기 출력."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        infolist = zf.infolist()
        total_size = sum(info.file_size for info in infolist)
        comp_size = sum(info.compress_size for info in infolist)

        print("\n📦 [ZIP 내부 파일 목록]")
        print("-" * 60)
        for info in infolist:
            print(f"  {info.filename:<38} {info.file_size:>10,} bytes")
        print("-" * 60)
        print(f"  총 파일 수: {len(infolist)}개")
        print(f"  압축 전 크기: {total_size / 1024:.1f} KB ({total_size:,} bytes)")
        print(f"  압축 후 크기: {comp_size / 1024:.1f} KB ({comp_size:,} bytes)")
        print("-" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="DACON submit.zip 패키징 도구")
    parser.add_argument(
        "--src",
        default="submit",
        help="압축 대상 소스 디렉토리 (기본값: submit)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="출력할 ZIP 파일 경로 (기본값: submit_YYYYMMDD_HHMMSS.zip)",
    )
    parser.add_argument(
        "--date-only",
        action="store_true",
        help="파일명에 시/분/초 없이 날짜만 포함 (submit_YYYYMMDD.zip)",
    )
    parser.add_argument(
        "--no-alias",
        action="store_true",
        help="submit.zip 사본을 생성하지 않음 (기본은 submit.zip도 함께 생성)",
    )
    args = parser.parse_args()

    # 파일명 결정
    now = datetime.datetime.now()
    if args.output:
        out_name = args.output
    elif args.date_only:
        out_name = f"submit_{now.strftime('%Y%m%d')}.zip"
    else:
        out_name = f"submit_{now.strftime('%Y%m%d_%H%M%S')}.zip"

    print(f"🚀 '{args.src}/' 디렉토리 패키징 시작...")
    success, err_msg = create_submission_zip(args.src, out_name)

    if not success:
        print(f"❌ 패키징 실패: {err_msg}", file=sys.stderr)
        return 1

    print(f"✅ 압축 완료: {out_name}")
    print_zip_info(out_name)

    # submit.zip 별칭 생성
    if not args.no_alias and out_name != "submit.zip":
        shutil.copy2(out_name, "submit.zip")
        print("🔗 DACON 제출용 기본 파일인 'submit.zip'으로도 함께 복사되었습니다.")

    print("\n🎉 제출 준비가 완료되었습니다! 생성된 파일을 DACON에 업로드하세요.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
