"""제출 ZIP과 코드 스냅샷을 보존하는 표준 라이브러리 전용 도구."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "submissions"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_info():
    def run(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"head_at_archive": run("rev-parse", "HEAD"),
            "status_at_archive": run("status", "--short")}


def save(args):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.name):
        raise ValueError("이름은 영문·숫자·하이픈·밑줄 1~80자로 입력하세요.")
    ARCHIVE.mkdir(exist_ok=True)
    destination = ARCHIVE / args.name
    if destination.exists():
        raise ValueError("이미 있는 이름입니다. 기존 기록을 덮어쓸 수 없습니다.")
    with tempfile.TemporaryDirectory(dir=ARCHIVE, prefix=".pending-") as tmp:
        staging = Path(tmp)
        archive = staging / "submit.zip"
        if args.zip:
            archive.write_bytes(Path(args.zip).read_bytes())
        else:
            source = ROOT / "open/baseline"
            files = [source / "script.py"]
            if (source / "requirements.txt").exists():
                files.append(source / "requirements.txt")
            if (source / "model").exists():
                files.extend(sorted(p for p in (source / "model").rglob("*") if p.is_file()))
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                for path in files:
                    if path.is_symlink():
                        raise ValueError(f"심볼릭 링크는 제출에 포함할 수 없습니다: {path}")
                    z.write(path, path.relative_to(source).as_posix())
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        hashes = {}
        with zipfile.ZipFile(archive) as z:
            if "script.py" not in z.namelist():
                raise ValueError("ZIP 최상위에 script.py가 필요합니다.")
            for item in z.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                    raise ValueError("안전하지 않은 ZIP 경로입니다.")
                if item.is_dir():
                    continue
                if path.as_posix() in hashes:
                    raise ValueError("ZIP에 중복 경로가 있습니다.")
                content = z.read(item)
                target = staging / "code" / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                hashes[path.as_posix()] = hashlib.sha256(content).hexdigest()
        write_json(staging / "metadata.json", {
            "name": args.name, "archived_at": now(), "note": args.note,
            "source": str(Path(args.zip).resolve()) if args.zip else "open/baseline",
            "sha256": digest, "files": hashes, **git_info(),
            "submission_status": "unknown" if args.zip else "prepared",
        })
        # 예약된 디렉터리만 사용하여 다른 실행의 기록도 덮어쓰지 않는다.
        destination.mkdir()
        try:
            for path in staging.iterdir():
                shutil.move(str(path), destination / path.name)
        except Exception:
            shutil.rmtree(destination)
            raise
    print(f"저장 완료: {destination / 'submit.zip'}")


def result(args):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.name):
        raise ValueError("잘못된 기록 이름입니다.")
    folder = ARCHIVE / args.name
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256((folder / "submit.zip").read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("보관된 ZIP이 변경되었습니다. 점수를 기록하지 않습니다.")
    if not 0 <= args.score <= 1:
        raise ValueError("점수는 0~1 사이여야 합니다.")
    results = folder / "results"
    results.mkdir(exist_ok=True)
    write_json(results / f"{uuid.uuid4().hex}.json", {
        "recorded_at": now(), "score": args.score, "kind": args.kind,
        "submission_id": args.submission_id, "note": args.note,
        "zip_sha256": metadata["sha256"],
    })
    print(f"결과 기록 완료: {args.name} ({args.kind}: {args.score})")


def listing(_args):
    for path in sorted(ARCHIVE.glob("*/metadata.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results = [json.loads(p.read_text(encoding="utf-8"))
                   for p in (path.parent / "results").glob("*.json")]
        results.sort(key=lambda r: r["recorded_at"])
        scores = ", ".join(f"{r['kind']}={r['score']}" for r in results) or "점수 미등록"
        print(f"{data['name']} | {scores} | {data['note']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("save", help="현재 코드 또는 기존 ZIP 보관")
    p.add_argument("name")
    p.add_argument("--zip", help="생략하면 open/baseline을 새로 패키징")
    p.add_argument("--note", required=True, help="변경 내용·실험 가설")
    p.set_defaults(func=save)
    p = commands.add_parser("result", help="해당 제출본에 점수 추가 (기존 결과 유지)")
    p.add_argument("name")
    p.add_argument("--score", type=float, required=True)
    p.add_argument("--kind", choices=["public", "dev"], required=True)
    p.add_argument("--submission-id", default="")
    p.add_argument("--note", default="")
    p.set_defaults(func=result)
    p = commands.add_parser("list", help="버전별 점수 조회")
    p.set_defaults(func=listing)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
