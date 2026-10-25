
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"

print("PROJECT ROOT:", ROOT)
print("\nChecking scripts for root-relative path definitions...\n")

for path in sorted(SCRIPTS_DIR.rglob("*.py")):
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8-sig")

    matches = []

    for line_number, line in enumerate(source.splitlines(), start=1):
        if (
            "Path(__file__)" in line
            or "os.path.dirname(__file__)" in line
            or "data" in line.lower()
            and ("Path(" in line or "join(" in line)
        ):
            matches.append((line_number, line.strip()))

    if matches:
        print(f"\n{path.relative_to(ROOT)}")
        for line_number, line in matches:
            print(f"  Line {line_number}: {line}")
