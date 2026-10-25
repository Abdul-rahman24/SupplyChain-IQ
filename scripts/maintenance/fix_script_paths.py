
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"

OLD = "Path(__file__).resolve().parent"
NEW = "Path(__file__).resolve().parents[2]"

updated = []
unchanged = []

for script in sorted(SCRIPTS_DIR.rglob("*.py")):
    if script.name in {"project_paths.py"}:
        continue

    source = script.read_text(encoding="utf-8")

    if OLD not in source:
        unchanged.append(script.relative_to(ROOT))
        continue

    revised = source.replace(OLD, NEW)
    script.write_text(revised, encoding="utf-8")
    updated.append(script.relative_to(ROOT))

print(f"Updated {len(updated)} scripts:")
for path in updated:
    print(f"  {path}")

if unchanged:
    print("\nNo matching path expression found in:")
    for path in unchanged:
        print(f"  {path}")

print("\nExpected project root:", ROOT)
print("Path update complete.")
