
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("PROJECT ROOT")
print("=" * 65)

for item in sorted(ROOT.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
    if item.is_dir():
        print(f"[DIR]  {item.name}/")
    else:
        print(f"[FILE] {item.name}")

print("\nSCRIPT FOLDERS")
print("=" * 65)

scripts_dir = ROOT / "scripts"

if scripts_dir.exists():
    for folder in sorted(scripts_dir.iterdir()):
        if folder.is_dir():
            count = len(list(folder.glob("*.py")))
            print(f"{folder.name}/ ({count} Python scripts)")
