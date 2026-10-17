"""Visualize your repo's tree structure"""

import sys
from pathlib import Path


def main():
    SKIP = {
        ".git",
        ".venv",
        "__pycache__",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        "dist",
        "build",
    }

    def tree(path: Path, prefix: str = "") -> None:
        entries = sorted(
            (p for p in path.iterdir() if p.name not in SKIP and not p.name.endswith(".egg-info")),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        for i, p in enumerate(entries):
            last = i == len(entries) - 1
            print(prefix + ("└── " if last else "├── ") + p.name + ("/" if p.is_dir() else ""))
            if p.is_dir():
                tree(p, prefix + ("    " if last else "│   "))

    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    print(root.name + "/")
    tree(root)


if __name__ == "__main__":
    main()
