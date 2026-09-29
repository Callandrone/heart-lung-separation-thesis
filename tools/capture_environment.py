"""Print a portable environment record; optionally write it to --output."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packages = sorted(
        {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}.items()
    )
    payload = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": dict(packages),
        "note": "Environment at capture time; not evidence of the original thesis training environment.",
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
