import sys
import unittest
from pathlib import Path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    start_dir = project_root / "parquet_tests"

    # Ensure local package imports work when launched from external cwd/debug adapters.
    sys.path.insert(0, str(project_root))

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(start_dir),
        pattern="test_*.py",
        top_level_dir=str(project_root),
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
