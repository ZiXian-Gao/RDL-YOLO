"""Allow ``python -m rdl_release`` from an installed checkout."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
