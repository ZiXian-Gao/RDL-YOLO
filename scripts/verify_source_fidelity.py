"""Compare selected class bodies with a supplied original nn source directory."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def definitions(path: Path) -> dict[str, ast.ClassDef | ast.FunctionDef]:
    """Use the last top-level binding for duplicate names, as Python does."""
    return {node.name: node for node in ast.parse(path.read_text(encoding="utf-8-sig")).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))}


def verify(source_nn: Path, checkout: Path) -> dict:
    """Verify ASTs and record full-file source hashes without importing legacy modules."""
    records = json.loads((checkout / "provenance/extracted_operators.json").read_text())
    results = []
    for record in records:
        relative, name = record["file"], record["definition"]
        original, current = source_nn / relative, checkout / "ultralytics/nn" / relative
        source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
        equal = ast.dump(definitions(original)[name], include_attributes=False) == ast.dump(
            definitions(current)[name], include_attributes=False)
        results.append({"file": relative, "definition": name, "ast_equal": equal,
                        "source_file_hash_matches": source_hash == record["source_sha256"]})
    core = []
    original = definitions(source_nn / "modules/block.py")
    current = definitions(checkout / "ultralytics/nn/modules/block.py")
    for name in ("C3k", "C3k2", "C2f", "C3", "Bottleneck", "C2PSA", "PSABlock", "Attention", "DFL"):
        core.append({"definition": name, "ast_equal": ast.dump(original[name], include_attributes=False)
                     == ast.dump(current[name], include_attributes=False)})
    return {"selected_operators": results, "core_inherited_classes": core,
            "all_equal": all(item["ast_equal"] and item["source_file_hash_matches"] for item in results)
                         and all(item["ast_equal"] for item in core),
            "historical_experiment_identity_established": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-nn", type=Path, required=True,
                        help="Original research checkout's ultralytics/nn directory")
    parser.add_argument("--output", type=Path, default=Path("docs/source_fidelity.json"))
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    result = verify(args.source_nn.resolve(strict=True), checkout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_equal": result["all_equal"],
                      "selected_operator_count": len(result["selected_operators"]),
                      "core_class_count": len(result["core_inherited_classes"])}))
    return 0 if result["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
