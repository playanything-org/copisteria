# Modified by Play Anything (https://github.com/playanything-org/copisteria): see README.md.
import ast
from pathlib import Path


def test_page_reader_has_no_opencv_imports():
    # Detector inference imports these modules, but never uses the crop dataset or
    # scribble augmentation, which runs only while the model is in training mode.
    training_only = {
        ("detector/small/arch7.py", "scribble"),
        ("detector/small/data.py", "CropDataset.__getitem__"),
    }
    source = Path(__file__).resolve().parents[1] / "src/copisteria"
    unexpected = []
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if not any(module.partition(".")[0] == "cv2" for module in modules):
                continue
            enclosing = []
            parent = parents[node]
            while parent is not tree:
                if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    enclosing.append(parent.name)
                parent = parents[parent]
            if (relative, ".".join(reversed(enclosing))) not in training_only:
                unexpected.append(f"{relative}:{node.lineno}")
    assert not unexpected, unexpected
