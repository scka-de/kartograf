"""AST-based surface extraction from a Python agent repo.

A "surface" is any natural-language text the developer wrote that declares what the agent
does or knows. Four kinds, each tagged so the report can color them differently:

  - prompt:      sentence from a top-level string constant whose name matches
                 *(INSTRUCTION|PROMPT|SYSTEM)*
  - tool:        first paragraph of a public function's docstring (the tool's intent)
  - tool_arg:    one item from the docstring's `Args:` block
  - docstring:   first paragraph of any other public function/class docstring
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

PROMPT_NAME_RE = re.compile(
    r"(?:^|_)(INSTRUCTION|INSTRUCTIONS|PROMPT|SYSTEM_PROMPT|GUIDELINES|POLICY_TEXT|ROLE_PROMPT)"
    r"(?:_|$)",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\-\d])")
ARGS_HEADER_RE = re.compile(r"^\s*Args:\s*$", re.MULTILINE)


@dataclass
class Surface:
    id: str
    text: str
    kind: str
    source_file: str
    source_line: int
    container: str = ""
    metadata: dict = field(default_factory=dict)


DEFAULT_EXCLUDE = (".venv", "tests", "test", "__pycache__", ".adk")


def extract_surfaces(
    repo_path: str | Path, exclude: tuple[str, ...] = DEFAULT_EXCLUDE
) -> list[Surface]:
    """Walk a repo, parse every .py file, return all surfaces."""
    root = Path(repo_path).resolve()
    surfaces: list[Surface] = []
    for py_file in sorted(root.rglob("*.py")):
        if any(part in exclude for part in py_file.parts):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        rel = py_file.relative_to(root)
        surfaces.extend(_walk_module(tree, str(rel)))
    return _dedupe(surfaces)


def _walk_module(tree: ast.Module, rel_path: str) -> list[Surface]:
    out: list[Surface] = []
    out.extend(_extract_prompt_constants(tree, rel_path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            out.extend(_extract_callable_surfaces(node, rel_path, kind_hint="tool"))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            out.extend(_extract_callable_surfaces(node, rel_path, kind_hint="docstring"))
    return out


def _extract_prompt_constants(tree: ast.Module, rel_path: str) -> list[Surface]:
    out: list[Surface] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not target_names:
            continue
        prompt_targets = [n for n in target_names if PROMPT_NAME_RE.search(n)]
        if not prompt_targets:
            continue
        text = _literal_string(node.value)
        if not text or len(text.strip()) < 30:
            continue
        for sentence in _split_sentences(text):
            if len(sentence.strip()) < 20:
                continue
            out.append(
                Surface(
                    id=_make_id("prompt", rel_path, node.lineno, len(out)),
                    text=sentence.strip(),
                    kind="prompt",
                    source_file=rel_path,
                    source_line=node.lineno,
                    container=prompt_targets[0],
                )
            )
    return out


def _extract_callable_surfaces(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    rel_path: str,
    kind_hint: str,
) -> list[Surface]:
    docstring = ast.get_docstring(node) or ""
    if not docstring or len(docstring.strip()) < 20:
        return []

    out: list[Surface] = []
    intent, args_block = _split_docstring(docstring)
    if intent and len(intent) >= 20:
        out.append(
            Surface(
                id=_make_id(kind_hint, rel_path, node.lineno, 0),
                text=intent,
                kind=kind_hint,
                source_file=rel_path,
                source_line=node.lineno,
                container=node.name,
            )
        )

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and args_block:
        for index, arg_line in enumerate(_iter_arg_descriptions(args_block)):
            if len(arg_line) < 15:
                continue
            out.append(
                Surface(
                    id=_make_id("tool_arg", rel_path, node.lineno, index + 1),
                    text=arg_line,
                    kind="tool_arg",
                    source_file=rel_path,
                    source_line=node.lineno,
                    container=node.name,
                )
            )
    return out


def _split_docstring(text: str) -> tuple[str, str]:
    """Returns (first paragraph, args block text)."""
    args_match = ARGS_HEADER_RE.search(text)
    if args_match:
        intent_part = text[: args_match.start()].strip()
        args_part = text[args_match.end() :]
        next_section = re.search(
            r"^\s*(Returns|Raises|Example|Yields):\s*$", args_part, re.MULTILINE
        )
        if next_section:
            args_part = args_part[: next_section.start()]
        return _first_paragraph(intent_part), args_part.strip()
    return _first_paragraph(text), ""


def _first_paragraph(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    for separator in ("\n\n", "\n    Args", "\n    Returns"):
        if separator in text:
            text = text.split(separator)[0]
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _iter_arg_descriptions(args_block: str):
    current: list[str] = []
    name = ""
    for raw in args_block.splitlines():
        line = raw.rstrip()
        if not line:
            if name and current:
                yield f"{name}: " + " ".join(current).strip()
                current, name = [], ""
            continue
        match = re.match(r"^\s+([a-zA-Z_][\w]*)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
        if match:
            if name and current:
                yield f"{name}: " + " ".join(current).strip()
            name = match.group(1)
            current = [match.group(2).strip()]
        else:
            stripped = line.strip()
            if stripped:
                current.append(stripped)
    if name and current:
        yield f"{name}: " + " ".join(current).strip()


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\n{2,}", "\n\n", text.strip())
    blocks = [block for block in cleaned.split("\n\n") if block.strip()]
    sentences: list[str] = []
    for block in blocks:
        flattened = " ".join(line.strip() for line in block.splitlines() if line.strip())
        flattened = re.sub(r"^(\d+\.|[*\-])\s*\*?\*?", "", flattened).strip()
        if not flattened:
            continue
        parts = SENTENCE_SPLIT_RE.split(flattened)
        sentences.extend(part.strip() for part in parts if part.strip())
    return sentences


def _literal_string(node: ast.expr) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("{...}")
        return "".join(parts)
    return ""


def _make_id(kind: str, rel_path: str, line: int, index: int) -> str:
    safe = rel_path.replace("/", "_").replace(".", "_")
    return f"{kind}_{safe}_L{line}_{index}"


def _dedupe(surfaces: list[Surface]) -> list[Surface]:
    """Drop only true duplicates: same text in the same file under the same container.

    Distinct callables that share boilerplate text are kept — they count as separate
    surfaces because their source location is informative for the developer reading
    the report.
    """
    seen: dict[tuple[str, str, str], Surface] = {}
    for surface in surfaces:
        key = (surface.text.strip().lower(), surface.source_file, surface.container)
        if key not in seen:
            seen[key] = surface
    return list(seen.values())
