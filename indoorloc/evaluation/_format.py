"""Plain-text and Markdown tables (standard library only), shared by ``report`` and ``literature``."""
from __future__ import annotations

import math

MISSING = "-"


def fmt(value, digits: int = 3) -> str:
    """A cell: numbers with ``digits`` decimals (ints and bools as is), None as ``-``."""
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return MISSING if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def render(header: list[str], rows: list[list[str]], style: str = "markdown") -> str:
    """``style="markdown"`` (GitHub pipe table) or ``"text"`` (space-aligned, numbers right-aligned)."""
    if style not in ("markdown", "text"):
        raise ValueError(f"style must be 'markdown' or 'text', got {style!r}")
    cells = [[str(c) for c in header]] + [[str(c) for c in row] for row in rows]
    if style == "markdown":
        esc = lambda s: s.replace("|", "\\|")  # noqa: E731
        lines = ["| " + " | ".join(esc(c) for c in cells[0]) + " |",
                 "|" + "|".join("---" for _ in header) + "|"]
        lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in cells[1:]]
        return "\n".join(lines)
    widths = [max(len(row[j]) for row in cells) for j in range(len(header))]

    def numeric(j: int) -> bool:
        return all(_is_number(row[j]) for row in cells[1:]) and len(cells) > 1

    align = [numeric(j) for j in range(len(header))]
    line = lambda row: "  ".join(c.rjust(w) if a else c.ljust(w) for c, w, a in zip(row, widths, align))  # noqa: E731
    return "\n".join([line(cells[0]).rstrip(), "  ".join("-" * w for w in widths)] +
                     [line(r).rstrip() for r in cells[1:]])


def _is_number(s: str) -> bool:
    if s == MISSING:
        return True
    try:
        float(s)
    except ValueError:
        return False
    return True


def cite(source: dict | None) -> str:
    """``Authors (Year), Venue. DOI`` in one line; ``-`` when there is no identifiable source."""
    if not source:
        return MISSING
    authors = source.get("authors") or ""
    names = [a.strip() for a in authors.split(",") if a.strip()]
    lead = (names[0].split()[-1] + (" et al." if len(names) > 2 else "" if len(names) == 1 else
                                    f" and {names[1].split()[-1]}")) if names else ""
    parts = [f"{lead} ({source.get('year')})" if lead else str(source.get("year") or "")]
    if source.get("venue"):
        parts.append(str(source["venue"]).split(",")[0])
    doi = source.get("doi")
    text = ", ".join(p for p in parts if p)
    return f"{text}. doi:{doi}" if doi else (f"{text}. {source['url']}" if source.get("url") else text)
