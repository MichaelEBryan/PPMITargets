import sys
from pathlib import Path
import re
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import paths

ROOT = paths.ROOT
ARTICLE = ROOT / "09_article"
FIG = ROOT / "figures"
MAIN_FIGURES = 5

MAX_W = 6.5
MAX_H = 7.2

PAGEBREAK = ("\n```{=openxml}\n"
             '<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n'
             "```\n")

GREEK = {r"\beta": "*β*", r"\alpha": "*α*", r"\lambda": "*λ*",
         r"\chi": "*χ*", r"\mu": "μ", r"\sigma": "σ", r"\Sigma": "Σ",
         r"\pm": "±", r"\times": "×", r"\geq": "≥", r"\leq": "≤",
         r"\approx": "≈", r"\sim": "~", r"\to": "→", r"\ldots": "…",
         r"\AA": "Å"}

ITALIC_VARS = {"P", "k", "r", "R", "t", "z", "D", "n"}


def math(m):
    t = m.group(1)
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    for k, v in GREEK.items():
        t = t.replace(k, v)
    t = re.sub(r"\^(?:\{([^}]*)\}|(-?\w))",
               lambda m: "^" + (m.group(1) or m.group(2)).replace(
                   "-", "\u2212") + "^", t)
    t = re.sub(r"_\{([^}]*)\}", r"\1", t)
    t = t.replace("\\", "").replace("{", "").replace("}", "")
    t = re.sub(r"(?<![\w*])([A-Za-z])(?![\w*])",
               lambda x: f"*{x.group(1)}*" if x.group(1) in ITALIC_VARS
               else x.group(1), t)
    t = re.sub(r"(?<=\s)-(?=\s)", "\u2212", t)
    return t.strip()


def inline(t):
    t = re.sub(r"\\gene\{([^}]*)\}", r"*\1*", t)
    t = re.sub(r"\\pep\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", t)
    t = re.sub(r"\\(emph|textit)\{([^}]*)\}", r"*\2*", t)
    t = re.sub(r"\$([^$]*)\$", math, t)
    t = t.replace("``", "\u201c").replace("''", "\u201d")
    t = re.sub(r"\\[,;:! ]", " ", t)
    t = re.sub(r"\\(vspace|hspace)\*?\{[^}]*\}", "", t)
    t = t.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_")
    t = t.replace("~", " ").replace(r"\quad", "  ")
    t = re.sub(r"\\(newpage|clearpage|par|noindent|vspace\*?)\b", "", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", t)
    t = t.replace("{", "").replace("}", "")
    t = t.replace("--", "\u2013")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def parse_tables():
    src = (ARTICLE / "tables.tex").read_text()
    out = []
    for label, body in re.findall(
            r"\\textbf\{(Table \d+)\}(.*?)\\end\{tabular\}", src, re.S):
        cap = inline(body.split(r"\begin{tabular}")[0])
        cap = re.sub(r"^\s*\\quad\s*", "", cap).strip()
        rows = []
        tab = body.split(r"\begin{tabular}", 1)[1].split("}", 1)[1]
        for line in tab.split(r"\\"):
            line = line.replace(r"\hline", "").strip()
            if not line:
                continue
            cells = [inline(c) for c in line.split("&")]
            if any(cells):
                rows.append(cells)
        out.append((label, cap, rows[0], rows[1:]))
    return out


def figure_width(path):
    with Image.open(path) as im:
        w, h = im.size
    width = min(MAX_W, MAX_H * w / h)
    return round(width, 2)


def heading(text):
    return f"**{text}**\n"


def main():
    src = (ARTICLE / "article.tex").read_text()
    marker = "% " + "-" * 64 + " figures " + "-" * 4
    body, figs = src.split(marker)[0], src.split(marker)[-1]
    md = []

    title = inline(re.search(r"\\selectfont\s*\n(.*?)\\par\}",
                             body, re.S).group(1)).replace("\n", " ")
    md.append(f"**{title}**\n")
    md.append("Tejas Khare^1^, Michael E. Bryan^1^\n")
    md.append("^1^ [Department, Institution/School, City, State, Country]\n")
    md.append(heading("Student Authors"))
    md.append("Tejas Khare, [middle or high school]\n")
    md.append(heading("Keywords"))
    md.append("Parkinson's, genetics, onset, polygenic, druggability\n")
    md.append(heading("Overview"))
    md.append(
        "Parkinson's disease that begins before 50 is often treated as a "
        "different illness from disease that begins after 60. We found that "
        "age at onset instead tracks how much inherited risk a patient "
        "carries, with the same variants and pathways acting on both. "
        "Reading the genetics gene by gene showed which direction a drug "
        "would need to act, and that direction differed between two "
        "lysosomal genes usually treated alike.\n")
    md.append(PAGEBREAK)

    summ = re.search(r"\\textbf\{Summary\.\}(.*?)\\par\s*\n\s*\\vspace",
                     body, re.S).group(1)
    md.append(heading("Summary"))
    md.append(inline(summ) + "\n")
    md.append(PAGEBREAK)

    rest = body[body.index(r"\section*{Introduction}"):]
    HEAD = r"(\\(?:sub)*section\*\{(?:[^{}]|\{[^{}]*\})*\})"
    for chunk in re.split(HEAD, rest):
        chunk = chunk.strip()
        if not chunk:
            continue
        h = re.match(r"\\(sub)*section\*\{((?:[^{}]|\{[^{}]*\})*)\}", chunk)
        if h:
            name = inline(h.group(2))
            if name == "Materials and Methods":
                name = "Materials & Methods"
            if name == "References":
                md.append(heading("Acknowledgments"))
                md.append(inline(re.search(r"\\textbf\{Funding\.\}(.*?)\\par\}",
                                           body, re.S).group(1)) + "\n")
            md.append(heading(name))
            continue
        if r"\begin{enumerate}" in chunk:
            for item in re.findall(
                    r"\\item (.*?)(?=\\item|\\end\{enumerate\})", chunk, re.S):
                md.append(inline(item) + "\n")
            continue
        for para in re.split(r"\n\s*\n", chunk):
            p = inline(para)
            if p and not p.startswith("input"):
                md.append(p + "\n")

    legends = re.findall(
        r"\\figpage\{\.\./figures/([\w]+)\.pdf\}\{\\figlead\{([^}]*)\}"
        r"(.*?)\}\s*\n\s*\n", figs, re.S)
    missing = []

    def emit(stem, text, number, prefix):
        png = FIG / (stem + ".png")
        if not png.exists():
            missing.append(png.name)
            return
        body = inline(text)
        split = re.match(r"(.+?\.)\s+(.*)", body, re.S)
        title, caption = (split.group(1).rstrip("."), split.group(2)) \
            if split else (body.rstrip("."), "")
        md.append(f"![]({png}){{width={figure_width(png)}in}}\n")
        md.append(f"**{prefix} {number}: {title}.** {caption}\n".rstrip() + "\n")
        md.append(PAGEBREAK)

    md.append(PAGEBREAK.strip() + "\n")
    for i, (stem, _lead, text) in enumerate(legends[:MAIN_FIGURES], 1):
        emit(stem, text, i, "Figure")

    md.append(heading("Tables"))
    for label, cap, head, rows in parse_tables():
        md.append(f"**{label}: {cap}**\n")
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "|".join(["---"] * len(head)) + "|")
        for r in rows:
            r = (r + [""] * len(head))[:len(head)]
            md.append("| " + " | ".join(r) + " |")
        md.append("")
    md.append(PAGEBREAK)

    md.append(heading("Appendix"))
    md.append("Supplementary figures supporting the analyses described in the "
              "main text.\n")
    md.append(PAGEBREAK)
    for i, (stem, _lead, text) in enumerate(legends[MAIN_FIGURES:], 1):
        emit(stem, text, i, "Supplementary Figure")

    out = ARTICLE / "build" / "article_jei.md"
    text = "\n".join(md)
    out.write_text(text)
    print(f"wrote {out.relative_to(ROOT)}: {len(text.split())} words, "
          f"{len(legends)} legends, {len(parse_tables())} tables")
    if missing:
        print("missing figures:", ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
