"""Convert the manuscript LaTeX to Markdown laid out to the JEI template.

Section order, headings and page breaks follow the journal's author template:
title page and summary on pages of their own, then Introduction through
Materials & Methods with no space between sections, then references, the main
figures, the tables, and the supplementary figures as an appendix.

Headings are emitted as bold paragraphs rather than Markdown headings, because
the template asks authors not to use Word's automated heading styles.
"""
import pathlib
import re
import sys

from PIL import Image

ROOT = pathlib.Path("/tmp/PPMITargets")
MS = ROOT / "manuscript"
FIG = ROOT / "figures"

MAIN = ["C0_summary", "C1_when_patients_develop_pd", "F3_eqtl_evidence",
        "C5_what_makes_a_target", "C4_what_the_model_learns"]
SUPP = ["C2_inherited_risk_and_onset", "C3_same_disease_different_dose",
        "C6_which_way_to_push", "C7_what_the_model_separated",
        "S1_effect_sizes", "E1_ranking_rule_behaviour",
        "E2_where_the_rule_fails", "E3_shared_features_null",
        "E4_full_ablation", "E5_geneset_and_models",
        "E6_galc_structure_methods", "E7_17q21_haplotype",
        "E8_how_the_arms_were_assigned", "E10_robustness_checks",
        "E11_claim_audit", "E12_galc_structure_detail",
        "E13_enrolment_proxy", "E14_model_stability",
        "E15_candidate_genes", "E16_galc_locus"]

MAX_W = 6.5      # text width at one-inch margins on US Letter
MAX_H = 7.2      # leaves room for the caption below the image

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
    """Render inline maths as Markdown, using real superscripts."""
    t = m.group(1)
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    for k, v in GREEK.items():
        t = t.replace(k, v)
    # one pass, so a converted exponent is not rewritten by the next rule
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
    """Return (label, caption, header, rows) for each tabular block."""
    src = (MS / "tables.tex").read_text()
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
    """Width in inches that keeps the figure inside the text block."""
    with Image.open(path) as im:
        w, h = im.size
    width = min(MAX_W, MAX_H * w / h)
    return round(width, 2)


def heading(text):
    return f"**{text}**\n"


def main():
    src = (MS / "manuscript_v2.tex").read_text()
    marker = "% " + "-" * 64 + " figures " + "-" * 4
    body, figs = src.split(marker)[0], src.split(marker)[-1]
    md = []

    # ---- title page -----------------------------------------------------
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

    # ---- summary --------------------------------------------------------
    summ = re.search(r"\\textbf\{Summary\.\}(.*?)\\par\s*\n\s*\\vspace",
                     body, re.S).group(1)
    md.append(heading("Summary"))
    md.append(inline(summ) + "\n")
    md.append(PAGEBREAK)

    # ---- introduction through references --------------------------------
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

    # ---- figures --------------------------------------------------------
    legends = re.findall(
        r"\\figpage\{figures/(\w+)\.pdf\}\{\\figlead\{([^}]*)\}(.*?)\}\s*\n\s*\n",
        figs, re.S)
    order = MAIN + SUPP
    missing = []

    def emit(i, lead, text, number, prefix):
        """One figure: the image, then its title and caption beneath it."""
        png = FIG / (order[i] + ".png")
        if not png.exists():
            missing.append(png.name)
            return
        # the LaTeX lead is only the figure number; the title opens the caption
        body = inline(text)
        split = re.match(r"(.+?\.)\s+(.*)", body, re.S)
        title, caption = (split.group(1).rstrip("."), split.group(2)) \
            if split else (body.rstrip("."), "")
        md.append(f"![]({png}){{width={figure_width(png)}in}}\n")
        md.append(f"**{prefix} {number}: {title}.** {caption}\n".rstrip() + "\n")
        md.append(PAGEBREAK)

    md.append(PAGEBREAK.strip() + "\n")
    for i, (_stem, lead, text) in enumerate(legends[:len(MAIN)]):
        emit(i, lead, text, i + 1, "Figure")

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
    for i, (_stem, lead, text) in enumerate(legends[len(MAIN):],
                                            start=len(MAIN)):
        if i >= len(order):
            break
        emit(i, lead, text, i - len(MAIN) + 1, "Supplementary Figure")

    out = ROOT / "tools" / "manuscript_jei.md"
    out.write_text("\n".join(md))
    text = "\n".join(md)
    print(f"wrote {out}: {len(text.split())} words, "
          f"{len(legends)} legends, {len(parse_tables())} tables")
    if missing:
        print("missing figures:", ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
