"""Convert the manuscript LaTeX to Markdown for pandoc, keeping figures and tables."""
import pathlib
import re
import sys

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

SUP = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")

GREEK = {r"\beta": "β", r"\alpha": "α", r"\lambda": "λ", r"\chi": "χ",
         r"\mu": "μ", r"\sigma": "σ", r"\Sigma": "Σ", r"\pm": "±",
         r"\times": "×", r"\geq": "≥", r"\leq": "≤", r"\approx": "≈",
         r"\sim": "~", r"\to": "→", r"\ldots": "…", r"\AA": "Å"}


def math(m):
    t = m.group(1)
    for k, v in GREEK.items():
        t = t.replace(k, v)
    t = re.sub(r"\^\{([^}]*)\}", lambda x: x.group(1).translate(SUP), t)
    t = re.sub(r"\^(-?\w)", lambda x: x.group(1).translate(SUP), t)
    t = re.sub(r"_\{([^}]*)\}", lambda x: x.group(1), t)
    t = t.replace("-", "\u2212")
    t = t.replace("\\", "").replace("{", "").replace("}", "")
    return t.strip()


def inline(t):
    t = re.sub(r"\\gene\{([^}]*)\}", r"*\1*", t)
    t = re.sub(r"\\pep\{([^}]*)\}", r"`\1`", t)
    t = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", t)
    t = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", t)
    t = re.sub(r"\\textit\{([^}]*)\}", r"*\1*", t)
    t = re.sub(r"\$([^$]*)\$", math, t)
    t = t.replace("``", "\u201c").replace("''", "\u201d")
    t = re.sub(r"\\[,;:! ]", " ", t)
    t = t.replace(r"\AA", "\u00c5")
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
    """Return (caption, headers, rows) for each tabular block in tables.tex."""
    src = (MS / "tables.tex").read_text()
    out = []
    for block in re.findall(r"\\textbf\{(Table \d+)\}(.*?)\\end\{tabular\}",
                            src, re.S):
        label, body = block
        cap = inline(body.split(r"\begin{tabular}")[0])
        cap = re.sub(r"^\s*\\quad\s*", "", cap).strip()
        rows = []
        tab = body.split(r"\begin{tabular}", 1)[1]
        tab = tab.split("}", 1)[1]
        for line in tab.split(r"\\"):
            line = line.replace(r"\hline", "").strip()
            if not line:
                continue
            cells = [inline(c) for c in line.split("&")]
            if any(cells):
                rows.append(cells)
        out.append((label, cap, rows[0], rows[1:]))
    return out


def main():
    src = (MS / "manuscript_v2.tex").read_text()
    body = src.split("% " + "-" * 64 + " figures " + "-" * 4)[0]
    figs = src.split("% " + "-" * 64 + " figures " + "-" * 4)[-1]

    md = []

    title = re.search(r"\\selectfont\s*\n(.*?)\\par\}", body, re.S)
    md.append("# " + inline(title.group(1)).replace("\n", " ") + "\n")

    summ = re.search(r"\\textbf\{Summary\.\}(.*?)\\par\s*\n\s*\\vspace",
                     body, re.S)
    md.append("**Summary.** " + inline(summ.group(1)) + "\n")
    fund = re.search(r"\\textbf\{Funding\.\}(.*?)\\par\}", body, re.S)
    md.append("**Funding.** " + inline(fund.group(1)) + "\n")

    rest = body[body.index(r"\section*{Introduction}"):]
    HEAD = r"(\\(?:sub)*section\*\{(?:[^{}]|\{[^{}]*\})*\})"
    for chunk in re.split(HEAD, rest):
        chunk = chunk.strip()
        if not chunk:
            continue
        h = re.match(r"\\(sub)*section\*\{((?:[^{}]|\{[^{}]*\})*)\}", chunk)
        if h:
            level = 2 if not h.group(1) else 3
            md.append("#" * level + " " + inline(h.group(2)) + "\n")
            continue
        if r"\begin{enumerate}" in chunk:
            for i, item in enumerate(
                    re.findall(r"\\item (.*?)(?=\\item|\\end\{enumerate\})",
                               chunk, re.S), 1):
                md.append(f"{i}. " + inline(item) + "\n")
            continue
        for para in re.split(r"\n\s*\n", chunk):
            p = inline(para)
            if p and not p.startswith("input"):
                md.append(p + "\n")

    md.append("\\newpage\n\n## Tables\n")
    for label, cap, head, rows in parse_tables():
        md.append(f"**{label}.** {cap}\n")
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "|".join(["---"] * len(head)) + "|")
        for r in rows:
            r = (r + [""] * len(head))[:len(head)]
            md.append("| " + " | ".join(r) + " |")
        md.append("")

    legends = re.findall(r"\\figpage\{figures/(\w+)\.pdf\}\{\\figlead\{([^}]*)\}"
                         r"(.*?)\}\s*\n\s*\n", figs, re.S)
    order = MAIN + SUPP
    md.append("\\newpage\n\n## Figures\n")
    for i, (stem, lead, text) in enumerate(legends):
        if i >= len(order):
            break
        png = FIG / (order[i] + ".png")
        if not png.exists():
            print(f"missing {png}", file=sys.stderr)
            continue
        if i == len(MAIN):
            md.append("\\newpage\n\n## Supplementary figures\n")
        md.append(f"![]({png})\n")
        md.append(f"**{inline(lead)}.** {inline(text)}\n")
        md.append("\\newpage\n")

    (ROOT / "tools" / "manuscript_v2.md").write_text("\n".join(md))
    print(f"wrote markdown: {len('\n'.join(md).split())} words, "
          f"{len(legends)} figure legends")


if __name__ == "__main__":
    main()
