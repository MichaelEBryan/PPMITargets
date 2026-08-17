"""Check the built manuscript against the JEI template requirements."""
import html
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

DOCX = Path(sys.argv[1] if len(sys.argv) > 1
            else "/tmp/PPMITargets/manuscript/Khare_Bryan_JEI.docx")

EMU = 914400
TWIP = 1440


def attrs(xml, tag):
    """Attribute dictionaries for every occurrence of a tag."""
    out = []
    for m in re.finditer(rf"<{tag}(\s[^>]*?)?/?>", xml):
        out.append(dict(re.findall(r'([\w:]+)="([^"]*)"', m.group(1) or "")))
    return out


def main():
    z = zipfile.ZipFile(DOCX)
    doc = z.read("word/document.xml").decode("utf8")
    styles = z.read("word/styles.xml").decode("utf8")
    ct = z.read("[Content_Types].xml").decode("utf8")
    checks = []

    def ok(name, cond, detail=""):
        checks.append((bool(cond), name, detail))

    ok("archive intact", z.testzip() is None)

    pg = attrs(doc, "w:pgSz")[0]
    ok("US Letter page", (pg["w:w"], pg["w:h"]) == ("12240", "15840"),
       f'{int(pg["w:w"])/TWIP:g} x {int(pg["w:h"])/TWIP:g} in')

    mar = attrs(doc, "w:pgMar")[0]
    sides = [mar[k] for k in ("w:top", "w:right", "w:bottom", "w:left")]
    ok("one-inch margins", set(sides) == {"1440"},
       f"{int(sides[0])/TWIP:g} in all sides")

    ln = attrs(doc, "w:lnNumType")
    ok("continuous line numbering",
       ln and ln[0].get("w:countBy") == "1"
       and ln[0].get("w:restart") == "continuous")

    fonts = Counter(re.findall(r'w:ascii="([^"]+)"', doc + styles))
    ok("Arial throughout", set(fonts) <= {"Arial"}, dict(fonts))

    sizes = Counter(a["w:val"] for a in attrs(doc + styles, "w:sz"))
    ok("11 pt body text", sizes.most_common(1)[0][0] == "22",
       ", ".join(f"{int(k)/2:g}pt x{v}" for k, v in sizes.most_common()))

    sp = [a for a in attrs(styles, "w:spacing")
          if a.get("w:line") == "360" and a.get("w:lineRule") == "auto"]
    ok("1.5 line spacing", sp, f"{len(sp)} styles")

    used = sorted(set(a["w:val"] for a in attrs(doc, "w:pStyle")))
    ok("no Word heading styles", not any(s.startswith("Heading")
                                         for s in used), used)

    breaks = len([a for a in attrs(doc, "w:br") if a.get("w:type") == "page"])
    ok("page breaks placed", breaks >= 25, f"{breaks}")

    media = [n for n in z.namelist() if n.startswith("word/media/")]
    ok("figures embedded", len(media) == 25, f"{len(media)} images")

    # Word needs a content type for every part, by extension or by name
    defaults = {a["Extension"].lower() for a in attrs(ct, "Default")}
    named = {a["PartName"].lstrip("/") for a in attrs(ct, "Override")}
    undeclared = [n for n in media
                  if n not in named
                  and n.rsplit(".", 1)[-1].lower() not in defaults]
    ok("image parts declared", not undeclared,
       undeclared[:3] or f"{len(media)} declared")

    ext = [(int(a["cx"]) / EMU, int(a["cy"]) / EMU)
           for a in attrs(doc, "wp:extent")]
    over = [(round(w, 2), round(h, 2)) for w, h in ext if w > 6.5 or h > 9.0]
    tall = max(ext, key=lambda p: p[1])
    ok("figures fit the text block", not over,
       f"largest {tall[0]:.2f} x {tall[1]:.2f} in")

    text = html.unescape(re.sub(r"<[^>]+>", "", doc.replace("</w:p>", "\n")))
    order = ["Summary", "Introduction", "Results", "Discussion",
             "Materials & Methods", "Acknowledgments", "References",
             "Tables", "Appendix"]
    pos = [text.find(h) for h in order]
    ok("JEI section order", all(p > 0 for p in pos) and pos == sorted(pos),
       {h: p for h, p in zip(order, pos) if p < 0} or "as specified")

    body = text[:text.find("Tables")]
    ok("no lists in the body", "<w:numPr>" not in doc[:doc.find("Tables")])
    ok("plural first person", " I " not in body)
    ok("summary within 250 words",
       len(text[text.find("Summary"):text.find("Introduction")].split()) <= 252,
       f"{len(text[text.find('Summary'):text.find('Introduction')].split())-1}"
       " words")

    width = max(len(n) for _, n, _ in checks)
    fails = sum(not c for c, _, _ in checks)
    for cond, name, detail in checks:
        print(f"  {'pass' if cond else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(checks) - fails}/{len(checks)} checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
