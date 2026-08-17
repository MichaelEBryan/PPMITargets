"""Build a pandoc reference document carrying JEI's formatting.

Takes the journal's own template, empties the body but keeps its section
properties (US Letter, one-inch margins, continuous line numbering), and sets
every style to Arial 11 at 1.5 line spacing with bold black headings.
"""
import re
import shutil
import zipfile
from pathlib import Path

TEMPLATE = Path("/tmp/jei_template.docx")
OUT = Path("/tmp/PPMITargets/tools/jei_reference.docx")

ARIAL = ('<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial" '
         'w:eastAsia="Arial"/>')
SPACING = '<w:spacing w:line="360" w:lineRule="auto" w:before="0" w:after="0"/>'


def body_styles():
    """Paragraph and character styles pandoc looks for, all Arial 11."""
    def para(sid, name, extra_ppr="", extra_rpr="", based="Normal"):
        return (f'<w:style w:type="paragraph" w:styleId="{sid}">'
                f'<w:name w:val="{name}"/><w:basedOn w:val="{based}"/>'
                f'<w:qFormat/><w:pPr>{SPACING}{extra_ppr}</w:pPr>'
                f'<w:rPr>{ARIAL}<w:sz w:val="22"/><w:szCs w:val="22"/>'
                f'<w:color w:val="000000"/>{extra_rpr}</w:rPr></w:style>')

    bold = "<w:b/>"
    keep = "<w:keepNext/>"
    styles = [
        f'<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
        f'<w:name w:val="Normal"/><w:qFormat/><w:pPr>{SPACING}'
        f'<w:jc w:val="both"/></w:pPr><w:rPr>{ARIAL}<w:sz w:val="22"/>'
        f'<w:szCs w:val="22"/><w:color w:val="000000"/></w:rPr></w:style>',
        para("Title", "Title", '<w:jc w:val="left"/>',
             bold + '<w:sz w:val="28"/><w:szCs w:val="28"/>'),
        para("Heading1", "heading 1", keep + '<w:jc w:val="left"/>', bold),
        para("Heading2", "heading 2", keep + '<w:jc w:val="left"/>', bold),
        para("Heading3", "heading 3", keep + '<w:jc w:val="left"/>', bold),
        para("BodyText", "Body Text"),
        para("Compact", "Compact"),
        para("SourceCode", "Source Code"),
        para("ImageCaption", "Image Caption", '<w:jc w:val="left"/>'),
        para("TableCaption", "Table Caption", keep + '<w:jc w:val="left"/>'),
        para("CaptionedFigure", "Captioned Figure",
             '<w:jc w:val="center"/>'),
        para("FigureWithCaption", "Figure With Caption",
             '<w:jc w:val="center"/>'),
        para("Figure", "Figure", '<w:jc w:val="center"/>'),
        para("Author", "Author", '<w:jc w:val="left"/>'),
        para("Date", "Date", '<w:jc w:val="left"/>'),
        para("Abstract", "Abstract"),
        para("FirstParagraph", "First Paragraph"),
        para("ListParagraph", "List Paragraph", '<w:jc w:val="left"/>'),
        f'<w:style w:type="table" w:styleId="Table"><w:name w:val="Table"/>'
        f'<w:tblPr><w:tblBorders>'
        f'<w:top w:val="single" w:sz="6" w:color="000000"/>'
        f'<w:bottom w:val="single" w:sz="6" w:color="000000"/>'
        f'<w:insideH w:val="single" w:sz="2" w:color="BFBFBF"/>'
        f'</w:tblBorders></w:tblPr>'
        f'<w:tcPr><w:vAlign w:val="center"/></w:tcPr></w:style>',
    ]
    return "".join(styles)


DOC_DEFAULTS = (
    "<w:docDefaults><w:rPrDefault><w:rPr>" + ARIAL +
    '<w:sz w:val="22"/><w:szCs w:val="22"/><w:color w:val="000000"/>'
    "</w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>" + SPACING +
    "</w:pPr></w:pPrDefault></w:docDefaults>")

MEDIA = re.compile(r"word/media/")
FONT = re.compile(r"word/fonts/")


def strip_font_embedding(xml):
    """Drop the template's embedded fonts so the file uses installed Arial."""
    xml = re.sub(r"<w:embed(Regular|Bold|Italic|BoldItalic)\b[^/]*/>", "", xml)
    xml = re.sub(r"<w:embedTrueTypeFonts\s*/>", "", xml)
    return xml


def main():
    z = zipfile.ZipFile(TEMPLATE)
    doc = z.read("word/document.xml").decode("utf8")
    sect = re.search(r"<w:sectPr\b.*?</w:sectPr>", doc, re.S).group(0)
    # keep the journal's page setup and line numbering, drop headers/footers
    sect = re.sub(r"<w:(headerReference|footerReference)[^/]*/>", "", sect)
    if "<w:lnNumType" not in sect:
        sect = sect.replace("</w:sectPr>",
                            '<w:lnNumType w:countBy="1" w:restart="continuous"/>'
                            "</w:sectPr>")

    body = ('<w:body><w:p><w:r><w:t></w:t></w:r></w:p>' + sect + "</w:body>")
    new_doc = re.sub(r"<w:body>.*</w:body>", body, doc, flags=re.S)

    styles = z.read("word/styles.xml").decode("utf8")
    styles = re.sub(r"<w:style\b.*?</w:style>", "", styles, flags=re.S)
    styles = re.sub(r"<w:docDefaults>.*?</w:docDefaults>", DOC_DEFAULTS,
                    styles, flags=re.S)
    styles = styles.replace("</w:styles>", body_styles() + "</w:styles>")

    # declare the image types pandoc will embed, and drop the font parts
    ct = z.read("[Content_Types].xml").decode("utf8")
    ct = re.sub(r'<Default Extension="odttf"[^>]*/>', "", ct)
    for ext, mime in (("png", "image/png"), ("jpeg", "image/jpeg"),
                      ("jpg", "image/jpeg"), ("tiff", "image/tiff")):
        if f'Extension="{ext}"' not in ct:
            ct = ct.replace("<Override", f'<Default Extension="{ext}" '
                                         f'ContentType="{mime}"/><Override', 1)

    dropped = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as out:
        for item in z.infolist():
            name = item.filename
            if MEDIA.match(name) or FONT.match(name):
                dropped += 1
                continue
            if name == "word/document.xml":
                out.writestr(item, new_doc)
            elif name == "word/styles.xml":
                out.writestr(item, styles)
            elif name == "[Content_Types].xml":
                out.writestr(item, ct)
            elif name in ("word/fontTable.xml", "word/settings.xml"):
                out.writestr(item, strip_font_embedding(
                    z.read(name).decode("utf8")))
            elif name.endswith(".rels"):
                rels = z.read(name).decode("utf8")
                rels = re.sub(r'<Relationship\b[^>]*Target="(media|fonts)/'
                              r'[^"]*"[^>]*/>', "", rels)
                out.writestr(item, rels)
            elif re.match(r"word/(header|footer)\d*\.xml$", name):
                # pandoc requires these to exist; the body no longer uses them
                tag = "hdr" if "header" in name else "ftr"
                out.writestr(item, re.sub(
                    rf"<w:{tag}(\s[^>]*)?>.*</w:{tag}>",
                    lambda m: f"<w:{tag}{m.group(1) or ''}><w:p/></w:{tag}>",
                    z.read(name).decode("utf8"), flags=re.S))
            else:
                out.writestr(item, z.read(name))

    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} kB, "
          f"{dropped} template media and font parts removed)")
    print("page setup:", re.findall(r"<w:pgSz[^/]*/>|<w:pgMar[^/]*/>", sect))
    print("line numbering:", re.findall(r"<w:lnNumType[^/]*/>", sect))


if __name__ == "__main__":
    main()
