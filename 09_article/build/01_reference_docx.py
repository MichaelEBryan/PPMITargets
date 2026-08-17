import sys
from pathlib import Path
import re
import zipfile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import paths

SOURCE = ("https://www.emerginginvestigators.org/documents/"
          "author_manuscript_template")
HERE = paths.ROOT / "09_article" / "build"
TEMPLATE = HERE / "jei_template.docx"
OUT = HERE / "jei_reference.docx"

BLANK_PROPS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<cp:coreProperties '
    'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/'
    'core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    "<dc:title/><dc:creator/><cp:lastModifiedBy/>"
    "</cp:coreProperties>")


AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def fetch_template():
    if TEMPLATE.exists():
        return TEMPLATE
    r = requests.get(SOURCE, headers={"User-Agent": AGENT}, timeout=60)
    if r.status_code != 200 or not r.content.startswith(b"PK"):
        raise SystemExit(
            f"could not download the author template ({r.status_code}).\n"
            f"Save it from {SOURCE}\nto {TEMPLATE} and run this again.")
    TEMPLATE.write_bytes(r.content)
    return TEMPLATE

ARIAL = ('<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial" '
         'w:eastAsia="Arial"/>')
SPACING = '<w:spacing w:line="360" w:lineRule="auto" w:before="0" w:after="0"/>'


def body_styles():
    def para(sid, name, extra_ppr="", extra_rpr="", based="Normal"):
        return (f'<w:style w:type="paragraph" w:styleId="{sid}">'
                f'<w:name w:val="{name}"/><w:basedOn w:val="{based}"/>'
                f'<w:qFormat/><w:pPr>{SPACING}{extra_ppr}</w:pPr>'
                f'<w:rPr>{ARIAL}<w:sz w:val="22"/><w:szCs w:val="22"/>'
                f'<w:color w:val="000000"/>{extra_rpr}</w:rPr></w:style>')

    bold = "<w:b/>"
    keep = "<w:keepNext/>"
    styles = [
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
        f'<w:name w:val="Normal"/><w:qFormat/><w:pPr>{SPACING}'
        f'<w:jc w:val="both"/></w:pPr><w:rPr>{ARIAL}<w:sz w:val="22"/>'
        '<w:szCs w:val="22"/><w:color w:val="000000"/></w:rPr></w:style>',
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
        '<w:style w:type="table" w:styleId="Table"><w:name w:val="Table"/>'
        '<w:tblPr><w:tblBorders>'
        '<w:top w:val="single" w:sz="6" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="6" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="2" w:color="BFBFBF"/>'
        '</w:tblBorders></w:tblPr>'
        '<w:tcPr><w:vAlign w:val="center"/></w:tcPr></w:style>',
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
    xml = re.sub(r"<w:embed(Regular|Bold|Italic|BoldItalic)\b[^/]*/>", "", xml)
    xml = re.sub(r"<w:embedTrueTypeFonts\s*/>", "", xml)
    return xml


def main():
    z = zipfile.ZipFile(fetch_template())
    doc = z.read("word/document.xml").decode("utf8")
    sect = re.search(r"<w:sectPr\b.*?</w:sectPr>", doc, re.S).group(0)
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

    ct = z.read("[Content_Types].xml").decode("utf8")
    ct = re.sub(r'<Default Extension="odtt"[^>]*/>', "", ct)
    for ext, mime in (("png", "image/png"), ("jpeg", "image/jpeg"),
                      ("jpg", "image/jpeg"), ("tif", "image/tif")):
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
            elif name == "docProps/core.xml":
                out.writestr(item, BLANK_PROPS)
            elif name in ("word/fontTable.xml", "word/settings.xml"):
                out.writestr(item, strip_font_embedding(
                    z.read(name).decode("utf8")))
            elif name.endswith(".rels"):
                rels = z.read(name).decode("utf8")
                rels = re.sub(r'<Relationship\b[^>]*Target="(media|fonts)/'
                              r'[^"]*"[^>]*/>', "", rels)
                out.writestr(item, rels)
            elif re.match(r"word/(header|footer)\d*\.xml$", name):
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
