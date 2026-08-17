import sys
from pathlib import Path
import html
import re
import zipfile
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import paths

DOCX = Path(sys.argv[1]) if len(sys.argv) > 1 \
    else paths.ROOT / "09_article" / "article_jei.docx"
ARIAL = "/System/Library/Fonts/Supplemental/Arial.tt"

PT = 64
SIZE = 11
COLUMN_PT = 6.5 * 72
LINE_PT = SIZE * 1.15 * 1.5
PAGE_PT = 9.0 * 72
PER_PAGE = int(PAGE_PT // LINE_PT)


def paragraphs(doc):
    for p in re.split(r"</w:p>", doc):
        yield ('w:type="page"' in p,
               html.unescape(re.sub(r"<[^>]+>", "", p)).strip())


def wrapped_lines(text, font, scale):
    if not text:
        return 1
    lines, cur = 1, 0.0
    space = font.getlength(" ") / scale
    for word in text.split():
        w = font.getlength(word) / scale
        if cur and cur + space + w > COLUMN_PT:
            lines += 1
            cur = w
        else:
            cur += (space if cur else 0) + w
    return lines


def main():
    doc = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf8")
    font = ImageFont.truetype(ARIAL, SIZE * PT)
    scale = PT

    counting, lines, pages_forced = False, 0, 0
    for is_break, text in paragraphs(doc):
        if text.startswith("Introduction"):
            counting = True
        if text.startswith("References"):
            break
        if not counting:
            continue
        if is_break:
            pages_forced += 1
            lines = (lines // PER_PAGE + 1) * PER_PAGE
            continue
        lines += wrapped_lines(text, font, scale)

    pages = lines / PER_PAGE
    print(f"text block   {COLUMN_PT/72:g} x {PAGE_PT/72:g} in, "
          f"{PER_PAGE} lines a page at 1.5 spacing")
    print(f"Introduction to end of Methods: {lines} lines, "
          f"{pages:.1f} pages")
    print("JEI limit 10 pages, 11.5 tolerated at first submission")
    print(f"over by {pages - 10:.1f} pages "
          f"({(pages - 10) / pages * 100:.0f}% of the text)")


if __name__ == "__main__":
    main()
