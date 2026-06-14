"""
generate_pdf_reports.py
=======================
Converts TECHNICAL_REPORT.md and PROJECT_REPORT.md to polished PDFs.
Libraries used: markdown, lxml, reportlab (all already installed).

Run from project root:
    python generate_pdf_reports.py
Outputs:
    TECHNICAL_REPORT.pdf
    PROJECT_REPORT.pdf
"""

import re
import sys
from pathlib import Path
from datetime import datetime

import markdown as md_lib
from lxml import html as lhtml
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Preformatted, HRFlowable, KeepTogether, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas as rl_canvas

ROOT = Path(__file__).resolve().parent

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


# ══════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════

def make_styles():
    base = getSampleStyleSheet()

    def add(name, **kw):
        if name not in base:
            base.add(ParagraphStyle(name=name, **kw))
        return base[name]

    add('CoverTitle',
        fontSize=28, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=10, spaceBefore=0,
        leading=34, alignment=TA_CENTER)

    add('CoverSubtitle',
        fontSize=14, fontName='Helvetica',
        textColor=colors.HexColor('#334155'),
        spaceAfter=6, spaceBefore=0,
        leading=20, alignment=TA_CENTER)

    add('CoverMeta',
        fontSize=11, fontName='Helvetica',
        textColor=colors.HexColor('#64748b'),
        spaceAfter=4, spaceBefore=0,
        leading=16, alignment=TA_CENTER)

    add('H1',
        fontSize=18, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6, spaceBefore=18,
        leading=24)

    add('H2',
        fontSize=15, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1e3a5f'),
        spaceAfter=5, spaceBefore=14,
        leading=20)

    add('H3',
        fontSize=13, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#374151'),
        spaceAfter=4, spaceBefore=10,
        leading=17)

    add('H4',
        fontSize=11, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#4b5563'),
        spaceAfter=3, spaceBefore=8,
        leading=15)

    add('H5',
        fontSize=10, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#6b7280'),
        spaceAfter=2, spaceBefore=6,
        leading=14)

    add('H6',
        fontSize=10, fontName='Helvetica-BoldOblique',
        textColor=colors.HexColor('#6b7280'),
        spaceAfter=2, spaceBefore=4,
        leading=14)

    add('Body',
        fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=5, spaceBefore=2,
        leading=15, wordWrap='LTR')

    add('BulletItem',
        fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=2, spaceBefore=0,
        leading=14, leftIndent=16, bulletIndent=4)

    add('BulletItem2',
        fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#374151'),
        spaceAfter=2, spaceBefore=0,
        leading=14, leftIndent=32, bulletIndent=20)

    add('BulletItem3',
        fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#374151'),
        spaceAfter=2, spaceBefore=0,
        leading=14, leftIndent=48, bulletIndent=36)

    add('NumberItem',
        fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=2, spaceBefore=0,
        leading=14, leftIndent=16, bulletIndent=4)

    add('CodeBlock',
        fontSize=8, fontName='Courier',
        textColor=colors.HexColor('#1e293b'),
        backColor=colors.HexColor('#f1f5f9'),
        spaceAfter=8, spaceBefore=4,
        leading=11, leftIndent=8, rightIndent=8,
        borderPad=6, borderColor=colors.HexColor('#cbd5e1'),
        borderWidth=0.5)

    add('InlineCode',
        fontSize=9, fontName='Courier',
        textColor=colors.HexColor('#be185d'),
        leading=13)

    add('TableHdr',
        fontSize=9, fontName='Helvetica-Bold',
        textColor=colors.white,
        leading=12, leftIndent=0)

    add('TableCell',
        fontSize=9, fontName='Helvetica',
        textColor=colors.HexColor('#1f2937'),
        leading=12, leftIndent=0)

    add('TableCellCode',
        fontSize=8, fontName='Courier',
        textColor=colors.HexColor('#1e293b'),
        leading=11, leftIndent=0)

    add('BlockQuote',
        fontSize=10, fontName='Helvetica-Oblique',
        textColor=colors.HexColor('#374151'),
        spaceAfter=6, spaceBefore=4,
        leading=15, leftIndent=20)

    add('FooterStyle',
        fontSize=8, fontName='Helvetica',
        textColor=colors.HexColor('#94a3b8'),
        leading=10, alignment=TA_CENTER)

    return base


# ══════════════════════════════════════════════════════════════════
# INLINE MARKUP CONVERTER (element → reportlab XML)
# ══════════════════════════════════════════════════════════════════

def _esc(text: str) -> str:
    """Escape characters that break reportlab Paragraph XML."""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text


def _inline(el) -> str:
    """
    Recursively convert an lxml element's content (text + inline children)
    into a reportlab-compatible XML string.
    Block-level children are skipped (they should have been extracted earlier).
    """
    BLOCK = {'p', 'div', 'ul', 'ol', 'li', 'table', 'thead', 'tbody',
             'tr', 'th', 'td', 'pre', 'blockquote', 'h1', 'h2', 'h3',
             'h4', 'h5', 'h6', 'hr', 'br'}

    parts = [_esc(el.text or '')]

    for child in el:
        tag = child.tag.lower() if isinstance(child.tag, str) else ''

        if tag in BLOCK:
            parts.append(_esc(child.tail or ''))
            continue

        inner = _inline(child)

        if tag in ('strong', 'b'):
            parts.append(f'<b>{inner}</b>')
        elif tag in ('em', 'i'):
            parts.append(f'<i>{inner}</i>')
        elif tag in ('u',):
            parts.append(f'<u>{inner}</u>')
        elif tag in ('s', 'del', 'strike'):
            parts.append(f'<strike>{inner}</strike>')
        elif tag == 'code':
            parts.append(f'<font name="Courier" size="8" color="#be185d">{inner}</font>')
        elif tag in ('a',):
            parts.append(inner)
        elif tag == 'br':
            parts.append('<br/>')
        elif tag in ('span', 'sup', 'sub'):
            parts.append(inner)
        else:
            parts.append(inner)

        parts.append(_esc(child.tail or ''))

    return ''.join(parts)


# ══════════════════════════════════════════════════════════════════
# BLOCK ELEMENT → FLOWABLES
# ══════════════════════════════════════════════════════════════════

def process_list(el, styles, depth=0):
    """Convert <ul>/<ol> element tree to list of flowables."""
    tag = el.tag.lower()
    items = []
    counter = 0

    style_map = {0: 'BulletItem', 1: 'BulletItem2', 2: 'BulletItem3'}
    bullet_style = style_map.get(depth, 'BulletItem3')

    for li in el:
        if li.tag.lower() != 'li':
            continue
        counter += 1

        # Collect direct text content of this <li> (before any nested list)
        text_parts = [_esc(li.text or '')]
        nested_lists = []

        for child in li:
            c_tag = child.tag.lower()
            if c_tag in ('ul', 'ol'):
                nested_lists.append(child)
            else:
                text_parts.append(_inline(child))
            text_parts.append(_esc(child.tail or ''))

        text = ''.join(text_parts).strip()
        if not text:
            text = '&nbsp;'

        if tag == 'ul':
            bullet = '•'
        else:
            bullet = f'{counter}.'

        para = Paragraph(f'<bullet>{_esc(bullet)}</bullet>{text}',
                         styles[bullet_style])
        items.append(para)

        # Process nested lists
        for nested in nested_lists:
            items.extend(process_list(nested, styles, depth + 1))

    return items


def process_table(el, styles):
    """Convert <table> element to a reportlab Table flowable."""
    data = []
    is_header_row = []

    # Handle thead / tbody / tr at any depth
    all_rows = []
    thead = el.find('.//thead')
    tbody = el.find('.//tbody')

    if thead is not None:
        for tr in thead.findall('.//tr'):
            all_rows.append((tr, True))
    if tbody is not None:
        for tr in tbody.findall('.//tr'):
            all_rows.append((tr, False))
    # Fallback: rows directly in table
    if not all_rows:
        for i, tr in enumerate(el.findall('.//tr')):
            all_rows.append((tr, i == 0))

    for tr, is_hdr in all_rows:
        row = []
        for cell in tr:
            c_tag = cell.tag.lower()
            text = _inline(cell)
            if c_tag == 'th' or is_hdr:
                row.append(Paragraph(text, styles['TableHdr']))
            else:
                # Detect code-heavy cells
                raw = ''.join(cell.itertext())
                if raw.count('\n') > 1:
                    row.append(Paragraph(text, styles['TableCellCode']))
                else:
                    row.append(Paragraph(text, styles['TableCell']))
        if row:
            data.append(row)
            is_header_row.append(is_hdr or (len(data) == 1))

    if not data:
        return []

    n_cols = max(len(row) for row in data)
    # Pad short rows
    for row in data:
        while len(row) < n_cols:
            row.append(Paragraph('', styles['TableCell']))

    col_w = CONTENT_W / n_cols

    tbl = Table(data, colWidths=[col_w] * n_cols, repeatRows=1,
                hAlign='LEFT', splitByRow=True)

    n_header = sum(1 for h in is_header_row if h)

    ts = TableStyle([
        # Header rows
        ('BACKGROUND', (0, 0), (-1, n_header - 1),
         colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, n_header - 1), colors.white),
        ('FONTNAME', (0, 0), (-1, n_header - 1), 'Helvetica-Bold'),
        # Alternating body rows
        ('ROWBACKGROUNDS', (0, n_header), (-1, -1),
         [colors.white, colors.HexColor('#f8fafc')]),
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#e2e8f0')),
        ('LINEBELOW', (0, n_header - 1), (-1, n_header - 1), 1,
         colors.HexColor('#334155')),
        # Padding
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ])
    tbl.setStyle(ts)
    return [Spacer(1, 3 * mm), tbl, Spacer(1, 3 * mm)]


def process_element(el, styles, flowables, list_depth=0):
    """Convert one lxml block element into reportlab flowables."""
    tag = el.tag.lower() if isinstance(el.tag, str) else ''

    # ── Headings ────────────────────────────────────────────────
    if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
        level = int(tag[1])
        h_styles = {1: 'H1', 2: 'H2', 3: 'H3', 4: 'H4', 5: 'H5', 6: 'H6'}
        text = _inline(el)

        group = []
        if level == 1:
            group.append(Spacer(1, 4 * mm))
            group.append(HRFlowable(
                width='100%', thickness=2,
                color=colors.HexColor('#0f172a'), spaceAfter=2))
        elif level == 2:
            group.append(Spacer(1, 2 * mm))

        group.append(Paragraph(text, styles[h_styles[level]]))

        if level == 1:
            group.append(HRFlowable(
                width='100%', thickness=0.5,
                color=colors.HexColor('#94a3b8'), spaceAfter=4))

        flowables.append(KeepTogether(group))

    # ── Paragraph ───────────────────────────────────────────────
    elif tag == 'p':
        text = _inline(el)
        if text.strip():
            flowables.append(Paragraph(text, styles['Body']))

    # ── Horizontal rule ─────────────────────────────────────────
    elif tag == 'hr':
        flowables.append(Spacer(1, 3 * mm))
        flowables.append(HRFlowable(
            width='100%', thickness=1,
            color=colors.HexColor('#94a3b8'), spaceAfter=3))
        flowables.append(Spacer(1, 3 * mm))

    # ── Code block ──────────────────────────────────────────────
    elif tag == 'pre':
        code_el = el.find('code')
        source = code_el if code_el is not None else el
        raw = ''.join(source.itertext())
        # Remove leading/trailing blank lines
        lines = raw.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        code_text = '\n'.join(lines)
        flowables.append(Preformatted(code_text, styles['CodeBlock']))

    # ── Unordered / ordered list ─────────────────────────────────
    elif tag in ('ul', 'ol'):
        items = process_list(el, styles, list_depth)
        if items:
            group = [Spacer(1, 1 * mm)] + items + [Spacer(1, 2 * mm)]
            flowables.extend(group)

    # ── Table ────────────────────────────────────────────────────
    elif tag == 'table':
        tbl_flowables = process_table(el, styles)
        flowables.extend(tbl_flowables)

    # ── Blockquote ───────────────────────────────────────────────
    elif tag == 'blockquote':
        for child in el:
            text = _inline(child)
            if text.strip():
                flowables.append(Paragraph(text, styles['BlockQuote']))

    # ── Div / section (recurse) ──────────────────────────────────
    elif tag in ('div', 'section', 'article', 'body', 'html'):
        for child in el:
            process_element(child, styles, flowables, list_depth)

    # ── Everything else: try to extract text ─────────────────────
    else:
        text = _inline(el)
        if text.strip():
            flowables.append(Paragraph(text, styles['Body']))


# ══════════════════════════════════════════════════════════════════
# MARKDOWN → FLOWABLES
# ══════════════════════════════════════════════════════════════════

def markdown_to_flowables(md_text: str, styles) -> list:
    """Parse full markdown text and return a list of reportlab flowables."""

    # Convert markdown → HTML
    converter = md_lib.Markdown(extensions=[
        'tables',
        'fenced_code',
        'nl2br',
        'sane_lists',
        'attr_list',
    ])
    html_body = converter.convert(md_text)

    # Wrap in full HTML document for lxml
    full_html = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html><body>' + html_body + '</body></html>'
    )
    tree = lhtml.document_fromstring(full_html.encode('utf-8'))
    body = tree.find('.//body')

    flowables = []
    if body is not None:
        for child in body:
            process_element(child, styles, flowables)

    return flowables


# ══════════════════════════════════════════════════════════════════
# PAGE TEMPLATE (header / footer / page numbers)
# ══════════════════════════════════════════════════════════════════

def make_page_template(title: str, date_str: str, is_confidential: bool):
    """Return a canvas callback that draws header/footer on every page."""

    def _draw(canvas, doc):
        canvas.saveState()

        # ── Footer ───────────────────────────────────────────────
        footer_y = MARGIN - 12 * mm

        # Left: title (shortened)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#94a3b8'))
        short = title[:60] + ('…' if len(title) > 60 else '')
        canvas.drawString(MARGIN, footer_y, short)

        # Centre: page number
        page_num = f'Page {doc.page}'
        canvas.drawCentredString(PAGE_W / 2, footer_y, page_num)

        # Right: date
        canvas.drawRightString(PAGE_W - MARGIN, footer_y, date_str)

        # Footer line
        canvas.setStrokeColor(colors.HexColor('#e2e8f0'))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, footer_y + 4 * mm, PAGE_W - MARGIN, footer_y + 4 * mm)

        # Confidential watermark
        if is_confidential:
            canvas.saveState()
            canvas.setFont('Helvetica-Bold', 60)
            canvas.setFillColor(colors.Color(0.9, 0.9, 0.9, alpha=0.3))
            canvas.translate(PAGE_W / 2, PAGE_H / 2)
            canvas.rotate(45)
            canvas.drawCentredString(0, 0, 'CONFIDENTIAL')
            canvas.restoreState()

        canvas.restoreState()

    return _draw


# ══════════════════════════════════════════════════════════════════
# COVER PAGE
# ══════════════════════════════════════════════════════════════════

def make_cover(title: str, subtitle: str, date_str: str,
               is_confidential: bool, styles) -> list:
    flowables = [Spacer(1, 6 * cm)]

    # Accent bar
    flowables.append(HRFlowable(
        width='60%', thickness=4,
        color=colors.HexColor('#1e3a5f'),
        hAlign='CENTER', spaceAfter=12))

    flowables.append(Paragraph(title, styles['CoverTitle']))
    flowables.append(Spacer(1, 4 * mm))
    flowables.append(Paragraph(subtitle, styles['CoverSubtitle']))
    flowables.append(Spacer(1, 8 * mm))

    flowables.append(HRFlowable(
        width='40%', thickness=1,
        color=colors.HexColor('#94a3b8'),
        hAlign='CENTER', spaceAfter=10))

    flowables.append(Spacer(1, 4 * mm))
    flowables.append(Paragraph(f'Generated: {date_str}', styles['CoverMeta']))

    if is_confidential:
        flowables.append(Spacer(1, 4 * mm))
        flowables.append(Paragraph(
            '<b><font color="#dc2626">CONFIDENTIAL — Personal Use Only</font></b>',
            styles['CoverMeta']))

    flowables.append(Spacer(1, 8 * mm))
    flowables.append(HRFlowable(
        width='60%', thickness=4,
        color=colors.HexColor('#1e3a5f'),
        hAlign='CENTER', spaceAfter=0))

    flowables.append(PageBreak())
    return flowables


# ══════════════════════════════════════════════════════════════════
# MAIN PDF BUILDER
# ══════════════════════════════════════════════════════════════════

def build_pdf(md_path: Path, pdf_path: Path,
              title: str, subtitle: str,
              is_confidential: bool = False):
    print(f'\n  [{md_path.name}]')
    print(f'   Reading markdown ...', end=' ', flush=True)
    md_text = md_path.read_text(encoding='utf-8')
    print(f'{len(md_text):,} chars, '
          f'{md_text.count(chr(10)):,} lines')

    styles = make_styles()
    date_str = datetime.now().strftime('%d %B %Y')

    print('   Converting markdown to flowables ...', end=' ', flush=True)

    # Remove the H1 title from body (it goes on the cover page)
    lines = md_text.split('\n')
    body_lines = []
    skip_first_h1 = True
    for line in lines:
        if skip_first_h1 and line.startswith('# '):
            skip_first_h1 = False
            continue
        body_lines.append(line)
    body_md = '\n'.join(body_lines)

    content_flowables = markdown_to_flowables(body_md, styles)
    print(f'{len(content_flowables)} flowables')

    # Full document = cover + content
    cover = make_cover(title, subtitle, date_str, is_confidential, styles)
    all_flowables = cover + content_flowables

    print(f'   Writing PDF -> {pdf_path.name} ...', end=' ', flush=True)
    page_cb = make_page_template(title, date_str, is_confidential)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN + 8 * mm,   # extra for footer
        title=title,
        author='NVH AI Platform',
        subject=subtitle,
        creator='generate_pdf_reports.py',
    )
    doc.build(all_flowables, onFirstPage=page_cb, onLaterPages=page_cb)

    size_kb = pdf_path.stat().st_size / 1024
    print(f'done  ({size_kb:.0f} KB, {doc.page} pages)')
    return pdf_path


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

REPORTS = [
    {
        'md':            ROOT / 'TECHNICAL_REPORT.md',
        'pdf':           ROOT / 'TECHNICAL_REPORT.pdf',
        'title':         'AI-Powered NVH Acoustic Optimisation Platform',
        'subtitle':      'Microwave Oven Noise Reduction - Comprehensive Technical Report',
        'confidential':  False,
    },
    {
        'md':            ROOT / 'PROJECT_REPORT.md',
        'pdf':           ROOT / 'PROJECT_REPORT.pdf',
        'title':         'NVH AI Platform - Project Deep-Dive & Interview Preparation',
        'subtitle':      'Personal Reference | All Phases | Technical Concepts | Q&A',
        'confidential':  True,
    },
]


def main():
    print('=' * 62)
    print('  NVH Report Generator -- Markdown to PDF')
    print('=' * 62)

    generated = []
    for cfg in REPORTS:
        md_path = cfg['md']
        if not md_path.exists():
            print(f'\n  [SKIP] {md_path.name} not found')
            continue
        pdf_path = cfg['pdf']
        build_pdf(
            md_path, pdf_path,
            title=cfg['title'],
            subtitle=cfg['subtitle'],
            is_confidential=cfg['confidential'],
        )
        generated.append(pdf_path)

    print()
    print('=' * 62)
    print(f'  Done - {len(generated)} PDF(s) created:')
    for p in generated:
        size_kb = p.stat().st_size / 1024
        print(f'    {p.name}  ({size_kb:.0f} KB)')
    print('=' * 62)


if __name__ == '__main__':
    main()
