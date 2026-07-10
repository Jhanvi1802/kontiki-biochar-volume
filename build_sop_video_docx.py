# Build the slow-video capture SOP as an editable Word document (.docx).
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = os.path.dirname(os.path.abspath(__file__))
FRAME = os.path.join(
    os.environ.get("TEMP", ""), ""  # placeholder; example image path set below
)
EXAMPLE_IMG = r"C:/Users/Admin/AppData/Local/Temp/claude/c--Users-Admin-OneDrive-Desktop-volume-detection/50cb09d8-7809-4cbb-a3ee-e147fb4c70ca/scratchpad/test_frames/frame_012.jpg"

RUST = RGBColor(0xA9, 0x4E, 0x28)
INK = RGBColor(0x20, 0x1C, 0x16)
GREEN = RGBColor(0x2E, 0x7D, 0x33)
RED = RGBColor(0xBE, 0x3B, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0x6A, 0x62, 0x56)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def bottom_border(paragraph, color="E4DED1", sz=8):
    pPr = paragraph._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single"); b.set(qn("w:sz"), str(sz)); b.set(qn("w:space"), "4"); b.set(qn("w:color"), color)
    pbdr.append(b); pPr.append(pbdr)


def run(p, text, *, bold=False, color=None, size=None, italic=False):
    r = p.add_run(text)
    r.bold = bold; r.italic = italic
    if color is not None: r.font.color.rgb = color
    if size is not None: r.font.size = Pt(size)
    return r


def h2(doc, num, title):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(4)
    run(p, f"{num}   ", bold=True, color=RUST, size=14)
    run(p, title, bold=True, color=RUST, size=14)
    bottom_border(p)
    return p


def bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet"); p.paragraph_format.space_after = Pt(3)
        _rich(p, it)


def numbered(doc, items):
    for i, it in enumerate(items, 1):
        p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.3); p.paragraph_format.space_after = Pt(3)
        run(p, f"{i}.  ", bold=True, color=INK)
        _rich(p, it)


def _rich(p, text):
    # split on ** for bold segments: "walk **slowly** around" -> bold 'slowly'
    for i, seg in enumerate(text.split("**")):
        if seg:
            run(p, seg, bold=(i % 2 == 1))


def table(doc, headers, rows, widths=None, header_fill="A94E28", do_dont=False):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Table Grid"
    hdr = t.rows[0].cells
    for j, htext in enumerate(headers):
        hdr[j].paragraphs[0].clear() if hdr[j].paragraphs[0].runs else None
        run(hdr[j].paragraphs[0], htext, bold=True, color=WHITE, size=10)
        shade(hdr[j], header_fill)
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            p = cells[j].paragraphs[0]
            col = None
            if do_dont and j == 1: col = GREEN
            if do_dont and j == 2: col = RED
            run(p, val, color=col, bold=(do_dont and j in (1, 2)), size=10)
    if widths:
        for j, w in enumerate(widths):
            for r in t.rows:
                r.cells[j].width = Inches(w)
    return t


doc = Document()
# base font
style = doc.styles["Normal"]; style.font.name = "Calibri"; style.font.size = Pt(10.5)
for s in doc.sections:
    s.top_margin = s.bottom_margin = Inches(0.6); s.left_margin = s.right_margin = Inches(0.8)

# ---- banner ----
bt = doc.add_table(rows=1, cols=1); bt.style = "Table Grid"
c = bt.rows[0].cells[0]; shade(c, "A94E28")
p = c.paragraphs[0]; run(p, "STANDARD OPERATING PROCEDURE", bold=True, color=WHITE, size=9)
p2 = c.add_paragraph(); run(p2, "Recording the Kiln Video for Biochar Volume", bold=True, color=WHITE, size=19)
p3 = c.add_paragraph(); run(p3, "Use: one slow video per biochar-filled kiln     Time: ~1 minute per kiln     For: Field data collectors", color=WHITE, size=9)

h2(doc, "1.", "Why this matters")
box = doc.add_table(rows=1, cols=1); box.style = "Table Grid"; shade(box.rows[0].cells[0], "FBEEE7")
bp = box.rows[0].cells[0].paragraphs[0]
_rich(bp, "The video is turned by software into a **3-D model** of the biochar, and that model is "
          "used to measure the **volume**. A slow, steady, complete video gives an accurate volume. "
          "A fast, shaky, or half-finished video gives a wrong number — or no result at all. "
          "Please follow every step exactly.")

h2(doc, "2.", "When to record")
bullets(doc, [
    "Record **after the biochar is fully quenched** (cooled with water) and the surface has settled.",
    "Record **before unloading** — while all the biochar is still inside the kiln.",
    "Record in **good daylight**. Never at night.",
    "Make sure the surface is clear — **no foam, smoke, or standing water** hiding the biochar.",
])

h2(doc, "3.", "What you need")
bullets(doc, [
    "A smartphone with a working camera and **enough free storage** for a short video.",
    "A **clean camera lens** — wipe it first (ash or dust blurs everything).",
    "**Recommended:** a **1-metre stick or measuring tape** to lay across the rim as a size reference (see Section 8).",
])

h2(doc, "4.", "Get ready (before you record)")
bullets(doc, [
    "Move **people, tools, hoses and buckets** away from the kiln — nothing should block it or move during the video.",
    "Stand so the **sun is behind you**, not facing the camera.",
    "Open the camera, switch to **Video**, and set the quality to **1080p or 4K** if your phone allows.",
    "Keep the zoom at **1× — do not zoom** at any time.",
])

h2(doc, "5.", "The technique — the slow circle")
numbered(doc, [
    "**Stand at the edge (rim)** of the kiln.",
    "**Tilt the phone down about 50–60°** so you look **into** the kiln — you must see the **black biochar surface AND the full round rim**.",
    "**Walk slowly all the way around** the kiln — a complete circle — keeping the biochar and the whole rim in view the entire time.",
    "Take about **30–40 seconds** for one full circle. Slow and smooth.",
    "Keep the kiln **filling most of the screen**.",
])
if os.path.exists(EXAMPLE_IMG):
    doc.add_picture(EXAMPLE_IMG, width=Inches(4.6))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run(cap, "Example: the kiln fills the frame with the rim and biochar visible. "
             "(Tilt a little more downward than this to see the surface better.)", italic=True, color=MUTED, size=9)

h2(doc, "6.", "The 6 golden rules")
rules = [
    ("1 · Go slow", "Move at a slow walk. Fast movement = blur = the software fails."),
    ("2 · Full circle", "Go all the way around (360°). Do not stop halfway."),
    ("3 · Look down into it", "Tilt ~50–60° down. See the black surface, not just the side."),
    ("4 · Keep the whole rim in view", "The rim is the ruler that sets the real size. Never cut it off."),
    ("5 · Hold steady", "Two hands. Move your feet smoothly — no jerks or swings."),
    ("6 · One take, no zoom", "Record in one continuous take. Never zoom in or out."),
]
rt = doc.add_table(rows=0, cols=2); rt.style = "Table Grid"
for i in range(0, 6, 2):
    cells = rt.add_row().cells
    for k in range(2):
        title, desc = rules[i + k]
        p = cells[k].paragraphs[0]; run(p, title, bold=True, color=RUST, size=11)
        dp = cells[k].add_paragraph(); run(dp, desc, size=9.5, color=MUTED)

h2(doc, "7.", "How to record — step by step")
numbered(doc, [
    "**Wipe the lens.** Open the camera, switch to **Video**, set **1080p/4K**, zoom at **1×**.",
    "**Stand at the rim** and tilt down ~50–60°. Check you can see the **biochar + the full rim**.",
    "**Press record** and hold still for **2 seconds** before you start moving.",
    "**Walk slowly all the way around** (~30–40 seconds), keeping the surface and full rim in frame the whole time.",
    "**Optional but better:** do a **second slow loop** at a slightly different height.",
    "**Stop recording.** Play it back and check it against Section 9.",
    "Record **2 videos** of the same kiln, in case one comes out shaky.",
])

h2(doc, "8.", "Add a size reference (recommended)")
p = doc.add_paragraph()
_rich(p, "Before recording, lay a **1-metre stick or an open measuring tape across the rim**, so it is "
         "clearly visible in the video. This gives the software an exact, known size and makes the volume "
         "more reliable. If you have none, the software can use the kiln's rim instead — but a 1-metre "
         "marker is **strongly recommended** for the most accurate result.")

h2(doc, "9.", "Check before you submit (must pass EVERY point)")
checks = [
    "You recorded a full circle — all the way around the kiln.",
    "The whole round rim is visible for the entire video.",
    "All the black biochar surface is visible (not hidden behind the rim).",
    "The video is smooth and sharp — not shaky, not blurry.",
    "Bright daylight — no deep shadow and no strong glare on the biochar.",
    "You did not zoom, and it is one continuous take.",
    "(If used) the 1-metre reference is visible in the video.",
]
ct = doc.add_table(rows=0, cols=2); ct.style = "Table Grid"
for chk in checks:
    cells = ct.add_row().cells
    cells[0].width = Inches(0.4)
    run(cells[0].paragraphs[0], "☐", size=13)
    run(cells[1].paragraphs[0], chk, size=10)
warn = doc.add_paragraph(); run(warn, "If any point fails — delete it and record again.", bold=True, color=RED)

h2(doc, "10.", "Do & Don't")
table(doc,
      ["Situation", "DO", "DON'T"],
      [["Speed", "Move at a slow, steady walk", "Rush or swing the phone (causes blur)"],
       ["Coverage", "Complete a full 360° circle", "Stop halfway or skip a side"],
       ["Angle", "Tilt down ~50–60° into the kiln", "Shoot flat from the side (rim hides biochar)"],
       ["The rim", "Keep the whole rim in view always", "Let any part of the rim leave the frame"],
       ["Zoom", "Keep zoom fixed at 1×", "Zoom in or out during the video"],
       ["Light", "Daylight, sun behind you", "Night, deep shadow, or facing the sun"],
       ["Scene", "One kiln, nothing moving", "People/tools moving through the shot"]],
      widths=[1.4, 2.6, 2.9], do_dont=True)

h2(doc, "11.", "Common mistakes and how to fix them")
table(doc,
      ["Mistake", "Why it's wrong", "Fix"],
      [["Moved too fast", "Frames are blurry — 3-D model fails", "Slow down; 30–40 s for one circle"],
       ["Did not go all the way around", "Part of the surface is never seen", "Always complete the full circle"],
       ["Rim went out of frame", "No size reference — size becomes wrong", "Step back; keep the full rim visible"],
       ["Zoomed during the video", "Confuses the 3-D software", "Keep zoom at 1× the whole time"],
       ["Shot from the side", "Rim hides the biochar surface", "Stand higher and tilt down into the kiln"],
       ["Too dark / glare", "Biochar not clearly visible", "Record in daylight, sun behind you"]],
      widths=[1.9, 2.7, 2.3])

foot = doc.add_paragraph(); foot.paragraph_format.space_before = Pt(16)
run(foot, "Record one good video for every biochar-filled kiln, following this procedure. "
          "A correct video takes about a minute and gives an accurate biochar volume. "
          "Questions? Contact your supervisor.   ·   SOP-KILN-VIDEO-01  v1.0", color=MUTED, size=9)

out = os.path.join(BASE, "SOP_Kiln_Video.docx")
doc.save(out)
print("wrote", out)
