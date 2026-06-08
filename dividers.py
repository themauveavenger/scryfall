import os
import time
from datetime import datetime
from decimal import Decimal
from os import makedirs, path

import click
import httpx
from fpdf import FPDF

from oracle_db import OracleDB

# ── Dimensions ──────────────────────────────────────────────────────────────
DIVIDER_WIDTH_MM = Decimal("66.04")   # 2.6"
DIVIDER_HEIGHT_MM = Decimal("95.25")  # 3.75"
PAGE_WIDTH_MM = Decimal("215.9")      # 8.5"
PAGE_HEIGHT_MM = Decimal("279.4")     # 11"
GAP_MM = Decimal("0.2")

TOTAL_DIVIDERS_WIDTH = (Decimal("3") * DIVIDER_WIDTH_MM) + (Decimal("2") * GAP_MM)
TOTAL_DIVIDERS_HEIGHT = (Decimal("2") * DIVIDER_HEIGHT_MM) + GAP_MM

START_X = (PAGE_WIDTH_MM - TOTAL_DIVIDERS_WIDTH) / Decimal("2")
START_Y = (PAGE_HEIGHT_MM - TOTAL_DIVIDERS_HEIGHT) / Decimal("2")

# Tab (file-folder style)
TAB_HEIGHT = Decimal("14")       # total height of the tab strip
TAB_PADDING = Decimal("2")       # padding inside tab
SVG_SIZE = Decimal("10")         # symbol is square (viewBox 800×800)
TAB_TEXT_FONT_PT = 28            # visually ~same height as SVG
BODY_CONTENT_HEIGHT = Decimal("20")  # approx: name + date + counts
SVG_CACHE_DIR = "./set_symbols"
PDF_BASE_DIR = "./pdfs/dividers"

ALLOWED_SET_TYPES = {
    "core",
    "expansion",
    "commander",
    "masters",
    "draft_innovation",
}
EXCLUDED_SET_CODES = {"plst"}


# ── Data helpers ────────────────────────────────────────────────────────────

def fetch_sets_from_api() -> dict[str, dict]:
    """Fetch all sets from Scryfall and return a dict keyed by lowercase code."""
    resp = httpx.get("https://api.scryfall.com/sets", timeout=30)
    resp.raise_for_status()
    return {s["code"].lower(): s for s in resp.json()["data"]}


def filter_sets(sets: list[dict]) -> list[dict]:
    """Keep only sets released since 2021-01-01 that match our criteria."""
    cutoff = datetime(2021, 1, 1)
    result = []
    for s in sets:
        released_at = s.get("released_at")
        if not released_at:
            continue
        try:
            release_date = datetime.strptime(released_at, "%Y-%m-%d")
        except ValueError:
            continue
        if release_date < cutoff or release_date > datetime.now():
            continue
        code = s.get("code", "").lower()
        if code in EXCLUDED_SET_CODES:
            continue
        if s.get("set_type", "") not in ALLOWED_SET_TYPES:
            continue
        result.append(s)
    # newest first
    result.sort(key=lambda s: s["released_at"], reverse=True)
    return result


def download_svg(icon_svg_uri: str, code: str) -> str | None:
    """Download and cache an SVG set symbol. Returns the local path or None."""
    cache_path = path.join(SVG_CACHE_DIR, f"{code}.svg")
    if path.exists(cache_path):
        return cache_path
    try:
        resp = httpx.get(icon_svg_uri, timeout=30)
        resp.raise_for_status()
        makedirs(SVG_CACHE_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        time.sleep(0.1)  # be polite to Scryfall's CDN
        return cache_path
    except Exception as exc:
        click.echo(f"Warning: failed to download SVG for {code}: {exc}")
        return None


def format_release_date(released_at: str) -> str:
    dt = datetime.strptime(released_at, "%Y-%m-%d")
    return dt.strftime("%B %Y")


# ── PDF generation ──────────────────────────────────────────────────────────

def generate_dividers_pdf(sets_data: list[dict], db: OracleDB) -> str:
    pdf = FPDF(orientation="P", unit="mm", format="letter")
    pdf.set_margin(0)
    pdf.set_auto_page_break(False)
    pdf.add_page()

    current_divider = 0
    total = len(sets_data)

    for s in sets_data:
        col = current_divider % 3
        row = (current_divider // 3) % 2

        x = START_X + Decimal(col) * (DIVIDER_WIDTH_MM + GAP_MM)
        y = START_Y + Decimal(row) * (DIVIDER_HEIGHT_MM + GAP_MM)

        # light border to guide cutting
        pdf.set_draw_color(180, 180, 180)
        pdf.rect(
            float(x),
            float(y),
            float(DIVIDER_WIDTH_MM),
            float(DIVIDER_HEIGHT_MM),
        )

        # ── Tab strip: symbol + set code, same horizontal line ──
        code = s["code"].upper()
        svg_path = download_svg(s["icon_svg_uri"], code.lower())

        svg_x = x + TAB_PADDING
        svg_y = y + TAB_PADDING

        if svg_path:
            try:
                pdf.image(
                    svg_path,
                    x=float(svg_x),
                    y=float(svg_y),
                    w=float(SVG_SIZE),
                    h=float(SVG_SIZE),
                )
            except Exception as exc:
                click.echo(f"Warning: failed to render SVG for {code}: {exc}")

        # set code text to the right of the symbol, same visual height
        text_x = svg_x + SVG_SIZE + TAB_PADDING
        text_y = svg_y
        pdf.set_xy(float(text_x), float(text_y))
        pdf.set_font("Times", "B", TAB_TEXT_FONT_PT)
        pdf.cell(
            float(DIVIDER_WIDTH_MM - SVG_SIZE - TAB_PADDING * 3),
            float(SVG_SIZE),
            code,
            align="L",
        )

        # separator line between tab and body
        tab_line_y = float(y + TAB_HEIGHT)
        pdf.line(float(x), tab_line_y, float(x + DIVIDER_WIDTH_MM), tab_line_y)

        # ── Body: name, date, rarity counts ──
        # center body content in the space below the tab
        body_top = (
            y
            + TAB_HEIGHT
            + (DIVIDER_HEIGHT_MM - TAB_HEIGHT - BODY_CONTENT_HEIGHT) / Decimal("2")
        )
        text_width = float(DIVIDER_WIDTH_MM - Decimal("4"))
        content_x = float(x + Decimal("2"))

        # set name (shrink font if necessary)
        set_name = s["name"]
        pdf.set_xy(content_x, float(body_top))
        pdf.set_font("Times", "B", 10)
        while pdf.get_string_width(set_name) > text_width and pdf.font_size_pt > 6:
            pdf.set_font("Times", "B", pdf.font_size_pt - 1)
        pdf.cell(text_width, 5, set_name, align="C")

        # release month & year
        pdf.set_xy(content_x, float(body_top + Decimal("6")))
        pdf.set_font("Times", "", 9)
        pdf.cell(text_width, 5, format_release_date(s["released_at"]), align="C")

        # rarity counts
        rarity_counts = db.get_set_rarity_counts(code.lower())
        counts_str = (
            f"C: {rarity_counts['common']}   "
            f"U: {rarity_counts['uncommon']}   "
            f"R: {rarity_counts['rare']}   "
            f"M: {rarity_counts['mythic']}"
        )
        pdf.set_xy(content_x, float(body_top + Decimal("12")))
        pdf.set_font("Times", "", 8)
        pdf.cell(text_width, 4, counts_str, align="C")

        current_divider += 1

        # start a new page after every 6 dividers
        if current_divider % 6 == 0 and current_divider < total:
            pdf.add_page()

    # write file
    if not path.exists(PDF_BASE_DIR):
        makedirs(PDF_BASE_DIR, exist_ok=True)
    filename = f"dividers-{int(datetime.now().timestamp())}.pdf"
    full_path = path.join(PDF_BASE_DIR, filename)
    click.echo(f"Writing PDF: {full_path}")
    pdf.output(full_path)
    return full_path


# ── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("set_codes", nargs=-1)
def dividers(set_codes: tuple[str, ...]):
    """Generate MTG set divider PDFs.

    With no arguments, generates dividers for all sets released since 2021
    that match the standard / commander / masters / draft-innovation criteria.

    Pass one or more set codes (e.g.  blb dsk fdh ) to generate only those sets.
    """
    db = OracleDB()
    all_api_sets = fetch_sets_from_api()

    if set_codes:
        selected = []
        for raw_code in set_codes:
            code = raw_code.lower().strip()
            s = all_api_sets.get(code)
            if s is None:
                click.echo(f"Warning: set '{raw_code}' not found in Scryfall API, skipping")
                continue
            selected.append(s)
    else:
        click.echo("Fetching sets from Scryfall API...")
        selected = filter_sets(all_api_sets.values())
        click.echo(f"Found {len(selected)} sets matching criteria.")

    # Filter out unreleased and empty sets (applies to both auto & explicit)
    today = datetime.now()
    valid_selected = []
    for s in selected:
        released_at = s.get("released_at")
        if not released_at:
            continue
        try:
            release_date = datetime.strptime(released_at, "%Y-%m-%d")
        except ValueError:
            continue
        if release_date > today:
            click.echo(f"Skipping unreleased set: {s['code'].upper()} ({s['name']})")
            continue
        if not db.has_set(s["code"]):
            click.echo(f"Skipping empty set: {s['code'].upper()} ({s['name']})")
            continue
        valid_selected.append(s)
    selected = valid_selected

    if not selected:
        click.echo("No sets to generate.")
        return

    generate_dividers_pdf(selected, db)


if __name__ == "__main__":
    dividers()
