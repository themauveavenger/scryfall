import os
import time
from datetime import datetime
from decimal import Decimal
from os import makedirs, path

import click
import httpx
from fpdf import FPDF
from PIL import Image, ImageDraw

from oracle_db import OracleDB

import re as _re

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
GRADIENT_CACHE_DIR = "./set_symbols/gradients"
PDF_BASE_DIR = "./pdfs/dividers"

# ── Theme colors ───────────────────────────────────────────────────────────
# RGB tuples for well-known sets. Fallback uses keyword heuristics.
SET_THEME_COLORS: dict[str, tuple[int, int, int]] = {
    "otj": (210, 170, 80),    # Outlaws of Thunder Junction – sandy gold western
    "big": (210, 170, 80),    # The Big Score – same western theme
    "blb": (80, 140, 60),     # Bloomburrow – woodland green
    "dsk": (60, 40, 80),      # Duskmourn – dark horror purple
    "fdn": (40, 80, 140),     # Foundations – classic blue
    "mkm": (100, 40, 60),     # Murders at Karlov Manor – dark red mystery
    "lci": (100, 80, 50),     # Lost Caverns of Ixalan – earthy cave brown
    "woe": (140, 80, 140),    # Wilds of Eldraine – fairy tale purple
    "mom": (50, 70, 50),      # March of the Machine – Phyrexian black-green
    "one": (40, 80, 40),      # Phyrexia: All Will Be One – compleation green
    "bro": (160, 90, 50),     # Brothers' War – rusty artifact brown
    "dmu": (50, 80, 140),     # Dominaria United – classic blue
    "snc": (180, 150, 60),    # Streets of New Capenna – art deco gold
    "neo": (60, 120, 180),    # Kamigawa: Neon Dynasty – neon blue
    "vow": (120, 30, 40),     # Innistrad: Crimson Vow – vampire red
    "mid": (180, 100, 30),    # Innistrad: Midnight Hunt – werewolf orange
    "afr": (60, 50, 120),     # Adventures in the Forgotten Realms – D&D dark blue
    "stx": (80, 60, 150),     # Strixhaven – magic school purple
    "khm": (100, 140, 180),   # Kaldheim – icy blue
    "mh3": (100, 60, 150),    # Modern Horizons 3 – purple/blue
    "mh2": (100, 60, 140),    # Modern Horizons 2 – purple
    "cmm": (180, 150, 50),    # Commander Masters – gold
    "2x2": (180, 150, 60),    # Double Masters 2022 – gold
    "dmr": (50, 80, 140),     # Dominaria Remastered – blue
    "inr": (80, 50, 80),      # Innistrad Remastered – horror purple
    "pio": (100, 80, 150),    # Pioneer Masters – purple
    "rvr": (180, 160, 80),    # Ravnica Remastered – guild gold
    "tsr": (80, 60, 150),     # Time Spiral Remastered – time purple
    "mb2": (120, 60, 140),    # Mystery Booster 2 – mystery purple
    "ltr": (80, 120, 60),     # Lord of the Rings – LOTR green
    "clb": (50, 60, 100),     # Commander Legends: Baldur's Gate – D&D dark blue
    "j22": (120, 100, 60),    # Jumpstart 2022 – neutral gold
    "j21": (100, 100, 100),   # Jumpstart: Historic Horizons – neutral grey
    "dbl": (100, 100, 100),   # Innistrad: Double Feature – neutral grey
    "h1r": (100, 60, 140),    # MH1 Timeshifts – purple
    "h2r": (100, 60, 140),    # MH2 Timeshifts – purple
    "clu": (100, 80, 120),    # Ravnica: Clue Edition – mystery purple
    # Commander
    "ltc": (80, 120, 60),     # Tales of Middle-earth Commander – LOTR green
    "who": (30, 50, 100),     # Doctor Who – sci-fi dark blue
    "woc": (140, 80, 140),    # Wilds of Eldraine Commander – purple
    "moc": (50, 70, 50),      # March of the Machine Commander – black-green
    "onc": (40, 80, 40),      # Phyrexia Commander – green
    "pip": (200, 60, 40),     # Fallout – post-apoc rusty red
    "mkc": (100, 40, 60),     # Murders at Karlov Manor Commander – dark red
    "lcc": (100, 80, 50),     # Lost Caverns Commander – cave brown
    "drc": (180, 160, 40),    # Aetherdrift Commander – desert gold
    "tdc": (180, 160, 40),    # Tarkir: Dragonstorm Commander – desert gold
    "fic": (80, 120, 160),    # Final Fantasy Commander – blue fantasy
    "eoc": (80, 120, 160),    # Edge of Eternities Commander – blue
    "ecc": (100, 80, 120),    # Lorwyn Eclipsed Commander – dark purple
    "soc": (80, 60, 150),     # Secrets of Strixhaven Commander – purple
    "msc": (180, 40, 40),     # Marvel Super Heroes Commander – red
    "hoc": (80, 120, 60),     # The Hobbit Commander – green
    "trc": (60, 100, 160),    # Star Trek Commander – space blue
    "40k": (40, 70, 40),      # Warhammer 40k – dark green
    "dmc": (50, 80, 140),     # Dominaria United Commander – blue
    "ncc": (180, 150, 60),    # New Capenna Commander – gold
    "nec": (60, 120, 180),    # Neon Dynasty Commander – blue
    "voc": (120, 30, 40),     # Crimson Vow Commander – red
    "mic": (180, 100, 30),    # Midnight Hunt Commander – orange
    "afc": (60, 50, 120),     # Forgotten Realms Commander – dark blue
    "c21": (80, 60, 140),     # Commander 2021 – purple
    "khc": (100, 140, 180),   # Kaldheim Commander – icy blue
    "brc": (160, 90, 50),     # Brothers' War Commander – rusty brown
    "scd": (80, 60, 140),     # Starter Commander Decks – purple
    "blc": (80, 140, 60),     # Bloomburrow Commander – green
    "otc": (210, 170, 80),    # Outlaws Commander – sandy gold
    "m3c": (100, 60, 150),    # Modern Horizons 3 Commander – purple
    "dsc": (60, 40, 80),      # Duskmourn Commander – horror purple
    "fdc": (40, 80, 140),     # Foundations Commander – blue
    # Eternal
    "tle": (80, 140, 180),    # Avatar Eternal – water/air blue
    "tmc": (60, 140, 60),     # TMNT Eternal – green
    "spe": (180, 40, 40),     # Spider-Man Eternal – red
    "rex": (80, 120, 60),     # Jurassic World – jungle green
    "bot": (120, 140, 160),   # Transformers – silver/blue
    # Expansions
    "tdm": (180, 160, 40),    # Tarkir: Dragonstorm – desert gold
    "dft": (180, 160, 40),    # Aetherdrift – desert racing gold
    "fin": (80, 120, 160),    # Final Fantasy – blue fantasy
    "eoe": (80, 120, 160),    # Edge of Eternities – space blue
    "om1": (100, 80, 120),    # Through the Omenpaths – mystery purple
    "om2": (100, 80, 120),    # Through the Omenpaths 2 – mystery purple
    "fra": (80, 60, 150),     # Reality Fracture – time/dimension purple
    "trk": (60, 100, 160),    # Star Trek – space blue
    "hob": (80, 120, 60),     # The Hobbit – green
    "msh": (180, 40, 40),     # Marvel Super Heroes – red
    "tmt": (60, 140, 60),     # TMNT – green
    "tla": (80, 140, 180),    # Avatar: The Last Airbender – blue
    "spm": (180, 40, 40),     # Spider-Man – red
    "sos": (80, 60, 150),     # Secrets of Strixhaven – purple
    "ecl": (100, 80, 120),    # Lorwyn Eclipsed – dark purple
    "mat": (50, 70, 50),      # Aftermath – black-green
    "mul": (120, 60, 140),    # Multiverse Legends – purple
    "wot": (140, 80, 140),    # Enchanting Tales – purple
    "spg": (180, 150, 60),    # Special Guests – gold
    "otp": (100, 80, 50),     # Breaking News – brown
}


def lighten_color(rgb: tuple[int, int, int], factor: float = 0.7) -> tuple[int, int, int]:
    """Blend a color toward white. factor=0.7 means 70% white, 30% original."""
    r, g, b = rgb
    return (
        min(255, int(r + (255 - r) * factor)),
        min(255, int(g + (255 - g) * factor)),
        min(255, int(b + (255 - b) * factor)),
    )


def is_dark_background(rgb: tuple[int, int, int]) -> bool:
    """Return True if the color is dark enough to need light text."""
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 130


def pick_theme_color(set_name: str, set_code: str) -> tuple[int, int, int]:
    """Return an RGB tuple for the divider gradient."""
    color = SET_THEME_COLORS.get(set_code.lower())
    if color:
        return color

    name_lower = set_name.lower()
    keyword_colors: list[tuple[tuple[str, ...], tuple[int, int, int]]] = [
        (("desert", "sand", "gold", "treasure", "western", "cowboy", "sun", "amber"), (210, 170, 80)),
        (("green", "forest", "nature", "jungle", "dinosaur", "druid", "elf", "tree"), (80, 140, 60)),
        (("blue", "water", "ocean", "sea", "ice", "sky", "air", "wind", "storm", "tempest"), (60, 120, 180)),
        (("red", "fire", "blood", "crimson", "volcano", "dragon", "inferno", "hell"), (180, 50, 40)),
        (("black", "dark", "death", "horror", "shadow", "night", "vampire", "zombie", "undead", "dread"), (60, 40, 60)),
        (("white", "light", "plains", "angel", "holy", "divine", "radiant"), (200, 180, 140)),
        (("artifact", "metal", "steel", "iron", "rust", "gear", "machine", "robot"), (160, 130, 100)),
        (("purple", "arcane", "mystic", "magic", "enchant", "spell", "wizard"), (120, 60, 150)),
        (("dungeon", "dragon", "adventure", "quest", "realm", "plane"), (80, 60, 120)),
    ]
    for keywords, rgb in keyword_colors:
        if any(k in name_lower for k in keywords):
            return rgb
    return (40, 70, 120)  # neutral dark blue fallback


# Gradient rendered at 150 DPI – plenty smooth for print.
GRADIENT_DPI = 150
GRADIENT_WIDTH_PX = int(float(DIVIDER_WIDTH_MM) / 25.4 * GRADIENT_DPI)
GRADIENT_HEIGHT_PX = int(float(DIVIDER_HEIGHT_MM) / 25.4 * GRADIENT_DPI)


def get_gradient_path(color: tuple[int, int, int]) -> str:
    """Return the cached path for a gradient PNG, generating it if needed."""
    r, g, b = color
    hex_name = f"{r:02x}{g:02x}{b:02x}"
    cache_path = path.join(GRADIENT_CACHE_DIR, f"{hex_name}.png")
    if path.exists(cache_path):
        return cache_path

    img = Image.new("RGB", (GRADIENT_WIDTH_PX, GRADIENT_HEIGHT_PX))
    draw = ImageDraw.Draw(img)
    for y in range(GRADIENT_HEIGHT_PX):
        ratio = y / (GRADIENT_HEIGHT_PX - 1) if GRADIENT_HEIGHT_PX > 1 else 0
        r_out = int(r + (255 - r) * ratio)
        g_out = int(g + (255 - g) * ratio)
        b_out = int(b + (255 - b) * ratio)
        draw.line([(0, y), (GRADIENT_WIDTH_PX - 1, y)], fill=(r_out, g_out, b_out))

    makedirs(GRADIENT_CACHE_DIR, exist_ok=True)
    img.save(cache_path, "PNG")
    return cache_path

ALLOWED_SET_TYPES = {
    "core",
    "expansion",
    "commander",
    "masters",
    "draft_innovation",
    "eternal",
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


def get_svg_aspect_ratio(svg_path: str) -> float:
    """Parse the SVG viewBox (or width/height) to return width/height ratio."""
    with open(svg_path, "r", encoding="utf-8") as f:
        text = f.read()
    viewbox = _re.search(r'viewBox=["\']([^"\']+)["\']', text)
    if viewbox:
        parts = viewbox.group(1).split()
        if len(parts) == 4:
            w, h = float(parts[2]), float(parts[3])
            if h:
                return w / h
    w = _re.search(r'width=["\']([\d.]+)["\']', text)
    h = _re.search(r'height=["\']([\d.]+)["\']', text)
    if w and h:
        return float(w.group(1)) / float(h.group(1))
    return 1.0


# ── PDF generation ──────────────────────────────────────────────────────────

def generate_dividers_pdf(sets_data: list[dict], db: OracleDB) -> str:
    pdf = FPDF(orientation="P", unit="mm", format="letter")
    pdf.add_font("Alegreya", "", "./fonts/Alegreya/static/Alegreya-Regular.ttf")
    pdf.add_font("Alegreya", "B", "./fonts/Alegreya/static/Alegreya-Bold.ttf")
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

        # ── Gradient background ──
        color = pick_theme_color(s["name"], s["code"])
        is_dark = is_dark_background(color)
        gradient_path = get_gradient_path(color)
        pdf.image(
            gradient_path,
            x=float(x),
            y=float(y),
            w=float(DIVIDER_WIDTH_MM),
            h=float(DIVIDER_HEIGHT_MM),
        )

        # Choose text color based on background brightness.
        if is_dark:
            tab_text_rgb = (255, 255, 255)
        else:
            tab_text_rgb = (0, 0, 0)

        # Medium gray border — visible on both dark colored tops and white bottoms.
        border_rgb = (160, 160, 160)

        # border to guide cutting
        pdf.set_draw_color(*border_rgb)
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

        # Compute natural aspect ratio so symbols aren't squished.
        if svg_path:
            aspect = get_svg_aspect_ratio(svg_path)
            svg_w = SVG_SIZE * Decimal(str(aspect))
            svg_h = SVG_SIZE
        else:
            svg_w = SVG_SIZE
            svg_h = SVG_SIZE

        # Tinted backing behind the black SVG when the gradient is dark.
        if is_dark:
            tint = lighten_color(color)
            pdf.set_fill_color(*tint)
            pdf.rect(
                float(svg_x - Decimal("0.5")),
                float(svg_y - Decimal("0.5")),
                float(svg_w + Decimal("1")),
                float(svg_h + Decimal("1")),
                style="F",
            )

        # Scryfall SVGs have no explicit fill; fpdf2 uses current fill_color.
        pdf.set_fill_color(0, 0, 0)
        if svg_path:
            try:
                pdf.image(
                    svg_path,
                    x=float(svg_x),
                    y=float(svg_y),
                    w=float(svg_w),
                    h=float(svg_h),
                )
            except Exception as exc:
                click.echo(f"Warning: failed to render SVG for {code}: {exc}")

        # set code text to the right of the symbol, same visual height
        text_x = svg_x + svg_w + TAB_PADDING
        text_y = svg_y
        pdf.set_xy(float(text_x), float(text_y))
        pdf.set_text_color(*tab_text_rgb)
        pdf.set_font("Alegreya", "B", TAB_TEXT_FONT_PT)
        pdf.cell(
            float(DIVIDER_WIDTH_MM - svg_w - TAB_PADDING * 3),
            float(svg_h),
            code,
            align="L",
        )
        pdf.set_text_color(0, 0, 0)  # reset for body text

        # separator line between tab and body
        tab_line_y = float(y + TAB_HEIGHT)
        pdf.set_draw_color(*border_rgb)
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
        pdf.set_font("Alegreya", "B", 10)
        while pdf.get_string_width(set_name) > text_width and pdf.font_size_pt > 6:
            pdf.set_font("Alegreya", "B", pdf.font_size_pt - 1)
        pdf.cell(text_width, 5, set_name, align="C")

        # release month & year
        pdf.set_xy(content_x, float(body_top + Decimal("6")))
        pdf.set_font("Alegreya", "", 9)
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
        pdf.set_font("Alegreya", "", 8)
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
