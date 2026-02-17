import glob
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from os import makedirs, path
from pathlib import Path
from typing import Literal, TypedDict, get_args

import click
import httpx
import pyperclip
from fpdf import FPDF
from PIL import Image, ImageEnhance
from requests import get

# Define card dimensions in inches and convert to mm
# 2.5" x 3.5" cards at 300 PPI = 750 x 1050 pixels
# But in PDF, we work with physical dimensions
CARD_WIDTH_IN = Decimal("2.5")
CARD_HEIGHT_IN = Decimal("3.5")

ONE_INCH_MM = Decimal("25.4") # 1 inch to mm conversion.

# Convert inches to mm (1 inch = 25.4 mm)
CARD_WIDTH_MM = CARD_WIDTH_IN * ONE_INCH_MM # Decimal("63.5")  # CARD_WIDTH_IN * 25.4
CARD_HEIGHT_MM = CARD_HEIGHT_IN * ONE_INCH_MM # Decimal("88.9")  # CARD_HEIGHT_IN * 25.4

# Gap between cards
GAP_MM = Decimal("0.2")

DOUBLE_GAP_MM = GAP_MM * Decimal(2)

# Page dimensions in mm
PAGE_WIDTH_MM = Decimal("215.9")  # 8.5"
PAGE_HEIGHT_MM = Decimal("279.4")  # 11"

# Calculate margins to center 3x3 grid
TOTAL_CARDS_WIDTH = (Decimal(3) * CARD_WIDTH_MM) + DOUBLE_GAP_MM
TOTAL_CARDS_HEIGHT = (Decimal(3) * CARD_HEIGHT_MM) + DOUBLE_GAP_MM

# Calculate starting x,y for top-left card to center the grid
# add two gaps in both directions
START_X = (PAGE_WIDTH_MM - TOTAL_CARDS_WIDTH) / Decimal(2)
START_Y = (PAGE_HEIGHT_MM - TOTAL_CARDS_HEIGHT) / Decimal(2)

type Rarity = Literal["common", "uncommon", "rare", "mythic"]


class ScryfallImageUris(TypedDict):
    png: str
    large: str


class ScryfallCardFace(TypedDict):
    name: str
    image_uris: ScryfallImageUris


type Legality = Literal["legal", "not_legal"]
type Format = Literal[
    "standard",
    "pioneer",
    "modern",
    "alchemy",
    "future",
    "historic",
    "timeless",
    "commander",
    "legacy",
    "oathbreaker",
]
type FrameEffect = Literal[
    "inverted", "legendary", "fullart"
]  # some effects listed, not all.
INVERTED_FRAME_EFFECT = "inverted"

FORMAT_CHOICES = get_args(Format.__value__)

BASIC_LANDS: list[str] = ["Island", "Mountain", "Plains", "Swamp", "Forest"]

type SetType = Literal[
    "token",
    "core",
    "expansion",
    "alchemy",
    "box",
    "promo",
    "funny",
    "eternal",
    "masters",
    "spellbook",
]

type Layout = Literal[
    "adventure",
    "normal",
    "transform",
    "omen",
    "battle",
    "flip",
    "modal_dfc",
    "class",
    "case",
    "saga",
    "token",
    "prototype",
    "mutate",
    "split",
]


class ScryfallCardResponse(TypedDict):
    name: str
    printed_name: str
    id: str
    oracle_id: str
    image_uris: ScryfallImageUris
    set: str  # the 3-4 letter set code
    set_name: str  # the full set name
    set_type: SetType
    rarity: Rarity
    card_faces: list[ScryfallCardFace]
    legalities: dict[Format, Legality]
    released_at: str
    promo: bool
    collector_number: str
    frame_effects: list[FrameEffect]
    type_line: str
    layout: Layout


class ScryfallResponse(TypedDict):
    data: list[ScryfallCardResponse]
    next_page: str
    total_cards: int
    has_more: bool


def store_upscaled_image(image_url: str, image_path: str):
    resp = get(image_url)

    with Image.open(BytesIO(resp.content)) as img:
        new_size = (img.width * 4, img.height * 4)

        upscaled = img.resize(new_size, resample=Image.Resampling.LANCZOS)

        color_enhance = ImageEnhance.Color(upscaled)
        upscaled = color_enhance.enhance(1.2)

        sharpness_enhance = ImageEnhance.Sharpness(upscaled)
        upscaled = sharpness_enhance.enhance(1.2)

        upscaled.save(image_path)

def upscale_image(image_path: str, new_name: str):
    with Image.open(image_path) as img:
        new_size = (img.width * 4, img.height * 4)

        upscaled = img.resize(new_size, resample=Image.Resampling.LANCZOS)

        color_enhance = ImageEnhance.Color(upscaled)
        upscaled = color_enhance.enhance(1.2)

        sharpness_enhance = ImageEnhance.Sharpness(upscaled)
        upscaled = sharpness_enhance.enhance(1.2)

        upscaled.save(new_name)


class ScryfallCard:
    def __init__(self, card_data: ScryfallCardResponse):
        self.card_data = card_data

    def id(self) -> str:
        return self.card_data.get("id")

    def name(self) -> str:
        printed_name = self.card_data.get("printed_name")
        if printed_name is not None:
            return printed_name
        return self.card_data.get("name")

    def rarity(self) -> Rarity:
        return self.card_data["rarity"]

    def set(self) -> str:
        return str.upper(self.card_data.get("set"))

    def is_transform(self) -> bool:
        """This card has multiple card faces and transforms. All card faces should be added to PDF"""
        card_layout: Layout = self.card_data.get("layout", "")
        return card_layout in ["transform", "modal_dfc", "reversible_card"]

    def card_face_uris(self) -> list[str] | None:
        faces = self.card_data.get("card_faces")
        if faces is not None and len(faces) > 0:
            return [face["image_uris"]["png"] for face in faces]
        return None

    def card_data_path(self) -> str:
        return f"./data_sets/card_data/{self.set()}/{self.id()}.json"

    def card_image_path(self) -> str:
        return f"./card_images/{self.set()}/{self.id()}.png"

    def image_file_exists(self) -> bool:
        return path.exists(self.card_image_path())

    def image(self) -> str:
        """the png image url on scryfall"""
        return self.card_data["image_uris"]["png"]

    def faces_directory(self) -> str:
        return path.join(".", "card_images", self.set(), self.id())

    def faces_image_files(self) -> list[str]:
        return glob.glob(self.faces_directory() + "/*.png")

    def store_card_faces(self):
        # make a directory with the id number, store all card faces in there.
        if not path.exists(self.faces_directory()):
            makedirs(self.faces_directory(), exist_ok=True)
        else:
            click.echo(f"Images already exist for {self.name()}")
            return

        uris = self.card_face_uris()
        if uris is not None and len(uris) > 0:
            count = 1

            for card_face_uri in uris:
                click.echo(f"fetching new image {self.name()} - {self.id()}")
                save_path = path.join(self.faces_directory(), f"face_{count}.png")
                store_upscaled_image(card_face_uri, save_path)
                count = count + 1

    def store_image_from_web(self):
        if not self.image_file_exists():
            # make the directories first
            p = Path(self.card_image_path())
            p.parent.mkdir(exist_ok=True, parents=True)

            # fetch next
            print(f"Fetching new image {self.name()} - {self.id()}")

            store_upscaled_image(self.image(), self.card_image_path())
        else:
            print(f"Image already exists for {self.name()} - {self.id()}")

    def store_image(self):
        if self.is_transform():
            self.store_card_faces()
        else:
            self.store_image_from_web()

    def release_date(self) -> datetime:
        return datetime.fromisoformat(self.card_data["released_at"])

    def collector_number(self) -> str:
        return self.card_data["collector_number"]


def extract_set_code(card_name: str) -> tuple[str | None, str]:
    """
    extracts the set code, if it exists, from the card_name.
    returns the set_code, if it was found, and the card name with set code removed.
    """
    # see if the card_name contains a set
    set_code_match = re.search("\\(\\w{3,4}\\)", card_name)
    set_code = None
    if set_code_match is not None:
        set_code = set_code_match[0]
        card_name = re.sub("\\(\\w{3,4}\\)", "", card_name).strip()

    # remove parentheses
    if set_code is not None:
        set_code = set_code.replace("(", "").strip()
        set_code = set_code.replace(")", "").strip()

    return set_code, card_name


def extract_collector_number(card_name: str) -> tuple[str | None, str]:
    """
    extracts the colletor number, if it exists, from the card name.
    returns the collector number,  if it was found, and the card name with the number removed.
    """
    collector_number_match = re.search("\\d+$", card_name)
    cn = None

    if collector_number_match is not None:
        cn = collector_number_match[0]
        card_name = re.sub("\\d+$", "", card_name).strip()

    return cn, card_name


class OracleDB:
    def __init__(self, data_set_path="./data_sets/default-cards-20260215221934.json"):
        click.echo(f"Opening Oracle Card DB at {data_set_path}")

        with open(data_set_path, encoding="utf-8") as json_file:
            json_data = json.load(json_file)
            self.data: list[ScryfallCardResponse] = json_data

    def get_set_cards(self, set_code: str) -> list[ScryfallCard]:
        set_code_search_value = set_code.upper()
        return [
            ScryfallCard(c)
            for c in self.data
            if c["set"].upper() == set_code_search_value
        ]

    def find_card(self, card_name: str) -> ScryfallCard | None:
        cn, card_name = extract_collector_number(card_name)

        set_code, card_name = extract_set_code(card_name)

        matched_card_names = (
            c
            for c in self.data
            if card_name in c.get("printed_name", "") or card_name in c.get("name", "")
        )

        # check for set code
        if set_code is not None:
            set_code = set_code.lower()
            matches_set = (
                c for c in matched_card_names if c["set"].lower() == set_code
            )
        else:
            matches_set = matched_card_names

        # check for collector number
        if cn is not None:
            matches_cn = (c for c in matches_set if c["collector_number"] == cn)
        else:
            matches_cn = matches_set

        results = list(ScryfallCard(c) for c in matches_cn)

        if results is None or len(results) == 0:
            return None

        return max(results, key=lambda c: c.release_date())


@dataclass
class CardPrintInstructions:
    num_copies: int
    card: ScryfallCard


@dataclass
class PDFImageLocation:
    x: Decimal
    y: Decimal


IMAGE_PAGE_LOCATIONS_FRONTS = [
    PDFImageLocation(Decimal("12.5"), Decimal("6.15")),
    PDFImageLocation(Decimal("76.2"), Decimal("6.15")),
    PDFImageLocation(Decimal("139.7"), Decimal("6.15")),
    PDFImageLocation(Decimal("12.5"), Decimal("95.25")),
    PDFImageLocation(Decimal("76.2"), Decimal("95.25")),
    PDFImageLocation(Decimal("139.7"), Decimal("95.25")),
    PDFImageLocation(Decimal("12.5"), Decimal("184.15")),
    PDFImageLocation(Decimal("76.2"), Decimal("184.15")),
    PDFImageLocation(Decimal("139.7"), Decimal("184.15")),
]

IMAGE_PAGE_LOCATIONS_BACKS = [
    PDFImageLocation(Decimal("139.7"), Decimal("6.15")),
    PDFImageLocation(Decimal("76.2"), Decimal("6.15")),
    PDFImageLocation(Decimal("12.5"), Decimal("6.15")),
    PDFImageLocation(Decimal("139.7"), Decimal("95.25")),
    PDFImageLocation(Decimal("76.2"), Decimal("95.25")),
    PDFImageLocation(Decimal("12.5"), Decimal("95.25")),
    PDFImageLocation(Decimal("139.7"), Decimal("184.15")),
    PDFImageLocation(Decimal("76.2"), Decimal("184.15")),
    PDFImageLocation(Decimal("12.5"), Decimal("184.15")),
]


PDF_BASE_DIR = "pdfs"


def write_pdf_file(pdf_doc: FPDF, directory: str, deck_or_list_name: str):
    pdf_directory = path.join(PDF_BASE_DIR, directory)
    filename = f"{deck_or_list_name}-{int(datetime.now().timestamp())}.pdf"
    if not path.exists(pdf_directory):
        makedirs(pdf_directory, exist_ok=True)
    full_path = path.join(pdf_directory, filename)
    click.echo(f"Writing file: {full_path}")
    pdf_doc.output(full_path)


def swap_first_third(lst: list[str]) -> list[str]:
    for i in range(0, len(lst), 3):
        print(lst)
        print("**************")
        lst[i], lst[i + 2] = lst[i + 2], lst[i]
    return lst


def generate_transforms_pdf(cards: list[CardPrintInstructions], deck_name: str):
    pdf_doc = FPDF(orientation="P", unit="mm", format="letter")
    pdf_doc.set_margin(0)

    # get the right amount of file paths
    paths: list[list[str]] = []
    for instr in cards:
        for _ in range(instr.num_copies):
            paths.append(instr.card.faces_image_files())

    while paths:
        pdf_doc.add_page()
        page1 = pdf_doc.page
        pdf_doc.add_page()
        page2 = pdf_doc.page

        # start y up here. it only resets when there is a new
        # page
        y = START_Y + 0 * CARD_HEIGHT_MM

        for row in range(3):
            # start x - it resets back to the left on each row
            x = START_X + 0 * CARD_WIDTH_MM

            for col in range(3):
                front_path, back_path = paths.pop()

                f_x = float(x)
                f_y = float(y)
                f_width = float(CARD_WIDTH_MM)
                f_height = float(CARD_HEIGHT_MM)
                print(f"Adding front face ({f_x}, {f_y}) :: {f_width}, {f_height}")

                # set front face
                pdf_doc.page = page1
                pdf_doc.image(
                    front_path,
                    x=f_x,
                    y=f_y,
                    w=f_width,
                    h=f_height,
                )

                # set the back face on the second page
                pdf_doc.page = page2
                # calculate the correct x value, all the y values (height) remain the same.
                back_x = PAGE_WIDTH_MM - (x + CARD_WIDTH_MM)
                f_back_x = float(back_x)
                print(f"Adding back face ({f_back_x}, {f_y}) :: {f_width}, {f_height}")
                pdf_doc.image(
                    back_path,
                    x=f_back_x,
                    y=f_y,
                    w=f_width,
                    h=f_height,
                )

                # increment x - this is the column
                x = x + CARD_WIDTH_MM + (GAP_MM if col < 2 else Decimal(0))

            # increment y, this is the row
            y = y + CARD_HEIGHT_MM + (GAP_MM if row < 2 else Decimal(0))

    write_pdf_file(pdf_doc, "lists", deck_name)


def generate_deck_list_pdf(cards: list[CardPrintInstructions], deck_name: str):
    pdf_doc = FPDF(orientation="P", unit="mm", format="letter")
    pdf_doc.set_margin(0)

    image_paths: list[str] = []
    for instr in cards:
        # print the right number of copies
        for _ in range(instr.num_copies):
            image_paths.append(instr.card.card_image_path())

    while image_paths:
        pdf_doc.add_page()

        # start y up here. it only resets when there is a new
        # page
        y = START_Y + 0 * CARD_HEIGHT_MM

        # Add 3x3 grid of images
        for row in range(3):
            x = START_X + 0 * CARD_WIDTH_MM

            for col in range(3):
                if len(image_paths) == 0:
                    break

                image_path: str = image_paths.pop()

                print(f"adding image to page - {x}, {y}")

                pdf_doc.image(
                    image_path,
                    x=float(x),
                    y=float(y),
                    w=float(CARD_WIDTH_MM),
                    h=float(CARD_HEIGHT_MM),
                )

                # increment x - this is the column
                x = x + CARD_WIDTH_MM + (GAP_MM if col < 2 else Decimal(0))

            # increment y, this is the row
            y = y + CARD_HEIGHT_MM + (GAP_MM if row < 2 else Decimal(0))

    write_pdf_file(pdf_doc, "decks", deck_name)


def generate_pdf(
    cards: list[ScryfallCard], set_code: str, pdf_type: Literal["query", "booster"]
):
    pdf_doc = FPDF(orientation="P", unit="mm", format="letter")
    pdf_doc.set_margin(0)

    click.echo(f"Creating PDF with {len(cards)} cards.")

    image_paths: list[str] = []
    for c in cards:
        if c.is_transform():
            [image_paths.append(p) for p in c.faces_image_files()]
        else:
            image_paths.append(c.card_image_path())

    while len(image_paths) > 0:
        pdf_doc.add_page()

        # Add 3x3 grid of images
        for row in range(3):
            for col in range(3):
                x = START_X + col * (CARD_WIDTH_MM + GAP_MM)
                y = START_Y + row * (CARD_HEIGHT_MM + GAP_MM)

                if len(image_paths) == 0:
                    break

                image_path: str = image_paths.pop()

                pdf_doc.image(
                    image_path,
                    x=float(x),
                    y=float(y),
                    w=float(CARD_WIDTH_MM),
                    h=float(CARD_HEIGHT_MM),
                )
            else:
                continue
            break
        else:
            continue
        break

    write_pdf_file(pdf_doc, set_code, pdf_type)


def generate_booster_list(card_set: list[ScryfallCard]) -> list[ScryfallCard]:
    # 5 common, 3 uncommon, 1 rare/mythic
    booster_cards: list[ScryfallCard] = []
    common_pool = [c for c in card_set if c.rarity() == "common"]
    [booster_cards.append(random.choice(common_pool)) for _ in range(5)]

    uncommon_pool = [c for c in card_set if c.rarity() == "uncommon"]
    [booster_cards.append(random.choice(uncommon_pool)) for _ in range(3)]

    rare_mythic_pool = [c for c in card_set if c.rarity() in ["rare", "mythic"]]
    booster_cards.append(random.choice(rare_mythic_pool))

    return booster_cards


def run_scryfall_query(scryfall_query: str) -> list[ScryfallCard]:
    params = {"q": scryfall_query}
    resp = httpx.get(
        "https://api.scryfall.com/cards/search", params=params, timeout=None
    )
    d: ScryfallResponse = resp.json()

    click.echo(f"{d['total_cards']} total cards.")

    cards: list[ScryfallCard] = [ScryfallCard(c) for c in d["data"]]
    while d["has_more"]:
        resp = httpx.get(d["next_page"], timeout=None)
        d: ScryfallResponse = resp.json()
        for c in d["data"]:
            cards.append(ScryfallCard(c))

    return cards


def generate_card_print_instructions(data: str) -> list[CardPrintInstructions]:
    """
    :param data: a mostly mtg arena formatted, new line separated list of cards.
    """
    click.echo("Generating deck list.")
    db = OracleDB()
    card_print_instrs: list[CardPrintInstructions] = []

    for line in data.splitlines():
        # skip this line if it doesn't start with a number.
        trimmed = line.strip()
        if len(trimmed) < 1 or not trimmed[0].isdigit():
            continue

        [num_copies, _, card_name] = trimmed.partition(" ")

        # skip basic lands
        if card_name in BASIC_LANDS:
            continue

        # parse the number of copies
        num_copies = int(num_copies)

        # look up the card in the db
        c = db.find_card(card_name)

        if c is not None:
            c.store_image()
            card_print_instrs.append(CardPrintInstructions(num_copies, c))
        else:
            click.echo(f"{card_name} was not found.")

    print("\nDeck List To Print\n")
    [
        print(c.num_copies, c.card.name(), c.card.set(), c.card.collector_number())
        for c in card_print_instrs
    ]

    return card_print_instrs


@click.group
def cli():
    pass


@cli.command(
    "booster",
    short_help="Generates a PDF of exactly 9 random cards from the specified set.",
)
@click.argument("set_code")
def booster(set_code: str):
    db = OracleDB()
    cards = db.get_set_cards(set_code)
    # cards = run_long_query(f"set:{set_code} -t:\"basic land\"")
    click.echo(f"Retrieved {len(cards)} cards for set {set_code}.")

    booster_cards = generate_booster_list(cards)

    for c in booster_cards:
        c.store_image()

    generate_pdf(booster_cards, set_code, "booster")


def cards_to_add(num_cards: int, cards_per_page=9) -> int:
    remainder = num_cards % cards_per_page
    if remainder == 0:
        return 0
    return cards_per_page - remainder


def add_copies(
    instrs: list[CardPrintInstructions], print_transforms: bool = False
) -> list[CardPrintInstructions]:
    new_instrs = (
        [i for i in instrs]
        if print_transforms
        else [i for i in instrs if not i.card.is_transform()]
    )

    total_cards = sum([i.num_copies for i in new_instrs])
    copies_to_add = cards_to_add(total_cards, 9)
    print(total_cards, "total cards -- ", copies_to_add, "number of copies to add")

    for _ in range(copies_to_add):
        inst: CardPrintInstructions = random.choice(new_instrs)
        print("adding copy of ", inst.card.name(), f"({inst.card.collector_number()})")
        inst.num_copies += 1

    return new_instrs


@cli.command("query", short_help="Queries scryfall and generates a PDF of the cards.")
@click.argument("q")
@click.option(
    "-s",
    "--set_code",
    default="BLB",
    type=str,
    help="The 3-4 letter set code to limit the query to.",
)
@click.option(
    "-t", "--transforms", help="Just print transform cards.", default=False, type=bool
)
@click.option(
    "-c",
    "--copies",
    help="number of copies of each card to add to the PDF",
    type=int,
    default=1,
)
def query(q: str, set_code: str, transforms: bool, copies: int):
    scryfall_query = f"{q}"
    click.echo(f"Using query {scryfall_query}")

    cards = run_scryfall_query(scryfall_query)

    for c in cards:
        c.store_image()

    # build instrs list
    # check if length is multiple of 9
    instrs = [CardPrintInstructions(copies, c) for c in cards]
    instrs = add_copies(instrs, transforms)

    if transforms:
        generate_transforms_pdf(instrs, f"transforms-{set_code}")
    else:
        generate_deck_list_pdf(instrs, f"list-{set_code}")


@cli.command("deck_list", short_help="Generates a PDF of a provided deck list.")
@click.option("-n", "--name", default="deck", type=str, help="The name of your deck!")
def deck_list(name: str):
    data: str = pyperclip.paste()

    instrs = generate_card_print_instructions(data)

    total_cards = sum([i.num_copies for i in instrs])
    copies_to_add = cards_to_add(total_cards, 9)
    print(total_cards, "total cards -- ", copies_to_add, "number of copies to add")

    for _ in range(copies_to_add):
        inst: CardPrintInstructions = random.choice(instrs)
        print("adding copy of ", inst.card.name(), f"({inst.card.collector_number()})")
        inst.num_copies += 1

    generate_deck_list_pdf(instrs, name)


@cli.command("transforms")
@click.option("-n", "--name", default="deck", type=str, help="The name of your deck!")
def transforms(name: str):
    click.echo("Generating PDF for transforms")
    data: str = pyperclip.paste()

    instrs = generate_card_print_instructions(data)

    total_cards = sum([i.num_copies for i in instrs])
    copies_to_add = cards_to_add(total_cards, 9)
    print(total_cards, "total cards -- ", copies_to_add, "number of copies to add")

    for _ in range(copies_to_add):
        inst: CardPrintInstructions = random.choice(instrs)
        print("adding copy of ", inst.card.name(), f"({inst.card.collector_number()})")
        inst.num_copies += 1

    generate_transforms_pdf(instrs, name)


@cli.command("debug_stuff")
def debug_stuff():
    db = OracleDB()
    rag = db.find_card("Singularity Rupture (EOE) 398")

    if rag is not None:
        print(rag.name(), rag.collector_number())
    else:
        print("not found")


@cli.command("generate_backs")
def generate_backs_pdf():
    pdf_doc = FPDF(orientation="P", unit="mm", format="letter")
    pdf_doc.set_margin(0)
    pdf_doc.add_page()

    y = START_Y + 0 * (CARD_HEIGHT_MM + GAP_MM)

    print(f"effective width & height {pdf_doc.epw}, {pdf_doc.eph}")
    print(f"hard-coded width & height {PAGE_WIDTH_MM}, {PAGE_HEIGHT_MM}")

    for row in range(3):
        x = START_X + 0 * CARD_WIDTH_MM
        for col in range(3):
            image_path: str = "./card_images/mtg_card_back_upscaled.png"

            back_x = PAGE_WIDTH_MM - (x + CARD_WIDTH_MM)
            f_back_x = float(back_x)

            print(f"adding back image at ({f_back_x}, {y})")
            pdf_doc.image(
                image_path,
                x=float(f_back_x),
                y=float(y),
                w=float(CARD_WIDTH_MM),
                h=float(CARD_HEIGHT_MM),
                keep_aspect_ratio=False
            )
            x = x + CARD_WIDTH_MM + (GAP_MM if col < 2 else Decimal(0))

        y = y + CARD_HEIGHT_MM + (GAP_MM if row < 2 else Decimal(0))

    pdf_doc.output("./pdfs/card_backs.pdf")


@cli.command("upscale_card_back")
def upscale_card_back(): 
    upscale_image("./card_images/mtg_card_back.png", "./card_images/mtg_card_back_upscaled.png")


if __name__ == "__main__":
    cli()
