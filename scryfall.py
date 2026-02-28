from typing import Literal, TypedDict, get_args

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
