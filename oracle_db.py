import glob
import os
import re

import click
import json

from scryfall import ScryfallCardResponse
from scryfall_card import ScryfallCard


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
    """
    Class that encapsulates a json file from scryfall's bulk download section.
    """

    def __init__(self, data_set_path=None):
        if data_set_path is None:
            data_set_path = self._find_latest_bulk_file()

        click.echo(f"Opening Oracle Card DB at {data_set_path}")

        with open(data_set_path, encoding="utf-8") as json_file:
            json_data = json.load(json_file)
            self.data: list[ScryfallCardResponse] = json_data

        # Index cards by set code for fast lookups.
        self._set_index: dict[str, list[ScryfallCardResponse]] = {}
        for c in self.data:
            s = c["set"].lower()
            self._set_index.setdefault(s, []).append(c)

    @staticmethod
    def _find_latest_bulk_file() -> str:
        files = glob.glob("./data_sets/default-cards-*.json")
        if not files:
            raise FileNotFoundError(
                "No default-cards-*.json file found in ./data_sets/"
            )
        return max(files, key=os.path.getmtime)

    def get_set_cards(self, set_code: str) -> list[ScryfallCard]:
        set_code_search_value = set_code.lower()
        cards = self._set_index.get(set_code_search_value, [])
        return [ScryfallCard(c) for c in cards]

    def get_set_info(self, set_code: str) -> dict | None:
        set_code = set_code.lower()
        cards = self._set_index.get(set_code)
        if not cards:
            return None
        c = cards[0]
        return {
            "name": c.get("set_name", ""),
            "released_at": c.get("released_at", ""),
            "set_type": c.get("set_type", ""),
        }

    def get_set_rarity_counts(self, set_code: str) -> dict[str, int]:
        set_code = set_code.lower()
        counts = {"common": 0, "uncommon": 0, "rare": 0, "mythic": 0}
        seen_oracle_ids = set()
        for c in self._set_index.get(set_code, []):
            if c.get("lang") != "en":
                continue
            rarity = c.get("rarity")
            oid = c.get("oracle_id")
            if rarity not in counts or not oid or oid in seen_oracle_ids:
                continue
            seen_oracle_ids.add(oid)
            counts[rarity] += 1
        return counts

    def has_set(self, set_code: str) -> bool:
        return set_code.lower() in self._set_index

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
