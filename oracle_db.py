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
    def __init__(self, data_set_path="./data_sets/default-cards-20260221100742.json"):
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
