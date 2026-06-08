# AGENTS.md

## Hidden data dependencies

- `OracleDB` expects `./data_sets/default-cards-*.json`, a Scryfall bulk data dump. This file is **not in the repo** and must be downloaded manually from Scryfall's bulk-data endpoint. Commands `booster`, `deck_list`, `transforms`, and `debug_stuff` fail without it.
- `generate_deck_list_pdf` and `add_card_backs_page` require `./card_images/back_upscaled.jpg`. This card-back image is **not fetched automatically**; it must exist as a local static asset.

## API constraints

- `scryfall_card.run_scryfall_query` paginates through search results without any delay between requests. Scryfall asks callers to sleep ~100 ms between requests; missing this risks rate-limiting or temporary API bans.

## Headless / automation constraints

- `deck_list` and `transforms` CLI commands read deck lists from the **system clipboard** via `pyperclip.paste()`. They cannot run headlessly or in CI unless the clipboard is pre-populated.
