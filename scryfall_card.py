import glob
import click
from os import path, makedirs
from pathlib import Path
from datetime import datetime

from image_utils import store_upscaled_image
from scryfall import ScryfallCardResponse, Layout, Rarity

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
