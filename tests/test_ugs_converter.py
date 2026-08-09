import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ugs_converter import (
    UgsError,
    build_ugs,
    create_backup,
    decode_ugs_pixel,
    encode_ugs_pixel,
    export_png_frames,
    extract_ugs,
    inspect_ugs,
    install_ugs,
    make_sprite_sheet,
    read_ugs,
    rotate_left_16,
    rotate_right_16,
    split_sprite_sheet,
    validate_frames,
    write_ugs,
)


class PixelFormatTests(unittest.TestCase):
    def test_transparent_black_has_game_sentinel(self) -> None:
        self.assertEqual(encode_ugs_pixel(0, 0, 0, 0), 0xAAAA)
        self.assertEqual(decode_ugs_pixel(0xAAAA), (0, 0, 0, 0))

    def test_known_opaque_red_value(self) -> None:
        self.assertEqual(encode_ugs_pixel(255, 0, 0, 255), 0x52AD)
        self.assertEqual(decode_ugs_pixel(0x52AD), (255, 0, 0, 255))

    def test_rotations_are_inverse(self) -> None:
        for value in (0x0000, 0x0001, 0x1234, 0xAAAA, 0xFFFF):
            self.assertEqual(rotate_right_16(rotate_left_16(value, 3), 3), value)


class ContainerRoundTripTests(unittest.TestCase):
    def test_extract_and_build_preserve_ugs_bytes(self) -> None:
        pixels = [
            encode_ugs_pixel(255, 0, 0, 255),
            encode_ugs_pixel(0, 255, 0, 136),
        ]
        original = struct.pack("<HH2H", 2, 1, *pixels)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.ugs"
            frames = root / "frames"
            rebuilt = root / "rebuilt.ugs"
            source.write_bytes(original)

            decoded = read_ugs(source)
            self.assertEqual(len(decoded), 1)
            self.assertEqual(decoded[0].size, (2, 1))

            details = inspect_ugs(source)
            self.assertEqual(details.frame_count, 1)
            self.assertEqual((details.width, details.height), (2, 1))
            self.assertEqual(details.file_size, len(original))

            extract_ugs(source, frames)
            build_ugs(frames, rebuilt)
            self.assertEqual(rebuilt.read_bytes(), original)

    def test_backup_keeps_original_and_uses_free_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "hero.ugs"
            source.write_bytes(b"original")

            first = create_backup(source)
            second = create_backup(source)

            self.assertEqual(first.name, "hero.ugs.bak")
            self.assertEqual(second.name, "hero.ugs.bak.1")
            self.assertEqual(first.read_bytes(), b"original")
            self.assertEqual(second.read_bytes(), b"original")

    def test_sprite_sheet_is_split_in_row_major_order(self) -> None:
        sheet = Image.new("RGBA", (16, 16))
        for index in range(64):
            color = (index, 255 - index, index // 2, 255)
            x = (index % 8) * 2
            y = (index // 8) * 2
            for pixel_y in range(y, y + 2):
                for pixel_x in range(x, x + 2):
                    sheet.putpixel((pixel_x, pixel_y), color)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "sheet.png"
            sheet.save(source)
            frames = split_sprite_sheet(source)

        self.assertEqual(len(frames), 64)
        self.assertTrue(all(frame.size == (2, 2) for frame in frames))
        self.assertEqual(frames[0].getpixel((0, 0)), (0, 255, 0, 255))
        self.assertEqual(frames[63].getpixel((0, 0)), (63, 192, 31, 255))

    def test_sprite_sheet_export_round_trips_all_frames(self) -> None:
        original = [
            Image.new("RGBA", (2, 3), (index, 255 - index, index // 2, 255))
            for index in range(64)
        ]
        sheet = make_sprite_sheet(original)
        self.assertEqual(sheet.size, (16, 24))

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "sheet.png"
            sheet.save(source)
            restored = split_sprite_sheet(source)

        self.assertEqual(
            [frame.getpixel((0, 0)) for frame in restored],
            [frame.getpixel((0, 0)) for frame in original],
        )

    def test_export_all_frames_uses_stable_names_and_guards_overwrite(self) -> None:
        original = [Image.new("RGBA", (2, 2), (index, 0, 0, 255)) for index in range(3)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            exported = export_png_frames(original, output)

            self.assertEqual(
                [path.name for path in exported],
                ["frame_000.png", "frame_001.png", "frame_002.png"],
            )
            self.assertEqual(Image.open(exported[2]).convert("RGBA").getpixel((0, 0)), (2, 0, 0, 255))
            with self.assertRaises(UgsError):
                export_png_frames(original, output)

    def test_validation_collects_size_error_and_count_warning(self) -> None:
        report = validate_frames(
            [Image.new("RGBA", (64, 64)), Image.new("RGBA", (32, 64))]
        )
        self.assertFalse(report.ok)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(len(report.warnings), 1)

    def test_install_replaces_target_and_preserves_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "new.ugs"
            target = root / "Hero-Knight.ugs"
            write_ugs([Image.new("RGBA", (2, 1), (255, 0, 0, 255))], source)
            write_ugs([Image.new("RGBA", (2, 1), (0, 255, 0, 255))], target)
            original = target.read_bytes()

            backup = install_ugs(source, target)

            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(backup.read_bytes(), original)

    def test_install_rejects_incompatible_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "new.ugs"
            target = root / "target.ugs"
            write_ugs([Image.new("RGBA", (2, 1))], source)
            write_ugs([Image.new("RGBA", (1, 1))], target)

            with self.assertRaises(UgsError):
                install_ugs(source, target)


if __name__ == "__main__":
    unittest.main()
