import struct
import tempfile
import unittest
from pathlib import Path

from ugs_converter import (
    build_ugs,
    create_backup,
    decode_ugs_pixel,
    encode_ugs_pixel,
    extract_ugs,
    inspect_ugs,
    read_ugs,
    rotate_left_16,
    rotate_right_16,
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


if __name__ == "__main__":
    unittest.main()
