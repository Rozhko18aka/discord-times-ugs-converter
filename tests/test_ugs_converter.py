import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ugs_converter import (
    LIT_MAGIC,
    LIT_VALID_FLAGS,
    LitError,
    UgsError,
    build_ugs,
    build_lit,
    calculate_window_size,
    choose_lit_flags,
    create_backup,
    decode_lit,
    decode_ugs_pixel,
    encode_lit,
    encode_ugs_pixel,
    export_png_frames,
    extract_lit,
    extract_ugs,
    find_lit_files,
    inspect_lit,
    inspect_ugs,
    install_ugs,
    make_preview,
    make_sprite_sheet,
    read_ugs,
    replace_all_frames,
    replace_selected_frame,
    rotate_left_16,
    rotate_right_16,
    split_sprite_sheet,
    validate_frames,
    write_ugs,
    write_lit,
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


class InterfaceSizingTests(unittest.TestCase):
    def test_window_size_adapts_to_common_scaled_desktops(self) -> None:
        self.assertEqual(calculate_window_size(1280, 720), (1100, 590))
        self.assertEqual(calculate_window_size(1920, 1080), (1100, 760))
        self.assertEqual(calculate_window_size(800, 600), (720, 500))

    def test_window_never_exceeds_small_screen(self) -> None:
        self.assertEqual(calculate_window_size(640, 480), (640, 480))


class LitFormatTests(unittest.TestCase):
    def test_all_known_lit_variants_encode_and_decode(self) -> None:
        opaque = Image.new("RGBA", (13, 11), (180, 70, 25, 255))
        transparent = Image.new("RGBA", (13, 11), (180, 70, 25, 96))

        for flags in sorted(LIT_VALID_FLAGS):
            source = transparent if flags in (8, 10) else opaque
            encoded = encode_lit(source, flags)
            magic, width, height, stored_flags = struct.unpack_from("<4sIII", encoded)
            restored = decode_lit(encoded)

            self.assertEqual(magic, LIT_MAGIC)
            self.assertEqual((width, height, stored_flags), (13, 11, flags))
            self.assertEqual(restored.size, source.size)
            self.assertEqual(restored.mode, "RGBA")
            if flags in (8, 10):
                alpha_min, alpha_max = restored.getchannel("A").getextrema()
                self.assertLessEqual(abs(alpha_min - 96), 2)
                self.assertLessEqual(abs(alpha_max - 96), 2)
            else:
                self.assertEqual(restored.getchannel("A").getextrema(), (255, 255))

    def test_raw_lit_stores_ycbcr_triplets_without_compression(self) -> None:
        source = Image.new("RGB", (2, 1))
        source.putdata([(255, 0, 0), (0, 255, 0)])
        encoded = encode_lit(source, 4)

        self.assertEqual(encoded[16:], source.convert("YCbCr").tobytes())
        restored = decode_lit(encoded).convert("RGB")
        restored_pixels = (
            restored.get_flattened_data() if hasattr(restored, "get_flattened_data") else restored.getdata()
        )
        source_pixels = (
            source.get_flattened_data() if hasattr(source, "get_flattened_data") else source.getdata()
        )
        for actual, expected in zip(restored_pixels, source_pixels):
            self.assertTrue(all(abs(a - e) <= 3 for a, e in zip(actual, expected)))

    def test_new_png_chooses_raw_or_alpha_format(self) -> None:
        self.assertEqual(choose_lit_flags(Image.new("RGBA", (1, 1), (0, 0, 0, 255))), 4)
        self.assertEqual(choose_lit_flags(Image.new("RGBA", (1, 1), (0, 0, 0, 0))), 8)
        self.assertEqual(
            choose_lit_flags(Image.new("RGBA", (1, 1), (0, 0, 0, 128)), preferred=10),
            10,
        )

    def test_lit_rejects_bad_header_size_and_flags(self) -> None:
        with self.assertRaises(LitError):
            decode_lit(b"not a lit")
        with self.assertRaises(LitError):
            decode_lit(struct.pack("<4sIII", LIT_MAGIC, 8, 8, 1))
        with self.assertRaises(LitError):
            decode_lit(struct.pack("<4sIII", LIT_MAGIC, 8, 8, 4))

    def test_lit_file_workflow_exports_and_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_png = root / "source.png"
            lit_path = root / "source.lit"
            exported_png = root / "exported.png"
            rebuilt_lit = root / "rebuilt.lit"
            Image.new("RGBA", (9, 7), (20, 90, 160, 120)).save(source_png)

            built = build_lit(source_png, lit_path)
            self.assertEqual(built.flags, 8)
            exported = extract_lit(lit_path, exported_png)
            self.assertEqual(exported.width, 9)
            rebuilt = write_lit(Image.open(exported_png), rebuilt_lit)
            self.assertEqual(rebuilt.flags, 8)
            self.assertEqual(inspect_lit(rebuilt_lit).height, 7)

    def test_lit_write_guards_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "image.lit"
            destination.write_bytes(b"existing")
            with self.assertRaises(LitError):
                write_lit(Image.new("RGBA", (2, 2)), destination)

    def test_lit_folder_scan_is_recursive_case_insensitive_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "Nested"
            nested.mkdir()
            (root / "zeta.LIT").write_bytes(b"")
            (root / "alpha.lit").write_bytes(b"")
            (nested / "middle.LiT").write_bytes(b"")
            (nested / "ignore.png").write_bytes(b"")

            files = find_lit_files(root)

            self.assertEqual(
                [str(path.relative_to(root)) for path in files],
                ["alpha.lit", str(Path("Nested") / "middle.LiT"), "zeta.LIT"],
            )

    def test_lit_folder_scan_rejects_missing_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(LitError):
                find_lit_files(Path(temporary) / "missing")


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
            [Image.new("RGBA", (64, 64)), Image.new("RGBA", (32, 64))],
            expected_count=64,
        )
        self.assertFalse(report.ok)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(len(report.warnings), 1)

    def test_non_unit_frame_counts_are_valid_without_unit_expectation(self) -> None:
        for count in (1, 8, 50):
            frames = [Image.new("RGBA", (4, 3)) for _ in range(count)]
            report = validate_frames(frames)
            self.assertTrue(report.ok)
            self.assertEqual(report.warnings, ())

    def test_preview_uses_only_needed_columns(self) -> None:
        wide = Image.new("RGBA", (896, 128))
        self.assertEqual(make_preview([wide]).size, (896, 128))
        arrows = [Image.new("RGBA", (32, 22)) for _ in range(8)]
        self.assertEqual(make_preview(arrows).size, (256, 22))

    def test_non_unit_ugs_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for count, size in ((1, (12, 7)), (8, (4, 3)), (50, (2, 2))):
                frames = [
                    Image.new("RGBA", size, ((index % 16) * 17, 34, 51, 255))
                    for index in range(count)
                ]
                source = root / f"source-{count}.ugs"
                rebuilt = root / f"rebuilt-{count}.ugs"
                write_ugs(frames, source)
                decoded = read_ugs(source)
                write_ugs(decoded, rebuilt)
                self.assertEqual(rebuilt.read_bytes(), source.read_bytes())

    def test_selected_frame_replacement_checks_size(self) -> None:
        frames = [Image.new("RGBA", (4, 3), (index, 0, 0, 255)) for index in range(2)]
        replacement = Image.new("RGB", (4, 3), (99, 0, 0))
        result = replace_selected_frame(frames, 1, replacement)

        self.assertEqual(result[1].mode, "RGBA")
        self.assertEqual(result[1].getpixel((0, 0)), (99, 0, 0, 255))
        self.assertEqual(frames[1].getpixel((0, 0)), (1, 0, 0, 255))
        with self.assertRaises(UgsError):
            replace_selected_frame(frames, 0, Image.new("RGBA", (3, 3)))

    def test_all_frame_replacement_checks_count_and_size(self) -> None:
        frames = [Image.new("RGBA", (4, 3)) for _ in range(2)]
        replacements = [Image.new("RGB", (4, 3)) for _ in range(2)]
        result = replace_all_frames(frames, replacements)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(frame.mode == "RGBA" for frame in result))

        with self.assertRaises(UgsError):
            replace_all_frames(frames, replacements[:1])
        with self.assertRaises(UgsError):
            replace_all_frames(frames, [Image.new("RGBA", (4, 3)), Image.new("RGBA", (3, 3))])

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
