#!/usr/bin/env python3
"""Converter for Discord Times UGS sprite containers and LIT images.

UGS layout used by the game:
    repeated records of <uint16 width><uint16 height><width*height uint16 pixels>

Pixels are transformed ARGB4444 values. The on-disk word is produced as
ROL16(ARGB4444, 3) XOR 0xAAAA. Fully transparent black therefore becomes
0xAAAA, but the format also supports 16 alpha levels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - depends on the local Python installation
    raise SystemExit(
        "Не найдена библиотека Pillow. Установите её командой: python -m pip install Pillow"
    ) from exc


XOR_MASK = 0xAAAA
FRAME_NAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)
GRID_COLUMNS = 8
GRID_ROWS = 8
UNIT_FRAME_COUNT = GRID_COLUMNS * GRID_ROWS
LIT_MAGIC = b"LIT\0"
LIT_VALID_FLAGS = frozenset((0, 2, 4, 6, 8, 10))
LIT_FLAG_SUBSAMPLED = 0x02
LIT_FLAG_RAW = 0x04
LIT_FLAG_ALPHA = 0x08
LIT_MAX_PIXELS = 64_000_000

# A conservative JPEG-style table.  The slightly larger DC divisor keeps the
# unshifted 0..255 LIT samples inside the signed-byte coefficient range.
LIT_QUANT_TABLE = bytes(
    (
        17, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    )
)
LIT_COEFFICIENT_ORDER = bytes(range(64))
LIT_DCT_MATRIX = tuple(
    tuple(
        math.cos((2 * sample + 1) * frequency * math.pi / 16)
        * (1 / math.sqrt(2) if frequency == 0 else 1)
        for frequency in range(8)
    )
    for sample in range(8)
)


class UgsError(Exception):
    """A user-facing validation error."""


class LitError(UgsError):
    """A user-facing LIT validation error."""


@dataclass(frozen=True)
class UgsInfo:
    frame_count: int
    width: int
    height: int
    file_size: int
    sha256: str


@dataclass(frozen=True)
class LitInfo:
    width: int
    height: int
    flags: int
    file_size: int
    sha256: str
    compressed: bool
    subsampled: bool
    has_alpha: bool

    @property
    def format_name(self) -> str:
        if not self.compressed:
            return "YCbCr без сжатия"
        color = "YCbCr 4:2:0" if self.subsampled else "YCbCr 4:4:4"
        return f"DCT {color}" + (" + Alpha" if self.has_alpha else "")


@dataclass(frozen=True)
class FrameSetReport:
    frame_count: int
    width: int | None
    height: int | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def calculate_window_size(screen_width: int, screen_height: int) -> tuple[int, int]:
    """Choose a useful window size without exceeding the current screen."""
    if screen_width <= 0 or screen_height <= 0:
        raise ValueError("Screen dimensions must be positive")
    width = min(screen_width, max(720, min(1100, int(screen_width * 0.88))))
    height = min(screen_height, max(500, min(760, int(screen_height * 0.82))))
    return width, height


def fit_image_inside(
    image_width: int, image_height: int, area_width: int, area_height: int
) -> tuple[int, int]:
    """Fit an image inside an area without ever enlarging it."""
    if min(image_width, image_height, area_width, area_height) <= 0:
        raise ValueError("Image and area dimensions must be positive")
    scale = min(1.0, area_width / image_width, area_height / image_height)
    return max(1, round(image_width * scale)), max(1, round(image_height * scale))


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _lit_layout(width: int, height: int, flags: int) -> tuple[int, int, int]:
    if flags not in LIT_VALID_FLAGS:
        raise LitError(f"Неподдерживаемые флаги LIT: {flags}.")
    if width <= 0 or height <= 0 or width * height > LIT_MAX_PIXELS:
        raise LitError(f"Некорректный размер LIT: {width}×{height}.")
    if flags & LIT_FLAG_RAW:
        return width, height, 16 + width * height * 3

    alignment = 16 if flags & LIT_FLAG_SUBSAMPLED else 8
    padded_width = _round_up(width, alignment)
    padded_height = _round_up(height, alignment)
    full_samples = padded_width * padded_height
    full_plane_size = 128 + full_samples
    if flags & LIT_FLAG_SUBSAMPLED:
        chroma_plane_size = 128 + full_samples // 4
    else:
        chroma_plane_size = full_plane_size
    expected_size = 16 + full_plane_size + 2 * chroma_plane_size
    if flags & LIT_FLAG_ALPHA:
        expected_size += full_plane_size
    return padded_width, padded_height, expected_size


def _parse_lit_header(data: bytes) -> tuple[int, int, int, int, int]:
    if len(data) < 16:
        raise LitError("Оборван заголовок LIT.")
    magic, width, height, flags = struct.unpack_from("<4sIII", data)
    if magic != LIT_MAGIC:
        raise LitError("Файл не является LIT: отсутствует сигнатура LIT\\0.")
    padded_width, padded_height, expected_size = _lit_layout(width, height, flags)
    if len(data) != expected_size:
        difference = len(data) - expected_size
        detail = f"лишних байт: {difference}" if difference > 0 else f"не хватает байт: {-difference}"
        raise LitError(
            f"Некорректный размер LIT: получено {len(data)}, ожидалось {expected_size} ({detail})."
        )
    return width, height, flags, padded_width, padded_height


def _decode_lit_plane(
    chunk: memoryview, blocks_x: int, blocks_y: int
) -> bytearray:
    block_count = blocks_x * blocks_y
    expected_size = 128 + block_count * 64
    if len(chunk) != expected_size:
        raise LitError("Оборвана сжатая плоскость LIT.")
    quantizers = bytes(chunk[:64])
    order = bytes(chunk[64:128])
    if 0 in quantizers:
        raise LitError("В таблице квантования LIT найден нулевой делитель.")
    if set(order) != set(range(64)):
        raise LitError("Таблица порядка коэффициентов LIT повреждена.")

    encoded = chunk[128:]
    width = blocks_x * 8
    result = bytearray(width * blocks_y * 8)
    matrix = LIT_DCT_MATRIX
    for block_index in range(block_count):
        coefficients: list[int] = []
        for index in range(64):
            value = encoded[order[index] * block_count + block_index]
            signed_value = value - 256 if value > 127 else value
            coefficients.append(signed_value * quantizers[index])

        # Separable, unshifted 8×8 inverse DCT used by the game.
        intermediate = [[0.0] * 8 for _ in range(8)]
        for y in range(8):
            row_matrix = matrix[y]
            for u in range(8):
                intermediate[y][u] = sum(
                    row_matrix[v] * coefficients[v * 8 + u] for v in range(8)
                )

        block_x = block_index % blocks_x * 8
        block_y = block_index // blocks_x * 8
        for y in range(8):
            destination = (block_y + y) * width + block_x
            intermediate_row = intermediate[y]
            for x in range(8):
                value = round(
                    sum(matrix[x][u] * intermediate_row[u] for u in range(8)) / 4
                )
                result[destination + x] = max(0, min(255, value))
    return result


def _smooth_lit_chroma(samples: bytearray, blocks_x: int, blocks_y: int) -> None:
    """Apply the block-boundary filter from the original game decoder."""
    width = blocks_x * 8
    for block_y in range(1, blocks_y):
        upper = (block_y * 8 - 1) * width
        lower = upper + width
        for x in range(width):
            correction = int((samples[upper + x] - samples[lower + x]) / 4)
            samples[upper + x] -= correction
            samples[lower + x] += correction
    for block_x in range(1, blocks_x):
        for y in range(blocks_y * 8):
            right = y * width + block_x * 8
            left = right - 1
            correction = int((samples[left] - samples[right]) / 4)
            samples[left] -= correction
            samples[right] += correction


def _upsample_lit_chroma(samples: bytearray, blocks_x: int, blocks_y: int) -> bytearray:
    """Reproduce the game's 2× chroma expansion and smoothing filter."""
    width = blocks_x * 8
    height = blocks_y * 8
    output_width = width * 2
    output_height = height * 2
    output = bytearray(output_width * output_height)
    for y in range(height):
        for x in range(width):
            value = samples[y * width + x]
            destination = y * 2 * output_width + x * 2
            output[destination] = value
            output[destination + 1] = value
            output[destination + output_width] = value
            output[destination + output_width + 1] = value

    for y in range(output_height):
        row = y * output_width
        previous = output[row]
        for x in range(1, output_width - 1):
            current = output[row + x]
            output[row + x] = (2 * current + previous + output[row + x + 1]) // 4
            previous = current
    for x in range(output_width):
        previous = output[x]
        for y in range(1, output_height - 1):
            position = y * output_width + x
            current = output[position]
            output[position] = (
                2 * current + previous + output[position + output_width]
            ) // 4
            previous = current
    return output


def decode_lit(data: bytes) -> Image.Image:
    """Decode one complete LIT file into an RGBA Pillow image."""
    width, height, flags, padded_width, padded_height = _parse_lit_header(data)
    if flags & LIT_FLAG_RAW:
        return Image.frombytes("YCbCr", (width, height), data[16:]).convert("RGBA")

    view = memoryview(data)
    offset = 16
    blocks_x = padded_width // 8
    blocks_y = padded_height // 8
    full_plane_size = 128 + padded_width * padded_height
    luminance = _decode_lit_plane(view[offset : offset + full_plane_size], blocks_x, blocks_y)
    offset += full_plane_size

    if flags & LIT_FLAG_SUBSAMPLED:
        chroma_blocks_x = blocks_x // 2
        chroma_blocks_y = blocks_y // 2
    else:
        chroma_blocks_x = blocks_x
        chroma_blocks_y = blocks_y
    chroma_plane_size = 128 + chroma_blocks_x * chroma_blocks_y * 64
    blue_chroma = _decode_lit_plane(
        view[offset : offset + chroma_plane_size], chroma_blocks_x, chroma_blocks_y
    )
    offset += chroma_plane_size
    red_chroma = _decode_lit_plane(
        view[offset : offset + chroma_plane_size], chroma_blocks_x, chroma_blocks_y
    )
    offset += chroma_plane_size
    _smooth_lit_chroma(blue_chroma, chroma_blocks_x, chroma_blocks_y)
    _smooth_lit_chroma(red_chroma, chroma_blocks_x, chroma_blocks_y)
    if flags & LIT_FLAG_SUBSAMPLED:
        blue_chroma = _upsample_lit_chroma(blue_chroma, chroma_blocks_x, chroma_blocks_y)
        red_chroma = _upsample_lit_chroma(red_chroma, chroma_blocks_x, chroma_blocks_y)

    planes = (
        Image.frombytes("L", (padded_width, padded_height), bytes(luminance)),
        Image.frombytes("L", (padded_width, padded_height), bytes(blue_chroma)),
        Image.frombytes("L", (padded_width, padded_height), bytes(red_chroma)),
    )
    image = Image.merge("YCbCr", planes).convert("RGBA").crop((0, 0, width, height))
    if flags & LIT_FLAG_ALPHA:
        alpha = _decode_lit_plane(
            view[offset : offset + full_plane_size], blocks_x, blocks_y
        )
        image.putalpha(
            Image.frombytes("L", (padded_width, padded_height), bytes(alpha)).crop(
                (0, 0, width, height)
            )
        )
    return image


def read_lit(path: Path) -> Image.Image:
    try:
        return decode_lit(path.read_bytes())
    except OSError as exc:
        raise LitError(f"Не удалось прочитать LIT: {exc}") from exc


def inspect_lit(path: Path) -> LitInfo:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LitError(f"Не удалось прочитать LIT: {exc}") from exc
    width, height, flags, _padded_width, _padded_height = _parse_lit_header(data)
    # Decode as part of validation so corrupt coefficient tables are also found.
    decode_lit(data)
    return LitInfo(
        width=width,
        height=height,
        flags=flags,
        file_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        compressed=not bool(flags & LIT_FLAG_RAW),
        subsampled=bool(flags & LIT_FLAG_SUBSAMPLED),
        has_alpha=bool(flags & LIT_FLAG_ALPHA),
    )


def _pad_lit_plane(plane: Image.Image, width: int, height: int) -> Image.Image:
    if plane.mode != "L":
        plane = plane.convert("L")
    if plane.size == (width, height):
        return plane.copy()
    padded = Image.new("L", (width, height))
    padded.paste(plane, (0, 0))
    if plane.width < width:
        edge = plane.crop((plane.width - 1, 0, plane.width, plane.height)).resize(
            (width - plane.width, plane.height), Image.Resampling.NEAREST
        )
        padded.paste(edge, (plane.width, 0))
    if plane.height < height:
        edge = padded.crop((0, plane.height - 1, width, plane.height)).resize(
            (width, height - plane.height), Image.Resampling.NEAREST
        )
        padded.paste(edge, (0, plane.height))
    return padded


def _encode_lit_plane(plane: Image.Image) -> bytes:
    width, height = plane.size
    if width % 8 or height % 8:
        raise LitError("Внутренняя плоскость LIT должна делиться на 8.")
    blocks_x = width // 8
    blocks_y = height // 8
    block_count = blocks_x * blocks_y
    frequency_data = [bytearray(block_count) for _ in range(64)]
    pixels = plane.load()
    matrix = LIT_DCT_MATRIX

    for block_index in range(block_count):
        origin_x = block_index % blocks_x * 8
        origin_y = block_index // blocks_x * 8
        intermediate = [[0.0] * 8 for _ in range(8)]
        for y in range(8):
            for u in range(8):
                intermediate[y][u] = sum(
                    matrix[x][u] * pixels[origin_x + x, origin_y + y] for x in range(8)
                )
        for v in range(8):
            for u in range(8):
                coefficient = sum(
                    matrix[y][v] * intermediate[y][u] for y in range(8)
                ) / 4
                quantized = max(
                    -128,
                    min(127, round(coefficient / LIT_QUANT_TABLE[v * 8 + u])),
                )
                frequency_data[v * 8 + u][block_index] = quantized & 0xFF

    return b"".join(
        (LIT_QUANT_TABLE, LIT_COEFFICIENT_ORDER, *(bytes(values) for values in frequency_data))
    )


def choose_lit_flags(image: Image.Image, preferred: int | None = None) -> int:
    """Choose a compatible format while preserving an original format when possible."""
    alpha = image.convert("RGBA").getchannel("A")
    has_transparency = alpha.getextrema()[0] < 255
    if preferred in LIT_VALID_FLAGS:
        if not has_transparency or preferred & LIT_FLAG_ALPHA:
            return preferred
    return 8 if has_transparency else 4


def encode_lit(image: Image.Image, flags: int | None = None) -> bytes:
    """Encode a Pillow image into a game-compatible LIT byte stream."""
    rgba = image.convert("RGBA")
    width, height = rgba.size
    selected_flags = choose_lit_flags(rgba) if flags is None else flags
    padded_width, padded_height, _expected_size = _lit_layout(width, height, selected_flags)
    has_transparency = rgba.getchannel("A").getextrema()[0] < 255
    if has_transparency and not selected_flags & LIT_FLAG_ALPHA:
        raise LitError("Выбранный вариант LIT не поддерживает прозрачность PNG.")

    header = struct.pack("<4sIII", LIT_MAGIC, width, height, selected_flags)
    ycbcr = rgba.convert("RGB").convert("YCbCr")
    if selected_flags & LIT_FLAG_RAW:
        return header + ycbcr.tobytes()

    luminance, blue_chroma, red_chroma = ycbcr.split()
    luminance = _pad_lit_plane(luminance, padded_width, padded_height)
    blue_chroma = _pad_lit_plane(blue_chroma, padded_width, padded_height)
    red_chroma = _pad_lit_plane(red_chroma, padded_width, padded_height)
    if selected_flags & LIT_FLAG_SUBSAMPLED:
        chroma_size = (padded_width // 2, padded_height // 2)
        blue_chroma = blue_chroma.resize(chroma_size, Image.Resampling.BOX)
        red_chroma = red_chroma.resize(chroma_size, Image.Resampling.BOX)

    parts = [
        header,
        _encode_lit_plane(luminance),
        _encode_lit_plane(blue_chroma),
        _encode_lit_plane(red_chroma),
    ]
    if selected_flags & LIT_FLAG_ALPHA:
        alpha = _pad_lit_plane(rgba.getchannel("A"), padded_width, padded_height)
        parts.append(_encode_lit_plane(alpha))
    encoded = b"".join(parts)
    if len(encoded) != _expected_size:
        raise LitError("Внутренняя ошибка при расчёте размера LIT.")
    return encoded


def write_lit(
    image: Image.Image,
    destination: Path,
    overwrite: bool = False,
    flags: int | None = None,
) -> LitInfo:
    if destination.exists() and not overwrite:
        raise LitError(f"Файл уже существует: {destination}. Используйте --force для замены.")
    data = encode_lit(image, flags)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    except OSError as exc:
        raise LitError(f"Не удалось сохранить LIT: {exc}") from exc
    return inspect_lit(destination)


def extract_lit(source: Path, destination: Path, overwrite: bool = False) -> LitInfo:
    if destination.exists() and not overwrite:
        raise LitError(f"Файл уже существует: {destination}. Используйте --force для замены.")
    image = read_lit(source)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, "PNG")
    except OSError as exc:
        raise LitError(f"Не удалось сохранить PNG: {exc}") from exc
    return inspect_lit(source)


def build_lit(
    source: Path,
    destination: Path,
    overwrite: bool = False,
    flags: int | None = None,
) -> LitInfo:
    try:
        with Image.open(source) as opened:
            image = opened.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise LitError(f"Не удалось открыть PNG: {exc}") from exc
    return write_lit(image, destination, overwrite, flags)


def find_lit_files(folder: Path) -> list[Path]:
    """Return all LIT files in a folder and its subfolders in stable order."""
    if not folder.is_dir():
        raise LitError(f"Папка не найдена: {folder}")
    try:
        files = [
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".lit"
        ]
    except OSError as exc:
        raise LitError(f"Не удалось прочитать папку LIT: {exc}") from exc
    return sorted(files, key=lambda path: str(path.relative_to(folder)).casefold())


def rotate_left_16(value: int, count: int) -> int:
    count %= 16
    return ((value << count) | (value >> (16 - count))) & 0xFFFF


def rotate_right_16(value: int, count: int) -> int:
    count %= 16
    return ((value >> count) | (value << (16 - count))) & 0xFFFF


def decode_ugs_pixel(value: int) -> tuple[int, int, int, int]:
    argb4444 = rotate_right_16(value ^ XOR_MASK, 3)
    alpha4 = (argb4444 >> 12) & 0x0F
    red4 = (argb4444 >> 8) & 0x0F
    green4 = (argb4444 >> 4) & 0x0F
    blue4 = argb4444 & 0x0F
    # Bit replication is the exact 4-bit to 8-bit expansion: n * 17.
    return red4 * 17, green4 * 17, blue4 * 17, alpha4 * 17


def encode_ugs_pixel(red: int, green: int, blue: int, alpha: int) -> int:
    # The original Discord Times converter truncates each channel to its high nibble.
    argb4444 = ((alpha >> 4) << 12) | ((red >> 4) << 8) | ((green >> 4) << 4) | (blue >> 4)
    return rotate_left_16(argb4444, 3) ^ XOR_MASK


def read_ugs(path: Path) -> list[Image.Image]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise UgsError(f"Не удалось прочитать файл: {exc}") from exc

    if not data:
        raise UgsError("UGS-файл пуст.")

    frames: list[Image.Image] = []
    offset = 0
    expected_size: tuple[int, int] | None = None

    while offset < len(data):
        if len(data) - offset < 4:
            raise UgsError(f"Оборван заголовок кадра {len(frames)} по адресу {offset}.")
        width, height = struct.unpack_from("<HH", data, offset)
        if width == 0 or height == 0 or width > 4096 or height > 4096:
            raise UgsError(
                f"Некорректный размер кадра {len(frames)}: {width}×{height} "
                f"(адрес {offset})."
            )
        if expected_size is None:
            expected_size = (width, height)
        elif expected_size != (width, height):
            raise UgsError(
                f"Кадр {len(frames)} имеет размер {width}×{height}, "
                f"ожидался {expected_size[0]}×{expected_size[1]}."
            )

        pixel_bytes = width * height * 2
        record_end = offset + 4 + pixel_bytes
        if record_end > len(data):
            raise UgsError(
                f"Оборваны пиксели кадра {len(frames)}: требуется ещё "
                f"{record_end - len(data)} байт."
            )

        values = struct.unpack_from(f"<{width * height}H", data, offset + 4)
        image = Image.new("RGBA", (width, height))
        image.putdata([decode_ugs_pixel(value) for value in values])
        frames.append(image)
        offset = record_end

    return frames


def inspect_ugs(path: Path) -> UgsInfo:
    """Validate a UGS file and return a compact summary."""
    frames = read_ugs(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise UgsError(f"Не удалось прочитать файл: {exc}") from exc
    return UgsInfo(
        frame_count=len(frames),
        width=frames[0].width,
        height=frames[0].height,
        file_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def create_backup(path: Path) -> Path:
    """Copy an existing file to the first available numbered .bak path."""
    if not path.is_file():
        raise UgsError(f"Нечего сохранять в резервную копию: {path}")

    candidate = path.with_name(path.name + ".bak")
    number = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak.{number}")
        number += 1
    try:
        shutil.copy2(path, candidate)
    except OSError as exc:
        raise UgsError(f"Не удалось создать резервную копию: {exc}") from exc
    return candidate


def install_ugs(source: Path, target: Path) -> Path:
    """Validate and atomically install a UGS over an existing game file."""
    source_info = inspect_ugs(source)
    if not target.is_file():
        raise UgsError(f"Файл игры не найден: {target}")
    target_info = inspect_ugs(target)
    source_layout = (source_info.frame_count, source_info.width, source_info.height)
    target_layout = (target_info.frame_count, target_info.width, target_info.height)
    if source_layout != target_layout:
        raise UgsError(
            "Новый UGS несовместим с заменяемым: "
            f"{source_info.frame_count} кадров {source_info.width}×{source_info.height} вместо "
            f"{target_info.frame_count} кадров {target_info.width}×{target_info.height}."
        )
    try:
        if source.resolve() == target.resolve():
            raise UgsError("Исходный файл и файл игры совпадают.")
    except OSError as exc:
        raise UgsError(f"Не удалось проверить пути: {exc}") from exc

    backup = create_backup(target)
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".install.tmp", dir=target.parent
        )
        with os.fdopen(fd, "wb") as temporary, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, temporary)
        os.replace(temporary_name, target)
    except OSError as exc:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise UgsError(
            f"Не удалось установить UGS: {exc}. Исходный файл игры сохранён в {backup}"
        ) from exc
    return backup


def open_in_file_manager(path: Path) -> None:
    """Open a directory (or select a file) using the platform file manager."""
    target = path if path.is_dir() else path.parent
    try:
        if sys.platform == "win32":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except OSError as exc:
        raise UgsError(f"Не удалось открыть папку: {exc}") from exc


def validate_frames(
    frames: list[Image.Image], expected_count: int | None = None
) -> FrameSetReport:
    """Collect all useful frame errors and non-blocking compatibility warnings."""
    if not frames:
        return FrameSetReport(0, None, None, ("Кадры не загружены.",), ())

    width, height = frames[0].size
    errors: list[str] = []
    warnings: list[str] = []
    if width <= 0 or height <= 0:
        errors.append("Размер первого кадра некорректен.")
    if width > 65535 or height > 65535:
        errors.append("Размер кадра не помещается в формат UGS.")

    wrong_sizes = [
        index for index, frame in enumerate(frames) if frame.size != (width, height)
    ]
    if wrong_sizes:
        shown = ", ".join(str(index) for index in wrong_sizes[:10])
        suffix = "…" if len(wrong_sizes) > 10 else ""
        errors.append(f"Кадры другого размера: {shown}{suffix}.")

    if expected_count is not None and len(frames) != expected_count:
        warnings.append(
            f"Для сетки 8×8 ожидается {expected_count} кадров, загружено {len(frames)}."
        )

    return FrameSetReport(
        frame_count=len(frames),
        width=width,
        height=height,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def load_png_frames(paths: list[Path]) -> list[Image.Image]:
    """Load PNG files in natural filename order."""
    if not paths:
        raise UgsError("PNG-файлы не выбраны.")

    def natural_key(path: Path) -> list[object]:
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]

    frames: list[Image.Image] = []
    for path in sorted(paths, key=natural_key):
        if path.suffix.lower() != ".png":
            continue
        try:
            with Image.open(path) as opened:
                frames.append(opened.convert("RGBA"))
        except OSError as exc:
            raise UgsError(f"Не удалось прочитать {path.name}: {exc}") from exc
    if not frames:
        raise UgsError("Среди выбранных файлов нет PNG.")
    return frames


def replace_selected_frame(
    frames: list[Image.Image], index: int, replacement: Image.Image
) -> list[Image.Image]:
    """Return a copy of a frame set with one size-compatible replacement."""
    if not frames:
        raise UgsError("Сначала загрузите кадры.")
    if index < 0 or index >= len(frames):
        raise UgsError(f"Кадр {index} отсутствует.")
    if replacement.size != frames[index].size:
        raise UgsError(
            f"Размер нового кадра {replacement.width}×{replacement.height}, "
            f"ожидался {frames[index].width}×{frames[index].height}."
        )
    result = list(frames)
    result[index] = replacement.convert("RGBA")
    return result


def replace_all_frames(
    frames: list[Image.Image], replacements: list[Image.Image]
) -> list[Image.Image]:
    """Validate and return a complete frame-set replacement."""
    if not frames:
        raise UgsError("Сначала загрузите исходный UGS или спрайт-лист.")
    if len(replacements) != len(frames):
        raise UgsError(
            f"Количество новых кадров: {len(replacements)}, ожидалось: {len(frames)}."
        )
    expected_size = frames[0].size
    wrong_sizes = [
        index for index, frame in enumerate(replacements) if frame.size != expected_size
    ]
    if wrong_sizes:
        shown = ", ".join(str(index) for index in wrong_sizes[:10])
        suffix = "…" if len(wrong_sizes) > 10 else ""
        raise UgsError(
            f"Новые кадры другого размера: {shown}{suffix}. "
            f"Ожидался размер {expected_size[0]}×{expected_size[1]}."
        )
    return [frame.convert("RGBA") for frame in replacements]


def split_sprite_sheet(
    source: Path, columns: int = GRID_COLUMNS, rows: int = GRID_ROWS
) -> list[Image.Image]:
    """Split a regularly spaced sprite sheet into row-major frames."""
    try:
        with Image.open(source) as opened:
            sheet = opened.convert("RGBA")
    except OSError as exc:
        raise UgsError(f"Не удалось прочитать спрайт-лист: {exc}") from exc
    if sheet.width % columns or sheet.height % rows:
        raise UgsError(
            f"Размер спрайт-листа {sheet.width}×{sheet.height} не делится на сетку "
            f"{columns}×{rows}."
        )
    frame_width = sheet.width // columns
    frame_height = sheet.height // rows
    if frame_width == 0 or frame_height == 0:
        raise UgsError("Ячейки спрайт-листа имеют нулевой размер.")
    return [
        sheet.crop(
            (
                column * frame_width,
                row * frame_height,
                (column + 1) * frame_width,
                (row + 1) * frame_height,
            )
        )
        for row in range(rows)
        for column in range(columns)
    ]


def make_preview(frames: list[Image.Image], columns: int | None = None) -> Image.Image:
    if not frames:
        raise UgsError("Невозможно создать предпросмотр без кадров.")
    if columns is None:
        columns = min(GRID_COLUMNS, len(frames))
    if columns <= 0:
        raise UgsError("Количество столбцов предпросмотра должно быть положительным.")
    width, height = frames[0].size
    rows = math.ceil(len(frames) / columns)
    preview = Image.new("RGBA", (columns * width, rows * height), (36, 36, 36, 255))

    # A subtle checkerboard makes transparent areas visible.
    draw = ImageDraw.Draw(preview)
    tile = 8
    for y in range(0, preview.height, tile):
        for x in range(0, preview.width, tile):
            shade = 52 if ((x // tile) + (y // tile)) % 2 else 68
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(shade, shade, shade, 255))

    for index, frame in enumerate(frames):
        preview.alpha_composite(frame, ((index % columns) * width, (index // columns) * height))
    return preview


def make_sprite_sheet(
    frames: list[Image.Image], columns: int = GRID_COLUMNS, rows: int = GRID_ROWS
) -> Image.Image:
    """Combine frames into a transparent, tightly packed sprite sheet."""
    report = validate_frames(frames, expected_count=columns * rows)
    if report.errors:
        raise UgsError("\n".join(report.errors))
    capacity = columns * rows
    if len(frames) > capacity:
        raise UgsError(
            f"В лист {columns}×{rows} помещается {capacity} кадров, загружено {len(frames)}."
        )
    assert report.width is not None and report.height is not None
    sheet = Image.new(
        "RGBA", (columns * report.width, rows * report.height), (0, 0, 0, 0)
    )
    for index, frame in enumerate(frames):
        sheet.alpha_composite(
            frame.convert("RGBA"),
            ((index % columns) * report.width, (index // columns) * report.height),
        )
    return sheet


def export_png_frames(
    frames: list[Image.Image], output_dir: Path, overwrite: bool = False
) -> list[Path]:
    """Export all loaded frames with stable frame_000.png numbering."""
    report = validate_frames(frames)
    if report.errors:
        raise UgsError("\n".join(report.errors))
    digits = max(3, len(str(len(frames) - 1)))
    targets = [output_dir / f"frame_{index:0{digits}d}.png" for index in range(len(frames))]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise UgsError(
            f"В папке уже есть экспортируемые кадры ({len(existing)}). Разрешите замену."
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame, target in zip(frames, targets):
            frame.convert("RGBA").save(target)
    except OSError as exc:
        raise UgsError(f"Не удалось экспортировать PNG: {exc}") from exc
    return targets


def extract_ugs(source: Path, output_dir: Path) -> tuple[int, tuple[int, int]]:
    frames = read_ugs(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    digits = max(3, len(str(len(frames) - 1)))
    for index, frame in enumerate(frames):
        frame.save(output_dir / f"frame_{index:0{digits}d}.png")

    make_preview(frames).save(output_dir / "preview_sheet.png")
    metadata = {
        "format": "Discord Times UGS",
        "converter_format_version": 2,
        "pixel_format": "ROL16(ARGB4444, 3) XOR 0xAAAA, little-endian",
        "alpha_levels": 16,
        "transparent_black": "0xAAAA",
        "frame_count": len(frames),
        "width": frames[0].width,
        "height": frames[0].height,
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (output_dir / "ugs_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(frames), frames[0].size


def find_frames(input_dir: Path) -> list[Path]:
    numbered: list[tuple[int, Path]] = []
    try:
        entries = list(input_dir.iterdir())
    except OSError as exc:
        raise UgsError(f"Не удалось открыть папку с PNG: {exc}") from exc

    for path in entries:
        match = FRAME_NAME_RE.match(path.name)
        if match and path.is_file():
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise UgsError("Не найдены файлы frame_000.png, frame_001.png и т. д.")

    numbered.sort(key=lambda item: item[0])
    indices = [index for index, _ in numbered]
    expected = list(range(len(numbered)))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        raise UgsError(
            "Нумерация кадров должна идти подряд от нуля. "
            + (f"Пропущены: {missing}." if missing else f"Получены номера: {indices}.")
        )
    return [path for _, path in numbered]


def read_expected_metadata(input_dir: Path) -> dict[str, object] | None:
    metadata_path = input_dir / "ugs_info.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UgsError(f"Не удалось прочитать ugs_info.json: {exc}") from exc
    if not isinstance(metadata, dict):
        raise UgsError("ugs_info.json должен содержать объект JSON.")
    return metadata


def write_ugs(
    frames: list[Image.Image], destination: Path, overwrite: bool = False
) -> tuple[int, tuple[int, int]]:
    report = validate_frames(frames)
    if report.errors:
        raise UgsError("\n".join(report.errors))
    if destination.exists() and not overwrite:
        raise UgsError(f"Файл уже существует: {destination}. Используйте --force для замены.")

    output = bytearray()
    expected_size = frames[0].size
    for index, source_frame in enumerate(frames):
        frame = source_frame.convert("RGBA")
        if frame.size != expected_size:
            raise UgsError(
                f"Кадр {index}: размер {frame.width}×{frame.height}, "
                f"ожидался {expected_size[0]}×{expected_size[1]}."
            )
        output.extend(struct.pack("<HH", frame.width, frame.height))
        pixels = frame.get_flattened_data() if hasattr(frame, "get_flattened_data") else frame.getdata()
        for red, green, blue, alpha in pixels:
            output.extend(struct.pack("<H", encode_ugs_pixel(red, green, blue, alpha)))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(fd, "wb") as temporary:
            temporary.write(output)
        os.replace(temporary_name, destination)
    except OSError as exc:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise UgsError(f"Не удалось записать UGS: {exc}") from exc
    return len(frames), expected_size


def build_ugs(input_dir: Path, destination: Path, overwrite: bool = False) -> tuple[int, tuple[int, int]]:
    frame_paths = find_frames(input_dir)
    metadata = read_expected_metadata(input_dir)
    if metadata is not None and metadata.get("pixel_format") == "BGR565 little-endian":
        raise UgsError(
            "Эти PNG были распакованы старой версией конвертера с неверными цветами. "
            "Распакуйте исходный UGS заново в новую папку или удалите ugs_info.json, "
            "если PNG были полностью заменены вашими изображениями."
        )
    if metadata is not None and isinstance(metadata.get("frame_count"), int):
        expected_count = int(metadata["frame_count"])
        if len(frame_paths) != expected_count:
            raise UgsError(
                f"Количество кадров не совпадает с ugs_info.json: "
                f"найдено {len(frame_paths)}, ожидалось {expected_count}."
            )
    frames = load_png_frames(frame_paths)
    if metadata is not None:
        metadata_width = metadata.get("width")
        metadata_height = metadata.get("height")
        if isinstance(metadata_width, int) and isinstance(metadata_height, int):
            if frames[0].size != (metadata_width, metadata_height):
                raise UgsError(
                    f"Размер кадров не совпадает с ugs_info.json: "
                    f"получен {frames[0].width}x{frames[0].height}, "
                    f"ожидался {metadata_width}x{metadata_height}."
                )
    return write_ugs(frames, destination, overwrite)


def resource_path(relative_path: str | Path) -> Path:
    """Return a bundled resource path for source and PyInstaller launches."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / relative_path


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
    except ImportError:
        DND_FILES = None
        TkinterDnD = None

    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    root.title("Discord Times — Graphics Converter")
    try:
        root.iconbitmap(default=str(resource_path("assets/app-icon.ico")))
    except (OSError, tk.TclError):
        # The converter still works when an unpackaged source copy has no icon.
        pass
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width, window_height = calculate_window_size(screen_width, screen_height)
    window_x = max(0, (screen_width - window_width) // 2)
    window_y = max(0, (screen_height - window_height) // 3)
    root.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
    root.minsize(min(900, window_width), min(520, window_height))
    last_directory = Path.cwd()
    frames: list[Image.Image] = []
    grid_photos: list[ImageTk.PhotoImage] = []
    animation_photo: ImageTk.PhotoImage | None = None
    animation_position = 0
    selected_frame_index = 0
    animation_running = True
    last_built_ugs: Path | None = None
    suggested_stem = "sprites"
    workspace_expected_count: int | None = UNIT_FRAME_COUNT
    grid_cell_size = 58
    grid_resize_job: str | None = None
    animation_preview_size = max(96, min(140, window_height // 5))
    lit_image: Image.Image | None = None
    lit_source_path: Path | None = None
    lit_original_flags: int | None = None
    lit_photo: ImageTk.PhotoImage | None = None
    lit_dirty = False
    lit_folder_root: Path | None = None
    lit_folder_files: list[Path] = []
    lit_folder_index = -1

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)
    ugs_tab = ttk.Frame(notebook)
    lit_tab = ttk.Frame(notebook)
    notebook.add(ugs_tab, text="UGS — спрайты")
    notebook.add(lit_tab, text="LIT — изображения")

    outer = ttk.Frame(ugs_tab, padding=12)
    outer.pack(fill="both", expand=True)
    controls_width = min(360, max(300, window_width // 3))
    controls_shell = ttk.Frame(outer, width=controls_width)
    controls_shell.pack(side="left", fill="y", padx=(0, 14))
    controls_shell.pack_propagate(False)
    controls_canvas = tk.Canvas(
        controls_shell,
        width=controls_width,
        highlightthickness=0,
        borderwidth=0,
        background=root.cget("background"),
    )
    controls_scrollbar = ttk.Scrollbar(
        controls_shell, orient="vertical", command=controls_canvas.yview
    )
    controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
    controls_scrollbar.pack(side="right", fill="y")
    controls_canvas.pack(side="left", fill="both", expand=True)
    controls = ttk.Frame(controls_canvas)
    controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")

    def update_controls_scrollregion(_event: object | None = None) -> None:
        controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

    def resize_controls(event: object) -> None:
        width = int(getattr(event, "width", controls_width))
        controls_canvas.itemconfigure(controls_window, width=width)

    controls.bind("<Configure>", update_controls_scrollregion)
    controls_canvas.bind("<Configure>", resize_controls)
    def scroll_controls(event: object) -> None:
        delta = int(getattr(event, "delta", 0))
        if delta:
            controls_canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    controls_shell.bind(
        "<Enter>", lambda _event: root.bind_all("<MouseWheel>", scroll_controls)
    )
    controls_shell.bind(
        "<Leave>", lambda _event: root.unbind_all("<MouseWheel>")
    )
    preview = ttk.Frame(outer)
    preview.pack(side="left", fill="both", expand=True)

    ttk.Label(controls, text="UGS Converter", font=("Segoe UI", 18, "bold")).pack(
        anchor="w", pady=(2, 4)
    )
    ttk.Label(
        controls,
        text="Загрузите UGS, папку кадров, отдельные PNG или спрайт-лист 8×8.",
        wraplength=controls_width - 35,
        justify="left",
    ).pack(anchor="w", pady=(0, 12))

    source_var = tk.StringVar(value="Кадры не загружены")
    status_var = tk.StringVar(
        value=(
            "Перетащите PNG или папку в окно."
            if DND_FILES is not None
            else "Перетаскивание отключено: установите tkinterdnd2."
        )
    )
    direction_var = tk.StringVar(value="1")
    speed_var = tk.IntVar(value=140)

    source_box = ttk.LabelFrame(controls, text="Текущий набор", padding=8)
    source_box.pack(fill="x", pady=(0, 10))
    ttk.Label(
        source_box, textvariable=source_var, wraplength=controls_width - 50
    ).pack(anchor="w")
    ttk.Label(
        source_box,
        textvariable=status_var,
        wraplength=controls_width - 50,
        foreground="#555555",
    ).pack(
        anchor="w", pady=(4, 0)
    )

    grid_box = ttk.LabelFrame(preview, text="Сетка кадров 8×8", padding=8)
    grid_box.pack(fill="both", expand=True)
    grid_frame = ttk.Frame(grid_box)
    grid_frame.pack(anchor="center", expand=True)
    grid_labels: list[ttk.Label] = []
    for index in range(UNIT_FRAME_COUNT):
        label = ttk.Label(
            grid_frame,
            text=f"{index:02d}",
            width=7,
            anchor="center",
            relief="ridge",
            padding=2,
        )
        label.grid(row=index // GRID_COLUMNS, column=index % GRID_COLUMNS, padx=1, pady=1)
        grid_labels.append(label)

    animation_box = ttk.LabelFrame(controls, text="Анимированный предпросмотр", padding=8)
    animation_box.pack(fill="x", pady=(0, 10))
    animation_label = ttk.Label(animation_box, text="Нет кадров", anchor="center")
    animation_label.pack(fill="x", pady=(0, 6))
    animation_controls = ttk.Frame(animation_box)
    animation_controls.pack(fill="x")
    ttk.Label(animation_controls, text="Строка:").pack(side="left")
    direction_combo = ttk.Combobox(
        animation_controls,
        textvariable=direction_var,
        values=[str(number) for number in range(1, GRID_ROWS + 1)],
        width=3,
        state="readonly",
    )
    direction_combo.pack(side="left", padx=(4, 8))
    play_button = ttk.Button(animation_controls, text="Пауза", width=8)
    play_button.pack(side="left")
    ttk.Label(animation_box, text="Скорость, мс:").pack(anchor="w", pady=(7, 0))
    ttk.Scale(animation_box, from_=60, to=500, variable=speed_var, orient="horizontal").pack(
        fill="x"
    )

    def remember(path: Path) -> None:
        nonlocal last_directory
        last_directory = path if path.is_dir() else path.parent

    def checker_preview(frame: Image.Image, size: int) -> Image.Image:
        background = Image.new("RGBA", (size, size), (68, 68, 68, 255))
        draw = ImageDraw.Draw(background)
        tile = max(6, size // 10)
        for y in range(0, size, tile):
            for x in range(0, size, tile):
                shade = 52 if ((x // tile) + (y // tile)) % 2 else 74
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(shade,) * 3 + (255,))
        scale = min(size / frame.width, size / frame.height)
        scaled_size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
        scaled = frame.convert("RGBA").resize(scaled_size, Image.Resampling.NEAREST)
        background.alpha_composite(
            scaled, ((size - scaled.width) // 2, (size - scaled.height) // 2)
        )
        return background

    def animation_indices() -> list[int]:
        if not frames:
            return []
        row = max(0, min(GRID_ROWS - 1, int(direction_var.get()) - 1))
        indices = [
            index
            for index in range(row * GRID_COLUMNS, (row + 1) * GRID_COLUMNS)
            if index < len(frames)
        ]
        return indices or list(range(min(GRID_COLUMNS, len(frames))))

    def show_animation_frame() -> None:
        nonlocal animation_photo, animation_position
        indices = animation_indices()
        if not indices:
            animation_label.configure(image="", text="Нет кадров")
            return
        animation_position %= len(indices)
        image = checker_preview(
            frames[indices[animation_position]], animation_preview_size
        )
        animation_photo = ImageTk.PhotoImage(image)
        animation_label.configure(image=animation_photo, text="")

    def animation_tick() -> None:
        nonlocal animation_position
        if animation_running and frames:
            show_animation_frame()
            animation_position += 1
        root.after(max(60, int(speed_var.get())), animation_tick)

    def toggle_animation() -> None:
        nonlocal animation_running
        animation_running = not animation_running
        play_button.configure(text="Пауза" if animation_running else "Старт")
        if animation_running:
            show_animation_frame()

    play_button.configure(command=toggle_animation)

    def select_cell(index: int) -> None:
        nonlocal animation_position, selected_frame_index
        if index >= len(frames):
            return
        selected_frame_index = index
        direction_var.set(str(index // GRID_COLUMNS + 1))
        animation_position = index % GRID_COLUMNS
        show_animation_frame()
        refresh_selection()
        status_var.set(f"Выбран кадр {index:03d}, строка {index // GRID_COLUMNS + 1}.")

    for index, label in enumerate(grid_labels):
        label.bind("<Button-1>", lambda _event, selected=index: select_cell(selected))

    def refresh_selection() -> None:
        for index, label in enumerate(grid_labels):
            label.configure(
                relief="sunken"
                if index == selected_frame_index and index < len(frames)
                else "ridge"
            )

    def refresh_grid() -> None:
        grid_photos.clear()
        for index, label in enumerate(grid_labels):
            if index < len(frames):
                photo = ImageTk.PhotoImage(
                    checker_preview(frames[index], grid_cell_size)
                )
                grid_photos.append(photo)
                label.configure(image=photo, text="", width=0)
            else:
                label.configure(image="", text=f"{index:02d}", width=7)
        refresh_selection()
        show_animation_frame()

    def apply_grid_size(new_size: int) -> None:
        nonlocal grid_cell_size, grid_resize_job
        grid_resize_job = None
        if new_size == grid_cell_size:
            return
        grid_cell_size = new_size
        if frames:
            refresh_grid()

    def schedule_grid_resize(event: object) -> None:
        nonlocal grid_resize_job
        width = int(getattr(event, "width", 600))
        height = int(getattr(event, "height", 600))
        available = min((width - 36) // GRID_COLUMNS, (height - 42) // GRID_ROWS)
        new_size = max(30, min(72, available - 6))
        if abs(new_size - grid_cell_size) < 2:
            return
        if grid_resize_job is not None:
            root.after_cancel(grid_resize_job)
        grid_resize_job = root.after(120, lambda: apply_grid_size(new_size))

    grid_box.bind("<Configure>", schedule_grid_resize)

    def set_frames(
        new_frames: list[Image.Image],
        caption: str,
        stem: str,
        expected_count: int | None = UNIT_FRAME_COUNT,
    ) -> None:
        nonlocal frames, animation_position, selected_frame_index
        nonlocal suggested_stem, workspace_expected_count
        frames = [frame.convert("RGBA") for frame in new_frames]
        animation_position = 0
        selected_frame_index = 0
        suggested_stem = stem or "sprites"
        workspace_expected_count = expected_count
        source_var.set(caption)
        report = validate_frames(frames, workspace_expected_count)
        if report.errors:
            status_var.set(f"Ошибок: {len(report.errors)}")
        elif report.warnings:
            status_var.set(report.warnings[0])
        else:
            status_var.set(
                f"Готово: {report.frame_count} кадров {report.width}×{report.height}."
            )
        refresh_grid()

    def report_text(report: FrameSetReport) -> str:
        lines = [
            f"Кадров: {report.frame_count}",
            f"Размер: {report.width}×{report.height}" if report.width else "Размер: —",
        ]
        if report.errors:
            lines.extend(["", "Ошибки:", *[f"• {message}" for message in report.errors]])
        if report.warnings:
            lines.extend(["", "Предупреждения:", *[f"• {message}" for message in report.warnings]])
        if report.ok and not report.warnings:
            lines.extend(["", "Ошибок не найдено."])
        return "\n".join(lines)

    def load_paths(paths: list[Path]) -> None:
        ugs_paths = [path for path in paths if path.is_file() and path.suffix.lower() == ".ugs"]
        png_paths = [path for path in paths if path.is_file() and path.suffix.lower() == ".png"]
        directories = [path for path in paths if path.is_dir()]
        if ugs_paths:
            if len(ugs_paths) != 1 or png_paths or directories:
                messagebox.showerror("Открытие UGS", "Перетащите один UGS-файл за раз.")
                return
            load_ugs_path(ugs_paths[0], offer_export=False)
            return
        if directories and not png_paths:
            folder = directories[0]
            try:
                has_numbered_frames = any(
                    path.is_file() and FRAME_NAME_RE.match(path.name)
                    for path in folder.iterdir()
                )
                if has_numbered_frames:
                    png_paths = find_frames(folder)
                else:
                    png_paths = [
                        path
                        for path in folder.glob("*.png")
                        if path.name.lower() != "preview_sheet.png"
                    ]
                loaded = load_png_frames(png_paths)
            except UgsError as exc:
                messagebox.showerror("Ошибка PNG", str(exc))
                return
            remember(folder)
            set_frames(loaded, f"Папка: {folder}", folder.name)
            return
        if not png_paths:
            messagebox.showerror("Ошибка", "Перетащите PNG-файлы или папку с кадрами.")
            return
        if len(png_paths) == 1 and messagebox.askyesno(
            "Импорт PNG", "Разделить этот PNG как спрайт-лист 8×8?"
        ):
            import_sheet_path(png_paths[0])
            return
        try:
            loaded = load_png_frames(png_paths)
        except UgsError as exc:
            messagebox.showerror("Ошибка PNG", str(exc))
            return
        remember(png_paths[0])
        set_frames(loaded, f"Перетащено PNG: {len(loaded)}", png_paths[0].stem)

    def choose_png_files() -> None:
        names = filedialog.askopenfilenames(
            title="Выберите PNG-кадры",
            initialdir=last_directory,
            filetypes=[("PNG", "*.png")],
        )
        if names:
            load_paths([Path(name) for name in names])

    def choose_frame_folder() -> None:
        name = filedialog.askdirectory(title="Выберите папку кадров", initialdir=last_directory)
        if name:
            load_paths([Path(name)])

    def import_sheet_path(path: Path) -> None:
        try:
            loaded = split_sprite_sheet(path)
        except UgsError as exc:
            messagebox.showerror("Ошибка спрайт-листа", str(exc))
            return
        remember(path)
        set_frames(loaded, f"Спрайт-лист: {path.name}", path.stem)

    def import_sheet_clicked() -> None:
        name = filedialog.askopenfilename(
            title="Выберите спрайт-лист 8×8",
            initialdir=last_directory,
            filetypes=[("PNG", "*.png"), ("Все файлы", "*.*")],
        )
        if name:
            import_sheet_path(Path(name))

    def load_ugs_path(source: Path, offer_export: bool) -> None:
        try:
            loaded = read_ugs(source)
        except UgsError as exc:
            messagebox.showerror("UGS повреждён", str(exc))
            return
        remember(source)
        set_frames(
            loaded,
            f"UGS: {source.name}",
            source.stem,
            expected_count=None,
        )
        if not offer_export or not messagebox.askyesno(
            "UGS открыт", "Сохранить распакованные PNG в папку?"
        ):
            return
        output_name = filedialog.askdirectory(
            title="Папка для PNG", initialdir=last_directory
        )
        if not output_name:
            return
        try:
            count, size = extract_ugs(source, Path(output_name))
        except UgsError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return
        status_var.set(f"Распаковано {count} кадров {size[0]}×{size[1]}.")

    def open_ugs_clicked() -> None:
        source_name = filedialog.askopenfilename(
            title="Открыть UGS",
            initialdir=last_directory,
            filetypes=[("UGS sprites", "*.ugs"), ("Все файлы", "*.*")],
        )
        if source_name:
            load_ugs_path(Path(source_name), offer_export=False)

    def validate_clicked() -> None:
        report = validate_frames(frames, workspace_expected_count)
        messagebox.showerror("Найдены ошибки", report_text(report)) if report.errors else messagebox.showinfo(
            "Проверка кадров", report_text(report)
        )

    def build_clicked() -> None:
        nonlocal last_built_ugs
        report = validate_frames(frames, workspace_expected_count)
        if report.errors:
            messagebox.showerror("Нельзя собрать UGS", report_text(report))
            return
        if report.warnings and not messagebox.askyesno(
            "Есть предупреждения", report_text(report) + "\n\nПродолжить сборку?"
        ):
            return
        destination_name = filedialog.asksaveasfilename(
            title="Сохранить UGS",
            initialdir=last_directory,
            initialfile=f"{suggested_stem}-new.ugs",
            defaultextension=".ugs",
            filetypes=[("UGS sprites", "*.ugs"), ("Все файлы", "*.*")],
        )
        if not destination_name:
            return
        destination = Path(destination_name)
        backup: Path | None = None
        if destination.exists() and not messagebox.askyesno(
            "Подтверждение", f"Файл уже существует:\n{destination}\n\nЗаменить его?"
        ):
            return
        try:
            if destination.exists():
                backup = create_backup(destination)
            count, size = write_ugs(frames, destination, overwrite=True)
        except UgsError as exc:
            messagebox.showerror("Ошибка сборки", str(exc))
            return
        remember(destination)
        last_built_ugs = destination
        backup_line = f"\nРезервная копия: {backup}" if backup else ""
        offer_to_open(
            destination,
            f"Собрано кадров: {count}\nРазмер: {size[0]}×{size[1]}\n"
            f"Файл: {destination}{backup_line}",
        )

    def install_clicked() -> None:
        source: Path | None = None
        if last_built_ugs is not None and last_built_ugs.is_file():
            if messagebox.askyesno(
                "Установка", f"Использовать последний собранный файл?\n\n{last_built_ugs}"
            ):
                source = last_built_ugs
        if source is None:
            name = filedialog.askopenfilename(
                title="Выберите новый UGS",
                initialdir=last_directory,
                filetypes=[("UGS sprites", "*.ugs")],
            )
            if not name:
                return
            source = Path(name)
        target_name = filedialog.askopenfilename(
            title="Выберите заменяемый UGS в папке игры",
            initialdir=last_directory,
            filetypes=[("UGS sprites", "*.ugs")],
        )
        if not target_name:
            return
        target = Path(target_name)
        if not messagebox.askyesno(
            "Подтвердите установку",
            f"Новый файл:\n{source}\n\nФайл игры:\n{target}\n\n"
            "Оригинал будет сохранён рядом как резервная копия. Продолжить?",
        ):
            return
        try:
            backup = install_ugs(source, target)
        except UgsError as exc:
            messagebox.showerror("Ошибка установки", str(exc))
            return
        remember(target)
        messagebox.showinfo(
            "Установка завершена",
            f"Установлено:\n{target}\n\nРезервная копия:\n{backup}",
        )

    def export_selected_clicked() -> None:
        if not frames:
            messagebox.showerror("Экспорт", "Сначала загрузите кадры.")
            return
        destination_name = filedialog.asksaveasfilename(
            title="Экспортировать выбранный кадр",
            initialdir=last_directory,
            initialfile=f"{suggested_stem}-frame-{selected_frame_index:03d}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not destination_name:
            return
        destination = Path(destination_name)
        if destination.exists() and not messagebox.askyesno(
            "Подтверждение", f"Файл уже существует:\n{destination}\n\nЗаменить его?"
        ):
            return
        try:
            frames[selected_frame_index].convert("RGBA").save(destination)
        except OSError as exc:
            messagebox.showerror("Ошибка экспорта", f"Не удалось сохранить PNG: {exc}")
            return
        remember(destination)
        status_var.set(f"Экспортирован кадр {selected_frame_index:03d}: {destination.name}")

    def export_all_clicked() -> None:
        if not frames:
            messagebox.showerror("Экспорт", "Сначала загрузите кадры.")
            return
        output_name = filedialog.askdirectory(
            title="Папка для всех кадров", initialdir=last_directory
        )
        if not output_name:
            return
        output_dir = Path(output_name)
        existing = list(output_dir.glob("frame_*.png"))
        overwrite = False
        if existing:
            digits = max(3, len(str(len(frames) - 1)))
            expected_names = {
                f"frame_{index:0{digits}d}.png" for index in range(len(frames))
            }
            extras = [path for path in existing if path.name not in expected_names]
            if extras:
                messagebox.showerror(
                    "Папка содержит лишние кадры",
                    f"Найдено лишних frame_*.png: {len(extras)}. "
                    "Выберите пустую папку, чтобы старые кадры не попали в следующую сборку.",
                )
                return
            overwrite = messagebox.askyesno(
                "Подтверждение",
                f"В папке уже есть frame_*.png: {len(existing)}.\n\nЗаменить совпадающие файлы?",
            )
            if not overwrite:
                return
        try:
            exported = export_png_frames(frames, output_dir, overwrite=overwrite)
        except UgsError as exc:
            messagebox.showerror("Ошибка экспорта", str(exc))
            return
        remember(output_dir)
        offer_to_open(output_dir, f"Экспортировано кадров: {len(exported)}\n{output_dir}")

    def export_sheet_clicked() -> None:
        if not frames:
            messagebox.showerror("Экспорт", "Сначала загрузите кадры.")
            return
        report = validate_frames(frames, expected_count=UNIT_FRAME_COUNT)
        if report.errors:
            messagebox.showerror("Нельзя создать лист", report_text(report))
            return
        if report.warnings and not messagebox.askyesno(
            "Неполная сетка", report_text(report) + "\n\nПустые ячейки останутся прозрачными. Продолжить?"
        ):
            return
        destination_name = filedialog.asksaveasfilename(
            title="Экспортировать спрайт-лист 8×8",
            initialdir=last_directory,
            initialfile=f"{suggested_stem}-sheet-8x8.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not destination_name:
            return
        destination = Path(destination_name)
        if destination.exists() and not messagebox.askyesno(
            "Подтверждение", f"Файл уже существует:\n{destination}\n\nЗаменить его?"
        ):
            return
        try:
            make_sprite_sheet(frames).save(destination)
        except (OSError, UgsError) as exc:
            messagebox.showerror("Ошибка экспорта", str(exc))
            return
        remember(destination)
        offer_to_open(destination, f"Спрайт-лист 8×8 сохранён:\n{destination}")

    def replace_selected_clicked() -> None:
        nonlocal frames
        if not frames:
            messagebox.showerror("Замена кадра", "Сначала откройте UGS или импортируйте лист.")
            return
        source_name = filedialog.askopenfilename(
            title=f"PNG для кадра {selected_frame_index:03d}",
            initialdir=last_directory,
            filetypes=[("PNG", "*.png")],
        )
        if not source_name:
            return
        source = Path(source_name)
        try:
            replacement = load_png_frames([source])[0]
            frames = replace_selected_frame(frames, selected_frame_index, replacement)
        except UgsError as exc:
            messagebox.showerror("Замена кадра", str(exc))
            return
        remember(source)
        refresh_grid()
        status_var.set(f"Кадр {selected_frame_index:03d} заменён: {source.name}")

    def replace_all_clicked() -> None:
        nonlocal frames
        if not frames:
            messagebox.showerror("Замена кадров", "Сначала откройте UGS или импортируйте лист.")
            return
        folder_name = filedialog.askdirectory(
            title="Папка с новыми кадрами", initialdir=last_directory
        )
        if not folder_name:
            return
        folder = Path(folder_name)
        try:
            has_numbered = any(
                path.is_file() and FRAME_NAME_RE.match(path.name)
                for path in folder.iterdir()
            )
            if has_numbered:
                paths = find_frames(folder)
            else:
                paths = [
                    path
                    for path in folder.glob("*.png")
                    if path.name.lower() != "preview_sheet.png"
                ]
            replacements = load_png_frames(paths)
            frames = replace_all_frames(frames, replacements)
        except (OSError, UgsError) as exc:
            messagebox.showerror("Замена кадров", str(exc))
            return
        remember(folder)
        refresh_grid()
        status_var.set(f"Заменены все кадры: {len(frames)} из папки {folder.name}")

    def offer_to_open(path: Path, summary: str) -> None:
        if not messagebox.askyesno("Готово", summary + "\n\nОткрыть папку с результатом?"):
            return
        try:
            open_in_file_manager(path)
        except UgsError as exc:
            messagebox.showerror("Ошибка", str(exc))

    def inspect_clicked() -> None:
        source_name = filedialog.askopenfilename(
            title="Проверить UGS",
            initialdir=last_directory,
            filetypes=[("UGS sprites", "*.ugs"), ("Все файлы", "*.*")],
        )
        if not source_name:
            return
        source = Path(source_name)
        remember(source)
        try:
            details = inspect_ugs(source)
        except UgsError as exc:
            messagebox.showerror("UGS повреждён", str(exc))
            return
        messagebox.showinfo(
            "UGS исправен",
            f"Кадров: {details.frame_count}\n"
            f"Размер кадра: {details.width}×{details.height}\n"
            f"Размер файла: {details.file_size:,} байт\n"
            f"SHA-256: {details.sha256}",
        )

    def add_button_grid(parent: ttk.LabelFrame, items: tuple[tuple[str, object], ...]) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        for index, (text, command) in enumerate(items):
            ttk.Button(parent, text=text, command=command).grid(
                row=index // 2,
                column=index % 2,
                columnspan=2 if len(items) % 2 and index == len(items) - 1 else 1,
                sticky="ew",
                padx=(0, 3) if index % 2 == 0 and index != len(items) - 1 else (3, 0),
                pady=2,
            )

    main_actions = ttk.LabelFrame(controls, text="Основное", padding=8)
    main_actions.pack(fill="x", pady=(0, 8))
    add_button_grid(
        main_actions,
        (
            ("Открыть UGS", open_ugs_clicked),
            ("Собрать UGS", build_clicked),
            ("Импорт листа 8×8", import_sheet_clicked),
            ("Проверить кадры", validate_clicked),
        ),
    )

    export_actions = ttk.LabelFrame(controls, text="Экспорт", padding=8)
    export_actions.pack(fill="x", pady=(0, 8))
    add_button_grid(
        export_actions,
        (
            ("Лист 8×8", export_sheet_clicked),
            ("Выбранный кадр", export_selected_clicked),
            ("Все кадры", export_all_clicked),
        ),
    )

    replace_actions = ttk.LabelFrame(controls, text="Замена", padding=8)
    replace_actions.pack(fill="x")
    add_button_grid(
        replace_actions,
        (
            ("Выбранный кадр", replace_selected_clicked),
            ("Все из папки", replace_all_clicked),
        ),
    )

    # LIT converter lives in its own workspace so the UGS controls stay compact.
    lit_outer = ttk.Frame(lit_tab, padding=12)
    lit_outer.pack(fill="both", expand=True)
    lit_controls = ttk.Frame(lit_outer, width=controls_width)
    lit_controls.pack(side="left", fill="y", padx=(0, 14))
    lit_controls.pack_propagate(False)
    lit_preview = ttk.LabelFrame(lit_outer, text="Предпросмотр", padding=10)
    lit_preview.pack(side="left", fill="both", expand=True)

    ttk.Label(lit_controls, text="LIT Converter", font=("Segoe UI", 18, "bold")).pack(
        anchor="w", pady=(2, 4)
    )
    ttk.Label(
        lit_controls,
        text="Открывайте игровые LIT, экспортируйте их в PNG и собирайте обратно после редактирования.",
        wraplength=controls_width - 25,
        justify="left",
    ).pack(anchor="w", pady=(0, 12))
    lit_source_var = tk.StringVar(value="Изображение не загружено")
    lit_info_var = tk.StringVar(
        value="Перетащите сюда LIT или PNG." if DND_FILES is not None else "Откройте LIT или импортируйте PNG."
    )
    lit_source_box = ttk.LabelFrame(lit_controls, text="Текущее изображение", padding=8)
    lit_source_box.pack(fill="x", pady=(0, 10))
    ttk.Label(
        lit_source_box,
        textvariable=lit_source_var,
        wraplength=controls_width - 50,
        justify="left",
    ).pack(anchor="w")
    ttk.Label(
        lit_source_box,
        textvariable=lit_info_var,
        wraplength=controls_width - 50,
        justify="left",
        foreground="#555555",
    ).pack(anchor="w", pady=(4, 0))

    lit_folder_var = tk.StringVar(value="Папка LIT не выбрана")
    lit_file_var = tk.StringVar()
    lit_folder_box = ttk.LabelFrame(lit_controls, text="Файлы в папке", padding=8)
    lit_folder_box.pack(fill="x", pady=(0, 10))
    ttk.Label(
        lit_folder_box,
        textvariable=lit_folder_var,
        wraplength=controls_width - 50,
        foreground="#555555",
    ).pack(anchor="w", pady=(0, 5))
    lit_file_combo = ttk.Combobox(
        lit_folder_box,
        textvariable=lit_file_var,
        state="readonly",
    )
    lit_file_combo.pack(fill="x")
    lit_navigation = ttk.Frame(lit_folder_box)
    lit_navigation.pack(fill="x", pady=(6, 0))
    lit_previous_button = ttk.Button(
        lit_navigation,
        text="← Назад",
        command=lambda: step_lit_folder(-1),
        state="disabled",
    )
    lit_previous_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
    lit_next_button = ttk.Button(
        lit_navigation,
        text="Вперёд →",
        command=lambda: step_lit_folder(1),
        state="disabled",
    )
    lit_next_button.pack(side="left", fill="x", expand=True, padx=(3, 0))

    lit_preview_label = ttk.Label(
        lit_preview,
        text="Нет изображения",
        anchor="center",
    )
    lit_preview_label.pack(fill="both", expand=True)
    lit_preview_job: str | None = None

    def make_lit_preview(image: Image.Image, width: int, height: int) -> Image.Image:
        width = max(64, width)
        height = max(64, height)
        background = Image.new("RGBA", (width, height), (68, 68, 68, 255))
        draw = ImageDraw.Draw(background)
        tile = max(8, min(width, height) // 18)
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                shade = 52 if ((x // tile) + (y // tile)) % 2 else 78
                draw.rectangle(
                    (x, y, x + tile - 1, y + tile - 1), fill=(shade,) * 3 + (255,)
                )
        size = fit_image_inside(image.width, image.height, width, height)
        if size == image.size:
            scaled = image.convert("RGBA")
        else:
            scaled = image.convert("RGBA").resize(size, Image.Resampling.LANCZOS)
        background.alpha_composite(
            scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2)
        )
        return background

    def refresh_lit_preview() -> None:
        nonlocal lit_photo, lit_preview_job
        lit_preview_job = None
        if lit_image is None:
            lit_preview_label.configure(image="", text="Нет изображения")
            return
        width = max(100, lit_preview.winfo_width() - 24)
        height = max(100, lit_preview.winfo_height() - 34)
        rendered = make_lit_preview(lit_image, width, height)
        lit_photo = ImageTk.PhotoImage(rendered)
        lit_preview_label.configure(image=lit_photo, text="")

    def schedule_lit_preview(_event: object | None = None) -> None:
        nonlocal lit_preview_job
        if lit_preview_job is not None:
            root.after_cancel(lit_preview_job)
        lit_preview_job = root.after(80, refresh_lit_preview)

    lit_preview.bind("<Configure>", schedule_lit_preview)

    def update_lit_description() -> None:
        if lit_image is None:
            lit_source_var.set("Изображение не загружено")
            lit_info_var.set("Откройте LIT или импортируйте PNG.")
            return
        flags = choose_lit_flags(lit_image, lit_original_flags)
        format_info = LitInfo(
            width=lit_image.width,
            height=lit_image.height,
            flags=flags,
            file_size=0,
            sha256="",
            compressed=not bool(flags & LIT_FLAG_RAW),
            subsampled=bool(flags & LIT_FLAG_SUBSAMPLED),
            has_alpha=bool(flags & LIT_FLAG_ALPHA),
        )
        name = lit_source_path.name if lit_source_path is not None else "Новый LIT из PNG"
        lit_source_var.set(name + (" • изменён" if lit_dirty else ""))
        lit_info_var.set(
            f"{lit_image.width}×{lit_image.height} • тип {flags} • {format_info.format_name}"
        )

    def set_lit_image(
        image: Image.Image,
        source: Path | None,
        flags: int | None,
        dirty: bool,
    ) -> None:
        nonlocal lit_image, lit_source_path, lit_original_flags, lit_dirty
        lit_image = image.convert("RGBA")
        lit_source_path = source
        lit_original_flags = choose_lit_flags(lit_image, flags)
        lit_dirty = dirty
        update_lit_description()
        schedule_lit_preview()

    def update_lit_folder_controls() -> None:
        if lit_folder_root is None or not lit_folder_files:
            lit_folder_var.set("Папка LIT не выбрана")
            lit_file_combo.configure(values=())
            lit_file_var.set("")
            lit_previous_button.configure(state="disabled")
            lit_next_button.configure(state="disabled")
            return
        lit_folder_var.set(
            f"{lit_folder_root.name or lit_folder_root} • файлов: {len(lit_folder_files)}"
        )
        names = [
            str(path.relative_to(lit_folder_root)) for path in lit_folder_files
        ]
        lit_file_combo.configure(values=names)
        if 0 <= lit_folder_index < len(lit_folder_files):
            lit_file_combo.current(lit_folder_index)
        else:
            lit_file_var.set("")
        lit_previous_button.configure(
            state="normal" if lit_folder_index > 0 else "disabled"
        )
        lit_next_button.configure(
            state=(
                "normal"
                if lit_folder_index < len(lit_folder_files) - 1
                else "disabled"
            )
        )

    def confirm_discard_lit_changes() -> bool:
        if not lit_dirty:
            return True
        return messagebox.askyesno(
            "Несохранённые изменения",
            "Импортированное PNG ещё не сохранено в LIT. Открыть другой файл и потерять изменения?",
        )

    def load_lit_path(source: Path, confirm_discard: bool = True) -> bool:
        nonlocal lit_folder_index
        if confirm_discard and not confirm_discard_lit_changes():
            update_lit_folder_controls()
            return False
        try:
            source = source.resolve()
            data = source.read_bytes()
            width, height, flags, _padded_width, _padded_height = _parse_lit_header(data)
            lit_info_var.set(f"Чтение {width}×{height}…")
            root.update_idletasks()
            image = decode_lit(data)
        except (OSError, LitError) as exc:
            messagebox.showerror("Ошибка LIT", str(exc))
            update_lit_description()
            update_lit_folder_controls()
            return False
        remember(source)
        set_lit_image(image, source, flags, False)
        try:
            lit_folder_index = lit_folder_files.index(source)
        except ValueError:
            lit_folder_index = -1
        update_lit_folder_controls()
        notebook.select(lit_tab)
        return True

    def open_lit_clicked() -> None:
        source_name = filedialog.askopenfilename(
            title="Открыть LIT",
            initialdir=last_directory,
            filetypes=[("Discord Times LIT", "*.lit"), ("Все файлы", "*.*")],
        )
        if source_name:
            load_lit_path(Path(source_name))

    def open_lit_folder(folder: Path | None = None) -> None:
        nonlocal lit_folder_root, lit_folder_files, lit_folder_index
        if folder is None:
            folder_name = filedialog.askdirectory(
                title="Открыть папку с LIT", initialdir=last_directory
            )
            if not folder_name:
                return
            folder = Path(folder_name)
        if not confirm_discard_lit_changes():
            return
        try:
            folder = folder.resolve()
            files = find_lit_files(folder)
        except (OSError, LitError) as exc:
            messagebox.showerror("Папка LIT", str(exc))
            return
        if not files:
            messagebox.showinfo(
                "Папка LIT", f"В папке и её подпапках не найдено файлов .lit:\n{folder}"
            )
            return
        lit_folder_root = folder
        lit_folder_files = files
        lit_folder_index = 0
        remember(folder)
        update_lit_folder_controls()
        notebook.select(lit_tab)
        load_lit_path(files[0], confirm_discard=False)

    def select_lit_folder_file(_event: object | None = None) -> None:
        index = lit_file_combo.current()
        if 0 <= index < len(lit_folder_files):
            load_lit_path(lit_folder_files[index])

    def step_lit_folder(offset: int) -> None:
        if not lit_folder_files:
            return
        start = lit_folder_index if lit_folder_index >= 0 else -1
        target = max(0, min(len(lit_folder_files) - 1, start + offset))
        load_lit_path(lit_folder_files[target])

    lit_file_combo.bind("<<ComboboxSelected>>", select_lit_folder_file)

    def import_lit_png(source: Path | None = None) -> None:
        if source is None:
            source_name = filedialog.askopenfilename(
                title="Импортировать PNG в LIT",
                initialdir=last_directory,
                filetypes=[("PNG", "*.png")],
            )
            if not source_name:
                return
            source = Path(source_name)
        try:
            with Image.open(source) as opened:
                image = opened.convert("RGBA")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Ошибка PNG", f"Не удалось открыть PNG: {exc}")
            return
        remember(source)
        preferred = lit_original_flags if lit_image is not None else None
        set_lit_image(image, lit_source_path, preferred, True)
        notebook.select(lit_tab)

    def export_lit_png_clicked() -> None:
        if lit_image is None:
            messagebox.showerror("Экспорт PNG", "Сначала откройте LIT.")
            return
        stem = lit_source_path.stem if lit_source_path is not None else "image"
        destination_name = filedialog.asksaveasfilename(
            title="Экспортировать LIT в PNG",
            initialdir=last_directory,
            initialfile=f"{stem}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not destination_name:
            return
        destination = Path(destination_name)
        if destination.exists() and not messagebox.askyesno(
            "Подтверждение", f"Файл уже существует:\n{destination}\n\nЗаменить его?"
        ):
            return
        try:
            lit_image.save(destination, "PNG")
        except OSError as exc:
            messagebox.showerror("Ошибка экспорта", f"Не удалось сохранить PNG: {exc}")
            return
        remember(destination)
        messagebox.showinfo("Готово", f"PNG сохранён:\n{destination}")

    def save_lit_clicked() -> None:
        nonlocal lit_source_path, lit_original_flags, lit_dirty
        if lit_image is None:
            messagebox.showerror("Сборка LIT", "Сначала импортируйте PNG или откройте LIT.")
            return
        if lit_source_path is not None:
            initial_file = f"{lit_source_path.stem}-new.lit"
        else:
            initial_file = "image-new.lit"
        destination_name = filedialog.asksaveasfilename(
            title="Собрать LIT",
            initialdir=last_directory,
            initialfile=initial_file,
            defaultextension=".lit",
            filetypes=[("Discord Times LIT", "*.lit")],
        )
        if not destination_name:
            return
        destination = Path(destination_name)
        if destination.exists() and not messagebox.askyesno(
            "Подтверждение", f"Файл уже существует:\n{destination}\n\nСоздать резервную копию и заменить?"
        ):
            return
        backup: Path | None = None
        try:
            original_data = None
            if not lit_dirty and lit_source_path is not None and lit_source_path.is_file():
                original_data = lit_source_path.read_bytes()
            if destination.exists():
                backup = create_backup(destination)
            lit_info_var.set("Сборка LIT…")
            root.update_idletasks()
            if original_data is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(original_data)
                details = inspect_lit(destination)
            else:
                flags = choose_lit_flags(lit_image, lit_original_flags)
                details = write_lit(lit_image, destination, overwrite=True, flags=flags)
        except (OSError, UgsError) as exc:
            messagebox.showerror("Ошибка сборки LIT", str(exc))
            update_lit_description()
            return
        remember(destination)
        lit_source_path = destination
        lit_original_flags = details.flags
        lit_dirty = False
        update_lit_description()
        backup_line = f"\nРезервная копия: {backup}" if backup is not None else ""
        messagebox.showinfo(
            "LIT собран",
            f"{details.width}×{details.height} • тип {details.flags}\n{destination}{backup_line}",
        )

    def validate_lit_clicked() -> None:
        if lit_image is None:
            messagebox.showerror("Проверка LIT", "Сначала откройте или соберите изображение.")
            return
        try:
            if lit_source_path is not None and not lit_dirty:
                details = inspect_lit(lit_source_path)
            else:
                flags = choose_lit_flags(lit_image, lit_original_flags)
                encoded = encode_lit(lit_image, flags)
                restored = decode_lit(encoded)
                details = LitInfo(
                    width=restored.width,
                    height=restored.height,
                    flags=flags,
                    file_size=len(encoded),
                    sha256=hashlib.sha256(encoded).hexdigest(),
                    compressed=not bool(flags & LIT_FLAG_RAW),
                    subsampled=bool(flags & LIT_FLAG_SUBSAMPLED),
                    has_alpha=bool(flags & LIT_FLAG_ALPHA),
                )
        except LitError as exc:
            messagebox.showerror("LIT повреждён", str(exc))
            return
        messagebox.showinfo(
            "LIT исправен",
            f"Размер: {details.width}×{details.height}\n"
            f"Тип: {details.flags} • {details.format_name}\n"
            f"Размер файла: {details.file_size:,} байт\n"
            f"SHA-256: {details.sha256}",
        )

    lit_actions = ttk.LabelFrame(lit_controls, text="Конвертер", padding=8)
    lit_actions.pack(fill="x")
    add_button_grid(
        lit_actions,
        (
            ("Открыть LIT", open_lit_clicked),
            ("Папка LIT", open_lit_folder),
            ("Экспорт PNG", export_lit_png_clicked),
            ("Импорт PNG", import_lit_png),
            ("Собрать LIT", save_lit_clicked),
            ("Проверить LIT", validate_lit_clicked),
        ),
    )

    if DND_FILES is not None:
        def on_drop(event: object) -> None:
            raw = getattr(event, "data", "")
            dropped = [Path(item) for item in root.tk.splitlist(raw)]
            if (
                len(dropped) == 1
                and dropped[0].is_dir()
                and notebook.select() == str(lit_tab)
            ):
                open_lit_folder(dropped[0])
            elif len(dropped) == 1 and dropped[0].suffix.lower() == ".lit":
                load_lit_path(dropped[0])
            elif (
                len(dropped) == 1
                and dropped[0].suffix.lower() == ".png"
                and notebook.select() == str(lit_tab)
            ):
                import_lit_png(dropped[0])
            else:
                load_paths(dropped)

        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", on_drop)

    def direction_changed(_event: object) -> None:
        nonlocal animation_position
        animation_position = 0
        show_animation_frame()

    direction_combo.bind("<<ComboboxSelected>>", direction_changed)
    root.after(100, animation_tick)
    root.mainloop()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Конвертер графики Discord Times UGS/LIT ↔ PNG")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract", help="распаковать UGS в PNG")
    extract_parser.add_argument("source", type=Path, help="исходный .ugs")
    extract_parser.add_argument("output_dir", type=Path, help="папка для PNG")

    build_parser = subparsers.add_parser("build", help="собрать UGS из frame_*.png")
    build_parser.add_argument("input_dir", type=Path, help="папка с PNG")
    build_parser.add_argument("destination", type=Path, help="выходной .ugs")
    build_parser.add_argument("--force", action="store_true", help="разрешить замену существующего файла")

    inspect_parser = subparsers.add_parser("inspect", help="проверить UGS и показать сведения")
    inspect_parser.add_argument("source", type=Path, help="проверяемый .ugs")

    install_parser = subparsers.add_parser("install", help="установить UGS поверх файла игры")
    install_parser.add_argument("source", type=Path, help="новый .ugs")
    install_parser.add_argument("target", type=Path, help="заменяемый .ugs в папке игры")
    install_parser.add_argument(
        "--yes", action="store_true", help="подтвердить замену и создание резервной копии"
    )

    lit_extract_parser = subparsers.add_parser(
        "lit-extract", help="преобразовать LIT в PNG"
    )
    lit_extract_parser.add_argument("source", type=Path, help="исходный .lit")
    lit_extract_parser.add_argument("destination", type=Path, help="выходной .png")
    lit_extract_parser.add_argument(
        "--force", action="store_true", help="разрешить замену существующего PNG"
    )

    lit_build_parser = subparsers.add_parser(
        "lit-build", help="собрать LIT из PNG"
    )
    lit_build_parser.add_argument("source", type=Path, help="исходный .png")
    lit_build_parser.add_argument("destination", type=Path, help="выходной .lit")
    lit_build_parser.add_argument(
        "--flags",
        type=int,
        choices=sorted(LIT_VALID_FLAGS),
        help="тип LIT 0/2/4/6/8/10; без параметра выбирается автоматически",
    )
    lit_build_parser.add_argument(
        "--force", action="store_true", help="разрешить замену существующего LIT"
    )

    lit_inspect_parser = subparsers.add_parser(
        "lit-inspect", help="проверить LIT и показать сведения"
    )
    lit_inspect_parser.add_argument("source", type=Path, help="проверяемый .lit")
    return parser


def main(argv: list[str] | None = None) -> int:
    # A legacy Windows console may not support every character used in messages.
    # Replacing an unsupported glyph is preferable to failing after conversion.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = create_parser().parse_args(argv)
    if args.command is None:
        run_gui()
        return 0
    try:
        if args.command == "extract":
            count, size = extract_ugs(args.source, args.output_dir)
            print(f"Готово: распаковано {count} кадров {size[0]}x{size[1]} в {args.output_dir}")
        elif args.command == "build":
            count, size = build_ugs(args.input_dir, args.destination, args.force)
            print(f"Готово: собрано {count} кадров {size[0]}x{size[1]} в {args.destination}")
        elif args.command == "inspect":
            details = inspect_ugs(args.source)
            print(
                f"UGS исправен: {details.frame_count} кадров "
                f"{details.width}x{details.height}, {details.file_size} байт\n"
                f"SHA-256: {details.sha256}"
            )
        elif args.command == "install":
            if not args.yes:
                raise UgsError("Для установки добавьте --yes после проверки выбранных путей.")
            backup = install_ugs(args.source, args.target)
            print(f"Готово: установлен {args.target}\nРезервная копия: {backup}")
        elif args.command == "lit-extract":
            details = extract_lit(args.source, args.destination, args.force)
            print(
                f"Готово: LIT {details.width}x{details.height}, тип {details.flags} "
                f"экспортирован в {args.destination}"
            )
        elif args.command == "lit-build":
            details = build_lit(
                args.source, args.destination, args.force, args.flags
            )
            print(
                f"Готово: собран LIT {details.width}x{details.height}, "
                f"тип {details.flags} в {args.destination}"
            )
        elif args.command == "lit-inspect":
            details = inspect_lit(args.source)
            print(
                f"LIT исправен: {details.width}x{details.height}, тип {details.flags}, "
                f"{details.format_name}, {details.file_size} байт\n"
                f"SHA-256: {details.sha256}"
            )
    except UgsError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
