from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from isa import (
    DATA_START,
    DMEM_SIZE,
    IMEM_SIZE,
    WORD_MASK,
    WORD_SIZE,
    decode,
    disassemble,
    encode,
    lower,
    upper,
)

TEXT = "text"
DATA = "data"
SECTIONS = (TEXT, DATA)
MAGIC = b"CSA4"
HEADER_SIZE = 12
LIMITS = {TEXT: (0, IMEM_SIZE), DATA: (DATA_START, DMEM_SIZE)}
PRINTABLE = range(0x20, 0x7F)


class LinkError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Placed:
    module: str
    section: str
    index: int
    address: int
    data: bytearray

    @property
    def end(self) -> int:
        return self.address + len(self.data)


@dataclass(frozen=True, slots=True)
class SourceInfo:
    module: str
    line: int
    text: str


@dataclass(slots=True)
class Program:
    text: bytearray
    data: bytearray
    symbols: dict[str, tuple[str, int]]
    source: dict[int, SourceInfo]
    covered: dict[str, list[tuple[int, int]]]


def align(value: int) -> int:
    return (value + WORD_SIZE - 1) // WORD_SIZE * WORD_SIZE


def place_modules(modules: list[dict[str, Any]]) -> dict[tuple[int, str, int], Placed]:
    placed: dict[tuple[int, str, int], Placed] = {}
    high = {section: LIMITS[section][0] for section in SECTIONS}
    for number, module in enumerate(modules):
        for section in SECTIONS:
            cursor = align(high[section])
            for index, fragment in enumerate(module["sections"][section]):
                data = bytearray.fromhex(fragment["data"])
                address = cursor if fragment["org"] is None else fragment["org"]
                placed[number, section, index] = Placed(module["name"], section, index, address, data)
                cursor = align(address + len(data))
                high[section] = max(high[section], address + len(data))
    check_layout(list(placed.values()))
    return placed


def check_layout(fragments: list[Placed]) -> None:
    for section in SECTIONS:
        low, high = LIMITS[section]
        used = sorted((item for item in fragments if item.section == section and item.data), key=lambda f: f.address)
        for item in used:
            if item.address < low or item.end > high:
                span = f"{item.address:#06x}..{item.end:#06x}"
                msg = f"{item.module}: {section} fragment {span} is out of {low:#x}..{high:#x}"
                raise LinkError(msg)
        for first, second in itertools.pairwise(used):
            if second.address < first.end:
                msg = f"{section} overlap: {first.module} and {second.module} at {second.address:#06x}"
                raise LinkError(msg)


def resolve_symbols(
    modules: list[dict[str, Any]], placed: dict[tuple[int, str, int], Placed]
) -> tuple[list[dict[str, int]], dict[str, tuple[str, int]]]:
    local: list[dict[str, int]] = []
    exported: dict[str, tuple[str, int]] = {}
    owners: dict[str, str] = {}
    for number, module in enumerate(modules):
        table = {
            name: placed[number, item["section"], item["fragment"]].address + item["offset"]
            for name, item in module["symbols"].items()
        }
        local.append(table)
        for name in module["exports"]:
            if name in owners:
                msg = f"symbol {name} is exported by both {owners[name]} and {module['name']}"
                raise LinkError(msg)
            owners[name] = module["name"]
            exported[name] = (module["symbols"][name]["section"], table[name])
    return local, exported


def target_address(
    module: dict[str, Any], table: dict[str, int], exported: dict[str, tuple[str, int]], name: str
) -> int:
    if name in table:
        return table[name]
    if name in exported:
        return exported[name][1]
    msg = f"{module['name']}: undefined external symbol {name}"
    raise LinkError(msg)


def patch_instruction(word: int, kind: str, target: int, place: int) -> int:
    instr = decode(word)
    imm = {"pcrel": target - place, "hi": upper(target), "lo": lower(target), "abs": target}[kind]
    try:
        return encode(replace(instr, imm=imm))
    except ValueError as error:
        hint = ", use la to load the address" if kind == "abs" else ""
        msg = f"cannot relocate {disassemble(instr, place)} at {place:#06x} to {target:#x}: {error}{hint}"
        raise LinkError(msg) from None


def apply_relocations(
    modules: list[dict[str, Any]],
    placed: dict[tuple[int, str, int], Placed],
    local: list[dict[str, int]],
    exported: dict[str, tuple[str, int]],
) -> None:
    for number, module in enumerate(modules):
        for relocation in module["relocations"]:
            fragment = placed[number, relocation["section"], relocation["fragment"]]
            offset = relocation["offset"]
            target = target_address(module, local[number], exported, relocation["symbol"]) + relocation["addend"]
            word = int.from_bytes(fragment.data[offset : offset + WORD_SIZE], "little")
            if relocation["kind"] == "word":
                word = target & WORD_MASK
            else:
                word = patch_instruction(word, relocation["kind"], target, fragment.address + offset)
            fragment.data[offset : offset + WORD_SIZE] = word.to_bytes(WORD_SIZE, "little")


def check_imports(modules: list[dict[str, Any]], exported: dict[str, tuple[str, int]]) -> None:
    for module in modules:
        for name in module["imports"]:
            if name not in exported:
                msg = f"{module['name']}: external symbol {name} is not exported by any module"
                raise LinkError(msg)


def link(modules: list[dict[str, Any]]) -> Program:
    if not modules:
        msg = "nothing to link"
        raise LinkError(msg)
    placed = place_modules(modules)
    local, exported = resolve_symbols(modules, placed)
    check_imports(modules, exported)
    apply_relocations(modules, placed, local, exported)
    images = {
        section: bytearray(max([0, *(f.end for f in placed.values() if f.section == section)])) for section in SECTIONS
    }
    for fragment in placed.values():
        images[fragment.section][fragment.address : fragment.end] = fragment.data
    symbols: dict[str, tuple[str, int]] = {}
    for number, module in enumerate(modules):
        for name, item in module["symbols"].items():
            symbols.setdefault(name, (item["section"], local[number][name]))
    source = {
        placed[number, line["section"], line["fragment"]].address + line["offset"]: SourceInfo(
            module["name"], line["line"], line["text"]
        )
        for number, module in enumerate(modules)
        for line in module["source"]
    }
    covered = {
        section: sorted((f.address, f.end) for f in placed.values() if f.section == section and f.data)
        for section in SECTIONS
    }
    return Program(images[TEXT], images[DATA], symbols, source, covered)


def to_binary(program: Program) -> bytes:
    header = MAGIC + len(program.text).to_bytes(4, "little") + len(program.data).to_bytes(4, "little")
    return header + bytes(program.text) + bytes(program.data)


def from_binary(binary: bytes) -> tuple[bytes, bytes]:
    if binary[:4] != MAGIC or len(binary) < HEADER_SIZE:
        msg = "not a CSA4 binary"
        raise ValueError(msg)
    text_size = int.from_bytes(binary[4:8], "little")
    data_size = int.from_bytes(binary[8:12], "little")
    if len(binary) != HEADER_SIZE + text_size + data_size:
        msg = "corrupted CSA4 binary: wrong size"
        raise ValueError(msg)
    return binary[HEADER_SIZE : HEADER_SIZE + text_size], binary[HEADER_SIZE + text_size :]


def labels_by_address(program: Program, section: str) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for name, (where, address) in sorted(program.symbols.items(), key=lambda item: (item[1][1], item[0])):
        if where == section:
            result.setdefault(address, []).append(name)
    return result


def text_listing(program: Program) -> list[str]:
    labels = labels_by_address(program, TEXT)
    lines = ["; instruction memory"]
    for start, end in program.covered[TEXT]:
        for address in range(start, end, WORD_SIZE):
            lines.extend(f"{' ' * 18}{name}:" for name in labels.get(address, []))
            word = int.from_bytes(program.text[address : address + WORD_SIZE], "little")
            mnemonic = disassemble(decode(word), address)
            info = program.source.get(address)
            comment = f"; {info.module}:{info.line} {info.text}" if info else ""
            lines.append(f"{address:04x} - {word:08X} - {mnemonic:<28}{comment}".rstrip())
    return lines


def data_listing(program: Program) -> list[str]:
    labels = labels_by_address(program, DATA)
    lines = ["; data memory"]
    for start, end in program.covered[DATA]:
        address = start
        while address < end:
            names = [
                name for offset in range(address, min(address + WORD_SIZE, end)) for name in labels.get(offset, [])
            ]
            lines.extend(f"{' ' * 18}{name}:" for name in names)
            chunk = program.data[address : min(address + WORD_SIZE, end)]
            text = "".join(chr(byte) if byte in PRINTABLE else "." for byte in chunk)
            lines.append(f"{address:04x} - {chunk.hex().upper():<8} - {text}")
            address += WORD_SIZE
    return lines


def listing(program: Program) -> str:
    return "\n".join([*text_listing(program), "", *data_listing(program)]) + "\n"


def debug_info(program: Program) -> dict[str, object]:
    return {
        "source": {
            f"{address}": [info.module, info.line, info.text] for address, info in sorted(program.source.items())
        },
        "labels": {f"{address}": names[0] for address, names in sorted(labels_by_address(program, TEXT).items())},
    }


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Linker: object modules -> binary, listing and debug info")
    parser.add_argument("-o", "--output", type=Path, required=True, help="binary file, e.g. prog.bin")
    parser.add_argument("objects", type=Path, nargs="+")
    options = parser.parse_args(args)
    modules = [json.loads(path.read_text(encoding="utf-8")) for path in options.objects]
    try:
        program = link(modules)
    except LinkError as error:
        parser.exit(1, f"error: {error}\n")
    output: Path = options.output
    output.write_bytes(to_binary(program))
    output.with_suffix(".lst").write_text(listing(program), encoding="utf-8")
    output.with_suffix(".dbg").write_text(json.dumps(debug_info(program), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()