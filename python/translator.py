from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from isa import FORMATS, LOADS, REGISTER_INDEX, WORD_MASK, WORD_SIZE, Format, Instruction, Opcode, encode, lower, upper

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

TEXT = "text"
DATA = "data"
BYTE_MIN = -128
BYTE_MAX = 0xFF
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}
IDENTIFIER = re.compile(r"[A-Za-z_.][\w.]*")
LABEL = re.compile(r"([A-Za-z_.][\w.]*):")
MEMORY = re.compile(r"(.*)\((\w+)\)")
MODIFIER = re.compile(r"%(hi|lo)\(([A-Za-z_.][\w.]*)\)")
USAGE = "usage: translator.py <input.s> <output.o>"


class AsmError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Ref:
    symbol: str
    modifier: str = ""


type Value = int | Ref
type Parsed = tuple[Instruction, Ref | None]
type Expansion = list[tuple[str, list[str]]]


@dataclass(frozen=True, slots=True)
class Location:
    section: str
    fragment: int
    offset: int

    def to_json(self) -> dict[str, str | int]:
        return {"section": self.section, "fragment": self.fragment, "offset": self.offset}


@dataclass(slots=True)
class Fragment:
    org: int | None
    data: bytearray = field(default_factory=bytearray)


@dataclass(frozen=True, slots=True)
class Relocation:
    location: Location
    kind: str
    symbol: str


@dataclass(frozen=True, slots=True)
class SourceLine:
    location: Location
    line: int
    text: str


def unquoted(text: str) -> Iterator[tuple[int, str]]:
    quote = ""
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif quote:
            escaped = char == "\\"
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        else:
            yield index, char


def strip_comment(line: str) -> str:
    for index, char in unquoted(line):
        if char == ";":
            return line[:index]
    return line


def split_operands(text: str) -> list[str]:
    if not text.strip():
        return []
    cuts = [index for index, char in unquoted(text) if char == ","]
    bounds = zip([-1, *cuts], [*cuts, len(text)], strict=True)
    return [text[start + 1 : end].strip() for start, end in bounds]


def expect(operands: list[str], count: int) -> None:
    if len(operands) != count:
        msg = f"expected {count} operands, got {len(operands)}"
        raise AsmError(msg)


def quoted_body(token: str, quote: str) -> str | None:
    if len(token) > 1 and token[0] == quote and token[-1] == quote:
        return token[1:-1]
    return None


def unescape(body: str) -> str:
    result: list[str] = []
    chars = iter(body)
    for char in chars:
        if char != "\\":
            result.append(char)
            continue
        code = next(chars, "")
        if code not in ESCAPES:
            msg = f"unknown escape sequence: \\{code}"
            raise AsmError(msg)
        result.append(ESCAPES[code])
    return "".join(result)


def parse_number(token: str) -> int | None:
    body = quoted_body(token, "'")
    if body is not None:
        char = unescape(body)
        if len(char) != 1:
            msg = f"bad character literal: {token}"
            raise AsmError(msg)
        return ord(char)
    try:
        return int(token, 0)
    except ValueError:
        return None


def split_half(value: int, part: str) -> int:
    return upper(value) if part == "hi" else lower(value)


def relocation_kind(opcode: Opcode, ref: Ref) -> str:
    if FORMATS[opcode] in {Format.BRANCH, Format.JUMP}:
        return "pcrel"
    return ref.modifier or "abs"


def build(opcode: Opcode, value: Value, **registers: int) -> Parsed:
    if isinstance(value, Ref):
        return Instruction(opcode, **registers), value
    return Instruction(opcode, imm=value, **registers), None


class Assembler:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sections: dict[str, list[Fragment]] = {TEXT: [], DATA: []}
        self.symbols: dict[str, Location] = {}
        self.equs: dict[str, int] = {}
        self.exports: set[str] = set()
        self.imports: set[str] = set()
        self.relocations: list[Relocation] = []
        self.source: list[SourceLine] = []
        self.section = TEXT

    def assemble(self, source: str) -> None:
        for number, raw in enumerate(source.splitlines(), start=1):
            try:
                self.statement(number, raw)
            except AsmError as error:
                msg = f"{self.name}:{number}: {error}"
                raise AsmError(msg) from None
        self.check()

    def statement(self, number: int, raw: str) -> None:
        line = strip_comment(raw).strip()
        while match := LABEL.match(line):
            self.define(match.group(1))
            line = line[match.end() :].strip()
        if not line:
            return
        head, *rest = line.split(maxsplit=1)
        head = head.lower()
        operands = split_operands("".join(rest))
        if head.startswith("."):
            directive = DIRECTIVES.get(head)
            if directive is None:
                msg = f"unknown directive: {head}"
                raise AsmError(msg)
            directive(self, operands)
            return
        self.source.append(SourceLine(self.location(), number, line))
        self.instruction(head, operands)

    def current(self) -> Fragment:
        fragments = self.sections[self.section]
        if not fragments:
            fragments.append(Fragment(None))
        return fragments[-1]

    def location(self) -> Location:
        fragment = self.current()
        return Location(self.section, len(self.sections[self.section]) - 1, len(fragment.data))

    def emit(self, data: bytes) -> None:
        self.current().data.extend(data)

    def identifier(self, token: str) -> str:
        if not IDENTIFIER.fullmatch(token):
            msg = f"bad identifier: {token}"
            raise AsmError(msg)
        return token

    def define(self, name: str) -> None:
        if name in self.symbols or name in self.equs:
            msg = f"duplicate symbol: {name}"
            raise AsmError(msg)
        if name in self.imports:
            msg = f"symbol is declared as extern: {name}"
            raise AsmError(msg)
        self.symbols[name] = self.location()

    def register(self, token: str) -> int:
        if token not in REGISTER_INDEX:
            msg = f"unknown register: {token}"
            raise AsmError(msg)
        return REGISTER_INDEX[token]

    def value(self, token: str) -> Value:
        number = parse_number(token)
        if number is not None:
            return number
        if token in self.equs:
            return self.equs[token]
        if match := MODIFIER.fullmatch(token):
            part, symbol = match.groups()
            if symbol in self.equs:
                return split_half(self.equs[symbol], part)
            return Ref(symbol, part)
        return Ref(self.identifier(token))

    def constant(self, token: str) -> int:
        value = self.value(token)
        if isinstance(value, Ref):
            msg = f"constant expected: {token}"
            raise AsmError(msg)
        return value

    def label(self, token: str) -> Ref:
        value = self.value(token)
        if not isinstance(value, Ref) or value.modifier:
            msg = f"label expected: {token}"
            raise AsmError(msg)
        return value

    def memory(self, token: str) -> tuple[Value, int]:
        match = MEMORY.fullmatch(token)
        if not match:
            msg = f"memory operand expected: {token}"
            raise AsmError(msg)
        offset = match.group(1).strip()
        return (self.value(offset) if offset else 0), self.register(match.group(2))

    def instruction(self, name: str, operands: list[str]) -> None:
        if self.section != TEXT:
            msg = "instructions are allowed only in .text"
            raise AsmError(msg)
        rewrite = PSEUDO.get(name)
        expansion = rewrite(operands) if rewrite else [(name, operands)]
        for mnemonic, args in expansion:
            self.real(mnemonic, args)

    def real(self, name: str, operands: list[str]) -> None:
        try:
            opcode = Opcode[name.upper()]
        except KeyError:
            msg = f"unknown instruction: {name}"
            raise AsmError(msg) from None
        try:
            instr, ref = self.parse(opcode, operands)
            word = encode(instr)
        except ValueError as error:
            raise AsmError(str(error)) from None
        if ref is not None:
            self.relocations.append(Relocation(self.location(), relocation_kind(opcode, ref), ref.symbol))
        self.emit(word.to_bytes(WORD_SIZE, "little"))

    def parse(self, opcode: Opcode, operands: list[str]) -> Parsed:
        return PARSERS[FORMATS[opcode]](self, opcode, operands)

    def parse_register(self, opcode: Opcode, operands: list[str]) -> Parsed:
        expect(operands, 3)
        rd, rs1, rs2 = (self.register(token) for token in operands)
        return Instruction(opcode, rd=rd, rs1=rs1, rs2=rs2), None

    def parse_immediate(self, opcode: Opcode, operands: list[str]) -> Parsed:
        if opcode in LOADS:
            expect(operands, 2)
            offset, base = self.memory(operands[1])
            return build(opcode, offset, rd=self.register(operands[0]), rs1=base)
        expect(operands, 3)
        return build(opcode, self.value(operands[2]), rd=self.register(operands[0]), rs1=self.register(operands[1]))

    def parse_store(self, opcode: Opcode, operands: list[str]) -> Parsed:
        expect(operands, 2)
        offset, base = self.memory(operands[1])
        return build(opcode, offset, rs2=self.register(operands[0]), rs1=base)

    def parse_branch(self, opcode: Opcode, operands: list[str]) -> Parsed:
        expect(operands, 3)
        rs1, rs2 = self.register(operands[0]), self.register(operands[1])
        return Instruction(opcode, rs1=rs1, rs2=rs2), self.label(operands[2])

    def parse_upper(self, opcode: Opcode, operands: list[str]) -> Parsed:
        expect(operands, 2)
        return build(opcode, self.value(operands[1]), rd=self.register(operands[0]))

    def parse_jump(self, opcode: Opcode, operands: list[str]) -> Parsed:
        expect(operands, 2)
        return Instruction(opcode, rd=self.register(operands[0])), self.label(operands[1])

    def parse_system(self, opcode: Opcode, operands: list[str]) -> Parsed:
        expect(operands, 0)
        return Instruction(opcode), None

    def data_only(self) -> None:
        if self.section != DATA:
            msg = "data directives are allowed only in .data"
            raise AsmError(msg)

    def directive_text(self, operands: list[str]) -> None:
        expect(operands, 0)
        self.section = TEXT

    def directive_data(self, operands: list[str]) -> None:
        expect(operands, 0)
        self.section = DATA

    def directive_org(self, operands: list[str]) -> None:
        expect(operands, 1)
        address = self.constant(operands[0])
        if address < 0 or (self.section == TEXT and address % WORD_SIZE):
            msg = f"bad .org address: {address:#x}"
            raise AsmError(msg)
        self.sections[self.section].append(Fragment(address))

    def directive_equ(self, operands: list[str]) -> None:
        expect(operands, 2)
        name = self.identifier(operands[0])
        if name in self.equs or name in self.symbols:
            msg = f"duplicate symbol: {name}"
            raise AsmError(msg)
        self.equs[name] = self.constant(operands[1])

    def directive_global(self, operands: list[str]) -> None:
        self.exports.update(self.identifier(token) for token in operands)

    def directive_extern(self, operands: list[str]) -> None:
        for token in operands:
            if self.identifier(token) in self.symbols:
                msg = f"extern symbol is defined locally: {token}"
                raise AsmError(msg)
            self.imports.add(token)

    def directive_byte(self, operands: list[str]) -> None:
        self.data_only()
        for token in operands:
            self.emit(self.byte_item(token))

    def byte_item(self, token: str) -> bytes:
        body = quoted_body(token, '"')
        if body is not None:
            try:
                return unescape(body).encode("ascii")
            except UnicodeEncodeError:
                msg = f"only ASCII strings are supported: {token}"
                raise AsmError(msg) from None
        value = self.constant(token)
        if not BYTE_MIN <= value <= BYTE_MAX:
            msg = f"byte out of range: {value}"
            raise AsmError(msg)
        return bytes([value & BYTE_MAX])

    def directive_word(self, operands: list[str]) -> None:
        self.data_only()
        fragment = self.current()
        if ((fragment.org or 0) + len(fragment.data)) % WORD_SIZE:
            msg = "unaligned .word, use .align 4"
            raise AsmError(msg)
        for token in operands:
            value = self.value(token)
            if isinstance(value, Ref):
                if value.modifier:
                    msg = f"modifier is not allowed in .word: {token}"
                    raise AsmError(msg)
                self.relocations.append(Relocation(self.location(), "word32", value.symbol))
                value = 0
            self.emit((value & WORD_MASK).to_bytes(WORD_SIZE, "little"))

    def directive_space(self, operands: list[str]) -> None:
        self.data_only()
        expect(operands, 1)
        size = self.constant(operands[0])
        if size < 0:
            msg = f"negative .space size: {size}"
            raise AsmError(msg)
        self.emit(bytes(size))

    def directive_align(self, operands: list[str]) -> None:
        self.data_only()
        expect(operands, 1)
        alignment = self.constant(operands[0])
        if alignment <= 0:
            msg = f"bad alignment: {alignment}"
            raise AsmError(msg)
        self.emit(bytes(-len(self.current().data) % alignment))

    def check(self) -> None:
        for name in sorted(self.exports - self.symbols.keys()):
            msg = f"{self.name}: global symbol is not defined: {name}"
            raise AsmError(msg)
        known = self.symbols.keys() | self.imports
        for relocation in self.relocations:
            if relocation.symbol not in known:
                msg = f"{self.name}: undefined symbol: {relocation.symbol}"
                raise AsmError(msg)

    def to_object(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sections": {
                section: [{"org": fragment.org, "data": fragment.data.hex()} for fragment in fragments]
                for section, fragments in self.sections.items()
            },
            "symbols": {name: location.to_json() for name, location in sorted(self.symbols.items())},
            "exports": sorted(self.exports),
            "imports": sorted(self.imports),
            "relocations": [
                {**relocation.location.to_json(), "kind": relocation.kind, "symbol": relocation.symbol}
                for relocation in self.relocations
            ],
            "source": [{**line.location.to_json(), "line": line.line, "text": line.text} for line in self.source],
        }


def alias(mnemonic: str, *template: str) -> Callable[[list[str]], Expansion]:
    arity = len({part for part in template if part.startswith("$")})

    def rewrite(operands: list[str]) -> Expansion:
        expect(operands, arity)
        return [(mnemonic, [operands[int(part[1:])] if part.startswith("$") else part for part in template])]

    return rewrite


def rewrite_li(operands: list[str]) -> Expansion:
    expect(operands, 2)
    rd, token = operands
    number = parse_number(token)
    if number is None:
        return [("lui", [rd, f"%hi({token})"]), ("addi", [rd, rd, f"%lo({token})"])]
    return [("lui", [rd, str(upper(number))]), ("addi", [rd, rd, str(lower(number))])]


def rewrite_jal(operands: list[str]) -> Expansion:
    if len(operands) == 1:
        return [("jal", ["ra", *operands])]
    return [("jal", operands)]


PSEUDO: dict[str, Callable[[list[str]], Expansion]] = {
    "li": rewrite_li,
    "jal": rewrite_jal,
    "mv": alias("addi", "$0", "$1", "0"),
    "nop": alias("addi", "zero", "zero", "0"),
    "j": alias("jal", "zero", "$0"),
    "jr": alias("jalr", "zero", "$0", "0"),
    "ret": alias("jalr", "zero", "ra", "0"),
    "beqz": alias("beq", "$0", "zero", "$1"),
    "bnez": alias("bne", "$0", "zero", "$1"),
    "bgt": alias("blt", "$1", "$0", "$2"),
    "ble": alias("bge", "$1", "$0", "$2"),
}

PARSERS: dict[Format, Callable[[Assembler, Opcode, list[str]], Parsed]] = {
    Format.REGISTER: Assembler.parse_register,
    Format.IMMEDIATE: Assembler.parse_immediate,
    Format.STORE: Assembler.parse_store,
    Format.BRANCH: Assembler.parse_branch,
    Format.UPPER: Assembler.parse_upper,
    Format.JUMP: Assembler.parse_jump,
    Format.SYSTEM: Assembler.parse_system,
}

DIRECTIVES: dict[str, Callable[[Assembler, list[str]], None]] = {
    ".text": Assembler.directive_text,
    ".data": Assembler.directive_data,
    ".org": Assembler.directive_org,
    ".equ": Assembler.directive_equ,
    ".global": Assembler.directive_global,
    ".extern": Assembler.directive_extern,
    ".byte": Assembler.directive_byte,
    ".word": Assembler.directive_word,
    ".space": Assembler.directive_space,
    ".align": Assembler.directive_align,
}


def translate(source: str, name: str) -> dict[str, object]:
    assembler = Assembler(name)
    assembler.assemble(source)
    return assembler.to_object()


def main(args: list[str]) -> None:
    match args:
        case [source, target]:
            source_path = Path(source)
            module = translate(source_path.read_text(encoding="utf-8"), source_path.name)
            Path(target).write_text(json.dumps(module, indent=2) + "\n", encoding="utf-8")
        case _:
            raise SystemExit(USAGE)


if __name__ == "__main__":
    main(sys.argv[1:])