from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

REGISTERS = (
    "zero",
    "ra",
    "sp",
    "gp",
    "tp",
    "t0",
    "t1",
    "t2",
    "s0",
    "s1",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
)
REGISTER_INDEX = {
    **{f"x{index}": index for index in range(len(REGISTERS))},
    **{name: index for index, name in enumerate(REGISTERS)},
    "fp": REGISTERS.index("s0"),
}

WORD_SIZE = 4
WORD_MASK = 0xFFFF_FFFF
SHIFT_LIMIT = 32
IMM12_BITS = 12
BRANCH_BITS = 13
JUMP_BITS = 21
UPPER_BITS = 20
UPPER_SHIFT = 12
LOWER_MASK = 0xFFF
ROUNDING = 0x800

MAJOR_OP = 0b0110011
MAJOR_OP_IMM = 0b0010011
MAJOR_LOAD = 0b0000011
MAJOR_STORE = 0b0100011
MAJOR_BRANCH = 0b1100011
MAJOR_LUI = 0b0110111
MAJOR_JAL = 0b1101111
MAJOR_JALR = 0b1100111
MAJOR_CUSTOM0 = 0b0001011


class Format(Enum):
    REGISTER = "R"
    IMMEDIATE = "I"
    STORE = "S"
    BRANCH = "B"
    UPPER = "U"
    JUMP = "J"
    SYSTEM = "SYS"


class Opcode(Enum):
    ADD = "add"
    SUB = "sub"
    SLL = "sll"
    SLT = "slt"
    SLTU = "sltu"
    XOR = "xor"
    SRL = "srl"
    SRA = "sra"
    OR = "or"
    AND = "and"
    MUL = "mul"
    DIV = "div"
    REM = "rem"

    ADDI = "addi"
    SLTI = "slti"
    XORI = "xori"
    ORI = "ori"
    ANDI = "andi"
    SLLI = "slli"
    SRLI = "srli"
    SRAI = "srai"

    LW = "lw"
    LBU = "lbu"
    SB = "sb"
    SW = "sw"

    BEQ = "beq"
    BNE = "bne"
    BLT = "blt"
    BGE = "bge"
    BLTU = "bltu"
    BGEU = "bgeu"

    LUI = "lui"
    JAL = "jal"
    JALR = "jalr"

    EI = "ei"
    DI = "di"
    IRET = "iret"
    HALT = "halt"

    @property
    def mnemonic(self) -> str:
        return self.value


class Spec(NamedTuple):
    fmt: Format
    major: int
    funct3: int | None = None
    funct7: int | None = None


SPECS: dict[Opcode, Spec] = {
    Opcode.ADD: Spec(Format.REGISTER, MAJOR_OP, 0b000, 0b0000000),
    Opcode.SUB: Spec(Format.REGISTER, MAJOR_OP, 0b000, 0b0100000),
    Opcode.SLL: Spec(Format.REGISTER, MAJOR_OP, 0b001, 0b0000000),
    Opcode.SLT: Spec(Format.REGISTER, MAJOR_OP, 0b010, 0b0000000),
    Opcode.SLTU: Spec(Format.REGISTER, MAJOR_OP, 0b011, 0b0000000),
    Opcode.XOR: Spec(Format.REGISTER, MAJOR_OP, 0b100, 0b0000000),
    Opcode.SRL: Spec(Format.REGISTER, MAJOR_OP, 0b101, 0b0000000),
    Opcode.SRA: Spec(Format.REGISTER, MAJOR_OP, 0b101, 0b0100000),
    Opcode.OR: Spec(Format.REGISTER, MAJOR_OP, 0b110, 0b0000000),
    Opcode.AND: Spec(Format.REGISTER, MAJOR_OP, 0b111, 0b0000000),
    Opcode.MUL: Spec(Format.REGISTER, MAJOR_OP, 0b000, 0b0000001),
    Opcode.DIV: Spec(Format.REGISTER, MAJOR_OP, 0b100, 0b0000001),
    Opcode.REM: Spec(Format.REGISTER, MAJOR_OP, 0b110, 0b0000001),
    Opcode.ADDI: Spec(Format.IMMEDIATE, MAJOR_OP_IMM, 0b000),
    Opcode.SLTI: Spec(Format.IMMEDIATE, MAJOR_OP_IMM, 0b010),
    Opcode.XORI: Spec(Format.IMMEDIATE, MAJOR_OP_IMM, 0b100),
    Opcode.ORI: Spec(Format.IMMEDIATE, MAJOR_OP_IMM, 0b110),
    Opcode.ANDI: Spec(Format.IMMEDIATE, MAJOR_OP_IMM, 0b111),
    Opcode.SLLI: Spec(Format.IMMEDIATE, MAJOR_OP_IMM, 0b001, 0b0000000),
    Opcode.SRLI: Spec(Format.IMMEDIATE, MAJOR_OP_IMM, 0b101, 0b0000000),
    Opcode.SRAI: Spec(Format.IMMEDIATE, MAJOR_OP_IMM, 0b101, 0b0100000),
    Opcode.LW: Spec(Format.IMMEDIATE, MAJOR_LOAD, 0b010),
    Opcode.LBU: Spec(Format.IMMEDIATE, MAJOR_LOAD, 0b100),
    Opcode.SB: Spec(Format.STORE, MAJOR_STORE, 0b000),
    Opcode.SW: Spec(Format.STORE, MAJOR_STORE, 0b010),
    Opcode.BEQ: Spec(Format.BRANCH, MAJOR_BRANCH, 0b000),
    Opcode.BNE: Spec(Format.BRANCH, MAJOR_BRANCH, 0b001),
    Opcode.BLT: Spec(Format.BRANCH, MAJOR_BRANCH, 0b100),
    Opcode.BGE: Spec(Format.BRANCH, MAJOR_BRANCH, 0b101),
    Opcode.BLTU: Spec(Format.BRANCH, MAJOR_BRANCH, 0b110),
    Opcode.BGEU: Spec(Format.BRANCH, MAJOR_BRANCH, 0b111),
    Opcode.LUI: Spec(Format.UPPER, MAJOR_LUI),
    Opcode.JAL: Spec(Format.JUMP, MAJOR_JAL),
    Opcode.JALR: Spec(Format.IMMEDIATE, MAJOR_JALR, 0b000),
    Opcode.EI: Spec(Format.SYSTEM, MAJOR_CUSTOM0, 0b000),
    Opcode.DI: Spec(Format.SYSTEM, MAJOR_CUSTOM0, 0b001),
    Opcode.IRET: Spec(Format.SYSTEM, MAJOR_CUSTOM0, 0b010),
    Opcode.HALT: Spec(Format.SYSTEM, MAJOR_CUSTOM0, 0b111),
}
FORMATS = {opcode: spec.fmt for opcode, spec in SPECS.items()}
DECODE = {(spec.major, spec.funct3, spec.funct7): opcode for opcode, spec in SPECS.items()}

SHIFTS = frozenset({Opcode.SLLI, Opcode.SRLI, Opcode.SRAI})
LOADS = frozenset({Opcode.LW, Opcode.LBU})
USES_RD = frozenset({Format.REGISTER, Format.IMMEDIATE, Format.UPPER, Format.JUMP})
USES_RS1 = frozenset({Format.REGISTER, Format.IMMEDIATE, Format.STORE, Format.BRANCH})
USES_RS2 = frozenset({Format.REGISTER, Format.STORE, Format.BRANCH})


@dataclass(frozen=True, slots=True)
class Instruction:
    opcode: Opcode
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0

    def __post_init__(self) -> None:
        for register in (self.rd, self.rs1, self.rs2):
            if not 0 <= register < len(REGISTERS):
                msg = f"register out of range: {register}"
                raise ValueError(msg)
        if self.opcode in SHIFTS and not 0 <= self.imm < SHIFT_LIMIT:
            msg = f"shift amount out of range: {self.imm}"
            raise ValueError(msg)


def bits(value: int, high: int, low: int) -> int:
    return (value >> low) & ((1 << (high - low + 1)) - 1)


def sign_extend(value: int, width: int) -> int:
    sign = 1 << (width - 1)
    return (value & (sign - 1)) - (value & sign)


def signed_field(value: int, width: int) -> int:
    limit = 1 << (width - 1)
    if not -limit <= value < limit:
        msg = f"immediate {value} does not fit into {width} bits"
        raise ValueError(msg)
    return value & ((1 << width) - 1)


def offset_field(value: int, width: int) -> int:
    if value % 2:
        msg = f"jump offset must be even: {value}"
        raise ValueError(msg)
    return signed_field(value, width)


def upper(value: int) -> int:
    return bits((value & WORD_MASK) + ROUNDING, UPPER_SHIFT + UPPER_BITS - 1, UPPER_SHIFT)


def lower(value: int) -> int:
    return sign_extend(value & LOWER_MASK, IMM12_BITS)


def immediate_i(instr: Instruction, spec: Spec) -> int:
    if instr.opcode in SHIFTS:
        return instr.imm | (spec.funct7 or 0) << 5
    return signed_field(instr.imm, IMM12_BITS)


def encode(instr: Instruction) -> int:
    spec = SPECS[instr.opcode]
    word = spec.major | (spec.funct3 or 0) << 12
    rd, rs1, rs2 = instr.rd << 7, instr.rs1 << 15, instr.rs2 << 20
    match spec.fmt:
        case Format.REGISTER:
            word |= rd | rs1 | rs2 | (spec.funct7 or 0) << 25
        case Format.IMMEDIATE:
            word |= rd | rs1 | immediate_i(instr, spec) << 20
        case Format.STORE:
            imm = signed_field(instr.imm, IMM12_BITS)
            word |= bits(imm, 4, 0) << 7 | rs1 | rs2 | bits(imm, 11, 5) << 25
        case Format.BRANCH:
            imm = offset_field(instr.imm, BRANCH_BITS)
            word |= bits(imm, 11, 11) << 7 | bits(imm, 4, 1) << 8 | rs1 | rs2
            word |= bits(imm, 10, 5) << 25 | bits(imm, 12, 12) << 31
        case Format.UPPER:
            if not 0 <= instr.imm < 1 << UPPER_BITS:
                msg = f"upper immediate {instr.imm} does not fit into {UPPER_BITS} bits"
                raise ValueError(msg)
            word |= rd | instr.imm << UPPER_SHIFT
        case Format.JUMP:
            imm = offset_field(instr.imm, JUMP_BITS)
            word |= rd | bits(imm, 19, 12) << 12 | bits(imm, 11, 11) << 20
            word |= bits(imm, 10, 1) << 21 | bits(imm, 20, 20) << 31
        case Format.SYSTEM:
            pass
    return word


IMMEDIATE_DECODERS: dict[Format, Callable[[int], int]] = {
    Format.IMMEDIATE: lambda word: sign_extend(bits(word, 31, 20), IMM12_BITS),
    Format.STORE: lambda word: sign_extend(bits(word, 31, 25) << 5 | bits(word, 11, 7), IMM12_BITS),
    Format.BRANCH: lambda word: sign_extend(
        bits(word, 31, 31) << 12 | bits(word, 7, 7) << 11 | bits(word, 30, 25) << 5 | bits(word, 11, 8) << 1,
        BRANCH_BITS,
    ),
    Format.UPPER: lambda word: bits(word, 31, 12),
    Format.JUMP: lambda word: sign_extend(
        bits(word, 31, 31) << 20 | bits(word, 19, 12) << 12 | bits(word, 20, 20) << 11 | bits(word, 30, 21) << 1,
        JUMP_BITS,
    ),
}


def find_opcode(word: int) -> Opcode:
    major, funct3, funct7 = bits(word, 6, 0), bits(word, 14, 12), bits(word, 31, 25)
    for key in ((major, funct3, funct7), (major, funct3, None), (major, None, None)):
        if key in DECODE:
            return DECODE[key]
    msg = f"illegal instruction: {word:08X}"
    raise ValueError(msg)


def decode(word: int) -> Instruction:
    opcode = find_opcode(word)
    fmt = FORMATS[opcode]
    if opcode in SHIFTS:
        imm = bits(word, 24, 20)
    else:
        decoder = IMMEDIATE_DECODERS.get(fmt)
        imm = decoder(word) if decoder else 0
    return Instruction(
        opcode,
        rd=bits(word, 11, 7) if fmt in USES_RD else 0,
        rs1=bits(word, 19, 15) if fmt in USES_RS1 else 0,
        rs2=bits(word, 24, 20) if fmt in USES_RS2 else 0,
        imm=imm,
    )


def disassemble(instr: Instruction, address: int) -> str:
    rd, rs1, rs2 = REGISTERS[instr.rd], REGISTERS[instr.rs1], REGISTERS[instr.rs2]
    target = f"0x{(address + instr.imm) & WORD_MASK:04x}"
    match FORMATS[instr.opcode]:
        case Format.REGISTER:
            operands = f"{rd}, {rs1}, {rs2}"
        case Format.IMMEDIATE if instr.opcode in LOADS:
            operands = f"{rd}, {instr.imm}({rs1})"
        case Format.IMMEDIATE:
            operands = f"{rd}, {rs1}, {instr.imm}"
        case Format.STORE:
            operands = f"{rs2}, {instr.imm}({rs1})"
        case Format.BRANCH:
            operands = f"{rs1}, {rs2}, {target}"
        case Format.UPPER:
            operands = f"{rd}, 0x{instr.imm:05x}"
        case Format.JUMP:
            operands = f"{rd}, {target}"
        case Format.SYSTEM:
            operands = ""
    return f"{instr.opcode.mnemonic} {operands}".rstrip()


def listing_line(address: int, word: int) -> str:
    return f"{address:04x} - {word:08X} - {disassemble(decode(word), address)}"


def words_to_bytes(words: Iterable[int]) -> bytes:
    return b"".join((word & WORD_MASK).to_bytes(WORD_SIZE, "little") for word in words)


def bytes_to_words(data: bytes) -> list[int]:
    if len(data) % WORD_SIZE:
        msg = f"binary size {len(data)} is not a multiple of {WORD_SIZE}"
        raise ValueError(msg)
    return [int.from_bytes(data[i : i + WORD_SIZE], "little") for i in range(0, len(data), WORD_SIZE)]