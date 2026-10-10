import pytest

from isa import (
    FORMATS,
    REGISTER_INDEX,
    WORD_MASK,
    Format,
    Instruction,
    Opcode,
    bytes_to_words,
    decode,
    encode,
    listing_line,
    lower,
    upper,
    words_to_bytes,
)

ZERO = REGISTER_INDEX["zero"]
RA = REGISTER_INDEX["ra"]
SP = REGISTER_INDEX["sp"]
T0 = REGISTER_INDEX["t0"]
T1 = REGISTER_INDEX["t1"]
T2 = REGISTER_INDEX["t2"]
A0 = REGISTER_INDEX["a0"]
A1 = REGISTER_INDEX["a1"]

KNOWN_ENCODINGS = [
    (Instruction(Opcode.ADDI, rd=T0, rs1=T0, imm=1), 0x00128293),
    (Instruction(Opcode.ADD, rd=A0, rs1=A0, rs2=A1), 0x00B50533),
    (Instruction(Opcode.SUB, rd=T0, rs1=T1, rs2=T2), 0x407302B3),
    (Instruction(Opcode.SRAI, rd=T0, rs1=T0, imm=3), 0x4032D293),
    (Instruction(Opcode.LW, rd=T0, rs1=A0, imm=0), 0x00052283),
    (Instruction(Opcode.SW, rs1=SP, rs2=T0, imm=4), 0x00512223),
    (Instruction(Opcode.BEQ, rs1=T0, rs2=ZERO, imm=16), 0x00028863),
    (Instruction(Opcode.BEQ, rs1=T0, rs2=ZERO, imm=-4), 0xFE028EE3),
    (Instruction(Opcode.JAL, rd=RA, imm=8), 0x008000EF),
    (Instruction(Opcode.JAL, rd=ZERO, imm=-16), 0xFF1FF06F),
    (Instruction(Opcode.LUI, rd=A0, imm=0x12345), 0x12345537),
    (Instruction(Opcode.JALR, rd=ZERO, rs1=RA, imm=0), 0x00008067),
]


def sample(opcode: Opcode) -> Instruction:
    match FORMATS[opcode]:
        case Format.REGISTER:
            return Instruction(opcode, rd=T0, rs1=T1, rs2=A0)
        case Format.IMMEDIATE:
            return Instruction(opcode, rd=T0, rs1=T1, imm=7)
        case Format.STORE | Format.BRANCH:
            return Instruction(opcode, rs1=T1, rs2=T0, imm=-8)
        case Format.UPPER:
            return Instruction(opcode, rd=T0, imm=0xABCDE)
        case Format.JUMP:
            return Instruction(opcode, rd=RA, imm=-0x100)
        case Format.SYSTEM:
            return Instruction(opcode)


@pytest.mark.parametrize("opcode", list(Opcode))
def test_roundtrip(opcode: Opcode) -> None:
    instr = sample(opcode)
    assert decode(encode(instr)) == instr


@pytest.mark.parametrize(("instr", "word"), KNOWN_ENCODINGS)
def test_matches_risc_v_specification(instr: Instruction, word: int) -> None:
    assert encode(instr) == word
    assert decode(word) == instr


def test_register_numbers_follow_risc_v() -> None:
    assert [REGISTER_INDEX[name] for name in ("zero", "ra", "sp", "t0", "s0", "a0", "a5")] == [0, 1, 2, 5, 8, 10, 15]
    assert REGISTER_INDEX["x10"] == REGISTER_INDEX["a0"]
    assert REGISTER_INDEX["fp"] == REGISTER_INDEX["s0"]


@pytest.mark.parametrize("value", [0, 0x84, 0x1000, 0x12345678, 0x12345FFF, 0xFFFFFFFF, -1, -2048])
def test_upper_lower_split(value: int) -> None:
    assert ((upper(value) << 12) + lower(value)) & WORD_MASK == value & WORD_MASK


def test_immediate_overflow() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        encode(Instruction(Opcode.ADDI, rd=T0, rs1=T0, imm=2048))


def test_odd_branch_offset() -> None:
    with pytest.raises(ValueError, match="must be even"):
        encode(Instruction(Opcode.BEQ, rs1=T0, rs2=ZERO, imm=3))


def test_zero_word_is_illegal() -> None:
    with pytest.raises(ValueError, match="illegal instruction"):
        decode(0)


def test_listing() -> None:
    assert listing_line(0x10, 0x00028863) == "0010 - 00028863 - beq t0, zero, 0x0020"
    assert listing_line(0x08, 0x00052283) == "0008 - 00052283 - lw t0, 0(a0)"
    assert listing_line(0x00, 0x12345537) == "0000 - 12345537 - lui a0, 0x12345"


def test_binary_roundtrip() -> None:
    words = [0x00128293, 0x0000700B, 0x00000000]
    assert bytes_to_words(words_to_bytes(words)) == words