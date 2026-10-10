from pathlib import Path

import pytest

from isa import REGISTER_INDEX, Instruction, Opcode, bytes_to_words, decode
from translator import AsmError, translate

T0 = REGISTER_INDEX["t0"]
A0 = REGISTER_INDEX["a0"]
RA = REGISTER_INDEX["ra"]
ZERO = REGISTER_INDEX["zero"]


def fragment(module: dict[str, object], section: str, index: int = 0) -> dict[str, object]:
    sections = module["sections"]
    assert isinstance(sections, dict)
    fragments = sections[section]
    assert isinstance(fragments, list)
    result = fragments[index]
    assert isinstance(result, dict)
    return result


def text(source: str) -> list[Instruction]:
    data = fragment(translate(source, "test.s"), "text")["data"]
    assert isinstance(data, str)
    return [decode(word) for word in bytes_to_words(bytes.fromhex(data))]


def data_bytes(source: str) -> bytes:
    data = fragment(translate(source, "test.s"), "data")["data"]
    assert isinstance(data, str)
    return bytes.fromhex(data)


def relocations(source: str) -> list[tuple[int, str, str]]:
    module = translate(source, "test.s")
    items = module["relocations"]
    assert isinstance(items, list)
    return [(item["offset"], item["kind"], item["symbol"]) for item in items]


def test_hello_program() -> None:
    module = translate(Path("programs/hello.s").read_text(encoding="utf-8"), "hello.s")
    assert fragment(module, "text")["org"] == 0
    assert fragment(module, "data")["data"] == b"Hello, World!\n\0".hex()
    assert module["exports"] == ["_start"]
    symbols = module["symbols"]
    assert isinstance(symbols, dict)
    assert symbols["_start"] == {"section": "text", "fragment": 0, "offset": 4}
    assert symbols["msg"] == {"section": "data", "fragment": 0, "offset": 0}
    assert relocations(Path("programs/hello.s").read_text(encoding="utf-8"))[:4] == [
        (0, "pcrel", "_start"),
        (4, "hi", "msg"),
        (8, "lo", "msg"),
        (12, "pcrel", "print_str"),
    ]


def test_cat_program() -> None:
    module = translate(Path("programs/cat.s").read_text(encoding="utf-8"), "cat.s")
    assert fragment(module, "text", 0)["org"] == 0
    assert fragment(module, "text", 1)["org"] == 4


def test_li_number() -> None:
    assert text("li t0, 0x12345678") == [
        Instruction(Opcode.LUI, rd=T0, imm=0x12345),
        Instruction(Opcode.ADDI, rd=T0, rs1=T0, imm=0x678),
    ]


def test_li_rounds_upper_part() -> None:
    assert text("li t0, 0x12345FFF") == [
        Instruction(Opcode.LUI, rd=T0, imm=0x12346),
        Instruction(Opcode.ADDI, rd=T0, rs1=T0, imm=-1),
    ]


def test_li_equ_and_char() -> None:
    assert text(".equ PORT, 0x84\nli t0, PORT\nli a0, 'A'")[1::2] == [
        Instruction(Opcode.ADDI, rd=T0, rs1=T0, imm=0x84),
        Instruction(Opcode.ADDI, rd=A0, rs1=A0, imm=ord("A")),
    ]


def test_pseudo_instructions() -> None:
    assert text("ret\nmv a0, t0\nnop") == [
        Instruction(Opcode.JALR, rd=ZERO, rs1=RA),
        Instruction(Opcode.ADDI, rd=A0, rs1=T0),
        Instruction(Opcode.ADDI),
    ]


def test_swapped_branch_operands() -> None:
    assert relocations("loop: bgt a0, t0, loop") == [(0, "pcrel", "loop")]
    assert text("loop: bgt a0, t0, loop") == [Instruction(Opcode.BLT, rs1=T0, rs2=A0)]


def test_memory_operand_with_constant() -> None:
    assert text(".equ IN, 0x80\nlw t0, IN(zero)") == [Instruction(Opcode.LW, rd=T0, imm=0x80)]


def test_strings_and_comments() -> None:
    assert data_bytes('.data\ns: .byte "a;b\\n", 0 ; comment') == b"a;b\n\0"


def test_word_with_symbol() -> None:
    assert relocations(".data\nptr: .word ptr, 7") == [(0, "word32", "ptr")]
    assert data_bytes(".data\n.word 7") == (7).to_bytes(4, "little")


def test_space_and_align() -> None:
    assert data_bytes(".data\n.byte 1\n.align 4\n.space 2") == bytes([1, 0, 0, 0, 0, 0])


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("j nowhere", "undefined symbol: nowhere"),
        (".data\nadd t0, t0, t0", "only in .text"),
        ("add t0, t0, x16", "unknown register"),
        ("addi t0, t0, 2048", "does not fit"),
        (".byte 1", "only in .data"),
        ("a:\na:", "duplicate symbol"),
        (".global main", "global symbol is not defined"),
        ("frobnicate t0", "unknown instruction"),
        (".org 2", "bad .org"),
        ("add t0, t0", "expected 3 operands"),
    ],
)
def test_errors(source: str, message: str) -> None:
    with pytest.raises(AsmError, match=message):
        translate(source, "test.s")


def test_error_has_line_number() -> None:
    with pytest.raises(AsmError, match=r"test\.s:2:"):
        translate("nop\nadd t0, t0", "test.s")