import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.isa.assembler import assemble, assemble_line
from cpu_simulator.isa.decoder import decode, to_asm


class TestAssembleLine:
    def test_the_section_43_worked_example(self):
        assert assemble_line("add x5, x6, x7") == 0x007302B3

    def test_matches_decoder_worked_example(self):
        assert assemble_line("add x5, x6, x7") == decode(0x007302B3).raw

    def test_rejects_more_than_one_instruction(self):
        with pytest.raises(ValueError):
            assemble_line("add x1, x2, x3\nsub x4, x5, x6")


class TestRoundTripEveryOperandShape:
    """Assemble a line, decode the result, and confirm to_asm() of the
    decoded instruction is the *same* text (modulo whitespace/case) --
    the strongest check that parsing and rendering actually agree.
    """

    @pytest.mark.parametrize("text", [
        "add x5, x6, x7",
        "sub x5, x6, x7",
        "and x1, x2, x3",
        "or x1, x2, x3",
        "xor x1, x2, x3",
        "sll x1, x2, x3",
        "srl x1, x2, x3",
        "sra x1, x2, x3",
        "slt x1, x2, x3",
        "sltu x1, x2, x3",
    ])
    def test_r_type(self, text):
        word = assemble_line(text)
        assert to_asm(decode(word)) == text

    @pytest.mark.parametrize("text", [
        "addi x5, x5, -1",
        "addi x5, x5, 100",
        "slti x1, x2, 10",
        "sltiu x1, x2, 10",
        "xori x1, x2, 10",
        "ori x1, x2, 10",
        "andi x1, x2, 10",
    ])
    def test_i_type_arithmetic(self, text):
        word = assemble_line(text)
        assert to_asm(decode(word)) == text

    @pytest.mark.parametrize("text,expected_shamt", [
        ("slli x1, x2, 5", 5),
        ("srli x1, x2, 31", 31),
        ("srai x1, x2, 1", 1),
    ])
    def test_shift_immediate(self, text, expected_shamt):
        word = assemble_line(text)
        inst = decode(word)
        assert inst.immediate == expected_shamt

    def test_load(self):
        word = assemble_line("lw x5, 12(x2)")
        assert to_asm(decode(word)) == "lw x5, 12(x2)"

    def test_load_negative_offset(self):
        word = assemble_line("lw x5, -8(x2)")
        assert decode(word).immediate == -8

    def test_store(self):
        word = assemble_line("sw x5, 12(x2)")
        assert to_asm(decode(word)) == "sw x5, 12(x2)"

    def test_jalr(self):
        word = assemble_line("jalr x2, x1, 4")
        assert to_asm(decode(word)) == "jalr x2, x1, 4"

    @pytest.mark.parametrize("text", [
        "beq x1, x2, 16", "bne x1, x2, 16", "blt x1, x2, 16",
        "bge x1, x2, 16", "bltu x1, x2, 16", "bgeu x1, x2, 16",
    ])
    def test_branch_with_numeric_target(self, text):
        word = assemble_line(text)
        assert to_asm(decode(word)) == text

    def test_jal_with_numeric_target(self):
        word = assemble_line("jal x1, 1024")
        assert to_asm(decode(word)) == "jal x1, 1024"

    def test_lui(self):
        word = assemble_line("lui x3, 74565")  # 0x12345
        assert decode(word).immediate == 0x12345000

    def test_auipc(self):
        word = assemble_line("auipc x5, 4096")  # 0x1000
        assert decode(word).immediate == 0x1000000


class TestCaseAndWhitespace:
    def test_mnemonic_is_case_insensitive(self):
        assert assemble_line("ADD x5, x6, x7") == assemble_line("add x5, x6, x7")
        assert assemble_line("Add x5, x6, x7") == assemble_line("add x5, x6, x7")

    def test_extra_whitespace_is_tolerated(self):
        assert assemble_line("add   x5,   x6,   x7") == assemble_line("add x5, x6, x7")

    def test_trailing_comment_is_stripped(self):
        assert assemble_line("add x5, x6, x7  # sum two registers") == assemble_line("add x5, x6, x7")

    def test_blank_lines_and_comment_only_lines_produce_no_instruction(self):
        program = """
        # this is a whole-line comment

        add x5, x6, x7

        """
        assert assemble(program) == [assemble_line("add x5, x6, x7")]


class TestLabelsAndTwoPassAssembly:
    def test_the_section_44_loop_example(self):
        program = """
loop:
    addi x5, x5, -1
    bne  x5, x0, loop
        """
        words = assemble(program)
        assert len(words) == 2

        addi_inst = decode(words[0])
        assert addi_inst.mnemonic == "ADDI"
        assert addi_inst.immediate == -1

        bne_inst = decode(words[1])
        assert bne_inst.mnemonic == "BNE"
        # loop is at address 0; the branch itself is the second
        # instruction, at address 4 -- so the displacement back to
        # loop is 0 - 4 = -4.
        assert bne_inst.immediate == -4

    def test_label_on_its_own_line_does_not_consume_an_address(self):
        program = """
loop:
    add x1, x2, x3
        """
        # If "loop:" consumed an address, the single ADD would land at
        # address 4 instead of 0, and this label-address check would
        # disagree with a branch computed against it.
        program_with_branch = """
loop:
    add x1, x2, x3
    beq x1, x2, loop
        """
        words = assemble(program_with_branch)
        beq_inst = decode(words[1])
        # beq is the second instruction (address 4); loop is address 0.
        assert beq_inst.immediate == -4

    def test_forward_reference_to_a_later_label(self):
        program = """
    beq x1, x2, done
    add x3, x4, x5
done:
    add x6, x7, x8
        """
        words = assemble(program)
        beq_inst = decode(words[0])
        # beq is at address 0; done is the third instruction, address 8.
        assert beq_inst.immediate == 8

    def test_jal_to_a_label(self):
        program = """
start:
    add x1, x1, x1
target:
    add x2, x2, x2
        """
        # Re-derive with an explicit jal referencing the forward label.
        program = """
    jal x1, target
    add x2, x2, x2
target:
    add x3, x3, x3
        """
        words = assemble(program)
        jal_inst = decode(words[0])
        # jal at address 0; target is the third instruction, address 8.
        assert jal_inst.immediate == 8

    def test_label_and_instruction_on_the_same_line(self):
        program = "loop: add x1, x2, x3"
        words = assemble(program)
        assert len(words) == 1
        assert decode(words[0]).mnemonic == "ADD"

    def test_duplicate_label_rejected(self):
        program = """
loop:
    add x1, x2, x3
loop:
    add x4, x5, x6
        """
        with pytest.raises(ValueError):
            assemble(program)

    def test_undefined_label_rejected(self):
        with pytest.raises(ValueError):
            assemble("bne x1, x2, nowhere")

    def test_base_address_shifts_label_addresses(self):
        program = """
loop:
    add x1, x2, x3
    beq x1, x2, loop
        """
        words = assemble(program, base_address=0x1000)
        beq_inst = decode(words[1])
        # Absolute addresses shift with base_address, but the
        # *displacement* between two instructions in the same program
        # does not.
        assert beq_inst.immediate == -4


class TestErrorMessages:
    def test_unknown_mnemonic(self):
        with pytest.raises(ValueError):
            assemble_line("mul x1, x2, x3")

    def test_malformed_register(self):
        with pytest.raises(ValueError):
            assemble_line("add r5, x6, x7")

    def test_register_out_of_range(self):
        with pytest.raises(ValueError):
            assemble_line("add x99, x6, x7")

    def test_malformed_offset_base_syntax(self):
        with pytest.raises(ValueError):
            assemble_line("lw x5, 12 x2")

    def test_non_numeric_immediate(self):
        with pytest.raises(ValueError):
            assemble_line("addi x5, x6, banana")
