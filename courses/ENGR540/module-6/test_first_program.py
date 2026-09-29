import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.pseudo_cpu import PseudoOpCPU as CPU
from cpu_simulator.isa.assembler import assemble
from cpu_simulator.isa.decoder import decode
from cpu_simulator.memory.memory import Memory
from first_program import (
    PROGRAM_ASM,
    PROGRAM_PSEUDO_OPS,
    assemble_and_store,
    fetch_and_decode,
)

# Exact known encodings -- verified with the encoder itself before being
# written here, then cross-checked by decoding each one back (below).
# Section 51's principle: test exact values, not just "it round-trips".
EXPECTED_WORDS = [
    0x00A00093,  # addi x1, x0, 10
    0x00700113,  # addi x2, x0, 7
    0x002081B3,  # add  x3, x1, x2
    0x40218233,  # sub  x4, x3, x2
    0x00402023,  # sw   x4, 0(x0)
]


class TestExactKnownEncoding:
    def test_assembles_to_the_exact_expected_words(self):
        assert assemble(PROGRAM_ASM) == EXPECTED_WORDS

    def test_decoding_each_word_gives_exactly_the_expected_fields(self):
        expected_fields = [
            dict(mnemonic="ADDI", rd=1, rs1=0, rs2=None, immediate=10),
            dict(mnemonic="ADDI", rd=2, rs1=0, rs2=None, immediate=7),
            dict(mnemonic="ADD", rd=3, rs1=1, rs2=2, immediate=None),
            dict(mnemonic="SUB", rd=4, rs1=3, rs2=2, immediate=None),
            dict(mnemonic="SW", rd=None, rs1=0, rs2=4, immediate=0),
        ]
        for word, expected in zip(EXPECTED_WORDS, expected_fields):
            inst = decode(word)
            assert inst.mnemonic == expected["mnemonic"]
            assert inst.rd == expected["rd"]
            assert inst.rs1 == expected["rs1"]
            assert inst.rs2 == expected["rs2"]
            assert inst.immediate == expected["immediate"]

    def test_the_literal_section_51_example(self):
        # Independent of this program: the exact example the spec gives.
        from cpu_simulator.isa.assembler import assemble_line
        word = assemble_line("add x5, x6, x7")
        assert word == 0x007302B3
        inst = decode(word)
        assert inst.mnemonic == "ADD"
        assert inst.rd == 5
        assert inst.rs1 == 6
        assert inst.rs2 == 7


class TestDecodeEncodeSymmetry:
    """decode(encode(x)) == x for every instruction in this program --
    the "very strong testing principle" section 51 calls out by name.
    """

    def test_every_instruction_round_trips(self):
        from cpu_simulator.isa.encoder import encode

        for word in EXPECTED_WORDS:
            inst = decode(word)
            kwargs = {k: v for k, v in dict(
                rd=inst.rd, rs1=inst.rs1, rs2=inst.rs2, immediate=inst.immediate,
            ).items() if v is not None}
            assert encode(inst.mnemonic, **kwargs) == word


class TestAssembleStoreDecodeThroughMemory:
    """The three-step pipeline from section 49, exercised through real
    Memory (not just Python lists): assemble -> store -> read back ->
    decode.
    """

    def test_words_survive_a_round_trip_through_memory(self):
        instruction_memory = Memory(size_bytes=64)
        words = assemble_and_store(PROGRAM_ASM, instruction_memory)
        assert words == EXPECTED_WORDS
        for i, expected_word in enumerate(EXPECTED_WORDS):
            assert instruction_memory.read_word(i * 4) == expected_word

    def test_fetch_and_decode_recovers_every_instruction(self):
        instruction_memory = Memory(size_bytes=64)
        assemble_and_store(PROGRAM_ASM, instruction_memory)
        instructions = fetch_and_decode(instruction_memory, base_address=0, count=5)
        assert [inst.mnemonic for inst in instructions] == ["ADDI", "ADDI", "ADD", "SUB", "SW"]


class TestArchitecturalResult:
    """The section-49 target: x1=10, x2=7, x3=17, x4=10, Memory[0]=10.
    Reached via Module 5's pseudo-instruction execution path, not by
    driving the datapath from decoded fields -- that's Module 7's job,
    not built yet.
    """

    def test_registers_and_memory_match_the_spec(self):
        cpu = CPU()
        cpu.run(PROGRAM_PSEUDO_OPS)
        assert cpu.registers.read(1) == 10
        assert cpu.registers.read(2) == 7
        assert cpu.registers.read(3) == 17
        assert cpu.registers.read(4) == 10
        assert cpu.data_memory.read_word(0) == 10

    def test_pseudo_ops_are_the_semantic_equivalent_of_the_real_program(self):
        # addi rd, x0, imm and LOAD_IMM compute the same thing (both are
        # "0 + imm" through the ALU) -- confirm that's actually true
        # here, not just asserted in a comment.
        cpu_real_equivalent = CPU()
        for word in EXPECTED_WORDS[:2]:  # the two addi/LOAD_IMM lines
            inst = decode(word)
            cpu_real_equivalent.step_load_immediate(inst.rd, inst.immediate)
        cpu_pseudo = CPU()
        for op in PROGRAM_PSEUDO_OPS[:2]:
            cpu_pseudo.execute(op)
        assert cpu_real_equivalent.registers.read(1) == cpu_pseudo.registers.read(1)
        assert cpu_real_equivalent.registers.read(2) == cpu_pseudo.registers.read(2)
