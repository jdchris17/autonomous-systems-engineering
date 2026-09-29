import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.cpu import CPU
from cpu_simulator.cpu.pipelined_cpu import PIPELINE_DEPTH, PipelinedCPU, load_program
from cpu_simulator.cpu.trace import format_compact, format_pipeline, format_verbose
from cpu_simulator.isa.assembler import assemble


def _stepped_cpu(program: str, base_pc: int = 0, registers: dict[int, int] | None = None):
    cpu = CPU()
    for reg, value in (registers or {}).items():
        cpu.registers.write(reg, value)
    cpu.pc.jump(base_pc)
    words = assemble(program)
    for i, word in enumerate(words):
        cpu.instruction_memory.write_word(base_pc + i * 4, word)
    return cpu


class TestSection31WorkedExample:
    """The exact verbose trace example from the spec."""

    def test_matches_the_spec_example_field_for_field(self):
        cpu = _stepped_cpu("add x5, x6, x7", base_pc=0x20, registers={6: 10, 7: 7})
        cpu.step()
        text = format_verbose(cpu.last_trace)

        assert "PC\n0x00000020" in text
        assert "FETCH\nInstruction: 0x007302B3" in text
        assert "Mnemonic: ADD" in text
        assert "rd:  x5" in text
        assert "rs1: x6" in text
        assert "rs2: x7" in text
        assert "x6 = 10" in text
        assert "x7 = 7" in text
        assert "RegWrite  = 1" in text
        assert "MemRead   = 0" in text
        assert "MemWrite  = 0" in text
        assert "ALUSrc    = REGISTER" in text
        assert "ALUOp     = ADD" in text
        assert "ResultSrc = ALU" in text
        assert "A = 10" in text
        assert "B = 7" in text
        assert "ALUResult = 17" in text
        assert "No access" in text
        assert "x5 <- 17" in text
        assert "NEXT PC\n0x00000024" in text

    def test_section_header_uses_the_actual_cycle_number(self):
        cpu = _stepped_cpu("add x5, x6, x7\nadd x5, x6, x7")
        cpu.step()
        assert format_verbose(cpu.last_trace).startswith("Cycle/Step 1")
        cpu.step()
        assert format_verbose(cpu.last_trace).startswith("Cycle/Step 2")


class TestVerboseOmitsAbsentFields:
    def test_lui_has_no_rs1_rs2_registers_section(self):
        cpu = _stepped_cpu("lui x3, 74565")
        cpu.step()
        text = format_verbose(cpu.last_trace)
        assert "REGISTERS" not in text  # no rs1/rs2 at all for U-type

    def test_store_has_no_writeback(self):
        cpu = _stepped_cpu("sw x1, 0(x2)")
        cpu.step()
        text = format_verbose(cpu.last_trace)
        assert "No writeback" in text

    def test_load_shows_a_memory_read(self):
        cpu = _stepped_cpu("lw x5, 0(x2)")
        cpu.data_memory.write_word(0, 0xCAFEBABE)
        cpu.step()
        text = format_verbose(cpu.last_trace)
        assert "Read  [0x00000000] -> 3405691582" in text  # 0xCAFEBABE as decimal

    def test_store_shows_a_memory_write(self):
        cpu = _stepped_cpu("sw x1, 0(x2)", registers={1: 0x1234})
        cpu.step()
        text = format_verbose(cpu.last_trace)
        assert "Write [0x00000000] <- 4660" in text  # 0x1234 as decimal


class TestSection32WorkedExample:
    """The exact compact trace example from the spec."""

    PROGRAM = """
add  x5, x6, x7
addi x6, x6, -1
bne  x6, x0, -4
    """

    def test_matches_the_spec_example_exactly(self):
        cpu = _stepped_cpu(self.PROGRAM, base_pc=0x20, registers={6: 10, 7: 7})
        lines = []
        for _ in range(3):
            cpu.step()
            lines.append(format_compact(cpu.last_trace))

        assert lines[0] == "PC=0x20 ADD x5,x6,x7  x5<-17"
        assert lines[1] == "PC=0x24 ADDI x6,x6,-1  x6<-9"
        assert lines[2] == "PC=0x28 BNE x6,x0,-4  TAKEN -> 0x24"


class TestCompactBranchNotTaken:
    def test_shows_not_taken(self):
        cpu = _stepped_cpu("beq x1, x2, 16", registers={1: 1})  # x1 != x2(=0)
        cpu.step()
        line = format_compact(cpu.last_trace)
        assert "NOT TAKEN" in line
        assert "TAKEN ->" not in line


class TestCompactStore:
    def test_shows_the_memory_write(self):
        cpu = _stepped_cpu("sw x1, 4(x2)", registers={1: 99})
        cpu.step()
        line = format_compact(cpu.last_trace)
        assert "MEM[0x4]<-99" in line


class TestTraceHistory:
    def test_accumulates_one_entry_per_step(self):
        cpu = _stepped_cpu("addi x1, x0, 1\naddi x1, x1, 1\naddi x1, x1, 1")
        cpu.run(3)
        assert len(cpu.trace_history) == 3
        assert cpu.trace_history[-1] is cpu.last_trace

    def test_full_run_compact_trace_is_readable_as_a_program_log(self):
        cpu = _stepped_cpu("addi x1, x0, 5\naddi x2, x0, 3\nadd x3, x1, x2")
        cpu.run(3)
        lines = [format_compact(t) for t in cpu.trace_history]
        assert lines == [
            "PC=0x0 ADDI x1,x0,5  x1<-5",
            "PC=0x4 ADDI x2,x0,3  x2<-3",
            "PC=0x8 ADD x3,x1,x2  x3<-8",
        ]


class TestFormatPipelineFullPipeline:
    """Once the pipeline is full, one line per stage, newest (IF) to
    oldest (WB) -- section 8's own worked example shape.
    """

    PROGRAM = """
addi x1, x0, 5
addi x2, x0, 5
add  x3, x1, x2
addi x4, x0, 1
addi x5, x0, 2
    """

    def test_five_distinct_instructions_shown_at_once(self):
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(5)
        text = format_pipeline(cpu.last_trace)
        assert text.startswith("Cycle 5\n")
        assert "IF     0x00200293  addi x5,x0,2" in text
        assert "ID     0x00100213  addi x4,x0,1" in text
        assert "EX     0x002081b3  add x3,x1,x2" in text
        assert "MEM    0x00500113  addi x2,x0,5" in text
        assert "WB     0x00500093  addi x1,x0,5" in text

    def test_no_stall_or_flush_shows_the_compact_no_lines(self):
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(5)
        text = format_pipeline(cpu.last_trace)
        assert "Stall:     No" in text
        assert "Flush:     No" in text

    def test_forward_labels_reflect_the_forwarding_decision(self):
        # add x3, x1, x2 depends on both preceding addi's: x1 (rs1) was
        # produced two instructions back, now in MEM/WB; x2 (rs2) was
        # produced immediately before it, now in EX/MEM -- the more
        # recent producer.
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(5)  # the cycle "add x3,x1,x2" is in EX
        text = format_pipeline(cpu.last_trace)
        assert "EX     0x002081b3  add x3,x1,x2" in text
        assert "Forward A: MEM/WB" in text
        assert "Forward B: EX/MEM" in text


class TestFormatPipelineBubble:
    def test_empty_stage_renders_as_bubble(self):
        cpu = PipelinedCPU()
        load_program(cpu, "addi x1, x0, 1")
        cpu.step()  # cycle 1: only IF is populated
        text = format_pipeline(cpu.last_trace)
        assert "ID     (bubble)" in text
        assert "EX     (bubble)" in text
        assert "MEM    (bubble)" in text
        assert "WB     (bubble)" in text


class TestFormatPipelineStall:
    PROGRAM = """
addi x1, x0, 100
sw   x1, 0(x0)
lw   x5, 0(x0)
add  x6, x5, x1
    """

    def test_load_use_stall_shows_the_explanatory_block(self):
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(5)  # the cycle "add x6,x5,x1" is decoded and stalled
        text = format_pipeline(cpu.last_trace)
        assert "STALL: load-use dependency on x5" in text
        assert "bubble inserted into EX" in text
        assert "Stall:     No" not in text


class TestFormatPipelineFlush:
    PROGRAM = """
addi x1, x0, 5
addi x2, x0, 5
beq  x1, x2, skip
addi x10, x0, 999
addi x11, x0, 999
skip:
addi x3, x0, 42
    """

    def test_taken_branch_shows_the_flush_block(self):
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(5)  # the cycle beq resolves in EX
        text = format_pipeline(cpu.last_trace)
        assert "BRANCH TAKEN" in text
        assert "flush IF/ID" in text
        assert "flush ID/EX" in text
        assert "next PC = 0x00000014" in text
        assert "Flush:     No" not in text
