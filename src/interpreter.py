VALID_BF_COMMANDS = set("[]<>+-,.")


def _near(code: str, pc, window=12):
    if not isinstance(pc, int) or pc < 0:
        return ""
    start = max(0, pc - window)
    end = min(len(code), pc + window)
    return code[start:end]


class BrainfuckError(Exception):
    def __init__(self, message, pc=None, cmd=None, near=None, kind=None, blocked_reason=None):
        super().__init__(message)
        self.message = message
        self.pc = pc
        self.cmd = cmd
        self.near = near
        self.kind = kind
        self.blocked_reason = blocked_reason


class BrainfuckSyntaxError(BrainfuckError):
    pass


class BrainfuckRuntimeError(BrainfuckError):
    pass


class BrainfuckState:
    """
    Represents the complete state of a Brainfuck program at a given point in time.
    """
    def __init__(self, memory_size=30000):
        self.memory = [0] * memory_size
        self.dp = 0
        self.pc = 0
        self.output = ""
        self.code = ""
        self.input_cursor = 0
        self.bracket_map = {}
        self.open_brackets = []
        self.blocked_reason = None
        self.hit_step_limit = False

    def clone(self):
        new_state = BrainfuckState(len(self.memory))
        new_state.memory = list(self.memory)
        new_state.dp = self.dp
        new_state.pc = self.pc
        new_state.output = self.output
        new_state.code = self.code
        new_state.input_cursor = self.input_cursor
        new_state.bracket_map = dict(self.bracket_map)
        new_state.open_brackets = list(self.open_brackets)
        new_state.blocked_reason = self.blocked_reason
        new_state.hit_step_limit = self.hit_step_limit
        return new_state

class BrainfuckInterpreter:
    def __init__(self, memory_size=30000, max_steps_per_run=20000, wrap_pointers=True):
        self.memory_size = memory_size
        self.max_steps_per_run = int(max_steps_per_run)
        self.wrap_pointers = bool(wrap_pointers)
        self.state = BrainfuckState(memory_size)
        self.history = [self.state.clone()]
        self.input_buffer = ""

    def add_input(self, inp: str):
        """
        Adds input to the input buffer.
        """
        self.input_buffer += inp

    def execute(self, chunk: str, max_steps=None) -> str:
        """
        Executes a chunk of code incrementally.
        Returns the new output generated during this execution.
        """
        new_output = ""
        if not chunk:
            # Just try to run with existing code (e.g., after adding input)
            out = self._run(max_steps=max_steps)
            self.state.output += out
            new_output += out
            self.history[-1] = self.state.clone()
            return new_output

        for char in chunk:
            backup_state = self.state.clone()
            try:
                self._append_code_char(char)
                out = self._run(max_steps=max_steps)
                self.state.output += out
                new_output += out
            except Exception as e:
                self.state = backup_state
                raise e

            self.history.append(self.state.clone())
            
        return new_output

    def backspace(self) -> bool:
        """
        Rolls back the last character added to the code.
        Returns True if successful, False if no history left.
        """
        if len(self.history) > 1:
            self.history.pop() # Remove current state
            self.state = self.history[-1].clone()
            
            # Catch up with any available input
            out = self._run()
            self.state.output += out
            self.history[-1] = self.state.clone()
            
            return True
        return False

    def validate_code(self):
        if self.state.open_brackets:
            pc = self.state.open_brackets[-1]
            raise BrainfuckSyntaxError(
                "Unmatched '['",
                pc=pc,
                cmd="[",
                near=_near(self.state.code, pc),
                kind="unmatched_open_bracket",
                blocked_reason=self.state.blocked_reason,
            )

    def _run(self, max_steps=None) -> str:
        """
        Runs the code starting from the current program counter.
        Returns the new output generated.
        """
        if max_steps is None:
            max_steps = self.max_steps_per_run
        max_steps = int(max_steps)

        self.state.blocked_reason = None
        self.state.hit_step_limit = False

        code = self.state.code
        out = ""
        steps = 0
        
        while self.state.pc < len(code):
            if steps >= max_steps:
                self.state.hit_step_limit = True
                self.state.blocked_reason = "step_limit"
                break

            cmd = code[self.state.pc]
            
            if cmd == '[':
                match = self.state.bracket_map.get(self.state.pc)
                if match is None:
                    self.state.blocked_reason = "awaiting_closing_bracket"
                    break

                if self.state.memory[self.state.dp] == 0:
                    self.state.pc = match
            elif cmd == ']':
                match = self.state.bracket_map.get(self.state.pc)
                if match is None:
                    raise BrainfuckSyntaxError(
                        "Unmatched ']'",
                        pc=self.state.pc,
                        cmd="]",
                        near=_near(code, self.state.pc),
                        kind="unmatched_close_bracket",
                        blocked_reason=self.state.blocked_reason,
                    )

                if self.state.memory[self.state.dp] != 0:
                    self.state.pc = match
            elif cmd == '>':
                next_dp = self.state.dp + 1
                if self.wrap_pointers:
                    self.state.dp = next_dp % self.memory_size
                else:
                    if next_dp >= self.memory_size:
                        raise BrainfuckRuntimeError(
                            "Data pointer out of bounds",
                            pc=self.state.pc,
                            cmd=">",
                            near=_near(code, self.state.pc),
                            kind="data_pointer_oob",
                        )
                    self.state.dp = next_dp
            elif cmd == '<':
                next_dp = self.state.dp - 1
                if self.wrap_pointers:
                    self.state.dp = next_dp % self.memory_size
                else:
                    if next_dp < 0:
                        raise BrainfuckRuntimeError(
                            "Data pointer out of bounds",
                            pc=self.state.pc,
                            cmd="<",
                            near=_near(code, self.state.pc),
                            kind="data_pointer_oob",
                        )
                    self.state.dp = next_dp
            elif cmd == '+':
                self.state.memory[self.state.dp] = (self.state.memory[self.state.dp] + 1) % 256
            elif cmd == '-':
                self.state.memory[self.state.dp] = (self.state.memory[self.state.dp] - 1) % 256
            elif cmd == '.':
                out += chr(self.state.memory[self.state.dp])
            elif cmd == ',':
                if self.state.input_cursor < len(self.input_buffer):
                    self.state.memory[self.state.dp] = ord(self.input_buffer[self.state.input_cursor])
                    self.state.input_cursor += 1
                else:
                    self.state.blocked_reason = "awaiting_input"
                    break # Blocked on input
            
            self.state.pc += 1
            steps += 1
            
        return out

    def _append_code_char(self, char: str):
        idx = len(self.state.code)
        self.state.code += char

        if char == '[':
            self.state.open_brackets.append(idx)
        elif char == ']':
            if not self.state.open_brackets:
                raise BrainfuckSyntaxError(
                    "Unmatched ']'",
                    pc=idx,
                    cmd="]",
                    near=_near(self.state.code, idx),
                    kind="unmatched_close_bracket",
                )
            open_idx = self.state.open_brackets.pop()
            self.state.bracket_map[open_idx] = idx
            self.state.bracket_map[idx] = open_idx
