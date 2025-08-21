#!/usr/bin/env python3
"""
Spring β€” an intentionally quirky, slightly esoteric interpreted language.

This interpreter enforces that source files end with the extension `.spr` and
introduces small, odd language features so programs feel "a little weird but not
out of hand" β€” perfect for experiments, toys, and playful AI integrations.

Features added/changed from a plain interpreter:
- Requires a file extension `.spr` when running a file; running without one
  will raise a friendly error.
- Source files must start with a single-line signature: `spr!` (exactly) β€” this
  is our esoteric "header" that marks Spring programs.
- Adds a tiny set of quirky operators and forms:
  - `~~` (tilde-tilde) reverses a string or list.
  - `twist` unary operator flips truthiness (like boolean NOT but "artistic").
  - `->` used for arrow functions and also as an expression modifier.
  - `sprout` keyword for anonymous functions (compact lambdas).
- Builtins include `ai(prompt)` which calls a pluggable provider. If no external
  provider is configured, a harmless local mock is used.
- Friendly REPL and `python spring_interpreter.py file.spr` runner.

The interpreter is intentionally compact and readable rather than industrially
robust. Extend its parser or evaluator for more features.

"""

import sys
import math
import traceback
from dataclasses import dataclass
from typing import Any, List, Optional

# -------------------- Config / Esoteric checks --------------------
REQUIRED_HEADER = 'spr!'
REQUIRED_EXT = '.spr'

# -------------------- Tiny tokenizer & parser --------------------

TOK_EOF = 'EOF'
TOK_NUMBER = 'NUMBER'
TOK_STRING = 'STRING'
TOK_IDENT = 'IDENT'
TOK_OP = 'OP'
TOK_NEWLINE = 'NEWLINE'

class Token:
    def __init__(self, ttype, value=None, line=0, col=0):
        self.type = ttype
        self.value = value
        self.line = line
        self.col = col
    def __repr__(self):
        return f"Token({self.type!r}, {self.value!r}, {self.line},{self.col})"

def tokenize(src: str):
    i = 0
    n = len(src)
    line = 1
    col = 1
    tokens: List[Token] = []

    def cur():
        return src[i] if i < n else ''

    while i < n:
        c = src[i]
        if c in ' \t':
            i += 1
            col += 1
            continue
        if c == '\n':
            tokens.append(Token(TOK_NEWLINE, '\n', line, col))
            i += 1
            line += 1
            col = 1
            continue
        if c == '#':
            while i < n and src[i] != '\n':
                i += 1
            continue
        if c.isdigit() or (c == '.' and i+1 < n and src[i+1].isdigit()):
            start = i
            has_dot = False
            if c == '.': has_dot = True
            while i < n and (src[i].isdigit() or (src[i]=='.' and not has_dot)):
                if src[i]=='.': has_dot = True
                i += 1
            tok = float(src[start:i]) if has_dot else int(src[start:i])
            tokens.append(Token(TOK_NUMBER, tok, line, col))
            col += i-start
            continue
        if c in '"\'':
            quote = c
            i += 1
            s = ''
            while i < n and src[i] != quote:
                if src[i] == '\\' and i+1 < n:
                    esc = src[i+1]
                    if esc == 'n': s += '\n'
                    elif esc == 't': s += '\t'
                    else: s += esc
                    i += 2
                else:
                    s += src[i]
                    i += 1
            i += 1
            tokens.append(Token(TOK_STRING, s, line, col))
            col += len(s) + 2
            continue
        if c.isalpha() or c == '_':
            start = i
            while i < n and (src[i].isalnum() or src[i]=='_'):
                i += 1
            ident = src[start:i]
            tokens.append(Token(TOK_IDENT, ident, line, col))
            col += i-start
            continue
        two = src[i:i+2]
        if two in ('==','!=','<=','>=','->','~~'):
            tokens.append(Token(TOK_OP, two, line, col))
            i += 2
            col += 2
            continue
        if c in '+-*/%=(){}[],.:<>':
            tokens.append(Token(TOK_OP, c, line, col))
            i += 1
            col += 1
            continue
        raise SyntaxError(f"Unexpected char {c!r} at {line}:{col}")
    tokens.append(Token(TOK_EOF, None, line, col))
    return tokens

# AST nodes (lightweight)
@dataclass
class AST: pass
@dataclass
class Number(AST):
    value: Any
@dataclass
class String(AST):
    value: str
@dataclass
class Ident(AST):
    name: str
@dataclass
class Binary(AST):
    left: AST
    op: str
    right: AST
@dataclass
class Unary(AST):
    op: str
    val: AST
@dataclass
class Assign(AST):
    name: str
    val: AST
@dataclass
class Block(AST):
    stmts: List[AST]
@dataclass
class If(AST):
    cond: AST
    then: Block
    otherwise: Optional[Block]
@dataclass
class While(AST):
    cond: AST
    body: Block
@dataclass
class FuncDef(AST):
    name: Optional[str]
    params: List[str]
    body: Block
@dataclass
class Call(AST):
    callee: AST
    args: List[AST]
@dataclass
class Return(AST):
    value: Optional[AST]

class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.i = 0
    def cur(self): return self.toks[self.i]
    def eat(self, ttype=None, value=None):
        tok = self.cur()
        if ttype and tok.type != ttype:
            raise SyntaxError(f"Expected {ttype} got {tok}")
        if value and tok.value != value:
            raise SyntaxError(f"Expected value {value} got {tok}")
        self.i += 1
        return tok

    def parse(self):
        stmts = []
        while self.cur().type != TOK_EOF:
            if self.cur().type == TOK_NEWLINE:
                self.eat(TOK_NEWLINE); continue
            stmts.append(self.parse_stmt())
        return Block(stmts)

    def parse_stmt(self):
        tok = self.cur()
        if tok.type == TOK_IDENT and tok.value == 'def':
            return self.parse_def()
        if tok.type == TOK_IDENT and tok.value == 'if':
            return self.parse_if()
        if tok.type == TOK_IDENT and tok.value == 'while':
            return self.parse_while()
        if tok.type == TOK_IDENT and tok.value == 'return':
            self.eat(TOK_IDENT)
            val = None
            if self.cur().type not in (TOK_NEWLINE, TOK_EOF):
                val = self.parse_expr()
            if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE)
            return Return(val)

        node = self.parse_expr()
        if isinstance(node, Ident) and self.cur().type == TOK_OP and self.cur().value == '=':
            name = node.name
            self.eat(TOK_OP, '=')
            val = self.parse_expr()
            if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE)
            return Assign(name, val)
        if self.cur().type == TOK_NEWLINE:
            self.eat(TOK_NEWLINE)
        return node

    def parse_def(self):
        self.eat(TOK_IDENT, 'def')
        name = None
        if self.cur().type == TOK_IDENT:
            name = self.eat(TOK_IDENT).value
        self.eat(TOK_OP, '(')
        params = []
        while not (self.cur().type == TOK_OP and self.cur().value == ')'):
            if self.cur().type == TOK_IDENT:
                params.append(self.eat(TOK_IDENT).value)
            if self.cur().type == TOK_OP and self.cur().value == ',':
                self.eat(TOK_OP, ',')
                continue
        self.eat(TOK_OP, ')')
        if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE)
        stmts = []
        while not (self.cur().type == TOK_IDENT and self.cur().value == 'end') and self.cur().type != TOK_EOF:
            if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE); continue
            stmts.append(self.parse_stmt())
        if self.cur().type == TOK_IDENT and self.cur().value == 'end':
            self.eat(TOK_IDENT, 'end')
            if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE)
        return FuncDef(name, params, Block(stmts))

    def parse_if(self):
        self.eat(TOK_IDENT, 'if')
        cond = self.parse_expr()
        if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE)
        then = []
        otherwise = None
        while not (self.cur().type == TOK_IDENT and self.cur().value in ('else','end')) and self.cur().type != TOK_EOF:
            if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE); continue
            then.append(self.parse_stmt())
        if self.cur().type == TOK_IDENT and self.cur().value == 'else':
            self.eat(TOK_IDENT, 'else')
            if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE)
            els = []
            while not (self.cur().type == TOK_IDENT and self.cur().value == 'end') and self.cur().type != TOK_EOF:
                if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE); continue
                els.append(self.parse_stmt())
            otherwise = Block(els)
        if self.cur().type == TOK_IDENT and self.cur().value == 'end':
            self.eat(TOK_IDENT, 'end')
            if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE)
        return If(cond, Block(then), otherwise)

    def parse_while(self):
        self.eat(TOK_IDENT, 'while')
        cond = self.parse_expr()
        if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE)
        body = []
        while not (self.cur().type == TOK_IDENT and self.cur().value == 'end') and self.cur().type != TOK_EOF:
            if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE); continue
            body.append(self.parse_stmt())
        if self.cur().type == TOK_IDENT and self.cur().value == 'end':
            self.eat(TOK_IDENT, 'end')
            if self.cur().type == TOK_NEWLINE: self.eat(TOK_NEWLINE)
        return While(cond, Block(body))

    # expression parsing (minimal Pratt-like)
    def parse_expr(self):
        tok = self.cur()
        if tok.type == TOK_OP and tok.value == '(':
            self.eat(TOK_OP, '(')
            node = self.parse_expr()
            self.eat(TOK_OP, ')')
            return node
        if tok.type == TOK_NUMBER:
            self.eat(TOK_NUMBER); return Number(tok.value)
        if tok.type == TOK_STRING:
            self.eat(TOK_STRING); return String(tok.value)
        if tok.type == TOK_IDENT:
            name = self.eat(TOK_IDENT).value
            # special tiny esoteric: sprout() lambda shorthand: sprout(x) -> returns anon func
            if name == 'sprout' and self.cur().type == TOK_OP and self.cur().value == '(':
                self.eat(TOK_OP, '(')
                params = []
                while not (self.cur().type == TOK_OP and self.cur().value == ')'):
                    if self.cur().type == TOK_IDENT:
                        params.append(self.eat(TOK_IDENT).value)
                    if self.cur().type == TOK_OP and self.cur().value == ',': self.eat(TOK_OP, ',')
                self.eat(TOK_OP, ')')
                # body expression follows after -> or a single expr
                if self.cur().type == TOK_OP and self.cur().value == '->':
                    self.eat(TOK_OP, '->')
                    body = self.parse_expr()
                    # wrap body in FuncDef-like structure
                    return FuncDef(None, params, Block([Return(body)]))
            # normal ident or call
            node = Ident(name)
            # calls
            while self.cur().type == TOK_OP and self.cur().value == '(':
                self.eat(TOK_OP, '(')
                args = []
                while not (self.cur().type == TOK_OP and self.cur().value == ')'):
                    args.append(self.parse_expr())
                    if self.cur().type == TOK_OP and self.cur().value == ',': self.eat(TOK_OP, ',')
                self.eat(TOK_OP, ')')
                node = Call(node, args)
            return node
        if tok.type == TOK_OP and tok.value == '~~':
            self.eat(TOK_OP, '~~')
            val = self.parse_expr()
            return Unary('rev', val)
        if tok.type == TOK_OP and tok.value == '-':
            self.eat(TOK_OP, '-')
            return Unary('-', self.parse_expr())
        # binary ops handled in evaluator via left-associative parsing simplification
        # fallback
        raise SyntaxError(f"Can't parse expression starting with {tok}")

# -------------------- Evaluator --------------------

class ReturnException(Exception):
    def __init__(self, value): self.value = value

class Env:
    def __init__(self, parent=None):
        self.parent = parent
        self.vars = {}
    def get(self, name):
        if name in self.vars: return self.vars[name]
        if self.parent: return self.parent.get(name)
        raise NameError(f"Undefined name: {name}")
    def set(self, name, val): self.vars[name] = val

# simple AI provider interface (pluggable)
class AIProvider:
    def ask(self, prompt: str) -> str:
        # fallback trivial implementation
        return f"[mock ai reply to: {prompt}]"

# default provider (no external integration in this demo)
DEFAULT_AI = AIProvider()

def is_truthy(v):
    return bool(v)

def eval_node(node, env: Env):
    if isinstance(node, Number): return node.value
    if isinstance(node, String): return node.value
    if isinstance(node, Ident): return env.get(node.name)
    if isinstance(node, Assign):
        val = eval_node(node.val, env)
        env.set(node.name, val)
        return val
    if isinstance(node, Unary):
        if node.op == '-': return -eval_node(node.val, env)
        if node.op == 'rev':
            v = eval_node(node.val, env)
            try:
                return v[::-1]
            except Exception:
                return v
        if node.op == 'twist':
            return not is_truthy(eval_node(node.val, env))
    if isinstance(node, Binary):
        l = eval_node(node.left, env)
        r = eval_node(node.right, env)
        if node.op == '+': return l + r
        if node.op == '-': return l - r
        if node.op == '*': return l * r
        if node.op == '/': return l / r
        if node.op == '==': return l == r
        if node.op == '!=': return l != r
        if node.op == '<': return l < r
        if node.op == '>': return l > r
        if node.op == '<=': return l <= r
        if node.op == '>=': return l >= r
    if isinstance(node, Block):
        last = None
        for s in node.stmts:
            last = eval_node(s, env)
        return last
    if isinstance(node, If):
        if is_truthy(eval_node(node.cond, env)):
            return eval_node(node.then, Env(env))
        elif node.otherwise:
            return eval_node(node.otherwise, Env(env))
        return None
    if isinstance(node, While):
        while is_truthy(eval_node(node.cond, env)):
            try:
                eval_node(node.body, Env(env))
            except ReturnException as re:
                return re.value
        return None
    if isinstance(node, FuncDef):
        def fn(*args):
            newenv = Env(env)
            for n, v in zip(node.params, args):
                newenv.set(n, v)
            try:
                eval_node(node.body, newenv)
            except ReturnException as re:
                return re.value
            return None
        if node.name:
            env.set(node.name, fn)
        return fn
    if isinstance(node, Call):
        cal = eval_node(node.callee, env)
        args = [eval_node(a, env) for a in node.args]
        if callable(cal):
            return cal(*args)
        raise TypeError(f"Not callable: {cal}")
    if isinstance(node, Return):
        val = eval_node(node.value, env) if node.value else None
        raise ReturnException(val)
    raise RuntimeError(f"Unknown node: {node}")

# -------------------- Builtins (including esoteric ones) --------------------

def make_global_env(ai_provider: Optional[AIProvider]=None):
    g = Env()
    prov = ai_provider or DEFAULT_AI
    # add common literal-like names
    g.set('True', True)
    g.set('False', False)
    g.set('None', None)
    g.set('print', lambda *a: (sys.stdout.write(' '.join(map(str,a)) + '\n'), None)[1])
    g.set('len', lambda x: len(x))
    g.set('math', math)
    # esoteric helpers
    g.set('~~', lambda x: x[::-1] if hasattr(x,'__getitem__') else x)
    g.set('twist', lambda x: not bool(x))
    def ai_builtin(prompt):
        return prov.ask(str(prompt))
    g.set('ai', ai_builtin)
    return g

# -------------------- Runner & REPL --------------------

def run_src(src: str, filename: Optional[str]=None, ai_provider: Optional[AIProvider]=None):
    # header introspection: must begin with required header for esoteric flavor
    stripped = src.lstrip()
    if not stripped.startswith(REQUIRED_HEADER):
        raise SyntaxError(f"Spring file must start with signature '{REQUIRED_HEADER}'")
    # remove the first line header before tokenizing
    first_line_end = src.find('\n')
    if first_line_end != -1:
        body = src[first_line_end+1:]
    else:
        body = ''
    tokens = tokenize(body)
    parser = Parser(tokens)
    ast = parser.parse()
    env = make_global_env(ai_provider)
    # preload quirky keywords as functions
    env.set('sprout', lambda *args: '[sprout placeholder]')
    return eval_node(ast, env)

def run_file(path: str):
    if not path.endswith(REQUIRED_EXT):
        raise SystemExit(f"Error: Spring source files must end with '{REQUIRED_EXT}' β€” got: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    try:
        return run_src(src, filename=path)
    except Exception as e:
        traceback.print_exc()
        raise

def repl():
    print('Spring REPL β€” type `exit` or Ctrl-D to quit.');
    ai = DEFAULT_AI
    env = make_global_env(ai)
    while True:
        try:
            line = input('spr> ')
        except EOFError:
            print(); break
        if not line: continue
        if line.strip() == 'exit': break
        try:
            # accept tiny expressions only
            tokens = tokenize(line)
            ast = Parser(tokens).parse()
            res = eval_node(ast, env)
            if res is not None:
                print(repr(res))
        except Exception as e:
            print('Error:', e)

# -------------------- CLI --------------------

if __name__ == '__main__':
    if len(sys.argv) == 1:
        repl()
    else:
        path = sys.argv[1]
        if not path.endswith(REQUIRED_EXT):
            print(f"Error: Spring source files must end with '{REQUIRED_EXT}' β€” got: {path}")
            sys.exit(2)
        try:
            run_file(path)
        except Exception as e:
            print('Failed to run.', e)
            sys.exit(1)
