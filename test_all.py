import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pytest

from bf2onnx import parse, compile_bf_to_onnx, run

def test_all():
    bf = '++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.'
    tree = parse(bf)
    model = compile_bf_to_onnx(tree)
    result = run(model)
    assert result == b'Hello World!\n'
