# Brainfuck to ONNX Compiler (`bf2onnx.py`)
`bf2onnx.py` is a script that compiles Brainfuck programs into the ONNX format. I recently learned that ONNX, a neural network model format, is Turing-complete. This discovery posed a natural challenge to me: to write a Brainfuck compiler targeting it.

## Usage
To compile a Brainfuck program into a `.onnx` file:

```console
$ python ./bf2onnx.py -o hello.onnx --bf '++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.'
```

To see the textual representation of the compiled model:

```console
$ python ./bf2onnx.py --bf '++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.'
<
   ir_version: 9,
   opset_import: ["" : 19]
>
prog () => (int32[?] output) {
   ...
   (snip)
   ...
   output_17 = Concat <axis: int = 0> (output_16, char_12)
   output = Identity (output_17)
}
```

To run the compiled model (with onnxruntime):

```console
$ python ./bf2onnx.py --run --bf '++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.'
Hello World!
```