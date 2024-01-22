`bf2onnx.py` is a script to compile Brainfuck programs into ONNX, the format for neural network models. This is of course a joke, as is common with programs related to Brainfuck.

## Background
Upon learning that ONNX has sufficient expressiveness as an intermediate representation for compiling Brainfuck programs, I could not resist the temptation to actually write a Brainfuck compiler targeting ONNX.

## Example Usage
For compiling a Brainfuck program to `.onnx` file:

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