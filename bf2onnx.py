# Brainfuck to ONNX Compiler
import onnx
import onnx.helper as oh
import typing as T
from collections import defaultdict

class IRBuilder:

    def __init__(self):
        self.cur_nodes = []
        self.nodes_stack = []
        self.used_names = set()
        self.name_count = defaultdict(int)

    def emit(self, op: str, *, inputs, outputs, **kwargs):
        node = oh.make_node(op, inputs=inputs, outputs=outputs, **kwargs)
        self.cur_nodes.append(node)
        return node

    def pop_graph(self, name: str, *, inputs, outputs, **kwargs):
        nodes = self.cur_nodes
        if self.nodes_stack:
            self.cur_nodes = self.nodes_stack.pop()
        else:
            self.cur_nodes = []
        return oh.make_graph(nodes, name, inputs=inputs, outputs=outputs, **kwargs)

    def enter_graph(self):
        self.nodes_stack.append(self.cur_nodes)
        self.cur_nodes = []

    def generate_name(self, name):
        attempt = name
        while attempt in self.used_names:
            self.name_count[name] += 1
            attempt = f'{name}_{self.name_count[name]}'
        self.used_names.add(attempt)
        return attempt

def parse(bf: str) -> list:
    def _parse(s: str) -> T.Optional[tuple[str, str]]:
        cur = s
        ret = []
        while cur:
            if cur[0] in '+-<>.,':
                ret.append(cur[0])
                cur = cur[1:]
                continue
            if cur[0] == ']':
                break
            if cur[0] != '[':
                cur = cur[1:]
                continue
            maybe_parsed = _parse(cur[1:])
            if maybe_parsed is None:
                return None
            subtree, new = maybe_parsed
            if not new or new[0] != ']':
                return None
            ret.append(subtree)
            cur = new[1:]
        return ret, cur
    maybe_parsed = _parse(bf)
    if maybe_parsed is None:
        raise ValueError('parse error')
    tree, rem = maybe_parsed
    if rem:
        raise ValueError('input is not exhausted after parsing')
    return tree

# Maybe the names should be managed by the builder
def compile_single_bf_inst(builder, inst, mem_name: str, pointer_name: str, output_name: str) -> tuple[str, str, str]:
    if isinstance(inst, list):
        cond = builder.generate_name('cond')
        iszero = builder.generate_name('iszero')
        isnonzero = builder.generate_name('isnonzero')
        value = builder.generate_name('value')
        succ_ptr = builder.generate_name('succ_ptr')

        builder.emit('Add', inputs=[pointer_name, 'one_i32'], outputs=[succ_ptr])
        builder.emit('Slice', inputs=[mem_name, pointer_name, succ_ptr], outputs=[value])
        builder.emit('Equal', inputs=[value, 'zero_i32'], outputs=[iszero])
        builder.emit('Not', inputs=[iszero], outputs=[isnonzero])

        builder.enter_graph()
        mem_inner = mem = builder.generate_name('mem')
        ptr_inner = ptr = builder.generate_name('pointer')
        out_inner = out = builder.generate_name('output')
        for i in inst:
            mem, ptr, out = compile_single_bf_inst(builder, i, mem, ptr, out)
        iszero_inner = builder.generate_name('iszero_inner')
        isnonzero_inner = builder.generate_name('isnonzero_inner')
        value_inner = builder.generate_name('value_inner')
        succ_ptr_inner = builder.generate_name('succ_ptr_inner')
        builder.emit('Add', inputs=[ptr, 'one_i32'], outputs=[succ_ptr_inner])
        builder.emit('Slice', inputs=[mem, ptr, succ_ptr], outputs=[value_inner])
        builder.emit('Equal', inputs=[value_inner, 'zero_i32'], outputs=[iszero_inner])
        builder.emit('Not', inputs=[iszero_inner], outputs=[isnonzero_inner])
        loop_body = builder.pop_graph(
            builder.generate_name('loop_body'),
            inputs=[
                oh.make_tensor_value_info("iter_count", oh.TensorProto.INT64, []),
                oh.make_tensor_value_info("cond_in", oh.TensorProto.BOOL, [1]),
                oh.make_tensor_value_info(mem_inner, oh.TensorProto.INT32, [None]),
                oh.make_tensor_value_info(ptr_inner, oh.TensorProto.INT32, [1]),
                oh.make_tensor_value_info(out_inner, oh.TensorProto.INT32, [None]),
            ],
            outputs=[
                oh.make_tensor_value_info(isnonzero_inner, oh.TensorProto.BOOL, [None]), ## ?
                oh.make_tensor_value_info(mem, oh.TensorProto.INT32, [None]),
                oh.make_tensor_value_info(ptr, oh.TensorProto.INT32, [1]),
                oh.make_tensor_value_info(out, oh.TensorProto.INT32, [None]),
            ]
        )
        mem_after_loop = builder.generate_name('mem_after_loop')
        ptr_after_loop = builder.generate_name('ptr_after_loop')
        out_after_loop = builder.generate_name('out_after_loop')
        loop_body = builder.emit(
            'Loop',
            inputs=['', isnonzero, mem_name, pointer_name, output_name],
            outputs=[mem_after_loop, ptr_after_loop, out_after_loop],
            body=loop_body
        )
        return mem_after_loop, ptr_after_loop, out_after_loop

    if inst in '><':
        ptr = builder.generate_name('pointer')
        if inst == '>':
            builder.emit('Add', inputs=[pointer_name, 'one_i32'], outputs=[ptr])
        else:
            builder.emit('Sub', inputs=[pointer_name, 'one_i32'], outputs=[ptr])
        return mem_name, ptr, output_name
    elif inst in '+-':
        value = builder.generate_name('value')
        succ_ptr = builder.generate_name('succ_ptr')
        builder.emit('Add', inputs=[pointer_name, 'one_i32'], outputs=[succ_ptr])
        builder.emit('Slice', inputs=[mem_name, pointer_name, succ_ptr], outputs=[value])
        mod_value = builder.generate_name('succ_value')
        if inst == '+':
            builder.emit('Add', inputs=[value, 'one_i32'], outputs=[mod_value])
        else:
            builder.emit('Sub', inputs=[value, 'one_i32'], outputs=[mod_value])
        left = builder.generate_name('left')
        right = builder.generate_name('right')
        builder.emit('Slice', inputs=[mem_name, 'zero_i32', pointer_name], outputs=[left])
        builder.emit('Slice', inputs=[mem_name, succ_ptr, 'mem_end'], outputs=[right])
        mem = builder.generate_name('mem')
        builder.emit('Concat', inputs=[left, mod_value, right], outputs=[mem], axis=0)
        return mem, pointer_name, output_name
    elif inst == '.':
        char = builder.generate_name('char')
        succ_ptr = builder.generate_name('succ_ptr')
        builder.emit('Add', inputs=[pointer_name, 'one_i32'], outputs=[succ_ptr])
        builder.emit('Slice', inputs=[mem_name, pointer_name, succ_ptr], outputs=[char])
        out = builder.generate_name('output')
        builder.emit('Concat', inputs=[output_name, char], outputs=[out], axis=0)
        return mem_name, pointer_name, out
    raise NotImplementedError(f'inst {inst} not implemented')

def compile_bf_to_onnx(tree):
    builder = IRBuilder()
    MEMORY_SIZE = 30000
    mem = builder.generate_name('mem')
    mem_end = 'mem_end'  # builder.generate_name('mem_end')
    final_output = builder.generate_name('output')
    output = builder.generate_name('output')
    pointer = builder.generate_name('pointer')
    builder.emit('Constant', inputs=[], outputs=[mem], value=oh.make_tensor(
        name="mem_init",
        data_type=oh.TensorProto.INT32,
        dims=(MEMORY_SIZE,),
        vals=[0]*MEMORY_SIZE,
    ))
    builder.emit('Constant', inputs=[], outputs=[mem_end], value=oh.make_tensor(
        name="mem_end_const",
        data_type=oh.TensorProto.INT32,
        dims=(1,),
        vals=[MEMORY_SIZE],
    ))
    builder.emit('Constant', inputs=[], outputs=[output], value=oh.make_tensor(
        name="output_init",
        data_type=oh.TensorProto.INT32,
        dims=(0,),
        vals=[],
    ))
    builder.emit('Constant', inputs=[], outputs=[pointer], value=oh.make_tensor(
        name="pointer_init",
        data_type=oh.TensorProto.INT32,
        dims=(1,),
        vals=[1],
    ))
    builder.emit('Constant', inputs=[], outputs=['zero_i32'], value=oh.make_tensor(
        name="zero_i32_const",
        data_type=oh.TensorProto.INT32,
        dims=(1,),
        vals=[0],
    ))
    builder.emit('Constant', inputs=[], outputs=['one_i32'], value=oh.make_tensor(
        name="one_i32_const",
        data_type=oh.TensorProto.INT32,
        dims=(1,),
        vals=[1],
    ))
    mem_name = mem
    pointer_name = pointer
    output_name = output
    for inst in tree:
        mem_name, pointer_name, output_name = compile_single_bf_inst(builder, inst, mem_name, pointer_name, output_name)
    builder.emit('Identity', inputs=[output_name], outputs=[final_output])
    graph = builder.pop_graph(
        'prog',
        inputs=[],
        outputs=[oh.make_tensor_value_info(final_output, oh.TensorProto.INT32, [None])]
    )
    model = oh.make_model(graph=graph, opset_imports=[oh.make_opsetid("", 19)])
    return model

def run(model):
    import onnxruntime
    session = onnxruntime.InferenceSession(model.SerializeToString())
    result = session.run(['output'], {})
    return bytes(result[0].tolist())

def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '-o', help='ONNX output path. If not given the textual form of the model is printed to stdout')
    parser.add_argument('--run', action='store_true', help='Run the compiled model rather than writing it')
    parser.add_argument('--bf', help='Brainfuck program')
    parser.add_argument('--file', '-f', help='Brainfuck program file')
    args = parser.parse_args()
    if (not args.bf) == (not args.file):
        print(parser.format_help(), file=sys.stderr)
        print('Exactly one of --bf and --file must be given', file=sys.stderr)
        sys.exit(1)
    if args.bf:
        bf = args.bf
    elif args.file:
        with open(args.file, 'rb') as fp:
            bf = fp.read().decode('utf-8')
    model = compile_bf_to_onnx(parse(bf))
    onnx.checker.check_model(model)

    if args.run:
        result = run(model)
        sys.stdout.buffer.write(result)
    else:
        if args.out:
            with open(args.out, 'wb') as fp:
                fp.write(model.SerializeToString())
        else:
            print(onnx.printer.to_text(model))

if __name__ == '__main__':
    main()
