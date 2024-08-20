import sys
import tempfile
import math
import triton
import torch
from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class Layout:
    layout_line: str
    name: str

    def __post_init__(self):
        self.name = self.layout_line.split('=')[0].split('#')[1].strip()


@dataclass
class BlockedLayout(Layout):
    size_per_thread: Optional[list[int]] = None
    threads_per_warp: Optional[list[int]] = None
    warps_per_cta: Optional[list[int]] = None
    order: Optional[list[int]] = None

    def __post_init__(self):
        super().__post_init__()
        # e.g., #blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}
        self.size_per_thread = list(map(int, self.layout_line.split(
            '=')[2].split(',')[0].split('[')[1].split(']')[0].split(',')))
        self.threads_per_warp = list(map(int, self.layout_line.split(
            '=')[3].split(',')[0].split('[')[1].split(']')[0].split(',')))
        self.warps_per_cta = list(map(int, self.layout_line.split(
            '=')[4].split(',')[0].split('[')[1].split(']')[0].split(',')))
        self.order = list(map(int, self.layout_line.split('=')[5].split(',')[
                          0].split('[')[1].split(']')[0].split(',')))


@ dataclass
class NvidiaMmaLayout(Layout):
    version_major: Optional[int] = None
    version_minor: Optional[int] = None
    warps_per_cta: Optional[int] = None
    instr_shape: Optional[list[int]] = None

    def __post_init__(self):
        super().__post_init__()
        # e.g., #mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
        self.version_major = int(self.layout_line.split('=')[2].split(',')[0])
        self.version_minor = int(self.layout_line.split('=')[3].split(',')[0])
        self.warps_per_cta = list(map(int, self.layout_line.split(
            '=')[4].split('[')[1].split(']')[0].split(',')))
        self.instr_shape = list(map(int, self.layout_line.split(
            '=')[5].split('[')[1].split(']')[0].split(',')))


@ dataclass
class SliceLayout(Layout):
    dim: Optional[int] = None
    parent: Optional[Layout] = None

    def __post_init__(self):
        pass


@ dataclass
class Tensor:
    # 64x64xbf16
    shape_and_dtype_str: str
    shape: Optional[list] = None
    dtype: Optional[str] = None
    layout: Optional[Union[BlockedLayout, NvidiaMmaLayout, SliceLayout]] = None

    def __post_init__(self):
        # separate by 'x'
        shape_str = self.shape_and_dtype_str.split('x')[:-1]
        self.dtype = self.shape_and_dtype_str.split('x')[-1]
        self.shape = list(map(int, shape_str))


@ dataclass
class ConvertLayout:
    input_tensor: Tensor
    output_tensor: Tensor
    warps_per_cta: int
    layout_lines: list[str]


def parse_layout(layout_line):
    # e.g., #blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
    layout_name = layout_line.split('=')[0].strip().split('#')[1]
    layout_identifier = layout_line.split('=')[1].strip().split('<')[0]
    if layout_identifier == "#triton_gpu.blocked":
        return layout_name, BlockedLayout(name=layout_name, layout_line=layout_line)
    elif layout_identifier == "#triton_gpu.nvidia_mma":
        return layout_name, NvidiaMmaLayout(name=layout_name, layout_line=layout_line)
    elif layout_identifier == "#triton_gpu.slice":
        return layout_name, SliceLayout(name=layout_name, layout_line=layout_line)
    else:
        raise ValueError(f"Unknown layout identifier: {layout_identifier}")


def extract_tensor_info(tensor_str, layout_dict):
    # e.g., tensor<256xf32, #blocked>
    shape_and_dtype_str = tensor_str.split('<')[1].split(',')[0]
    layout_name = tensor_str.split('<')[1].split(
        ',')[1].split('>')[0].split('#')[1]

    # e.g., tensor<256xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    if "triton_gpu.slice" in tensor_str:
        dim = int(tensor_str.split('<')[2].split(',')[0].split('=')[1])
        parent_str = tensor_str.split('<')[2].split(',')[
            1].split('=')[1].split('}')[0].split('#')[1]
        parent_layout = layout_dict[parent_str]
        layout_name = f"triton_gpu.slice<dim = {dim}, parent = #{parent_str}>"
        layout = SliceLayout(
            layout_line=tensor_str, name=layout_name, dim=dim, parent=parent_layout)
    else:
        layout = layout_dict[layout_name]

    return shape_and_dtype_str, layout


def parse_convert_layout(convert_layout_line, layout_dict, layout_lines):
    # e.g., %149 = triton_gpu.convert_layout %145 : tensor<256xf32, #blocked> -> tensor<256xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    # remove = if exists
    convert_layout_line = convert_layout_line.split(':')[1].strip()
    input_tensor_str = convert_layout_line.split('->')[0].strip()
    output_tensor_str = convert_layout_line.split('->')[1].strip()  
    input_tensor_shape_and_dtype_str, input_tensor_layout = extract_tensor_info(
        input_tensor_str, layout_dict)
    output_tensor_shape_and_dtype_str, output_tensor_layout = extract_tensor_info(
        output_tensor_str, layout_dict)

    input_tensor = Tensor(
        shape_and_dtype_str=input_tensor_shape_and_dtype_str, layout=input_tensor_layout)
    output_tensor = Tensor(
        shape_and_dtype_str=output_tensor_shape_and_dtype_str, layout=output_tensor_layout)
    warps_per_cta = 4
    for _, layout in layout_dict.items():
        if isinstance(layout, (NvidiaMmaLayout, BlockedLayout)):
            warps_per_cta = math.prod(layout.warps_per_cta)
            break
    return ConvertLayout(input_tensor, output_tensor, warps_per_cta, layout_lines)


def parse_file(input_file):
    convert_layouts = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        sections = []
        cur_section = []
        for i in range(len(lines)):
            if lines[i].startswith("// --"):
                if len(cur_section) > 0:
                    sections.append(cur_section)
                cur_section = []
            elif len(lines[i].strip()) > 0:
                cur_section.append(lines[i])
        if len(cur_section) > 0:
            sections.append(cur_section)

        for section in sections:
            layout_dict = {}
            layout_lines = section[:-1]
            convert_layout_line = section[-1]
            for layout_line in layout_lines:
                name, layout = parse_layout(layout_line)
                layout_dict[name] = layout
            convert_layout = parse_convert_layout(
                convert_layout_line, layout_dict, layout_lines)
            convert_layouts.append(convert_layout)
    return convert_layouts


def generate_ttgir1d(kernel_name: str, convert_layout: ConvertLayout):
    M = convert_layout.input_tensor.shape[0]
    dtype = convert_layout.input_tensor.dtype
    src_layout = convert_layout.input_tensor.layout.name
    dst_layout = convert_layout.output_tensor.layout.name
    warps_per_cta = convert_layout.warps_per_cta

    layout_lines = "".join(convert_layout.layout_lines)

    ir = layout_lines + f"""module attributes {{"triton_gpu.num-warps" = {warps_per_cta} : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32}} {{
tt.func public @{kernel_name}(%arg0: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}) {{
    %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{src_layout}>
    %1 = tt.splat %arg0 : !tt.ptr<{dtype}> -> tensor<{M}x!tt.ptr<{dtype}>, #{src_layout}>
    %2 = tt.splat %arg1 : !tt.ptr<{dtype}> -> tensor<{M}x!tt.ptr<{dtype}>, #{dst_layout}>
    %3 = tt.addptr %1, %0 : tensor<{M}x!tt.ptr<{dtype}>, #{src_layout}>, tensor<{M}xi32, #{src_layout}>
    %4 = tt.load %3 : tensor<{M}x!tt.ptr<{dtype}>, #{src_layout}>
    %tmp = triton_gpu.convert_layout %4 : tensor<{M}x{dtype}, #{src_layout}> -> tensor<{M}x{dtype}, #{dst_layout}>
    %idx = arith.constant 0 : i32
    %ub = arith.constant 1024 : i32
    %step = arith.constant 1 : i32
    %5 = scf.for %i = %idx to %ub step %step iter_args(%arg = %tmp) -> (tensor<{M}x{dtype}, #{dst_layout}>) : i32 {{
        %result = triton_gpu.convert_layout %4 : tensor<{M}x{dtype}, #{src_layout}> -> tensor<{M}x{dtype}, #{dst_layout}>
        scf.yield %result : tensor<{M}x{dtype}, #{dst_layout}>
    }}
    %6 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{dst_layout}>
    %7 = tt.addptr %2, %6 : tensor<{M}x!tt.ptr<{dtype}>, #{dst_layout}>, tensor<{M}xi32, #{dst_layout}>
    tt.store %7, %5 : tensor<{M}x!tt.ptr<{dtype}>, #{dst_layout}>
    tt.return
    }}
}}
"""
    return ir


def generate_ttgir2d(kernel_name: str, convert_layout: ConvertLayout):
    M, N = convert_layout.input_tensor.shape
    dtype = convert_layout.input_tensor.dtype
    src_layout = convert_layout.input_tensor.layout.name
    dst_layout = convert_layout.output_tensor.layout.name
    warps_per_cta = convert_layout.warps_per_cta

    layout_lines = "".join(convert_layout.layout_lines)

    ir = layout_lines + f"""module attributes {{"triton_gpu.num-warps" = {warps_per_cta} : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32}} {{
tt.func public @{kernel_name}(%arg0: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}) {{
    %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #{src_layout}>
    %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #triton_gpu.slice<{{dim=1, parent=#{src_layout}}}>>
    %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #triton_gpu.slice<{{dim=0, parent=#{src_layout}}}>>
    %2 = tt.splat %arg0 : !tt.ptr<{dtype}> -> tensor<{M}x{N}x!tt.ptr<{dtype}>, #{src_layout}>
    %3 = tt.splat %arg1 : !tt.ptr<{dtype}> -> tensor<{M}x{N}x!tt.ptr<{dtype}>, #{dst_layout}>
    %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #{src_layout}}}>> -> tensor<{M}x1xi32, #{src_layout}>
    %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #{src_layout}}}>> -> tensor<1x{N}xi32, #{src_layout}>
    %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #{src_layout}>
    %7 = tt.broadcast %6 : tensor<1x{N}xi32, #{src_layout}> -> tensor<{M}x{N}xi32, #{src_layout}>
    %8 = tt.broadcast %5 : tensor<{M}x1xi32, #{src_layout}> -> tensor<{M}x{N}xi32, #{src_layout}>
    %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #{src_layout}>
    %10 = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<{dtype}>, #{src_layout}>, tensor<{M}x{N}xi32, #{src_layout}>
    %11 = tt.load %10 : tensor<{M}x{N}x!tt.ptr<{dtype}>, #{src_layout}>
    %tmp = triton_gpu.convert_layout %11 : tensor<{M}x{N}x{dtype}, #{src_layout}> -> tensor<{M}x{N}x{dtype}, #{dst_layout}>
    %idx = arith.constant 0 : i32
    %ub = arith.constant 1024 : i32
    %step = arith.constant 1 : i32
    %12 = scf.for %i = %idx to %ub step %step iter_args(%arg = %tmp) -> (tensor<{M}x{N}x{dtype}, #{dst_layout}>) : i32 {{
        %result = triton_gpu.convert_layout %11 : tensor<{M}x{N}x{dtype}, #{src_layout}> -> tensor<{M}x{N}x{dtype}, #{dst_layout}>
        scf.yield %result : tensor<{M}x{N}x{dtype}, #{dst_layout}>
    }}
    %13 = triton_gpu.convert_layout %9 : tensor<{M}x{N}xi32, #{src_layout}> -> tensor<{M}x{N}xi32, #{dst_layout}>
    %14 = tt.addptr %3, %13 : tensor<{M}x{N}x!tt.ptr<{dtype}>, #{dst_layout}>, tensor<{M}x{N}xi32, #{dst_layout}>
    tt.store %14, %12 : tensor<{M}x{N}x!tt.ptr<{dtype}>, #{dst_layout}>
    tt.return
  }}
}}
"""
    return ir


def generate_ttgir(kernel_name: str, convert_layout: ConvertLayout):
    if len(convert_layout.input_tensor.shape) == 1:
        return generate_ttgir1d(kernel_name, convert_layout)
    elif len(convert_layout.input_tensor.shape) == 2:
        return generate_ttgir2d(kernel_name, convert_layout)
    else:
        raise ValueError("Only support 1D or 2D tensor for now")


def compile_ttgir(ttgir):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ttgir)
        f.flush()
        kernel = triton.compile(f.name)
    return kernel


def triton_dtype_to_torch_dtype(dtype: str):
    if dtype == "f16":
        return torch.float16
    elif dtype == "f32":
        return torch.float32
    elif dtype == "f64":
        return torch.float64
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "i64":
        return torch.int64
    elif dtype == "i32":
        return torch.int32
    elif dtype == "i16":
        return torch.int16
    elif dtype == "i8":
        return torch.int8
    elif dtype.startswith("f8"):
        # Any fp8 type should work in our test cases
        return torch.float8_e5m2
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def execute(index, kernel, convert_layout: ConvertLayout):
    torch_dtype = triton_dtype_to_torch_dtype(
        convert_layout.input_tensor.dtype)
    src = torch.randn(convert_layout.input_tensor.shape,
                      device='cuda').to(torch_dtype)
    dst = torch.zeros(convert_layout.output_tensor.shape,
                      device='cuda').to(torch_dtype)
    kernel[(1, 1, 1)](src.data_ptr(), dst.data_ptr())
    torch.testing.assert_close(
        dst, src, msg="Mismatch between src and dst")

    time = triton.testing.do_bench_cudagraph(
        lambda: kernel[(1, 1, 1)](src.data_ptr(), dst.data_ptr()), rep=100)
    print(f"Kernel {index} execution time: {time}")


input_file = sys.argv[1]
convert_layouts = parse_file(input_file)

for i, convert_layout in enumerate(convert_layouts):
    ttgir = generate_ttgir("kernel" + str(i), convert_layout)
    kernel = compile_ttgir(ttgir)
    execute(i, kernel, convert_layout)
