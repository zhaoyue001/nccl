#!/usr/bin/env python3
"""
Patched generate.py for NCCL — adds per-group .so registration file generation.
Each collective operation compiles into its own independent .so:
  - Simple ops (AllGather, Broadcast, SendRecv): one .so per collective
  - Complex ops (AllReduce, Reduce, ReduceScatter): one .so per (redop, type) variant
"""
import os, sys, json

# Order of redops, tys, protos, algos must match src/include/device.h
all_colls =  ["Broadcast","Reduce","AllGather","ReduceScatter","AllReduce","SendRecv"]
all_redops = ["Sum","Prod","MinMax","PreMulSum","SumPostDiv"]
all_tys =    ["i8","u8","i32","u32","i64","u64","f16","f32","f64","bf16"]
all_protos = ["LL","LL128","SIMPLE"]
all_algos =  ["TREE","RING","COLLNET_DIRECT","COLLNET_CHAIN","NVLS","NVLS_TREE"]

gensrc = sys.argv[1]

if os.path.exists(gensrc):
  for name in os.listdir(gensrc):
    os.remove(os.path.join(gensrc, name))
else:
  os.mkdir(gensrc)

func_pattern = sys.argv[2:3]
if func_pattern and func_pattern[0]:
  import re
  func_pattern = func_pattern[0]
  func_pattern = func_pattern.replace("*", "[^ ]*")
  func_pattern += "$"
  def func_filter(*fn):
    return None is not re.match(func_pattern, paste(" ", *fn), flags=re.IGNORECASE)
else:
  def func_filter(coll, redop, ty, algo, proto):
    return True

def paste(sep, *args):
  return sep.join(x for x in args if x is not None)

algos_of_coll = {
  "AllGather":     ["RING","COLLNET_DIRECT","NVLS"],
  "AllReduce":     all_algos,
  "Broadcast":     ["RING"],
  "Reduce":        ["RING"],
  "ReduceScatter": ["RING","COLLNET_DIRECT","NVLS"],
  "SendRecv":      [None]
}

coll_camel_to_lower = {
  "AllGather":     "all_gather",
  "AllReduce":     "all_reduce",
  "Broadcast":     "broadcast",
  "Reduce":        "reduce",
  "ReduceScatter": "reduce_scatter",
  "SendRecv":      "sendrecv"
}
coll_lower_to_camel = {coll_camel_to_lower[x]: x for x in coll_camel_to_lower}

def required_cuda(coll, redop, ty, algo, proto):
  cudart, arch = 0, 0
  if coll in ("SendRecv", "Generic", "Nop"): return (cudart, arch)
  if proto!="SIMPLE" and algo not in ("RING","TREE"): return None
  if coll in ("AllReduce","Reduce","ReduceScatter"):
    if redop=="SumPostDiv" and ty[0] not in ("i","u"): return None
    if ty=="bf16": cudart = max(cudart, 11000)
  if "NVLS" in algo:
    if coll in ("AllReduce","Reduce","ReduceScatter"):
      nvls_ok = ((ty in ("i32","u32","i64","u64") and redop in ("Sum","MinMax")) or
                 (ty in ("f32","f64") and redop=="Sum") or
                 (ty in ("f16","bf16") and redop in ("Sum","MinMax")))
      if not nvls_ok: return None
    cudart = max(cudart, 12010)
    arch = max(arch, 900)
  return (cudart, arch)

def equivalent_primary(coll, redop, ty, algo, proto):
  if coll in ("AllReduce", "Reduce", "ReduceScatter"):
    if redop in ("Sum","Prod","PreMulSum") and ty[0]=="i":
      return (coll, redop, "u"+ty[1:], algo, proto)
    if redop=="MinMax" and ty[0]=="i" and ("NVLS" not in algo):
      return (coll, redop, "u"+ty[1:], algo, proto)
  return (coll, redop, ty, algo, proto)

def best_kernel(coll, redop, ty, algo, proto):
  def best(coll, redop, ty, algo, proto):
    if coll=="Nop": return ("Generic", None, None, None, None)
    if coll=="SendRecv": return ("SendRecv", None, None, None, None)
    if coll in ("AllGather","Broadcast"): return (coll, None, None, "RING", proto)
    return (coll, "Sum", ty, ("TREE" if algo=="TREE" else "RING"), proto)
  kfn = equivalent_primary(*best(coll, redop, ty, algo, proto))
  if not func_filter(*kfn): return ("Generic", None, None, None, None)
  return kfn

def enumerate_func_rows():
  yield ("SendRecv", None, None, None, None)
  for coll in ("AllGather", "Broadcast"):
    for algo in algos_of_coll[coll]:
      for proto in all_protos:
        yield (coll, None, None, algo, proto)
  for coll in ("AllReduce", "Reduce", "ReduceScatter"):
    for redop in all_redops:
      for ty in all_tys:
        for algo in algos_of_coll[coll]:
          for proto in all_protos:
            yield (coll, redop, ty, algo, proto)

def is_built(coll, redop, ty, algo, proto):
  built = required_cuda(coll, redop, ty, algo, proto)
  return built and func_filter(coll, redop, ty, algo, proto)

def validate(coll, redop, ty, algo, proto):
  valid = required_cuda(coll, redop, ty, algo, proto)
  built = valid and func_filter(coll, redop, ty, algo, proto)
  if built: return (coll, redop, ty, algo, proto)
  if valid: return ("Nop", None, None, None, None)
  return None

func_rows = [validate(*fn) for fn in enumerate_func_rows()]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn is not None))
primary_to_index = {fn: i for (i,fn) in zip(range(len(primary_funcs)), primary_funcs)}
kernel_funcs = sorted(set(best_kernel(*fn) for fn in primary_funcs))

# device_table.cu no longer generated — device-side dispatch table removed.
# All dispatch is now host-side via specialized kernel + per-op launch function.

# Generate host_table.cc (ncclDevFuncRowToId mapping only — runtime arrays in ops_core.cc)
with open(os.path.join(gensrc, "host_table.cc"), "w") as f:
  out = f.write
  out('#include "device.h"\n\n')
  out("extern int const ncclDevFuncRowToId[] = {\n")
  index = 0
  for fn in func_rows:
    fn_id, comment = -1, ""
    if fn is not None:
      fn_id = primary_to_index[equivalent_primary(*fn)]
      comment = " // " + paste(" ", *fn)
    out("/*%4d*/ %d,%s\n" % (index, fn_id, comment))
    index += 1
  out("-1};\n")

def impl_filename(coll, redop, ty, algo, proto):
  return "%s.cu" % paste("_", coll_camel_to_lower[coll], redop and redop.lower(), ty)

def partition_by_name(fns):
  ans = {}
  for fn in fns:
    name = impl_filename(*fn)
    coll = fn[0]
    if name not in ans:
      ans[name] = (coll, [])
    ans[name][1].append(fn)
  return ans

name_to_funcs = partition_by_name(fn for fn in primary_funcs if fn[0]!="Nop")
name_to_kernels = partition_by_name(kfn for kfn in kernel_funcs if kfn[0]!="Generic")
redop_to_cxx = {
  None: "FuncCopy",
  "Sum": "FuncSum", "Prod": "FuncProd", "MinMax": "FuncMinMax",
  "PreMulSum": "FuncPreMulSum", "SumPostDiv": "FuncSumPostDiv"
}
ty_to_cxx = {
  None: "int8_t",
  "i8": "int8_t", "u8": "uint8_t", "i32": "int32_t", "u32": "uint32_t",
  "i64": "int64_t", "u64": "uint64_t", "f16": "half", "f32": "float",
  "f64": "double", "bf16": "__nv_bfloat16"
}

# ===== BUILD THE GROUPING =====
# Simple ops: one .so per collective (all algos/protos)
# Complex ops: one .so per (coll, redop, type) variant (all algos/protos)
group_info = {}  # group_name -> {coll, redop, ty, files: [filenames], funcs: [...], kernels: [...]}

for name in name_to_funcs.keys():
  (coll, fns) = name_to_funcs[name]
  (_, kfns) = name_to_kernels.get(name) or (None, [])
  
  if coll in ("AllReduce", "Reduce", "ReduceScatter"):
    # Complex ops: group by (coll, redop, type)
    # Each .cu file belongs to ONE (redop, type) combo based on its name
    for fn in fns:
      if fn[0] == "Nop": continue
      c, redop, ty, algo, proto = fn
      group_key = (c, redop, ty)
      group_name = paste("_", coll_camel_to_lower[c], redop.lower(), ty)
      if group_name not in group_info:
        group_info[group_name] = {"coll": c, "redop": redop, "ty": ty, 
                                   "files": set(), "funcs": [], "kernels": []}
      group_info[group_name]["files"].add(name)
      group_info[group_name]["funcs"].append(fn)
    
    for kfn in kfns:
      if kfn[0] == "Generic": continue
      c, redop, ty, algo, proto = kfn
      group_key = (c, redop, ty)
      group_name = paste("_", coll_camel_to_lower[c], redop.lower(), ty)
      if group_name not in group_info:
        group_info[group_name] = {"coll": c, "redop": redop, "ty": ty,
                                   "files": set(), "funcs": [], "kernels": []}
      group_info[group_name]["files"].add(name)
      group_info[group_name]["kernels"].append(kfn)
  else:
    # Simple ops: one .so for all variants
    group_name = coll_camel_to_lower[coll]
    if group_name not in group_info:
      group_info[group_name] = {"coll": coll, "redop": None, "ty": None,
                                 "files": set(), "funcs": [], "kernels": []}
    group_info[group_name]["files"].add(name)
    for fn in fns:
      if fn[0] != "Nop":
        group_info[group_name]["funcs"].append(fn)
    for kfn in kfns:
      if kfn[0] != "Generic":
        group_info[group_name]["kernels"].append(kfn)

# Write group info as JSON for the Makefile
with open(os.path.join(gensrc, "group_info.json"), "w") as f:
  json.dump({k: {"coll": v["coll"], "redop": v["redop"], "ty": v["ty"],
                  "files": sorted(v["files"]), 
                  "func_count": len(v["funcs"]),
                  "kernel_count": len(v["kernels"])}
             for k, v in sorted(group_info.items())}, f, indent=2)

# ===== Generate per-group register files =====
for group_name, info in sorted(group_info.items()):
  coll = info["coll"]
  fns = info["funcs"]
  kfns = info["kernels"]
  
  with open(os.path.join(gensrc, f"{group_name}_register.cc"), "w") as f:
    out = f.write
    out('#include "device.h"\n')
    out('#include <cuda_runtime.h>\n')
    # Forward-declare the shared memory setters defined in per-op .cu files
    for name in sorted(info["files"]):
      base = name.replace('.cu', '')
      out(f'extern "C" void ncclOpSetShmem_{base}(int smem);\n')
    out('\n')
    
    # Forward declarations for kernels used in this group
    seen_kernels = set()
    for kfn in kfns:
      sym = paste("_", "ncclDevKernel", *kfn)
      if sym in seen_kernels: continue
      seen_kernels.add(sym)
      cudart, arch = required_cuda(*kfn)
      out(f"__global__ void {sym}(struct ncclDevComm*, uint64_t, struct ncclWork*);\n")
    
    out(f"\n// Group: {group_name}  coll={coll}  funcs={len(fns)}  kernels={len(kfns)}\n\n")
    
    # Kernel list for this group
    out(f"static void* _{group_name}_kernelList[] = {{\n")
    for kfn in kfns:
      sym = paste("_", "ncclDevKernel", *kfn)
      out(f"  (void*){sym},\n")
    out(f"  nullptr\n}};\n\n")
    
    # Launch function — simple delegate to cudaLaunchKernel.
    # Shared memory attribute is set in the .cu file (same CUDA module as kernel).
    out(f"static void ncclOpLaunch_{group_name}(void* fn, dim3 grid, dim3 block, void** args, size_t smem, cudaStream_t stream) {{\n")
    out(f"  cudaLaunchKernel(fn, grid, block, args, smem, stream);\n")
    out(f"}}\n")
    
    # Offsets into the global tables — compute from primary_to_index
    out(f"// Register this group's device functions and kernels with the main runtime\n")
    out(f"extern \"C\" __attribute__((visibility(\"default\"))) void ncclOpRegister_{group_name}() {{\n")
    # Set shared memory on kernels BEFORE registering them.
    # ncclOpSetShmem_* functions are defined in the .cu files (same CUDA module).
    out(f"  int _ncclSmem = ncclShmemDynamicSize(860); // sm_86\n")
    for name in sorted(info["files"]):
      base = name.replace('.cu', '')
      out(f"  ncclOpSetShmem_{base}(_ncclSmem);\n")
    out(f"  \n")
    
    # Register func→kernel mappings for each primary function in this group
    # Pass the launch function so main .so delegates cudaLaunchKernel to this .so
    out(f"  void* launchFn = (void*)ncclOpLaunch_{group_name};\n")
    for fn in fns:
      fn_id = primary_to_index.get(fn, -1)
      if fn_id < 0: continue
      kfn = best_kernel(*fn)
      specialized = "true"
      kfn_id = kernel_funcs.index(kfn) if kfn in kernel_funcs else -1
      if kfn_id >= 0:
        for ki, loc_kfn in enumerate(kfns):
          if loc_kfn == kfn:
            out(f"  ncclRegisterFuncKernel({fn_id}, _{group_name}_kernelList[{ki}], {specialized}, launchFn);\n")
            break
    
    out(f"}}\n")

# ===== Generate group_build.mk =====
with open(os.path.join(gensrc, "group_build.mk"), "w") as f:
  out = f.write
  out("# Auto-generated per-group .so build rules — fully parallel\n")
  out("# Each group .so is self-contained: its .cu files + common/onerank + register + launch\n")
  out(f"OPSDIR := $(BUILDDIR)/ops\n")
  out(f"GROUPS := {' '.join(sorted(group_info.keys()))}\n\n")
  
  common_o = "$(OBJDIR)/common.cu.o"
  onerank_o = "$(OBJDIR)/onerank.cu.o"
  
  for group_name, info in sorted(group_info.items()):
    files = sorted(info["files"])
    cu_objs = " ".join(f"$(OBJDIR)/genobj/{f}.o" for f in files)
    group_glue = f"$(OBJDIR)/genobj/{group_name}_glue.o"
    
    out(f"# --- {group_name} (self-contained) ---\n")
    out(f"{group_glue}: {cu_objs} {common_o} {onerank_o}\n")
    out(f"\t@printf \"%-15s %s\\n\" \"Dlink\" \"{group_name}\"\n")
    out(f"\t$(NVCC) $(NVCUFLAGS) -dlink {cu_objs} {common_o} {onerank_o} -o $@\n\n")
    
    out(f"$(OPSDIR)/nccl_{group_name}.so: {group_glue} $(OBJDIR)/genobj/{group_name}_register.o {cu_objs} {common_o} {onerank_o}\n")
    out(f"\t@printf \"%-15s %s\\n\" \"Linking\" \"{group_name}\"\n")
    out(f"\t@mkdir -p $(OPSDIR)\n")
    out(f"\t$(NVCC) -shared -o $@ $(OBJDIR)/genobj/{group_name}_register.o {cu_objs} {common_o} {onerank_o} {group_glue} -L$(BUILDDIR)/lib -lnccl -Xlinker -rpath -Xlinker $(BUILDDIR)/lib\n\n")
  
  out(f"ops: $(GROUPS:%=$(OPSDIR)/nccl_%.so)\n")
  out(f".PHONY: ops\n")

# ===== Generate register files build rules =====
with open(os.path.join(gensrc, "register_rules.mk"), "w") as f:
  out = f.write
  for group_name in sorted(group_info.keys()):
    out(f"$(OBJDIR)/genobj/{group_name}_register.o: $(OBJDIR)/gensrc/{group_name}_register.cc\n")
    out(f"\t$(NVCC) $(NVCUFLAGS) -dc $< -o $@\n\n")

# ===== Generate the original .cu files =====
with open(os.path.join(gensrc, "rules.mk"), "w") as f:
  out = f.write
  impl_names = sorted(name_to_funcs.keys())
  names = impl_names  # device_table.cu excluded — no device-side dispatch table needed
  out("LIB_OBJS_GEN = $(patsubst %, $(OBJDIR)/genobj/%.o, {names})\n"
      .format(names=" ".join(names)))
  out("\n")
  for name in impl_names:
    coll = name_to_funcs[name][0]
    out(
      "$(OBJDIR)/genobj/{name}.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/{lower_coll}.cu.d\n"
      "\t" "$(call COMPILE,$@,$(OBJDIR)/gensrc/{name})\n"
      "\n"
      .format(name=name, lower_coll=coll_camel_to_lower[coll])
    )

for coll in set(coll for (coll,_,_,_,_) in primary_funcs if coll!="Nop"):
  name = impl_filename(coll, None, None, None, None)
  if name not in name_to_funcs:
    name_to_funcs[name] = (coll, [])

for name in name_to_funcs.keys():
  (coll, fns) = name_to_funcs[name]
  with open(os.path.join(gensrc, name), "w") as f:
    out = f.write
    out('#include "common.h"\n'
        '#include "{lower_coll}.h"\n'
        .format(lower_coll=coll_camel_to_lower[coll]))
    (_, kfns) = name_to_kernels.get(name) or (None, [])
    for kfn in kfns:
      (coll, redop, ty, algo, proto) = kfn
      sym = paste("_", coll, redop, ty, algo, proto)
      fn_id = primary_to_index[kfn]
      cudart, arch = required_cuda(*kfn)
      if (cudart, arch) != (0, 0):
        out("#if CUDART_VERSION >= %d && __CUDA_ARCH__ >= %d\n" % (cudart, arch))
      out(
        "DEFINE_ncclDevKernel({sym}, ncclFunc{coll}, {redop_cxx}, {ty_cxx}, NCCL_ALGO_{algo}, NCCL_PROTO_{proto}, {fn_id})\n"
        .format(sym=sym, coll=coll, redop_cxx=redop_to_cxx[redop], ty_cxx=ty_to_cxx[ty],
                algo=(algo or "RING"), proto=(proto or "SIMPLE"), fn_id=fn_id)
      )
      if (cudart, arch) != (0, 0):
        out("#endif\n")
    for fn in fns:
      (coll, redop, ty, algo, proto) = fn
      sym = paste("_", coll, redop, ty, algo, proto)
      cudart, arch = required_cuda(*fn)
      if (cudart, arch) != (0, 0):
        out("#if CUDART_VERSION >= %d && __CUDA_ARCH__ >= %d\n" % (cudart, arch))
      out(
        "DEFINE_ncclDevFunc({sym}, ncclFunc{coll}, {redop_cxx}, {ty_cxx}, NCCL_ALGO_{algo}, NCCL_PROTO_{proto})\n"
        .format(sym=sym, coll=coll, redop_cxx=redop_to_cxx[redop], ty_cxx=ty_to_cxx[ty],
                algo=(algo or "RING"), proto=(proto or "SIMPLE"))
      )
      if (cudart, arch) != (0, 0):
        out("#endif\n")

    # Add shared memory attribute setter — MUST be in same .cu as kernel definitions.
    # cudaFuncSetAttribute only works when called from the same CUDA module
    # as the kernel definition. Cross-.so calls fail with INVALID_RESOURCE_HANDLE.
    # Always define the function (even if kfns is empty) so the register file
    # can call it unconditionally.
    out('\n// Per-op shared memory init — called from register function after dlopen.\n')
    out('// Uses LOCAL kernel symbols (same .cu, same CUDA module).\n')
    out('#include <cuda_runtime.h>\n')
    out('extern "C" __attribute__((visibility("default"))) void %s(int smem) {\n'
        % ('ncclOpSetShmem_' + name.replace('.cu', '')))
    if kfns:
      seen = set()
      for kfn in kfns:
        sym = paste("_", "ncclDevKernel", *kfn)
        if sym in seen: continue
        seen.add(sym)
        out('  cudaFuncSetAttribute((const void*)%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n' % sym)
    out('}\n')

print(f"Generated {len(group_info)} groups in {gensrc}/group_info.json")
print(f"Simple ops: {sum(1 for v in group_info.values() if v['coll'] in ('AllGather','Broadcast','SendRecv'))}")
print(f"Complex ops: {sum(1 for v in group_info.values() if v['coll'] in ('AllReduce','Reduce','ReduceScatter'))}")
