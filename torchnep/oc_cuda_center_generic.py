"""Runtime-C center-owned analytic sector family and mixed pullback."""
from __future__ import annotations
import importlib
import os
from pathlib import Path
import threading
import torch

_EXT=None
_LOCK=threading.Lock()
def _ext():
    global _EXT
    if _EXT is not None:return _EXT
    with _LOCK:
        if _EXT is None:
            if os.environ.get("TORCHNEP_OC_CUDA_PREBUILT","0")=="1":
                _EXT=importlib.import_module("torchnep_oc_center_sector_generic_v1")
            else:
                from torch.utils.cpp_extension import load
                src=Path(__file__).with_name("csrc")/"oc_center_sector_generic.cu"
                _EXT=load(name="torchnep_oc_center_sector_generic_v1",sources=[str(src)],extra_cflags=["-O3"],extra_cuda_cflags=["-O3","-lineinfo"],verbose=False)
    return _EXT

class _Vjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx,go,bank,spins,mix):
        ctx.save_for_backward(go,bank,spins,mix)
        return tuple(_ext().backward(go.contiguous(),bank.contiguous(),spins.contiguous(),mix.contiguous()))
    @staticmethod
    def backward(ctx,hb,hs,hm):
        go,bank,spins,mix=ctx.saved_tensors
        hb=torch.zeros_like(bank) if hb is None else hb
        hs=torch.zeros_like(spins) if hs is None else hs
        hm=torch.zeros_like(mix) if hm is None else hm
        p=_ext().double_backward(go.contiguous(),bank.contiguous(),spins.contiguous(),mix.contiguous(),hb.contiguous(),hs.contiguous(),hm.contiguous())
        n=ctx.needs_input_grad
        return p[0] if n[0] else None,p[1] if n[1] else None,p[2] if n[2] else None,p[3] if n[3] else None

class _Center(torch.autograd.Function):
    @staticmethod
    def forward(ctx,bank,spins,mix):
        ctx.save_for_backward(bank,spins,mix)
        return _ext().forward(bank.contiguous(),spins.contiguous(),mix.contiguous())
    @staticmethod
    def backward(ctx,go):
        bank,spins,mix=ctx.saved_tensors
        C=mix.shape[1]
        if go is None:go=bank.new_zeros((bank.shape[0],13*C+C*(C+1)//2))
        p=_Vjp.apply(go,bank,spins,mix);n=ctx.needs_input_grad
        return p[0] if n[0] else None,p[1] if n[1] else None,p[2] if n[2] else None

def center_sector(bank,spins,mix):
    if not bank.is_cuda or bank.shape[1]==85:return None
    return _Center.apply(bank,spins,mix)
