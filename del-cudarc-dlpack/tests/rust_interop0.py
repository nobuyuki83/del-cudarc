import torch
import ctypes
from torch.utils.dlpack import to_dlpack
import del_cudarc_dlpack
import numpy as np

if __name__ == "__main__":
    len = 100
    for len in [128,129,512,513]:
        trg = torch.arange(0, len, 1).to(torch.uint32)
        a = torch.zeros(size=(len,), dtype=torch.uint32).cpu()
        del_cudarc_dlpack.set_consecutive_sequence(to_dlpack(a))
        assert torch.equal(a, trg)

        if torch.cuda.is_available():
            a = torch.zeros(size=(len,), dtype=torch.uint32).cuda()
            del_cudarc_dlpack.set_consecutive_sequence(to_dlpack(a))
            assert torch.equal(a.cpu(), trg)

