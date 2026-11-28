"""Checkpoint I/O and cross-rank reductions used by the trainer."""

import csv
import json

import torch
import torch.distributed as dist


def json_save(path, value):
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def torch_save(path, value):
    pending = path.with_name(path.name + ".tmp")
    torch.save(value, pending)
    pending.replace(path)


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def collective_sum(values, device, world):
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(tensor)
    return tensor.cpu().tolist()
