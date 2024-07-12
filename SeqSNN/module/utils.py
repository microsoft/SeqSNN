from torch import nn


def get_cell(cell_type):
    try:
        Cell = getattr(nn, cell_type.upper())
    except Exception:
        raise ValueError(f"Unknown RNN cell type {cell_type}")
    return Cell
