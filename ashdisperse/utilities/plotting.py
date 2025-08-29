from numpy import ceil, sqrt


def plot_rowscols(panels):
    rows = int(ceil(sqrt(panels)))
    cols = int(ceil(panels/rows))
    return rows, cols


def pad_window(window, height, width, pad=1):
    row_start = max(window.row_off - pad, 0)
    row_stop = min(window.row_off + window.height - 1 + pad, height - 1)
    rows = row_stop - row_start + 1
    col_start = max(window.col_off - pad, 0)
    col_stop = min(window.col_off + window.width - 1 + pad, width - 1)
    columns = col_stop - col_start + 1
    return {'row_start': row_start, 'row_stop': row_stop, 'rows': rows,
            'col_start': col_start, 'col_stop': col_stop, 'columns': columns}
