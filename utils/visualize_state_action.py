import os
import curses
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def select_file(stdscr, files, current=0):
    """Curses-based menu. Returns selected index, or None if user quits."""
    curses.curs_set(0)
    stdscr.keypad(True)

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, "↑/↓ navigate · Enter plot · q quit"[:w - 1], curses.A_BOLD)
        stdscr.addstr(1, 0, f"({len(files)} files)"[:w - 1], curses.A_DIM)

        page_size = h - 3
        page_start = max(0, current - page_size // 2)
        page_start = min(page_start, max(0, len(files) - page_size))

        for i in range(page_start, min(page_start + page_size, len(files))):
            line = files[i][:w - 3]
            y = 3 + (i - page_start)
            if i == current:
                stdscr.addstr(y, 0, f"> {line}", curses.A_REVERSE)
            else:
                stdscr.addstr(y, 0, f"  {line}")

        stdscr.refresh()
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            current = max(0, current - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            current = min(len(files) - 1, current + 1)
        elif key == curses.KEY_PPAGE:
            current = max(0, current - page_size)
        elif key == curses.KEY_NPAGE:
            current = min(len(files) - 1, current + page_size)
        elif key == curses.KEY_HOME:
            current = 0
        elif key == curses.KEY_END:
            current = len(files) - 1
        elif key in (10, 13, curses.KEY_ENTER):
            return current
        elif key in (ord('q'), 27):
            return None


def plot_file(filepath):
    df = pd.read_parquet(filepath)
    state = np.stack(df['observation.state'])
    action = np.stack(df['action'])
    n_rows, n_cols = state.shape
    print(f'Rows: {n_rows}, Cols: {n_cols}')

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap('hsv', n_cols)
    x = np.arange(n_rows)

    for j in range(n_cols):
        color = cmap(j)
        ax.plot(x, state[:, j],  color=color, linewidth=0.8, alpha=0.9)             # solid = state
        ax.plot(x, action[:, j], color=color, linewidth=0.8, alpha=0.5, ls='--')    # dashed = action

    ax.set_xlabel('row index (frame)')
    ax.set_ylabel('value (rad)')
    ax.set_title(f'{filepath}: {n_cols} joints — state (solid) vs action (dashed)')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_cols - 1))
    fig.colorbar(sm, ax=ax, label='joint index (0–47)')

    plt.tight_layout()
    plt.show()


base_dir = os.path.expanduser(input('Enter the root directory that contains parquet files: ').strip())
files = sorted(f for f in os.listdir(base_dir) if f.endswith('.parquet'))
assert files, f'No .parquet files in {base_dir}'

current = 0
while True:
    result = curses.wrapper(lambda stdscr: select_file(stdscr, files, current))
    if result is None:
        break
    current = result
    plot_file(f'{base_dir}/{files[current]}')
