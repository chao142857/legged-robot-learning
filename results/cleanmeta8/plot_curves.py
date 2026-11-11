"""Render per-seed strict success curves and their equal-weight mean."""

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ('uniform', 'Uniform', '#2563eb', 'o'),
    ('fresh', 'Fresh', '#65748b', 's'),
    ('cleanmeta8', 'CleanMeta8', '#dc582f', '^'),
)
NOTES = {
    42: 'Uniform on RTX 3060. Fresh and CleanMeta8 on RTX 5060 Laptop.',
    47: 'Uniform and Fresh on RTX 3060. CleanMeta8 on RTX 5060 Laptop, restored at update 400.',
    48: 'Uniform and CleanMeta8 on RTX 5060 Laptop. Fresh on RTX 3060.',
    49: 'All three methods on RTX 3060.',
}


def render(data, seed=None):
    aggregate = seed is None
    seeds = data['student_seeds'] if aggregate else [seed]
    updates = np.asarray(data['updates']) / 1000
    figure, axes = plt.subplots(figsize=(10.6, 5.9))
    for key, name, color, marker in METHODS:
        values = np.asarray([data['seeds'][str(s)][key] for s in seeds]) * 100
        assert values.shape == (len(seeds), len(updates))
        assert np.isfinite(values).all() and ((0 <= values) & (values <= 100)).all()
        mean = values.mean(axis=0)
        if aggregate:
            deviation = values.std(axis=0, ddof=1)
            axes.fill_between(updates, np.maximum(0, mean - deviation), mean + deviation,
                              color=color, alpha=.11, linewidth=0, zorder=1)
        axes.plot(updates, mean, color=color, marker=marker, linewidth=2.3, markersize=5.5,
                  zorder=3, label=f'{name}  ·  10k {mean[-1]:.2f}%')

    axes.set_xlim(.75, 10.25)
    axes.set_ylim(0, 45)
    axes.set_xticks(updates)
    axes.set_yticks(range(0, 46, 5))
    axes.set_xlabel('Student PPO updates (thousands)', labelpad=10)
    axes.set_ylabel('Strict joint success (%)', labelpad=10)
    axes.grid(axis='y', color='#e1e7ef', linewidth=.8)
    axes.set_axisbelow(True)
    axes.spines[['top', 'right']].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color('#cbd5e1')
    title = 'WBC learning curves averaged across four seeds' if aggregate else f'WBC learning curves for seed {seed}'
    figure.suptitle(title, x=.5, y=.965, fontsize=16, fontweight='bold')
    figure.text(.5, .905, 'Strict 5 cm / 80% criterion  |  1,024 fixed test sequences  |  Mean policy actions',
                ha='center', fontsize=10, color='#475569')
    handles, labels = axes.get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .865),
                  ncol=3, frameon=False, fontsize=10.4)
    if aggregate:
        caption = 'Seeds 42, 47, 48 and 49, weighted equally. Shading shows ±1 sample standard deviation across seeds.'
    else:
        caption = NOTES[seed]
    figure.text(.096, .065, caption, fontsize=9, color='#475569')
    figure.subplots_adjust(left=.096, right=.978, top=.755, bottom=.19)
    png = io.BytesIO()
    figure.savefig(png, format='png', dpi=155, facecolor='white')
    plt.close(figure)
    return png.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stdout', action='store_true')
    args = parser.parse_args()
    data = globals().get('EMBEDDED_DATA')
    if data is None:
        data = json.loads(Path(__file__).with_name('curve_data.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelcolor': '#243247', 'text.color': '#172337',
                         'xtick.color': '#475569', 'ytick.color': '#475569'})
    images = {f'seed{seed}_strict_success.png': render(data, seed) for seed in data['student_seeds']}
    images['four_seed_mean_strict_success.png'] = render(data)
    if args.stdout:
        print(json.dumps({name: base64.b64encode(content).decode('ascii') for name, content in images.items()}))
    else:
        for name, content in images.items():
            Path(__file__).with_name(name).write_bytes(content)


if __name__ == '__main__':
    main()
