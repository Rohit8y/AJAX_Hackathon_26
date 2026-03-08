#!/usr/bin/env python3
"""
Match Event Visualiser  (2-D pitch top-down view)
==================================================
Shows a top-down 2-D pitch for any event code.

Change EVENT_CODES below to any code(s) from the XML.
Run with --list to see all available event codes.

Run:
    .venv/bin/python3 visualize_goal.py
    .venv/bin/python3 visualize_goal.py --event 2       # 0-based event index
    .venv/bin/python3 visualize_goal.py --list          # print all event codes

Common codes:
    "AJAX | DOELPUNT"               Ajax goals
    "FORTUNA SITTARD | DOELPUNT"    Fortuna goals
    "AJAX | SCHOT"                  Ajax shots
    "FORTUNA SITTARD | SCHOT"       Fortuna shots
    "AJAX | CORNER"                 Ajax corners
    "AJAX | KANS"                   Ajax chances
"""

import sys
import argparse

sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button

from skeleton_data import SkeletonData, BodyPart, Team
from event_data import EventParser

# ── Config — change these to visualise different events ───────────────────────

PARQUET = '/Users/rohityadav/Documents/AJAX/HackathonData/anonymized-limbtracking.parquet'
XML     = 'data/XML_anonymized.xml'

EVENT_CODES = [
    # 'AJAX | DOELPUNT',
    'FORTUNA SITTARD | DOELPUNT',
    #'AJAX | SCHOT'
    #'FORTUNA SITTARD | SCHOT',

]

PAD_BEFORE = 3.0   # seconds before event start
PAD_AFTER  = 2.0   # seconds after event end

TEAM_STYLE = {
    Team.TEAM_A:  ('#457b9d', 'Fortuna'),
    Team.REFEREE: ('#e63946', 'AJAX'),
    Team.TEAM_B:  ('#f4d35e', 'Referee'),
    Team.HOME:    ('#aaaaaa', 'Other'),
}

BALL_BASE_SIZE  = 120
BALL_SIZE_PER_M = 40


# ── Pitch drawing ─────────────────────────────────────────────────────────────

def _draw_pitch(ax, hx: float, hy: float):
    ax.set_facecolor('#1a6b2a')
    kw = dict(color='white', lw=1.5, zorder=1)

    def ln(xs, ys, **extra):
        ax.plot(xs, ys, **{**kw, **extra})

    ln([-hx, hx, hx, -hx, -hx], [-hy, -hy, hy, hy, -hy], lw=2)
    ln([0, 0], [-hy, hy])
    th = np.linspace(0, 2 * np.pi, 72)
    ln(9.15 * np.sin(th), 9.15 * np.cos(th))
    ax.scatter([0], [0], color='white', s=20, zorder=3)

    for sx in (-1, 1):
        gx = sx * hx
        for d, w in [(16.5, 20.16), (5.5, 9.16)]:
            dx = -sx * d
            ln([gx, gx+dx, gx+dx, gx, gx], [-w, -w, w, w, -w], lw=1.2)
        ax.scatter([sx * (hx - 11.0)], [0], color='white', s=12, zorder=3)
        ln([gx, gx], [-3.66, 3.66], lw=4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--event', type=int, default=0,
                     help='0-based index of the event to visualise')
    cli.add_argument('--list', action='store_true',
                     help='Print all available event codes and exit')
    args = cli.parse_args()

    print('Loading skeleton data…')
    skel_data  = SkeletonData(PARQUET)
    evt_parser = EventParser(XML, skel_data)
    meta       = skel_data.metadata

    if args.list:
        print('\nAll available event codes:')
        for code in sorted(evt_parser.all_codes):
            print(f'  {code!r}')
        return

    hx  = meta.pitch_long  * 0.1 / 2
    hy  = meta.pitch_short * 0.1 / 2
    fps = meta.framerate
    print(f'Pitch: {hx*2:.1f} m x {hy*2:.1f} m  |  {fps} fps')

    results    = evt_parser.get_frames_for_events(EVENT_CODES, PAD_BEFORE, PAD_AFTER)
    event_list = list(results.items())

    if not event_list:
        print(f'No events found for: {EVENT_CODES}')
        return

    print(f'\nFound {len(event_list)} event(s):')
    for i, (ev, fr) in enumerate(event_list):
        marker = ' <-- selected' if i == args.event else ''
        print(f'  [{i}] {ev}{marker}')

    if args.event >= len(event_list):
        print(f'\nError: --event {args.event} out of range (0–{len(event_list)-1})')
        return

    event, frames = event_list[args.event]
    n_frames = len(frames)
    default_idx = min(round(PAD_BEFORE * fps), n_frames - 1)

    print(f'\nVisualising: {event}')
    print(f'{n_frames} frames  ({PAD_BEFORE}s before + duration + {PAD_AFTER}s after)')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor='#0d1117')
    fig.suptitle(f'{event.code}  (event {args.event}/{len(event_list)-1})',
                 color='white', fontsize=11, y=0.97)

    ax = fig.add_axes([0.04, 0.15, 0.92, 0.80])
    ax.set_aspect('equal')
    ax.set_xlim(-hx * 1.08, hx * 1.08)
    ax.set_ylim(-hy * 1.12, hy * 1.12)
    ax.tick_params(colors='#555', labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    _draw_pitch(ax, hx, hy)

    # Pre-allocate one scatter per team
    player_scatts = {}
    for team, (color, _) in TEAM_STYLE.items():
        player_scatts[team] = ax.scatter([], [], color=color, s=80, zorder=6,
                                         edgecolors='white', linewidths=0.5)

    ball_scat = ax.scatter([], [], color='#ff9f1c', s=BALL_BASE_SIZE,
                           edgecolors='white', linewidths=0.8, zorder=10)
    ball_text = ax.text(0, 0, '', color='#ff9f1c', fontsize=7,
                        ha='left', va='bottom', zorder=11, fontfamily='monospace')
    frame_lbl = ax.text(0.01, 0.99, '', transform=ax.transAxes,
                        color='white', fontsize=8, va='top', fontfamily='monospace')

    legend_patches = [mpatches.Patch(color=c, label=l) for _, (c, l) in TEAM_STYLE.items()]
    legend_patches.append(mpatches.Patch(color='#ff9f1c', label='Ball'))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8,
              facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)

    player_texts = []

    # ── Draw function ─────────────────────────────────────────────────────────
    def draw(idx: int):
        nonlocal player_texts
        idx = int(np.clip(idx, 0, n_frames - 1))
        frame = frames[idx]

        for t in player_texts:
            t.remove()
        player_texts = []

        for team, (color, _) in TEAM_STYLE.items():
            xs, ys = [], []
            for player in frame.players:
                if player.team != team:
                    continue
                pelvis = player.parts.get(BodyPart.PELVIS)
                if pelvis is None:
                    continue
                xs.append(pelvis.x)
                ys.append(pelvis.y)
                t = ax.text(pelvis.x, pelvis.y + 1.2, str(player.jersey_number),
                            color='white', fontsize=6, ha='center', va='bottom',
                            fontweight='bold', zorder=7)
                player_texts.append(t)
            scat = player_scatts[team]
            scat.set_offsets(np.column_stack([xs, ys]) if xs else np.empty((0, 2)))

        if frame.ball:
            b = frame.ball.position
            ball_scat.set_offsets([[b.x, b.y]])
            ball_scat.set_sizes([BALL_BASE_SIZE + max(0.0, b.z) * BALL_SIZE_PER_M])
            ball_text.set_position((b.x + 0.8, b.y + 0.8))
            ball_text.set_text(f'z={b.z:.1f}m')
        else:
            ball_scat.set_offsets(np.empty((0, 2)))
            ball_text.set_text('')

        rel_sec = (idx - round(PAD_BEFORE * fps)) / fps
        frame_lbl.set_text(f'Frame {frame.frame_number}   t = {rel_sec:+.2f}s  (0 = event start)')
        fig.canvas.draw_idle()

    # ── Slider ────────────────────────────────────────────────────────────────
    ax_slider = fig.add_axes([0.12, 0.07, 0.76, 0.025], facecolor='#1a1a2e')
    slider = Slider(ax_slider, '', 0, n_frames - 1,
                    valinit=default_idx, valstep=1, color='#e63946')
    ax_slider.set_title('Frame', color='white', fontsize=7, pad=2)
    slider.valtext.set_color('white')
    slider.on_changed(lambda v: draw(v))

    # ── Play / Pause ──────────────────────────────────────────────────────────
    state = {'playing': False, 'timer': None}

    def _advance():
        nxt = int(slider.val) + 1
        if nxt >= n_frames:
            toggle_play(None)
            return
        slider.set_val(nxt)

    def toggle_play(_):
        if state['playing']:
            if state['timer']:
                state['timer'].stop()
            state['playing'] = False
            btn_play.label.set_text('Play')
        else:
            state['playing'] = True
            btn_play.label.set_text('Pause')
            state['timer'] = fig.canvas.new_timer(interval=int(1000 / fps))
            state['timer'].add_callback(_advance)
            state['timer'].start()

    ax_play = fig.add_axes([0.455, 0.02, 0.09, 0.035], facecolor='#1a1a2e')
    btn_play = Button(ax_play, 'Play', color='#1a1a2e', hovercolor='#2a2a4e')
    btn_play.label.set_color('white')
    btn_play.on_clicked(toggle_play)

    draw(default_idx)

    print()
    print('Controls:')
    print('  Drag slider   — scrub through frames')
    print('  Play button   — animate at match speed')
    print()

    plt.show()


if __name__ == '__main__':
    main()
