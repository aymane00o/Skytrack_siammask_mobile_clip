"""Report on one processed run: coverage, continuity, speed, and a sheet to audit.

    python report_run.py car_test.mp4 delivery/car_siammask_4min.csv \
        --frames 7193 --seconds 240 --out delivery --title "car_test 0-4:00, SiamMask"

Writes a markdown report, and a contact sheet of frames sampled across the run
with the tracked box drawn on them, so the one thing telemetry cannot answer -
whether the box is on the right car - can be answered by looking.
"""

import argparse
import csv
import os

import cv2
import numpy as np

HELD = ("locked", "reacquired")


def rows_of(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[int(r["frame"])] = {"state": r["state"],
                                "box": tuple(float(r[k]) for k in "xywh"),
                                "centre": (float(r["cx"]), float(r["cy"])),
                                "speed": float(r["speed_px_s"])}
    return out


def segments(held, total, begin=0):
    """Unbroken stretches of frames with a box, as (first, last)."""
    runs, start, previous = [], None, None
    for f in range(begin, begin + total):
        if f in held:
            if start is None:
                start = f
            previous = f
        elif start is not None:
            runs.append((start, previous))
            start = None
    if start is not None:
        runs.append((start, previous))
    return runs


def sheet(video, held, picks, path, width=320, columns=4):
    cap = cv2.VideoCapture(video)
    tiles = []
    for index in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, image = cap.read()
        if not ok:
            continue
        row = held.get(int(index))
        if row:
            x, y, w, h = (int(v) for v in row["box"])
            cv2.rectangle(image, (x - 2, y - 2), (x + w + 2, y + h + 2), (80, 220, 120), 2)
        label = f"frame {index}" + ("" if row else "  no box")
        cv2.putText(image, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (80, 220, 120) if row else (40, 130, 240), 2, cv2.LINE_AA)
        scale = width / image.shape[1]
        tiles.append(cv2.resize(image, (width, round(image.shape[0] * scale))))
    cap.release()
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + columns])
                                 for i in range(0, len(tiles), columns)]))
    return picks


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("csv")
    p.add_argument("--frames", type=int, required=True, help="frames the run covered")
    p.add_argument("--seconds", type=float, required=True, help="the span in seconds")
    p.add_argument("--took", type=float, default=0.0, help="wall-clock seconds the run took")
    p.add_argument("--out", default="delivery")
    p.add_argument("--title", default="run")
    p.add_argument("--samples", type=int, default=12)
    a = p.parse_args()

    rows = rows_of(a.csv)
    held = {f: r for f, r in rows.items() if r["state"] in HELD}
    begin = min(rows) if rows else 0
    runs = segments(held, a.frames, begin)
    fps_video = a.frames / a.seconds
    picks = np.linspace(begin, begin + a.frames - 1, a.samples).astype(int)
    sheet(a.video, held, picks, os.path.join(a.out, "accuracy_sheet.jpg"))

    lines = [f"# {a.title}", "",
             f"* frames processed: **{a.frames}** ({a.seconds:.0f} s at {fps_video:.2f} fps)",
             f"* frames with the target held: **{len(held)}** "
             f"({100 * len(held) / a.frames:.1f}%)",
             f"* unbroken stretches: **{len(runs)}**", ""]
    if a.took:
        lines.insert(3, f"* processing speed: **{a.frames / a.took:.1f} fps** "
                        f"({a.took:.0f} s for {a.seconds:.0f} s of video)")
    lines += ["| # | first frame | last frame | frames | seconds of video |",
              "| --- | --- | --- | --- | --- |"]
    for i, (first, last) in enumerate(sorted(runs, key=lambda r: r[0] - r[1])[:12], 1):  # longest first
        lines.append(f"| {i} | {first} | {last} | {last - first + 1} | "
                     f"{(last - first + 1) / fps_video:.1f} |")
    if held:
        speeds = [r["speed"] for r in held.values()]
        lines += ["", f"Target speed while held: median {np.median(speeds):.0f} px/s, "
                      f"95th percentile {np.percentile(speeds, 95):.0f} px/s."]
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
