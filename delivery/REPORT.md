# car_test.mp4, 0:00–4:00, tracked with SiamMask

**Target:** the grey SUV the helicopter is following.
**Source:** `car_test.mp4`, 640×360, 29.97 fps, 7925 frames (4:24 total).
**Span processed:** 0:00–4:00. Both runs below were locked onto the target at
frame 300 (0:10) — the first frame where the grey SUV is unambiguous — and ran to
frame 7192 (4:00), 6893 frames.
**Machine:** CPU only, no GPU.

## What was run

| | run A | run B |
| --- | --- | --- |
| follows the target | SiamMask (DAVIS weights, 255 px search window) | SiamMask, same settings |
| finds it again after a loss | nothing | YOLO26 detections + ByteTrack identities + appearance memory |
| command | `siamtrack.py` | `track_draw_box.py --algo siammask` |

Run A is SiamMask on its own, which is what this repository is for. Run B adds
the detector-based recovery from the companion repository, so the difference
measures exactly what recovery is worth on this clip.

## Performance

| | run A — SiamMask alone | run B — SiamMask + recovery |
| --- | --- | --- |
| frames processed | 6893 | 6893 |
| speed while actually following the car | 10.7 fps | 9.3 fps |
| wall-clock time for the whole 3:50 span | 40 s | 12 min 22 s |
| frames with a box on something | 229 (3.3%) | 1752 (25.4%) |
| unbroken tracking stretches | 1 | 25 |
| longest stretch | 229 frames (7.6 s) | 392 frames (13.1 s) |
| times the target was taken back | 0 | 15 |
| of those, on the right car (audited) | - | most, see Accuracy |

Run A finished the span in 40 seconds only because it stops working once the
target is lost - after frame 528 it is reading and re-encoding frames, not
tracking. Timed over the frames where it was following the car, it runs at
10.7 fps. Run B is slower per frame because the detector runs throughout, and it
keeps working for the whole four minutes.

## Accuracy

Coverage is not accuracy. The clip has no ground truth, so 20 frames were
sampled evenly across the frames each run reported as tracked, and each was
checked by eye against the target. `delivery/audit_zoom.jpg` and
`delivery/audit_wide.jpg` are those frames.

**Run A — SiamMask alone.** All 229 tracked frames (frames 300–528, 0:10–0:17)
are on the grey SUV. The box follows the car cleanly through traffic. At frame
528 the helicopter's camera swings and the car leaves the search window; from
there the run reports SEARCHING for the remaining 3 minutes 43 seconds, because
nothing in SiamMask can rediscover a target — it only knows what the car looked
like in the window where it was last seen.

**Run B — with recovery.** Of 20 sampled tracked frames, **14 were on the grey
target and 6 were on another vehicle** — a white van at 1:12, a white car at
1:49, a white truck at 1:52, a small car at 2:22, a car at the frame edge at
2:39, a yellow school bus at 2:47. The pattern follows what the footage does:

| section of the clip | what run B does |
| --- | --- |
| 0:10–0:35, aerial, target in clear view | on the target |
| 0:35–3:05, after the hard camera cut at frame 1044 into wide shots of freeway traffic | mostly on other vehicles, taken back 11 times on look-alikes |
| 3:05–4:00, the vehicle stopped, ground-level and close-up footage | on the grey SUV again, the two longest stretches of the run (13.1 s and 12.1 s) |

On that sample, roughly **70% of the frames run B reports as tracked are on the
right car** - about 18% of the four minutes with a correct box, and about 8% with
a box on a wrong vehicle. Twenty samples is a small audit: read it as "most of
the tracked frames are right, and a meaningful minority are not", not as a
precise figure.

The middle section is the known hard case for this clip: a hard cut to a wide
shot of near-identical cars, where the target is about 20 px across. Measured
previously, neither the appearance memory, a vehicle re-identification network
trained for that task, nor "the car the camera keeps centred" ranks the right car
first more than about half the time. The pipeline's answer for that case is a
person pressing **R** and drawing the box again, which is not part of these runs —
both were fully automatic, with no hand corrections.

## Files

| file | what it is |
| --- | --- |
| `car_siammask_pure.mp4` | run A, the full 0:10–4:00 annotated, 640×360 |
| `car_siammask_pure.csv` | run A per-frame telemetry |
| `car_siammask_recovery.mp4` | run B, same span, annotated |
| `car_siammask_recovery.csv` | run B per-frame telemetry |
| `timeline_pure.svg`, `timeline_recovery.svg` | path of the target and a timeline of the run: green where a box was held, orange where not |
| `audit_zoom.jpg`, `audit_wide.jpg` | the sampled frames the accuracy above was judged from |
| `recovery_sheet.jpg`, `pure_sheet.jpg` | frames sampled evenly across the whole span, tracked or not |
| `car_siammask_4min.mp4` | a third run, from frame 0 on the grey car visible at the start; that car leaves the frame after 130 frames (4.3 s) as the camera pans |

The annotated `.mp4` files are not in the repository: they are the source footage
with the boxes drawn on, and the footage stays local. The telemetry (`.csv`) is
here, and the commands under *Reproducing* regenerate the videos.

Telemetry columns: `frame, time_s, track_id, state, x, y, w, h, cx, cy, speed_px_s`,
in the video's own coordinates. `state` is `locked`, `reacquired` or `uncertain`;
frames with no row are frames with no box.

## Reproducing

```bash
python siamtrack.py car_test.mp4 --start-frame 300 --bbox 291,162,29,28 \
    --max-frames 6893 --no-show --output run_a.mp4 --csv run_a.csv
```

```bash
python track_draw_box.py car_test.mp4 --algo siammask --target car \
    --start-frame 300 --bbox 291,162,29,28 --max-frames 6893 --no-show \
    --yolo-imgsz 448 --exclude 0,0,640,50 --exclude 0,300,95,60 \
    --output run_b.mp4 --csv run_b.csv
```

The second command is from the companion repository,
`Skytrack_yolo26_mobile_clip_bytrack`.

## What this says

For this clip, SiamMask is the right tool for **following** the grey SUV while it
is in view — the box stays wrapped on the car, and it never once jumped to a
neighbouring vehicle while it had the target. It is the wrong tool on its own for
a four-minute chase, because one camera swing ends it permanently. Recovery
turns 7.6 seconds of tracking into roughly 40 seconds of correct tracking spread
over four minutes, at half the speed and with false locks in the crowded middle
section.
