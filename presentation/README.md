# YOLO26 + ByteTrack versus SiamMask — the technical review

A 14-slide deck comparing the two tracking pipelines in this project and its
companion, [Skytrack_yolo26_mobile_clip_bytrack](https://github.com/aymane00o/Skytrack_yolo26_mobile_clip_bytrack).
Every number in it comes from a run of the actual pipeline on the two test clips,
CPU only — none are taken from either algorithm's published benchmarks.

## What is in this folder

| path | what it is |
| --- | --- |
| `deck.json` | the deck's index: slide order, sections, typefaces (IBM Plex Sans, JetBrains Mono) |
| `slides/*.html` | one file per slide, in the Slides artifact format (1920×1080, inline styles) |
| `assets/*.png` | the four still frames the demo slide shows |

The slides are the source of the deck, not a rendering of it: they open in the
Slides artifact viewer rather than in a browser. This page is the same content as
text, so it can be read here.

On the demo slide, the images the slides reference as `/_blob/<id>` are:

| `/_blob/…` | file |
| --- | --- |
| `201220d2…` | `assets/yolo26_car.png` |
| `84d72964…` | `assets/yolo26_drone.png` |
| `510a49de…` | `assets/siammask_car.png` |
| `dba5a9f7…` | `assets/siammask_drone.png` |

## The deck, slide by slide

**1 · Cover.** Two ways to track one target through real footage, measured on a
CPU rather than read off a spec sheet.

**2 · What this covers.** How each pipeline is built · what each is built for ·
what was actually measured · which one to use, and the demo clips.

**3 · One target, one clip, no GPU.** The two test clips:

* *Car* — police-chase footage, 640×360, 7,925 frames (4:24). A helicopter follows
  a grey SUV through freeway traffic, with a hard camera cut at frame 1,044 into
  a wide shot of near-identical cars. The hard case is telling one grey car from
  another at 20 px across.
* *Drone* — a micro-talon VTOL, phone-shot, 1080×1920, 3,403 frames. Parked on a
  mat, taking off, then climbing away until it is a speck against cloud. The hard
  case is surviving takeoff and then shrinking to a few pixels.

**4 · YOLO26 + ByteTrack — detector-driven.** Every frame is re-examined; identity
is inferred, not assumed. YOLO26 finds every car, truck, aircraft or bird in the
frame · ByteTrack (Kalman motion + Hungarian matching) gives each detection an
identity and keeps it through weak frames · target memory recognises the picked
target after a gap and refuses look-alikes that lead by too little · CSRT carries
the box between detection passes. Output: box, track ID, position and speed,
motion trail, prediction arrow.

**5 · SiamMask — template-driven.** One template, set once. The box you draw
becomes a fixed ResNet-50 template · each frame only a 255 px window around where
the target was last seen is searched · the best match wins and a pixel mask comes
with it. Output: mask, real outline, box. No detector, no identity, no memory —
nothing here knows what a car *is*.

**6 · Same six stages, two very different answers.**

| | input | sense | identify | recover | follow | output |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26 + ByteTrack | every frame | detector, whole frame | Kalman + Hungarian | appearance memory | CSRT | box + ID + trail |
| SiamMask | every frame | 255 px window only | none | none | SiamMask itself | mask + box |

SiamMask spends nothing on identify or recover — which is why it is faster while
it works, and why it cannot get the target back once it is gone.

**7 · What each is actually built for.**

| property | YOLO26 + ByteTrack | SiamMask |
| --- | --- | --- |
| knows the object's class | yes — COCO classes (car, truck, airplane, bird…) | no — only what it looked like when picked |
| finds the target after a full loss | yes — detector + identity + memory | no alone; yes, paired with a detector |
| output per frame | box, track ID, speed, trail, prediction arrow | pixel mask — the target's real outline |
| per-frame cost | one CNN detector pass over the whole frame, plus matching | one siamese pass over one small window |
| needs a trained class list | yes — COCO, or fine-tune for others | no — works on any object shown once |
| weights | `yolo26m.pt`, 44 MB, downloaded on first use | ResNet-50 DAVIS, 106 MB, fetched by hand |
| setup | `pip install ultralytics` | clone the SiamMask repo, place weights by hand |

**8 · Measured — drone clip, 300 frames.**

| | speed | frames held | re-locks | starts |
| --- | --- | --- | --- | --- |
| YOLO26 + ByteTrack | 4.7 fps | 94% (281/300) | 1 | frame 60, from the mat, through takeoff |
| SiamMask | 7.9 fps | 100% (300/300) | 0 needed | frame 230, already airborne |

Not a fair race: SiamMask's window starts once the aircraft is already flying,
YOLO26 + ByteTrack's starts on the ground, through the hardest seconds of the
clip, and still holds it 94% of the time.

**9 · Measured — car clip, 300 frames, before the hard cut.**

| | speed | frames held | re-locks |
| --- | --- | --- | --- |
| YOLO26 + ByteTrack | 6.2 fps | 85% (254/300) | 1 — lost the car once, took it back |
| SiamMask | 13.2 fps | 76% (229/300) | 0 — lost the car once, never took it back |

The same moment breaks both: the camera swings around frame 528.

**10 · Coverage is not accuracy — the full 3:50, audited by eye.**

| | held | on the right car |
| --- | --- | --- |
| SiamMask alone | 3.3% — one stretch, 7.6 s | 100% of it |
| SiamMask + recovery, reported | 25.4% — 25 stretches, 15 re-locks | — |
| SiamMask + recovery, audited | — | ~18% of the clip on target, ~8% more on a wrong vehicle |

| section of the clip | what is happening | SiamMask + recovery |
| --- | --- | --- |
| 0:10 – 0:35 | aerial, target in clear view | on the target |
| 0:35 – 3:05 | hard cut into wide, crowded freeway traffic | mostly on look-alikes |
| 3:05 – 4:00 | vehicle stopped, ground-level, close-up | on the target again |

The full report, with the audit frames, is in
[`../delivery/REPORT.md`](../delivery/REPORT.md).

**11 · Which one to reach for.**

*YOLO26 + ByteTrack when* the target can leave the frame and come back · similar
objects are nearby and you need to filter by class · the footage runs minutes, not
seconds · an ID, speed and a trail matter more than an exact outline.

*SiamMask when* the target stays visible for the stretch that matters · you need a
precise outline, not just a box · the target is not a class any detector was
trained on · it is a short, close, single-subject follow shot.

**12 · SiamMask follows, ByteTrack recovers.** The pairing already exists
(`--algo siammask` in the companion repository). A detection lane — YOLO26, then
ByteTrack identities, then target memory — runs alongside a following lane of
SiamMask; when memory confirms a re-lock it hands SiamMask a fresh template.
Correct tracking over the 3:50: 7.6 s alone, about 40 s paired. The cost is
speed: 10.7 → 9.3 fps.

**13 · Short output clips.** Both algorithms on both clips — YOLO26 + ByteTrack
on the car (20 s, frames 300–900) and the drone (20 s, frames 60–660); SiamMask on
the car (10 s, frames 300–600) and the drone (14 s, frames 230–650). Stills only
in the deck; the `.mp4` files are not in this repository.

**14 · Two repositories, ready to run.**
`python skytrack.py clip.mp4 --target car` (YOLO26 + ByteTrack) and
`python siamtrack.py clip.mp4` (SiamMask). A detector recovers a target it never
expected to lose; a siamese tracker follows one it was shown, precisely, until it
looks away. Coverage numbers alone hid that difference — the audited frames did
not.
