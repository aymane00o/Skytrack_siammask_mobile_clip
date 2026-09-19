"""Track one target through a phone clip on a CPU: SiamMask, and nothing else.

You pick the target once - scrub to a frame, drag a box around it - and SiamMask
follows it frame to frame, segmenting it and drawing its real outline rather than
the box you drew.

    python siamtrack.py "micro talon.mov"
    python siamtrack.py car_test.mp4 --output tracked.mp4 --csv telemetry.csv

Press R during a run to draw the box again; Q or Esc quits.

WHAT RUNS
    SiamMask   a ResNet-50 siamese network: it keeps a template of the target from
               the moment you picked it, searches a window around where it was
               last seen, and returns the best match plus a mask of the object

    That is the whole pipeline. There is no detector telling it what a car or an
    aircraft is, no identities, no memory of the target's appearance - which is
    the point of this repository: what SiamMask on its own is worth on real phone
    footage, measured. The detector-based pipeline is a separate repository,
    Skytrack_yolo26_mobile_clip_bytrack, and the README compares the two.

    A tracker with no detector cannot rediscover a target it has lost. When the
    target goes out of view this run says so and stops following, unless you pass
    --reacquire, which turns on the contrast-based search for a target against sky.

MOBILE CLIPS
    Phone footage is tall and large - 1080x1920 at 30 fps. Frames are worked on at
    1080 px tall (--work-height), and the window SiamMask searches each frame is
    255 px square (--search); 191 is about a third faster. Output and telemetry are
    written in the video's own coordinates.

SIAMMASK'S CODE AND WEIGHTS ARE NOT IN THIS REPOSITORY
    git clone https://github.com/foolwood/SiamMask third_party/SiamMask
    download SiamMask_DAVIS.pth from that project's releases, next to this file
"""

import argparse
import sys

import cv2

import siammask
import track

WINDOW = "Pick the frame  -  drag slider, then press ENTER"


def choose_frame(video, start_frame, window_height):
    """Scrub to a frame, and return it with its index."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    def read_at(index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        return frame if ok else None

    index = min(max(start_frame, 0), total - 1)
    frame = read_at(index)
    if frame is None:
        raise SystemExit("Video has no frames")

    track.show(WINDOW, frame, window_height)
    cv2.createTrackbar("frame", WINDOW, index, max(total - 1, 1), lambda _: None)
    print(f"{total} frames. Drag the slider (or A/D, W/S) to a frame where the target is "
          f"clearly visible, then press ENTER. Q quits.")

    shown = -1
    while True:
        wanted = cv2.getTrackbarPos("frame", WINDOW)
        if wanted != shown:
            candidate = read_at(wanted)
            if candidate is not None:
                frame, index, shown = candidate, wanted, wanted
        preview = frame.copy()
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(preview, f"frame {index} / {total - 1}   ENTER = use this frame",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        track.show(WINDOW, preview, window_height)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32):
            break
        if key in (ord("q"), 27):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")
        step = {ord("a"): -1, ord("d"): 1, ord("s"): -30, ord("w"): 30}.get(key)
        if step:
            cv2.setTrackbarPos("frame", WINDOW, min(max(index + step, 0), total - 1))

    cap.release()
    cv2.destroyWindow(WINDOW)
    return index, frame


def draw_box(frame, window_height):
    """Drag a box around the target. Returns x, y, w, h."""
    print("Drag a box around the target, then press ENTER. C cancels.")
    picker = "Draw a box around the target"
    # Size the window before selectROI, or it opens at the video's own resolution
    # and portrait phone footage runs off the screen.
    track.show(picker, frame, window_height)
    box = cv2.selectROI(picker, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if box[2] == 0 or box[3] == 0:
        raise SystemExit("No target selected.")
    return tuple(int(v) for v in box)


def build_args(video, start, box, chosen):
    """The full argument set track.run_lock expects, with this pipeline's choices."""
    args = track.build_parser().parse_args([video])
    args.start_frame, args.lock_bbox = start, ",".join(str(v) for v in box)
    args.output, args.csv = chosen.output, chosen.csv
    args.show = not chosen.no_show
    args.max_frames = chosen.max_frames
    args.no_pace = chosen.no_pace
    args.window_height = chosen.window_height
    args.exclude = chosen.exclude
    args.repick = chosen.repick

    # The pipeline itself: SiamMask follows, and nothing detects.
    args.tracker = chosen.follower
    args.siammask_search = chosen.search
    args.bytetrack = False
    args.no_reid = True
    args.mode = "sky"                      # only used when --reacquire is on
    args.no_reacquire = not chosen.reacquire
    args.proc_height = chosen.work_height
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="input video file")
    p.add_argument("--output", help="write the annotated video here")
    p.add_argument("--csv", help="write per-frame telemetry here")
    p.add_argument("--start-frame", type=int, default=0, help="open the frame picker here")
    p.add_argument("--bbox", help="skip the picker and lock onto x,y,w,h - to repeat a run")
    p.add_argument("--no-show", action="store_true",
                   help="do not display anything; still writes --output and --csv")
    p.add_argument("--no-pace", action="store_true",
                   help="play as fast as it computes instead of at the video's own speed")
    p.add_argument("--max-frames", type=int, default=0, help="stop after this many frames")
    p.add_argument("--window-height", type=int, default=900,
                   help="tallest the window may be, in px; the saved video is unaffected")
    p.add_argument("--work-height", type=int, default=1080,
                   help="work on frames this tall; smaller is faster and loses small targets")
    p.add_argument("--follower", choices=["siammask", "csrt"], default="siammask",
                   help="what follows the target: siammask, or OpenCV's CSRT correlation "
                        "tracker for comparison - see the table in the README")
    p.add_argument("--search", type=int, default=255,
                   help="side of the window SiamMask searches each frame; 191 is about a "
                        "third faster, but a fast target or camera can escape it")
    p.add_argument("--reacquire", action="store_true",
                   help="when the target is lost, look for it again with the contrast search "
                        "for a target against sky - SiamMask itself cannot rediscover anything")
    p.add_argument("--repick", action="append", default=[], metavar="FRAME:X,Y,W,H",
                   help="pick the target again at this frame, as pressing R does. Repeatable")
    p.add_argument("--exclude", action="append", default=[], metavar="X,Y,W,H",
                   help="a region the search may never pick from, for a burned-in HUD or logo. "
                        "Repeatable")
    chosen = p.parse_args()

    if chosen.follower == "siammask" and not siammask.available():
        raise SystemExit(
            "SiamMask's code and weights are missing:\n"
            "  git clone https://github.com/foolwood/SiamMask third_party/SiamMask\n"
            "  put SiamMask_DAVIS.pth next to siamtrack.py")

    if chosen.bbox:
        parts = chosen.bbox.split(",")
        if len(parts) != 4:
            raise SystemExit("--bbox expects x,y,w,h")
        box, start = tuple(int(v) for v in parts), chosen.start_frame
    else:
        start, frame = choose_frame(chosen.video, chosen.start_frame, chosen.window_height)
        box = draw_box(frame, chosen.window_height)
        print(f"Repeat this exact run with:  --start-frame {start} "
              f"--bbox {box[0]},{box[1]},{box[2]},{box[3]}")

    print(f"{chosen.follower}, frames worked on at {chosen.work_height}px, search window "
          f"{chosen.search}px, reacquire {'on' if chosen.reacquire else 'off'}")
    track.run_lock(build_args(chosen.video, start, box, chosen))


if __name__ == "__main__":
    sys.exit(main())
