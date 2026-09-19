"""SiamMask as a drop-in replacement for the CSRT correlation tracker.

SiamMask (Wang et al., CVPR 2019) is a Siamese tracker: it takes a reference
crop of the target when it is initialised and, on every later frame, searches
a larger region around the last position for the place that best matches it.
Unlike a correlation filter it also predicts a pixel mask of the target, and the
box is read from that mask - so it follows the object's real outline as it
turns and changes shape, where CSRT keeps roughly the box it was given.

The model and weights are the authors' (github.com/foolwood/SiamMask, MIT
licence), cloned into third_party/SiamMask with SiamMask_DAVIS.pth beside this
file. What is here is the tracking step from their tools/test.py, trimmed to what
tracking needs - their file also imports a compiled benchmark-scoring module - and
updated for current NumPy and PyTorch.

One model is loaded per process and shared by every tracker. SiamMask keeps the
target's template inside the network itself, so each tracker holds its own copy
and puts it back before it searches; without that, two trackers sharing the model
would each end up following the other's target.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "third_party", "SiamMask")
WEIGHTS = os.path.join(HERE, "SiamMask_DAVIS.pth")

# The authors' DAVIS configuration (experiments/siammask_sharp/config_davis.json).
HP = {"instance_size": 255, "base_size": 8, "out_size": 127, "seg_thr": 0.35,
      "penalty_k": 0.04, "window_influence": 0.4, "lr": 1.0}
ANCHORS = {"stride": 8, "ratios": [0.33, 0.5, 1, 2, 3], "scales": [8], "round_dight": 0}

_model = None


def available():
    """Can SiamMask run here? (code cloned, weights downloaded, torch installed)"""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return os.path.isdir(REPO) and os.path.isfile(WEIGHTS)


def model():
    """The shared network, loaded on first use."""
    global _model
    if _model is not None:
        return _model
    if not available():
        raise SystemExit(
            "SiamMask needs its code and weights:\n"
            "  git clone https://github.com/foolwood/SiamMask third_party/SiamMask\n"
            "  download http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth next to track.py")
    import torch
    for path in (REPO, os.path.join(REPO, "experiments", "siammask_sharp")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from custom import Custom

    net = Custom(anchors=ANCHORS)
    try:
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    except Exception:                                   # noqa: BLE001 - older pickle format
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, _ = net.load_state_dict(state, strict=False)
    if [k for k in missing if not k.endswith("num_batches_tracked")]:
        raise SystemExit(f"SiamMask weights did not load: {len(missing)} missing parameters")
    net.eval()
    torch.set_grad_enabled(False)
    # Measured on a 16-core CPU tracking the chase clip: every core 7.7 -> 9.8 fps,
    # channels-last memory layout 9.5 -> 10.9, with the box on the car on exactly
    # the same 225 of 240 frames either way.
    torch.set_num_threads(os.cpu_count() or 1)
    net.to(memory_format=torch.channels_last)
    _model = net
    return net


def _anchors(score_size):
    from utils.anchors import Anchors
    anchors = Anchors(ANCHORS)
    a = anchors.anchors
    x1, y1, x2, y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    a = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
    stride, count = anchors.stride, a.shape[0]
    a = np.tile(a, score_size * score_size).reshape((-1, 4))
    ori = -(score_size // 2) * stride
    xx, yy = np.meshgrid([ori + stride * d for d in range(score_size)],
                         [ori + stride * d for d in range(score_size)])
    a[:, 0] = np.tile(xx.flatten(), (count, 1)).flatten().astype(np.float32)
    a[:, 1] = np.tile(yy.flatten(), (count, 1)).flatten().astype(np.float32)
    return a


def _subwindow(im, pos, model_sz, original_sz, avg):
    """A square crop centred on `pos`, padded with the mean colour, as a tensor."""
    import torch
    sz = original_sz
    c = (sz + 1) / 2
    x0, y0 = round(pos[0] - c), round(pos[1] - c)
    x1, y1 = x0 + sz - 1, y0 + sz - 1
    left, top = int(max(0., -x0)), int(max(0., -y0))
    right, bottom = int(max(0., x1 - im.shape[1] + 1)), int(max(0., y1 - im.shape[0] + 1))
    x0, x1, y0, y1 = x0 + left, x1 + left, y0 + top, y1 + top
    if any((top, bottom, left, right)):
        r, cc, k = im.shape
        padded = np.empty((r + top + bottom, cc + left + right, k), np.uint8)
        padded[:] = avg
        padded[top:top + r, left:left + cc] = im
        patch = padded[int(y0):int(y1 + 1), int(x0):int(x1 + 1)]
    else:
        patch = im[int(y0):int(y1 + 1), int(x0):int(x1 + 1)]
    if model_sz != original_sz:
        patch = cv2.resize(patch, (model_sz, model_sz))
    return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).float().unsqueeze(0)


class SiamMaskTracker:
    """Same interface as cv2.TrackerCSRT: init(frame, box), update(frame) -> (ok, box).

    After each update, `mask` is the target's pixel mask over the whole frame (or
    None), `polygon` its rotated outline, and `score` the network's confidence.
    """

    def __init__(self, min_score=0.2, search=255):
        """`search` is the side of the region searched each frame, in the network's
        input pixels. The published 255 costs 9.5 fps here; 191 runs at 13.5 fps and
        kept the box on the car on the same frames of the chase clip, but covers less
        ground per frame, so a target or a camera that moves fast can escape it."""
        self.net = model()                     # also puts SiamMask's code on the import path
        from utils.tracker_config import TrackerConfig
        p = TrackerConfig()
        p.update(dict(HP, instance_size=search), self.net.anchors)
        p.renew()
        p.anchor = _anchors(p.score_size)
        p.anchor_num = self.net.anchor_num
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        self.window = np.tile(window.flatten(), p.anchor_num)
        self.p = p
        self.min_score = min_score
        self.zf = None
        self.mask = self.polygon = None
        self.score = 0.0

    def init(self, frame, box):
        x, y, w, h = (float(v) for v in box)
        p = self.p
        self.size = (frame.shape[1], frame.shape[0])
        self.avg = np.mean(frame, axis=(0, 1))
        self.pos = np.array([x + w / 2, y + h / 2])
        self.sz = np.array([w, h])
        wc = w + p.context_amount * (w + h)
        hc = h + p.context_amount * (w + h)
        crop = _subwindow(frame, self.pos, p.exemplar_size, round(np.sqrt(wc * hc)), self.avg)
        self.net.template(crop)
        self.zf = self.net.zf
        return True

    def update(self, frame):
        import torch.nn.functional as F
        p, net = self.p, self.net
        net.zf = self.zf                       # this tracker's target, not the last one's
        wc = self.sz[1] + p.context_amount * self.sz.sum()
        hc = self.sz[0] + p.context_amount * self.sz.sum()
        s_x = np.sqrt(wc * hc)
        scale = p.exemplar_size / s_x
        s_x = s_x + 2 * ((p.instance_size - p.exemplar_size) / 2) / scale
        crop_box = [self.pos[0] - round(s_x) / 2, self.pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

        search = _subwindow(frame, self.pos, p.instance_size, round(s_x), self.avg)
        score, delta, _ = net.track_mask(search)
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).numpy()
        score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1)[:, 1].numpy()

        a = p.anchor
        delta[0] = delta[0] * a[:, 2] + a[:, 0]
        delta[1] = delta[1] * a[:, 3] + a[:, 1]
        delta[2] = np.exp(delta[2]) * a[:, 2]
        delta[3] = np.exp(delta[3]) * a[:, 3]

        def change(r):
            return np.maximum(r, 1. / r)

        def pad_size(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        sz_crop = self.sz * scale
        s_c = change(pad_size(delta[2], delta[3]) / pad_size(*sz_crop))
        r_c = change((sz_crop[0] / sz_crop[1]) / (delta[2] / delta[3]))
        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * score * (1 - p.window_influence) + self.window * p.window_influence
        best = int(np.argmax(pscore))

        pred = delta[:, best] / scale
        lr = penalty[best] * score[best] * p.lr
        self.pos = self.pos + pred[:2]
        self.sz = self.sz * (1 - lr) + pred[2:] * lr
        self.score = float(score[best])

        # The mask, read back into frame coordinates, and the box taken from it.
        _, dy, dx = np.unravel_index(best, (5, p.score_size, p.score_size))
        mask = net.track_refine((dy, dx)).sigmoid().squeeze().view(p.out_size, p.out_size).numpy()
        s = crop_box[2] / p.instance_size
        sub = [crop_box[0] + (dx - p.base_size / 2) * p.total_stride * s,
               crop_box[1] + (dy - p.base_size / 2) * p.total_stride * s,
               s * p.exemplar_size, s * p.exemplar_size]
        k = p.out_size / sub[2]
        back = [-sub[0] * k, -sub[1] * k]
        fw, fh = self.size
        ax, ay = (fw - 1) / (fw * k), (fh - 1) / (fh * k)
        mapping = np.array([[ax, 0, -ax * back[0]], [0, ay, -ay * back[1]]], dtype=np.float64)
        full = cv2.warpAffine(mask, mapping, (fw, fh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        binary = (full > p.seg_thr).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        self.mask, self.polygon = None, None
        if contours and max(areas) > 20:
            outline = contours[int(np.argmax(areas))].reshape(-1, 2)
            self.mask = binary
            self.polygon = cv2.boxPoints(cv2.minAreaRect(outline))
            bx, by, bw, bh = cv2.boundingRect(outline)
        else:
            bx, by = self.pos[0] - self.sz[0] / 2, self.pos[1] - self.sz[1] / 2
            bw, bh = self.sz

        self.pos[0] = max(0, min(fw, self.pos[0]))
        self.pos[1] = max(0, min(fh, self.pos[1]))
        self.sz[0] = max(10, min(fw, self.sz[0]))
        self.sz[1] = max(10, min(fh, self.sz[1]))
        ok = self.score >= self.min_score
        return ok, (float(bx), float(by), float(bw), float(bh))
