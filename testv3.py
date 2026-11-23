#!/usr/bin/env python3
"""
txrx.py — Chunked acoustic FSK test rig.

Frame:  SYNC(2) | SEQ(1) | LEN(1) | PAYLOAD (LEN bytes) | CRC32(4)
Each frame is preceded by an up-chirp and followed by a down-chirp,
so the receiver can locate every frame independently and reassemble
by sequence number.

Install:  pip install numpy sounddevice matplotlib
"""

import argparse
import binascii
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
except ImportError:
    sd = None


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
SR_DEFAULT      = 96000
F0_DEFAULT      = 21000.0
F1_DEFAULT      = 21500.0
BIT_DUR_DEFAULT = 0.010
AMP_DEFAULT     = 0.5
MAX_PAYLOAD     = 255

CHIRP_F0  = 18000.0
CHIRP_F1  = 22000.0
CHIRP_DUR = 0.100
GAP_DUR   = 0.020
TAIL_DUR  = 0.250

SYNC_WORD = bytes([0xB5, 0x5A])


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def apply_fade(w, sr, fade_ms=2.0):
    n = max(1, int(sr * fade_ms / 1000))
    if 2 * n >= len(w):
        return w
    w = w.copy()
    f = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n)))
    w[:n]  *= f
    w[-n:] *= f[::-1]
    return w


def chirp(sr, f0, f1, dur, amp=1.0, fade_ms=2.0):
    n = int(round(sr * dur))
    t = np.arange(n) / sr
    k = (f1 - f0) / dur
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    return apply_fade(amp * np.sin(phase), sr, fade_ms)


def bytes_to_bits(b):
    return [(byte >> i) & 1 for byte in b for i in range(7, -1, -1)]


def bits_to_bytes(bits):
    n = len(bits) // 8 * 8
    out = bytearray()
    for i in range(0, n, 8):
        v = 0
        for j in range(8):
            v = (v << 1) | bits[i + j]
        out.append(v)
    return bytes(out)


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #
def build_frame(payload: bytes, seq: int) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too long ({len(payload)} > {MAX_PAYLOAD})")
    if not 0 <= seq <= 255:
        raise ValueError("seq must be 0..255")
    header = SYNC_WORD + bytes([seq, len(payload)])
    crc = binascii.crc32(header + payload) & 0xFFFFFFFF
    return header + payload + crc.to_bytes(4, "big")


def chunk_payload(data: bytes, max_payload: int = MAX_PAYLOAD):
    return [data[i:i + max_payload] for i in range(0, len(data), max_payload)] or [b""]


# --------------------------------------------------------------------------- #
# modulation
# --------------------------------------------------------------------------- #
def fsk_modulate(bits, sr, f0, f1, bit_dur, amp):
    spb = int(round(sr * bit_dur))
    freqs = np.concatenate([np.full(spb, f1 if b else f0) for b in bits])
    phase = 2 * np.pi * np.cumsum(freqs) / sr
    return amp * np.sin(phase)


def build_waveform_chunked(text, sr, f0, f1, bit_dur, amp, max_payload=MAX_PAYLOAD):
    """Returns (waveform, chunks, per_frame_bit_counts)."""
    payload = text.encode("utf-8")
    chunks  = chunk_payload(payload, max_payload)

    up   = chirp(sr, CHIRP_F0, CHIRP_F1, CHIRP_DUR, amp)
    down = chirp(sr, CHIRP_F1, CHIRP_F0, CHIRP_DUR, amp)
    gap  = np.zeros(int(sr * GAP_DUR))

    waves, bit_counts = [], []
    for seq, ch in enumerate(chunks):
        frame = build_frame(ch, seq)
        bits  = bytes_to_bits(frame)
        data  = apply_fade(fsk_modulate(bits, sr, f0, f1, bit_dur, amp), sr, 1.0)
        waves.append(np.concatenate([up, gap, data, gap, down]))
        bit_counts.append(len(bits))

    return np.concatenate(waves), chunks, bit_counts


# --------------------------------------------------------------------------- #
# demodulation
# --------------------------------------------------------------------------- #
def find_chirp(rec, ref):
    corr       = np.correlate(rec, ref, mode="valid")
    win_energy = np.convolve(rec * rec, np.ones(len(ref)), mode="valid")
    ref_energy = float(np.sum(ref * ref))
    denom = np.sqrt(np.maximum(win_energy, 1e-20) * ref_energy)
    norm  = np.abs(corr) / denom
    return norm


def find_all_chirps(rec, ref, sr, min_gap_s=0.100, rel_thresh=0.35):
    """Scan non-overlapping windows of min_gap_s; return peaks above threshold."""
    norm = find_chirp(rec, ref)
    if not len(norm):
        return [], norm
    thresh  = rel_thresh * float(norm.max())
    min_gap = max(1, int(sr * min_gap_s))
    peaks, i = [], 0
    while i < len(norm):
        j = min(i + min_gap, len(norm))
        local = norm[i:j]
        if local.size and float(local.max()) >= thresh:
            peaks.append(i + int(np.argmax(local)))
        i = j
    return peaks, norm


def demod_fsk(rec, sr, f0, f1, bit_dur, start_sample, n_bits):
    spb = int(round(sr * bit_dur))
    n   = np.arange(spb)
    win = np.hanning(spb)
    lo  = np.exp(-2j * np.pi * f0 * n / sr)
    hi  = np.exp(-2j * np.pi * f1 * n / sr)

    bits, confs = [], []
    for i in range(n_bits):
        a = start_sample + i * spb
        b = a + spb
        if b > len(rec):
            break
        seg = rec[a:b] * win
        e0 = abs(np.sum(seg * lo))
        e1 = abs(np.sum(seg * hi))
        bits.append(1 if e1 > e0 else 0)
        confs.append((e1 - e0) / (e1 + e0 + 1e-12))
    return bits, np.asarray(confs)


def decode_one_frame(rec, sr, f0, f1, bit_dur, chirp_start, up_len):
    """Decode exactly one frame located at chirp_start. Returns dict."""
    start = chirp_start + up_len + int(sr * GAP_DUR)

    header_bits, _ = demod_fsk(rec, sr, f0, f1, bit_dur, start, 32)
    if len(header_bits) < 32:
        return {"ok": False, "reason": "short header"}

    header_bytes = bits_to_bytes(header_bits)
    if header_bytes[:2] != SYNC_WORD:
        return {"ok": False, "reason": f"sync mismatch ({header_bytes[:2].hex()})"}

    seq    = header_bytes[2]
    length = header_bytes[3]
    total_bits = 32 + length * 8
    all_bits, confs = demod_fsk(rec, sr, f0, f1, bit_dur, start, total_bits)

    data = bits_to_bytes(all_bits)
    need = 5 + length + 4
    if len(data) < need:
        return {"ok": False, "reason": "short frame", "seq": seq}

    frame_body = data[:5 + length]
    crc_rx     = int.from_bytes(data[5 + length:9 + length], "big")
    crc_calc   = binascii.crc32(frame_body) & 0xFFFFFFFF

    return {
        "ok":         crc_rx == crc_calc,
        "seq":        seq,
        "payload":    data[5:5 + length],
        "length":     length,
        "chirp_peak": chirp_start,
        "mean_conf":  float(confs.mean()) if len(confs) else 0.0,
        "confs":      confs,
        "crc_rx":     crc_rx,
        "crc_calc":   crc_calc,
    }


def decode_all(rec, sr, f0, f1, bit_dur, verbose=True):
    """Find every chirp and decode every frame; reassemble by seq."""
    up_ref = chirp(sr, CHIRP_F0, CHIRP_F1, CHIRP_DUR, 1.0)
    peaks, norm = find_all_chirps(rec, up_ref, sr)
    if verbose:
        print(f"  found {len(peaks)} chirp(s): {peaks}")

    frames = []
    for p in peaks:
        r = decode_one_frame(rec, sr, f0, f1, bit_dur, p, len(up_ref))
        r["chirp_peak"] = p
        frames.append(r)
        if verbose:
            status = "ok" if r.get("ok") else f"FAIL ({r.get('reason','crc')})"
            print(f"    frame @{p:>8}: seq={r.get('seq','?'):>3} "
                  f"len={r.get('length','?'):>3} conf={r.get('mean_conf',0):+.2f} "
                  f"-> {status}")

    good = {}
    missing = []
    for r in frames:
        if r.get("ok") and "seq" in r:
            good[r["seq"]] = r["payload"]

    if not good:
        return {"ok": False, "reason": "no frames decoded",
                "frames": frames, "peaks": peaks}

    n = max(good)
    for i in range(n + 1):
        if i not in good:
            missing.append(i)

    if missing:
        return {"ok": False, "reason": f"missing seq {missing}",
                "frames": frames, "peaks": peaks, "good": good}

    return {
        "ok":      True,
        "payload": b"".join(good[i] for i in range(n + 1)),
        "frames":  frames,
        "peaks":   peaks,
        "good":    good,
        "n_frames": n + 1,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_result(r):
    if r.get("ok"):
        txt = r["payload"].decode("utf-8", errors="replace")
        head = txt[:60].replace("\n", "⏎")
        print(f"  ✓ reassembled {len(r['payload'])} bytes from "
              f"{r['n_frames']} frame(s): {head!r}{'…' if len(txt)>60 else ''}")
    else:
        print(f"  ✗ {r.get('reason','decode failed')}")
    for f in r.get("frames", []):
        if f.get("ok"):
            print(f"    seq={f['seq']:>3}  len={f['length']:>3}  "
                  f"conf={f['mean_conf']:+.2f}  "
                  f"crc={f['crc_rx']:08x}✓")


def save_report(rec, sr, result, path, f0, f1):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    ax = axes[0]
    ax.specgram(rec, NFFT=1024, Fs=sr, noverlap=896, cmap="viridis")
    ax.set_ylim(15000, min(sr / 2, 25000))
    ax.axhline(f0, color="r",      ls="--", lw=0.7, alpha=0.7, label=f"f0={f0:.0f}")
    ax.axhline(f1, color="orange", ls="--", lw=0.7, alpha=0.7, label=f"f1={f1:.0f}")
    for p in result.get("peaks", []):
        ax.axvline(p / sr, color="cyan", lw=0.5, alpha=0.5)
    ax.set_ylabel("Hz")
    ax.set_title("Spectrogram (cyan lines = chirp peaks)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    if result.get("frames"):
        for f in result["frames"]:
            if "confs" in f and len(f["confs"]):
                ax.plot(f["confs"], lw=0.7,
                        label=f"seq={f.get('seq','?')} "
                              f"{'✓' if f.get('ok') else '✗'}")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Bit confidence per frame")
    ax.set_xlabel("bit index within frame")
    ax.set_ylabel("confidence")

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  report: {path}")


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def _warn_sr(sr, f1):
    if sr < 2.2 * f1:
        print(f"[warn] sample rate {sr} Hz is low for a {f1:.0f} Hz carrier. "
              f"Use >= {int(2.5 * f1)} Hz.", file=sys.stderr)


def _estimate_time(n_bytes, bit_dur, chunk_bytes=MAX_PAYLOAD):
    n_chunks = max(1, -(-n_bytes // chunk_bytes))
    data_bits = sum((8 + min(chunk_bytes, n_bytes - i * chunk_bytes)) * 8
                    for i in range(n_chunks))
    overhead = n_chunks * (2 * CHIRP_DUR + 2 * GAP_DUR)
    return data_bits * bit_dur + overhead, n_chunks


def cmd_loopback(args):
    text = args.text
    wave, chunks, bit_counts = build_waveform_chunked(
        text, args.sr, args.f0, args.f1, args.bit_dur, args.amp)

    t, n_chunks = _estimate_time(len(text.encode("utf-8")), args.bit_dur)
    print(f"Loopback: {len(text.encode('utf-8'))} bytes -> {n_chunks} frame(s), "
          f"{len(wave)/args.sr:.2f}s waveform")

    result = decode_all(wave, args.sr, args.f0, args.f1, args.bit_dur)
    print_result(result)

    if args.report:
        save_report(wave, args.sr, result, args.report, args.f0, args.f1)
    return 0 if result.get("ok") else 2


def cmd_send(args):
    if sd is None:
        print("error: sounddevice not installed", file=sys.stderr)
        return 1
    _warn_sr(args.sr, args.f1)

    text = args.text
    wave, chunks, bit_counts = build_waveform_chunked(
        text, args.sr, args.f0, args.f1, args.bit_dur, args.amp)

    payload_bytes = len(text.encode("utf-8"))
    dur = len(wave) / args.sr
    print(f"Transmitting {payload_bytes} B -> {len(chunks)} frame(s), "
          f"{dur:.2f}s @ {args.f0:.0f}/{args.f1:.0f} Hz, sr={args.sr}")
    for i, (c, b) in enumerate(zip(chunks, bit_counts)):
        print(f"  frame {i+1}/{len(chunks)}: {len(c)} B, {b} bits")

    sd.play(wave.astype(np.float32), args.sr, device=args.device)
    sd.wait()
    return 0


def _record_and_decode(args, rec):
    result = decode_all(rec, args.sr, args.f0, args.f1, args.bit_dur)
    print_result(result)
    if args.report:
        save_report(rec, args.sr, result, args.report, args.f0, args.f1)
    return 0 if result.get("ok") else 2


def cmd_recv(args):
    if sd is None:
        print("error: sounddevice not installed", file=sys.stderr)
        return 1
    _warn_sr(args.sr, args.f1)
    n = int(args.sr * args.duration)
    print(f"Recording {args.duration:.2f}s …")
    rec = sd.rec(n, samplerate=args.sr, channels=1,
                 dtype="float32", device=args.device)
    sd.wait()
    return _record_and_decode(args, rec.ravel())


def cmd_sendrecv(args):
    if sd is None:
        print("error: sounddevice not installed", file=sys.stderr)
        return 1
    _warn_sr(args.sr, args.f1)

    text = args.text
    wave, chunks, bit_counts = build_waveform_chunked(
        text, args.sr, args.f0, args.f1, args.bit_dur, args.amp)
    wave = wave.astype(np.float32)
    dur  = len(wave) / args.sr

    print(f"Send+recv: {len(text.encode('utf-8'))} B -> {len(chunks)} frame(s), "
          f"{dur:.2f}s @ {args.f0:.0f}/{args.f1:.0f} Hz, sr={args.sr}")
    for i, (c, b) in enumerate(zip(chunks, bit_counts)):
        print(f"  frame {i+1}/{len(chunks)}: {len(c)} B, {b} bits, "
              f"{b*args.bit_dur*1000:.0f} ms on-air")

    in_dev  = args.in_device  if args.in_device  is not None else args.device
    out_dev = args.out_device if args.out_device is not None else args.device

    chunks_rec = []
    def cb(indata, frames, t, status):
        chunks_rec.append(indata.copy())

    stream = sd.InputStream(samplerate=args.sr, channels=1,
                            dtype="float32", device=in_dev, callback=cb)
    with stream:
        sd.play(wave, args.sr, device=out_dev)
        sd.wait()
        time.sleep(args.tail)

    if not chunks_rec:
        print("error: no audio captured", file=sys.stderr)
        return 1
    rec = np.concatenate(chunks_rec).ravel()
    return _record_and_decode(args, rec)


def cmd_sendfile(args):
    p = Path(args.path)
    if not p.exists():
        print(f"error: {p} not found", file=sys.stderr)
        return 1
    text = p.read_text(encoding="utf-8", errors="replace")
    args.text = text
    return cmd_send(args)


def cmd_sweep(args):
    if sd is None:
        print("error: sounddevice not installed", file=sys.stderr)
        return 1
    _warn_sr(args.sr, args.sweep_f1)

    sweep = apply_fade(chirp(args.sr, args.sweep_f0, args.sweep_f1,
                             args.duration, args.amp, fade_ms=10.0),
                       args.sr, 10.0)

    print(f"Sweeping {args.sweep_f0/1000:.1f} → {args.sweep_f1/1000:.1f} kHz "
          f"over {args.duration:.2f}s")

    chunks_rec = []
    def cb(indata, frames, t, status):
        chunks_rec.append(indata.copy())

    in_dev  = args.in_device  if args.in_device  is not None else args.device
    out_dev = args.out_device if args.out_device is not None else args.device

    stream = sd.InputStream(samplerate=args.sr, channels=1,
                            dtype="float32", device=in_dev, callback=cb)
    with stream:
        sd.play(sweep, args.sr, device=out_dev)
        sd.wait()
        time.sleep(args.tail)

    if not chunks_rec:
        print("error: no audio captured", file=sys.stderr)
        return 1
    rec = np.concatenate(chunks_rec).ravel()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.specgram(rec, NFFT=2048, Fs=args.sr, noverlap=1536, cmap="magma")
    ax.set_ylim(args.sweep_f0 - 1000, min(args.sr / 2, args.sweep_f1 + 1000))
    ax.set_title(f"Sweep response  {args.sweep_f0/1000:.1f}–"
                 f"{args.sweep_f1/1000:.1f} kHz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")
    plt.tight_layout()
    plt.savefig(args.report, dpi=140)
    plt.close(fig)
    print(f"  report: {args.report}")
    return 0


def cmd_list(_args):
    if sd is None:
        print("error: sounddevice not installed", file=sys.stderr)
        return 1
    print(sd.query_devices())
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_common(p):
    p.add_argument("--sr",       type=int,   default=SR_DEFAULT)
    p.add_argument("--f0",       type=float, default=F0_DEFAULT)
    p.add_argument("--f1",       type=float, default=F1_DEFAULT)
    p.add_argument("--bit-dur",  type=float, default=BIT_DUR_DEFAULT)
    p.add_argument("--amp",      type=float, default=AMP_DEFAULT)
    p.add_argument("--device",   type=int, default=None)
    p.add_argument("--in-device",  type=int, default=None)
    p.add_argument("--out-device", type=int, default=None)
    p.add_argument("--tail",     type=float, default=TAIL_DUR)
    p.add_argument("--report",   type=Path, default=None)


def main():
    ap = argparse.ArgumentParser(prog="txrx",
                                 description="Chunked acoustic FSK test rig")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("loopback", help="DSP-only test, no hardware")
    p.add_argument("text", nargs="?", default="hello 21.5 kHz")
    add_common(p); p.set_defaults(func=cmd_loopback)

    p = sub.add_parser("send", help="transmit a string")
    p.add_argument("text", nargs="?", default="hello 21.5 kHz")
    add_common(p); p.set_defaults(func=cmd_send)

    p = sub.add_parser("sendfile", help="transmit the contents of a text file")
    p.add_argument("path")
    add_common(p); p.set_defaults(func=cmd_sendfile)

    p = sub.add_parser("recv", help="record and try to decode")
    p.add_argument("--duration", type=float, default=4.0)
    add_common(p); p.set_defaults(func=cmd_recv)

    p = sub.add_parser("sendrecv", help="send + record simultaneously")
    p.add_argument("text", nargs="?", default="hello 21.5 kHz")
    add_common(p); p.set_defaults(func=cmd_sendrecv)

    p = sub.add_parser("sweep", help="frequency sweep + report")
    p.add_argument("--sweep-f0", type=float, default=15000.0)
    p.add_argument("--sweep-f1", type=float, default=23000.0)
    p.add_argument("--duration", type=float, default=3.0)
    add_common(p)
    p.set_defaults(report=Path("sweep.png"))
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("list", help="list audio devices")
    p.set_defaults(func=cmd_list)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)