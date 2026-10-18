"""Flow data: a synthetic CICFlowMeter-style generator and a CICIDS2017 loader.

The synthetic generator exists so the whole project runs offline with no
download. It is NOT a substitute for real traffic: numbers produced on it show
that the pipeline works and how the methods compare under controlled, overlapping
class structure, not how they will behave on your network. Use
`load_cicids2017()` to run the same pipeline on real CSVs.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

CLASSES = ["BENIGN", "DoS", "PortScan", "BruteForce", "Botnet", "WebAttack"]

FEATURES = [
    "dst_port", "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_len_fwd", "total_len_bwd", "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "fwd_pkt_len_max", "flow_bytes_per_s", "flow_pkts_per_s", "flow_iat_mean",
    "flow_iat_std", "fwd_iat_mean", "syn_flag_count", "ack_flag_count",
    "rst_flag_count", "psh_flag_count", "fin_flag_count", "init_win_fwd",
    "init_win_bwd", "down_up_ratio", "pkt_len_variance", "active_mean",
    "idle_mean",
]

# Human-readable meaning of each feature (used in analyst-facing explanations).
FEATURE_DESC = {
    "dst_port": "destination port",
    "flow_duration": "flow duration (us)",
    "total_fwd_packets": "packets sent by initiator",
    "total_bwd_packets": "packets sent by responder",
    "total_len_fwd": "bytes sent by initiator",
    "total_len_bwd": "bytes sent by responder",
    "fwd_pkt_len_mean": "mean initiator packet size (B)",
    "bwd_pkt_len_mean": "mean responder packet size (B)",
    "fwd_pkt_len_max": "largest initiator packet (B)",
    "flow_bytes_per_s": "byte rate (B/s)",
    "flow_pkts_per_s": "packet rate (pkt/s)",
    "flow_iat_mean": "mean inter-arrival time (us)",
    "flow_iat_std": "inter-arrival time std-dev (us)",
    "fwd_iat_mean": "mean initiator inter-arrival (us)",
    "syn_flag_count": "SYN flags",
    "ack_flag_count": "ACK flags",
    "rst_flag_count": "RST flags",
    "psh_flag_count": "PSH flags",
    "fin_flag_count": "FIN flags",
    "init_win_fwd": "initiator TCP window (B)",
    "init_win_bwd": "responder TCP window (B)",
    "down_up_ratio": "responder/initiator packet ratio",
    "pkt_len_variance": "packet-length variance",
    "active_mean": "mean active time (us)",
    "idle_mean": "mean idle time (us)",
    "capture_session": "capture session id (dataset artifact)",
}

# What a SOC analyst would expect to see for each attack type. This is a small,
# *broad* (6-7 of 25 features) domain-knowledge dictionary used (a) by the
# simulated analyst in the triage experiment and (b) for explanation/expectation
# agreement. It is written from general knowledge of these attacks, not fitted
# to the model; but note the synthetic generator is also written from the same
# general knowledge, so agreement on synthetic data is optimistic.
EXPERT_SIGNATURES = {
    "DoS": {"flow_pkts_per_s", "syn_flag_count", "flow_iat_mean", "idle_mean",
            "flow_duration", "total_fwd_packets", "flow_bytes_per_s"},
    "PortScan": {"total_fwd_packets", "rst_flag_count", "dst_port",
                 "flow_duration", "total_len_fwd", "init_win_fwd",
                 "fwd_pkt_len_mean"},
    "BruteForce": {"dst_port", "psh_flag_count", "total_fwd_packets",
                   "total_bwd_packets", "flow_duration", "fwd_pkt_len_mean"},
    "Botnet": {"flow_iat_std", "idle_mean", "active_mean", "fwd_iat_mean",
               "flow_duration", "bwd_pkt_len_mean"},
    "WebAttack": {"fwd_pkt_len_max", "fwd_pkt_len_mean", "total_len_fwd",
                  "psh_flag_count", "dst_port", "pkt_len_variance"},
}

CLASS_MIX = {"BENIGN": 0.78, "DoS": 0.07, "PortScan": 0.06,
             "BruteForce": 0.04, "Botnet": 0.025, "WebAttack": 0.025}


# --------------------------------------------------------------------------
# synthetic generator
# --------------------------------------------------------------------------
def _ln(rng, median, sigma, n, lo=None, hi=None):
    x = rng.lognormal(np.log(median), sigma, n)
    if lo is not None or hi is not None:
        x = np.clip(x, lo, hi)
    return x


def _ri(rng, a, b, n):
    return rng.integers(a, b + 1, n).astype(float)


def _assemble(rng, n, *, fwd_pkts, bwd_pkts, fwd_len, bwd_len, dur_us, port,
              syn, ack, rst, psh, fin, win_f, win_b, iat_cv, active_frac,
              idle_us):
    """Build the 25 features from primitives so they stay mutually consistent."""
    fwd_pkts = np.maximum(np.round(fwd_pkts), 1.0)
    bwd_pkts = np.maximum(np.round(bwd_pkts), 0.0)
    dur_us = np.maximum(dur_us, 1.0)
    fwd_len = np.maximum(fwd_len, 0.0)
    bwd_len = np.where(bwd_pkts > 0, np.maximum(bwd_len, 0.0), 0.0)
    tot = fwd_pkts + bwd_pkts
    tlf, tlb = fwd_pkts * fwd_len, bwd_pkts * bwd_len
    secs = dur_us / 1e6
    iat_mean = dur_us / np.maximum(tot - 1, 1)
    fmax = np.where(fwd_len > 0,
                    np.minimum(1460, fwd_len * (1 + np.abs(rng.normal(0.4, 0.3, n)))),
                    0.0)
    return pd.DataFrame({
        "dst_port": port.astype(float),
        "flow_duration": dur_us,
        "total_fwd_packets": fwd_pkts,
        "total_bwd_packets": bwd_pkts,
        "total_len_fwd": tlf,
        "total_len_bwd": tlb,
        "fwd_pkt_len_mean": fwd_len,
        "bwd_pkt_len_mean": bwd_len,
        "fwd_pkt_len_max": fmax,
        "flow_bytes_per_s": (tlf + tlb) / secs,
        "flow_pkts_per_s": tot / secs,
        "flow_iat_mean": iat_mean,
        "flow_iat_std": iat_mean * iat_cv,
        "fwd_iat_mean": dur_us / np.maximum(fwd_pkts - 1, 1),
        "syn_flag_count": np.round(syn),
        "ack_flag_count": np.round(ack),
        "rst_flag_count": np.round(rst),
        "psh_flag_count": np.round(psh),
        "fin_flag_count": np.round(fin),
        "init_win_fwd": win_f,
        "init_win_bwd": win_b,
        "down_up_ratio": np.floor(bwd_pkts / fwd_pkts),
        "pkt_len_variance": ((fwd_len - bwd_len) ** 2) * rng.uniform(0.15, 0.35, n),
        "active_mean": dur_us * active_frac,
        "idle_mean": idle_us,
    })[FEATURES]


def _wins(rng, n, choices=(8192, 29200, 64240, 65535)):
    return rng.choice(choices, n).astype(float)


# ---- benign profiles -----------------------------------------------------
def _b_web(rng, n):
    fp = _ri(rng, 4, 30, n); bp = fp * rng.uniform(0.8, 3, n)
    tot = fp + bp
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 90, .4, n, 20, 700),
                     bwd_len=_ln(rng, 600, .6, n, 40, 1460), dur_us=_ln(rng, 3e5, 1.0, n, 1e3, 3e7),
                     port=rng.choice([443, 80, 8443], n, p=[.7, .25, .05]), syn=np.ones(n),
                     ack=tot * rng.uniform(.8, 1, n), rst=rng.binomial(1, .05, n),
                     psh=_ri(rng, 1, 8, n), fin=rng.choice([1, 2], n), win_f=_wins(rng, n),
                     win_b=_wins(rng, n), iat_cv=rng.uniform(1, 3, n),
                     active_frac=rng.uniform(.1, .6, n), idle_us=np.where(rng.random(n) < .3, _ln(rng, 5e5, 1, n), 0))


def _b_dns(rng, n):
    return _assemble(rng, n, fwd_pkts=np.ones(n), bwd_pkts=rng.choice([0, 1], n, p=[.05, .95]),
                     fwd_len=_ln(rng, 45, .2, n, 28, 120), bwd_len=_ln(rng, 110, .4, n, 40, 512),
                     dur_us=_ln(rng, 3e4, .8, n, 200, 3e6), port=np.full(n, 53), syn=np.zeros(n),
                     ack=np.zeros(n), rst=np.zeros(n), psh=np.zeros(n), fin=np.zeros(n),
                     win_f=np.full(n, -1.0), win_b=np.full(n, -1.0), iat_cv=rng.uniform(0, .3, n),
                     active_frac=rng.uniform(.5, 1, n), idle_us=np.zeros(n))


def _b_ssh(rng, n):
    fp = _ln(rng, 80, 1.0, n, 8, 3000); bp = fp * rng.uniform(.7, 1.3, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 60, .4, n, 30, 200),
                     bwd_len=_ln(rng, 110, .6, n, 30, 800), dur_us=_ln(rng, 6e7, 1.2, n, 1e6, 2e9),
                     port=np.full(n, 22), syn=np.ones(n), ack=fp + bp, rst=np.zeros(n),
                     psh=fp * rng.uniform(.4, .9, n), fin=rng.choice([1, 2], n), win_f=_wins(rng, n),
                     win_b=_wins(rng, n), iat_cv=rng.uniform(2, 6, n), active_frac=rng.uniform(.01, .2, n),
                     idle_us=_ln(rng, 8e6, 1, n))


def _b_bulk(rng, n):
    fp = _ln(rng, 600, .9, n, 60, 8000); bp = fp * rng.uniform(.05, .6, n)
    if rng.random() < .5:
        fp, bp = bp + 1, fp
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 1100, .3, n, 200, 1460),
                     bwd_len=_ln(rng, 60, .4, n, 0, 300), dur_us=_ln(rng, 4e6, .9, n, 2e5, 3e8),
                     port=rng.choice([443, 445, 21, 873], n, p=[.5, .25, .15, .1]), syn=np.ones(n),
                     ack=fp + bp, rst=np.zeros(n), psh=fp * rng.uniform(.01, .1, n), fin=np.ones(n),
                     win_f=_wins(rng, n), win_b=_wins(rng, n), iat_cv=rng.uniform(.5, 2, n),
                     active_frac=rng.uniform(.5, .95, n), idle_us=np.zeros(n))


def _b_api_poll(rng, n):  # hard: periodic beacons, but benign (botnet-like)
    fp = _ri(rng, 3, 12, n); bp = _ri(rng, 3, 10, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 110, .4, n, 40, 400),
                     bwd_len=_ln(rng, 350, .5, n, 60, 1400), dur_us=_ln(rng, 8e6, 1.0, n, 1e5, 2e8),
                     port=rng.choice([443, 8080, 80], n, p=[.7, .2, .1]), syn=np.ones(n),
                     ack=fp + bp, rst=np.zeros(n), psh=_ri(rng, 2, 6, n), fin=np.ones(n),
                     win_f=_wins(rng, n), win_b=_wins(rng, n), iat_cv=rng.uniform(1.5, 4, n),
                     active_frac=rng.uniform(.02, .2, n), idle_us=_ln(rng, 5e6, 1, n))


def _b_probe(rng, n):  # hard: health-check/monitoring (scan-like)
    return _assemble(rng, n, fwd_pkts=rng.choice([1, 2], n), bwd_pkts=rng.choice([0, 1], n),
                     fwd_len=np.zeros(n), bwd_len=np.zeros(n), dur_us=_ln(rng, 2e4, 1.2, n, 5, 2e6),
                     port=rng.choice([22, 80, 443, 3306, 8080], n), syn=np.ones(n),
                     ack=np.zeros(n), rst=rng.binomial(1, .6, n), psh=np.zeros(n), fin=np.zeros(n),
                     win_f=rng.choice([1024, 29200, 64240], n).astype(float),
                     win_b=np.zeros(n), iat_cv=np.zeros(n), active_frac=np.ones(n), idle_us=np.zeros(n))


def _b_failed_login(rng, n):  # hard: a few mistyped logins (brute-force-like)
    fp = _ri(rng, 6, 18, n); bp = _ri(rng, 6, 18, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 70, .3, n, 30, 150),
                     bwd_len=_ln(rng, 90, .4, n, 30, 300), dur_us=_ln(rng, 3e6, .8, n, 1e5, 6e7),
                     port=rng.choice([22, 21, 3389], n, p=[.6, .2, .2]), syn=np.ones(n),
                     ack=fp + bp, rst=rng.binomial(1, .1, n), psh=_ri(rng, 3, 10, n), fin=rng.choice([1, 2], n),
                     win_f=_wins(rng, n), win_b=_wins(rng, n), iat_cv=rng.uniform(1, 3, n),
                     active_frac=rng.uniform(.1, .5, n), idle_us=_ln(rng, 1e6, 1, n))


def _b_upload(rng, n):  # hard: large legitimate POST/upload (web-attack-like)
    fp = _ri(rng, 8, 60, n); bp = _ri(rng, 5, 30, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 600, .4, n, 150, 1400),
                     bwd_len=_ln(rng, 300, .6, n, 40, 1400), dur_us=_ln(rng, 1e6, .9, n, 5e4, 3e7),
                     port=rng.choice([443, 80, 8080], n, p=[.7, .25, .05]), syn=np.ones(n),
                     ack=fp + bp, rst=np.zeros(n), psh=_ri(rng, 3, 20, n), fin=np.ones(n),
                     win_f=_wins(rng, n), win_b=_wins(rng, n), iat_cv=rng.uniform(1, 3, n),
                     active_frac=rng.uniform(.3, .8, n), idle_us=np.zeros(n))


def _b_idle(rng, n):  # hard: long idle keep-alive sessions (slow-DoS-like)
    fp = _ri(rng, 5, 20, n); bp = _ri(rng, 3, 15, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 60, .5, n, 0, 300),
                     bwd_len=_ln(rng, 200, .6, n, 0, 800), dur_us=_ln(rng, 4e7, .8, n, 5e6, 6e8),
                     port=rng.choice([443, 80], n), syn=np.ones(n), ack=fp + bp, rst=np.zeros(n),
                     psh=_ri(rng, 0, 4, n), fin=np.ones(n), win_f=_wins(rng, n), win_b=_wins(rng, n),
                     iat_cv=rng.uniform(2, 5, n), active_frac=rng.uniform(.005, .05, n),
                     idle_us=_ln(rng, 1.5e7, .7, n))


def _b_other(rng, n):  # NTP / SMTP / misc
    fp = _ri(rng, 1, 15, n); bp = _ri(rng, 1, 15, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 120, .8, n, 20, 1400),
                     bwd_len=_ln(rng, 150, .8, n, 20, 1400), dur_us=_ln(rng, 2e5, 1.3, n, 100, 5e7),
                     port=rng.choice([123, 25, 587, 993, 5353, 3478], n), syn=rng.binomial(1, .6, n),
                     ack=(fp + bp) * rng.uniform(.5, 1, n), rst=rng.binomial(1, .05, n),
                     psh=_ri(rng, 0, 5, n), fin=rng.binomial(1, .6, n), win_f=_wins(rng, n),
                     win_b=_wins(rng, n), iat_cv=rng.uniform(.5, 3, n), active_frac=rng.uniform(.1, .8, n),
                     idle_us=np.zeros(n))


_BENIGN = [(_b_web, .38), (_b_dns, .15), (_b_ssh, .05), (_b_bulk, .06), (_b_api_poll, .08),
           (_b_probe, .04), (_b_failed_login, .03), (_b_upload, .05), (_b_idle, .05),
           (_b_other, .11)]


# ---- attack profiles -----------------------------------------------------
def _dos_flood(rng, n):
    fp = _ri(rng, 2, 12, n); bp = _ri(rng, 0, 3, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 20, 1.0, n, 0, 200),
                     bwd_len=_ln(rng, 40, .8, n, 0, 300), dur_us=_ln(rng, 2e4, 1.1, n, 50, 2e6),
                     port=rng.choice([80, 443, 8080], n, p=[.6, .3, .1]), syn=_ri(rng, 1, 4, n),
                     ack=fp * rng.uniform(0, .6, n), rst=rng.binomial(2, .3, n), psh=_ri(rng, 0, 2, n),
                     fin=rng.binomial(1, .15, n), win_f=_wins(rng, n, (512, 1024, 8192, 29200)),
                     win_b=np.where(rng.random(n) < .5, 0, _wins(rng, n)), iat_cv=rng.uniform(0, 1.5, n),
                     active_frac=rng.uniform(.5, 1, n), idle_us=np.zeros(n))


def _dos_syn(rng, n):
    fp = _ri(rng, 1, 3, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=rng.choice([0, 1], n, p=[.8, .2]),
                     fwd_len=np.zeros(n), bwd_len=np.zeros(n), dur_us=_ln(rng, 3e3, 1.3, n, 1, 5e5),
                     port=rng.choice([80, 443, 53, 8080], n, p=[.45, .35, .1, .1]), syn=fp,
                     ack=np.zeros(n), rst=rng.binomial(1, .25, n), psh=np.zeros(n), fin=np.zeros(n),
                     win_f=_wins(rng, n, (512, 1024, 8192)), win_b=np.zeros(n), iat_cv=np.zeros(n),
                     active_frac=np.ones(n), idle_us=np.zeros(n))


def _dos_slow(rng, n):
    fp = _ri(rng, 5, 25, n); bp = _ri(rng, 2, 12, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 25, .6, n, 0, 150),
                     bwd_len=_ln(rng, 150, .7, n, 0, 800), dur_us=_ln(rng, 5e7, .8, n, 5e6, 6e8),
                     port=rng.choice([80, 443], n, p=[.7, .3]), syn=np.ones(n), ack=fp + bp,
                     rst=rng.binomial(1, .1, n), psh=_ri(rng, 0, 3, n), fin=rng.binomial(1, .5, n),
                     win_f=_wins(rng, n), win_b=_wins(rng, n), iat_cv=rng.uniform(.5, 2, n),
                     active_frac=rng.uniform(.002, .03, n), idle_us=_ln(rng, 1.2e7, .7, n))


def _portscan(rng, n, slow=False):
    return _assemble(rng, n, fwd_pkts=rng.choice([1, 2], n, p=[.85, .15]),
                     bwd_pkts=rng.choice([0, 1], n, p=[.5, .5]), fwd_len=_ln(rng, 2, 1.0, n, 0, 12) * rng.binomial(1, .3, n),
                     bwd_len=np.zeros(n), dur_us=_ln(rng, 4e5 if slow else 1.5e3, 1.2, n, 1, 5e6),
                     port=np.where(rng.random(n) < .6, rng.integers(1, 65536, n),
                                   rng.choice([22, 23, 80, 443, 445, 3389, 8080, 21, 25, 3306], n)),
                     syn=np.ones(n), ack=np.zeros(n), rst=rng.binomial(1, .6, n), psh=np.zeros(n),
                     fin=np.zeros(n), win_f=rng.choice([1024, 2048, 4096, 5840, 16384], n).astype(float),
                     win_b=np.zeros(n), iat_cv=np.zeros(n), active_frac=np.ones(n), idle_us=np.zeros(n))


def _brute(rng, n, port_choices, p, fp_rng, dur_med, stealth=False):
    fp = _ri(rng, *fp_rng, n); bp = fp * rng.uniform(.8, 1.2, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 60, .25, n, 30, 150),
                     bwd_len=_ln(rng, 80, .3, n, 30, 200), dur_us=_ln(rng, dur_med, .6, n, 1e5, 6e8),
                     port=rng.choice(port_choices, n, p=p), syn=np.ones(n), ack=fp + bp,
                     rst=rng.binomial(1, .1, n), psh=fp * rng.uniform(.4, .8, n), fin=rng.choice([1, 2], n),
                     win_f=_wins(rng, n), win_b=_wins(rng, n), iat_cv=rng.uniform(.3, 1.5, n),
                     active_frac=rng.uniform(.4, .9, n) if not stealth else rng.uniform(.02, .1, n),
                     idle_us=np.zeros(n) if not stealth else _ln(rng, 3e6, .8, n))


def _bot(rng, n):
    fp = _ri(rng, 3, 12, n); bp = _ri(rng, 3, 10, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 100, .35, n, 40, 400),
                     bwd_len=_ln(rng, 300, .5, n, 60, 1400), dur_us=_ln(rng, 2.5e7, .9, n, 1e6, 3e8),
                     port=rng.choice([443, 8080, 6667, 4444, 80], n, p=[.35, .25, .15, .15, .1]),
                     syn=np.ones(n), ack=fp + bp, rst=rng.binomial(1, .05, n), psh=_ri(rng, 2, 6, n),
                     fin=np.ones(n), win_f=_wins(rng, n), win_b=_wins(rng, n), iat_cv=rng.uniform(2, 6, n),
                     active_frac=rng.uniform(.005, .05, n), idle_us=_ln(rng, 1.5e7, .7, n))


def _web_attack(rng, n):
    fp = _ri(rng, 6, 40, n); bp = _ri(rng, 4, 30, n)
    return _assemble(rng, n, fwd_pkts=fp, bwd_pkts=bp, fwd_len=_ln(rng, 480, .5, n, 100, 1400),
                     bwd_len=_ln(rng, 450, .6, n, 40, 1460), dur_us=_ln(rng, 8e5, .9, n, 5e4, 2e7),
                     port=rng.choice([80, 443, 8080], n, p=[.7, .2, .1]), syn=np.ones(n), ack=fp + bp,
                     rst=rng.binomial(1, .05, n), psh=_ri(rng, 4, 25, n), fin=np.ones(n),
                     win_f=_wins(rng, n), win_b=_wins(rng, n), iat_cv=rng.uniform(1, 3, n),
                     active_frac=rng.uniform(.3, .8, n), idle_us=np.zeros(n))


def _mk(parts):
    return lambda rng, n: _mix(rng, n, parts)


def _mix(rng, n, parts):
    fns, ws = zip(*parts)
    ws = np.array(ws, float) / np.sum(ws)
    which = rng.choice(len(fns), n, p=ws)
    out = []
    for i, fn in enumerate(fns):
        m = int((which == i).sum())
        if m:
            out.append(fn(rng, m))
    df = pd.concat(out, ignore_index=True)
    return df.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


_ATTACK = {
    "DoS": [(_dos_flood, .5), (_dos_syn, .2), (_dos_slow, .3)],
    "PortScan": [(lambda r, n: _portscan(r, n, False), .7), (lambda r, n: _portscan(r, n, True), .3)],
    "BruteForce": [(lambda r, n: _brute(r, n, [22], [1], (10, 30), 4e6), .5),
                   (lambda r, n: _brute(r, n, [21], [1], (6, 15), 2e6), .4),
                   (lambda r, n: _brute(r, n, [22, 21, 3389], [.5, .25, .25], (5, 14), 4e7, True), .1)],
    "Botnet": [(_bot, 1.0)],
    "WebAttack": [(_web_attack, 1.0)],
}


def _rand_ip(rng, n, private):
    if private:
        return [f"10.{a}.{b}.{c}" for a, b, c in zip(rng.integers(0, 4, n), rng.integers(0, 256, n), rng.integers(1, 255, n))]
    return [f"{a}.{b}.{c}.{d}" for a, b, c, d in zip(rng.integers(11, 223, n), rng.integers(0, 256, n), rng.integers(0, 256, n), rng.integers(1, 255, n))]


def generate_flows(n: int = 40000, seed: int = 0, label_noise: float = 0.003):
    """Return (X, y, meta). `label_noise` flips that fraction of labels uniformly
    at random to a different class, imitating imperfect ground truth."""
    rng = np.random.default_rng(seed)
    counts = {c: int(round(n * w)) for c, w in CLASS_MIX.items()}
    counts["BENIGN"] += n - sum(counts.values())
    frames, labels = [], []
    for c in CLASSES:
        parts = _BENIGN if c == "BENIGN" else _ATTACK[c]
        frames.append(_mix(rng, counts[c], parts))
        labels += [c] * counts[c]
    X = pd.concat(frames, ignore_index=True)
    y = np.array(labels, dtype=object)
    perm = rng.permutation(len(X))
    X, y = X.iloc[perm].reset_index(drop=True), y[perm]
    flip = rng.random(len(y)) < label_noise
    for i in np.where(flip)[0]:
        y[i] = rng.choice([c for c in CLASSES if c != y[i]])
    k = len(X)
    attack = y != "BENIGN"
    meta = pd.DataFrame({
        "src_ip": [a if att else b for a, b, att in zip(_rand_ip(rng, k, False), _rand_ip(rng, k, True), attack)],
        "dst_ip": _rand_ip(rng, k, True),
        "flow_id": np.arange(k),
    })
    return X, pd.Series(y, name="label"), meta


# --------------------------------------------------------------------------
# dataset-artifact injection (for the shortcut-learning audit)
# --------------------------------------------------------------------------
def add_capture_artifact(y, seed: int, leak: float):
    """A fake `capture_session` column. leak=0.97: sessions 3-5 hold (97%) attack
    flows and 0-2 hold (97%) benign flows, i.e. attacks were captured in a
    different session from normal traffic, a classic dataset artifact.
    leak=0: session id is independent of the label."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    if leak <= 0:
        return pd.Series(rng.integers(0, 6, n).astype(float), name="capture_session")
    attack = y != "BENIGN"
    use_label = rng.random(n) < leak
    attack_session = np.where(attack, rng.integers(3, 6, n), rng.integers(0, 3, n))
    random_session = rng.integers(0, 6, n)
    return pd.Series(np.where(use_label, attack_session, random_session).astype(float), name="capture_session")


# --------------------------------------------------------------------------
# real data: CICIDS2017 (MachineLearningCVE / TrafficLabelling CSVs)
# --------------------------------------------------------------------------
_CIC_COLS = {
    "Destination Port": "dst_port", "Flow Duration": "flow_duration",
    "Total Fwd Packets": "total_fwd_packets", "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_len_fwd", "Total Length of Bwd Packets": "total_len_bwd",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean", "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Fwd Packet Length Max": "fwd_pkt_len_max", "Flow Bytes/s": "flow_bytes_per_s",
    "Flow Packets/s": "flow_pkts_per_s", "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std", "Fwd IAT Mean": "fwd_iat_mean",
    "SYN Flag Count": "syn_flag_count", "ACK Flag Count": "ack_flag_count",
    "RST Flag Count": "rst_flag_count", "PSH Flag Count": "psh_flag_count",
    "FIN Flag Count": "fin_flag_count", "Init_Win_bytes_forward": "init_win_fwd",
    "Init_Win_bytes_backward": "init_win_bwd", "Down/Up Ratio": "down_up_ratio",
    "Packet Length Variance": "pkt_len_variance", "Active Mean": "active_mean",
    "Idle Mean": "idle_mean",
}


def _map_cic_label(s: str):
    s = s.strip().lower()
    if s == "benign":
        return "BENIGN"
    if "dos" in s or "ddos" in s:
        return "DoS"
    if "portscan" in s:
        return "PortScan"
    if "patator" in s:
        return "BruteForce"
    if s == "bot":
        return "Botnet"
    if "web attack" in s or "web attack" in s.replace("�", "-"):
        return "WebAttack"
    return None  # Heartbleed, Infiltration, SQL-injection stub etc. are dropped


def load_cicids2017(path: str, max_rows: int | None = None, seed: int = 0):
    """Load CICIDS2017 CSV file(s) (a file, or a directory of *.csv) into the
    same (X, y, meta) shape as generate_flows(). Column names are matched after
    stripping whitespace. Labels are collapsed into this project's 6 classes;
    other attack types are dropped. inf/NaN are cleaned; duplicate rows removed.

    NOTE: written against the documented CICIDS2017 column names; it has only
    been exercised on a tiny fabricated CSV here, not on the full dataset.
    """
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    if not files:
        raise FileNotFoundError(f"no CSV files found at {path}")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in list(_CIC_COLS) + ["Label"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected CICIDS2017 columns: {missing}")
    y = df["Label"].map(_map_cic_label)
    X = df[list(_CIC_COLS)].rename(columns=_CIC_COLS)[FEATURES].apply(pd.to_numeric, errors="coerce")
    keep = y.notna() & np.isfinite(X).all(axis=1)
    X, y = X[keep], y[keep]
    dup = X.assign(_y=y).duplicated()
    X, y = X[~dup].reset_index(drop=True), y[~dup].reset_index(drop=True)
    if max_rows and len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    meta = pd.DataFrame({"src_ip": "n/a", "dst_ip": "n/a", "flow_id": np.arange(len(X))})
    return X, y.rename("label"), meta
