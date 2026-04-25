"""
Full Experiment: VirtuosoNet-Augmented Score Following
=====================================================
Evaluates whether VirtuosoNet-generated expressive MIDI as reference
improves online audio score following vs baseline MusicXML synthesis.

Design: 25 performer-piece pairs × 4 brain conditions = 100 runs.
Primary metric: AR@500ms (Wilcoxon signed-rank test, VN vs XML).
"""

import json
import os
import sys
import tempfile
import time as tmod
import warnings

warnings.filterwarnings("ignore")

import mido
import numpy as np
import pandas as pd
import partitura
from scipy import stats

# ---------------------------------------------------------------------------
# Matchmaker imports
# ---------------------------------------------------------------------------
from matchmaker import Matchmaker
from matchmaker.utils.eval import (
    transfer_from_score_to_predicted_perf,
    TOLERANCES_IN_MILLISECONDS,
)
from matchmaker.utils import misc as matchmaker_misc
from matchmaker.utils.misc import generate_score_audio, get_current_note_bpm
from matchmaker.matchmaker import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "datasets", "asap-dataset")
ASAP_JSON = os.path.join(DATASET_ROOT, "asap_annotations.json")
VN_BASE = os.path.join(os.path.expanduser("~"), "virtuosoNet", "test_result")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "experiment_results")

TOLERANCES = [50, 100, 500, 1000, 2000]

PIECES = [
    {
        "id": "chopin_ballade1",
        "name": "Chopin Ballade No.1",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Ballades", "1"),
        "vn_midi": os.path.join(VN_BASE, "chopin_ballade1_by_isgn_z0.mid"),
        "performers": [
            "BuiJL04M", "JIA06M", "KimSuyeon04M", "Mo07M",
            "Richardson13M", "Sladek04M", "Zhou06M",
        ],
    },
    {
        "id": "chopin_ballade_2",
        "name": "Chopin Ballade No.2",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Ballades", "2"),
        "vn_midi": "",
        "performers": [
            "Denisova17M", "Gasanov08M", "Tetzloff06M",
        ],
    },
    {
        "id": "chopin_ballade_3",
        "name": "Chopin Ballade No.3",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Ballades", "3"),
        "vn_midi": "",
        "performers": [
            "Ko11M", "LinPeng06M",
        ],
    },
    {
        "id": "chopin_etude_10_12",
        "name": "Chopin Etude Op.10/12",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Etudes_op_10", "12"),
        "vn_midi": os.path.join(VN_BASE, "chopin_etude_10_12_by_isgn_z0.mid"),
        "performers": [
            "Bult-ItoS04M", "Floril02", "Gintov02", "Hirata02", "HuNY03M",
            "KimBenjamin04", "LuoJ08M", "Mordvinov04", "Na_2006_02",
            "Staupe02", "WuuE06M", "Yarden04", "ZhangYunling02M",
        ],
    },
    {
        "id": "beethoven_sonata_5_1",
        "name": "Beethoven Sonata 5-1",
        "asap_dir": os.path.join(DATASET_ROOT, "Beethoven", "Piano_Sonatas", "5-1"),
        "vn_midi": os.path.join(VN_BASE, "bps_5_1_by_isgn_z0.mid"),
        "performers": ["Colafelice02M", "SunD02M", "Wong01"],
    },
    {
        "id": "chopin_barcarolle",
        "name": "Chopin Barcarolle",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Barcarolle"),
        "vn_midi": os.path.join(VN_BASE, "chopin_barcarolle_by_isgn_z0.mid"),
        "performers": ["Na03"],
    },
    {
        "id": "bach_bwv858",
        "name": "Bach Prelude BWV 858",
        "asap_dir": os.path.join(DATASET_ROOT, "Bach", "Prelude", "bwv_858"),
        "vn_midi": os.path.join(VN_BASE, "bwv_858_prelude_by_isgn_z0.mid"),
        "performers": ["VuV01M"],
    },
    # ---- Expanded dataset (MAESTRO-linked) ----
    {
        "id": "bps_21_1",
        "name": "Beethoven Sonata 21-1 (Waldstein)",
        "asap_dir": os.path.join(DATASET_ROOT, "Beethoven", "Piano_Sonatas", "21-1_no_repeat"),
        "vn_midi": os.path.join(VN_BASE, "bps_21_1_by_isgn_z0.mid"),
        "performers": ["GonzalezJ06M", "KimSY02M", "LeeN02M", "RichardsonC02M", "Shychko02M", "Sladek02M", "Song03M", "SunY02M", "Tetzloff05M", "ToA02M", "TongB03M", "WangA02M", "YeZ02M", "ZhangW01M"],
    },
    {
        "id": "schubert_impromptu_3",
        "name": "Schubert Impromptu Op.90/3",
        "asap_dir": os.path.join(DATASET_ROOT, "Schubert", "Impromptu_op.90_D.899", "3"),
        "vn_midi": os.path.join(VN_BASE, "schubert_impromptu_3_by_isgn_z0.mid"),
        "performers": ["Hou06M", "JeonH06M", "Ko08M", "Kociuban10M", "LEE_K04M", "LeeSH08M", "Mizumoto07M", "Woo10M", "WuuE10M", "ZhangW07M", "ZhaoK10M"],
    },
    {
        "id": "liszt_campanella",
        "name": "Liszt La Campanella",
        "asap_dir": os.path.join(DATASET_ROOT, "Liszt", "Gran_Etudes_de_Paganini", "2_La_campanella"),
        "vn_midi": os.path.join(VN_BASE, "liszt_campanella_by_isgn_z0.mid"),
        "performers": ["BuiJL03M", "LeungR04M", "Levitsky06M", "Lin07M", "LiuC05M", "LiuY03M", "MunA08M", "VuV06M", "Yang05M", "ZhangH07M"],
    },
    {
        "id": "chopin_etude_25_11",
        "name": "Chopin Etude Op.25/11",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Etudes_op_25", "11"),
        "vn_midi": os.path.join(VN_BASE, "chopin_etude_25_11_by_isgn_z0.mid"),
        "performers": ["KimSuyeon03M", "KyykhynenT10M", "Lisiecki07M", "MiyashitaM03M", "Richardson_2013_08M", "Richardson_2015_03M", "Sham02M", "Shychko03M", "Song02M", "WangA02M"],
    },
    {
        "id": "chopin_scherzo_31",
        "name": "Chopin Scherzo Op.31",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Scherzos", "31"),
        "vn_midi": os.path.join(VN_BASE, "chopin_scherzo_31_by_isgn_z0.mid"),
        "performers": ["ChenW04M", "HuNY08M", "Jussow11M", "LeeN04M", "LinTami06M", "LuoJ12M", "SunD04M", "TongB04M", "TuanS05M", "ZhangW04M"],
    },
    {
        "id": "bach_prelude_848",
        "name": "Bach Prelude BWV 848",
        "asap_dir": os.path.join(DATASET_ROOT, "Bach", "Prelude", "bwv_848"),
        "vn_midi": os.path.join(VN_BASE, "bach_prelude_848_by_isgn_z0.mid"),
        "performers": ["Denisova06M", "Lee01M", "LeeSH01M", "Lin04M", "Lou01M", "MiyashitaM01M", "Mizumoto03M", "SunY01M", "Zhou01M"],
    },
    {
        "id": "bach_fugue_848",
        "name": "Bach Fugue BWV 848",
        "asap_dir": os.path.join(DATASET_ROOT, "Bach", "Fugue", "bwv_848"),
        "vn_midi": os.path.join(VN_BASE, "bach_fugue_848_by_isgn_z0.mid"),
        "performers": ["Denisova06M", "Lee01M", "LeeSH01M", "Lin04M", "Lou01M", "MiyashitaM01M", "Mizumoto03M", "SunY01M", "Zhou01M"],
    },
    {
        "id": "chopin_etude_10_4",
        "name": "Chopin Etude Op.10/4",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Etudes_op_10", "4"),
        "vn_midi": os.path.join(VN_BASE, "chopin_etude_10_4_by_isgn_z0.mid"),
        "performers": ["Duepree06M", "Knoll03M", "Kurz03M", "LeeSH03M", "Richardson_2013_03M", "Sladek03M", "Yi03M", "ZhangE03M", "ZhaoA03M"],
    },
    {
        "id": "bps_31_1",
        "name": "Beethoven Sonata 31-1",
        "asap_dir": os.path.join(DATASET_ROOT, "Beethoven", "Piano_Sonatas", "31-1"),
        "vn_midi": os.path.join(VN_BASE, "bps_31_1_by_isgn_z0.mid"),
        "performers": ["ADIG05M", "AbdelmoulaJS03M", "Ahfat02M", "BrendleA03M", "HuangSW08M", "LeungM03M", "Na06M", "Zuber04M"],
    },
    {
        "id": "chopin_etude_10_1",
        "name": "Chopin Etude Op.10/1",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Etudes_op_10", "1"),
        "vn_midi": os.path.join(VN_BASE, "chopin_etude_10_1_by_isgn_z0.mid"),
        "performers": ["ChenW03M", "KaiRuiR03M", "LiYZ02M", "LuM02M", "MunA07M", "SunY03M", "YeF03M", "YuP02M"],
    },
    {
        "id": "haydn_50_1",
        "name": "Haydn Sonata 50-1",
        "asap_dir": os.path.join(DATASET_ROOT, "Haydn", "Keyboard_Sonatas", "50-1_no_repeat"),
        "vn_midi": os.path.join(VN_BASE, "haydn_50_1_by_isgn_z0.mid"),
        "performers": ["GalantM02M", "HamSY02M", "Huang05M", "JeonH03M", "KARYAG06M", "Shi07M", "YoungS02M"],
    },
    {
        "id": "chopin_ballade_4",
        "name": "Chopin Ballade No.4",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Ballades", "4"),
        "vn_midi": os.path.join(VN_BASE, "chopin_ballade_4_by_isgn_z0.mid"),
        "performers": ["ChenC04M", "Kociuban07M", "Levitsky18M", "LuEric04M", "MiyashitaM04M", "Mizumoto02M", "Nikiforov13M"],
    },
    {
        "id": "chopin_etude_10_8",
        "name": "Chopin Etude Op.10/8",
        "asap_dir": os.path.join(DATASET_ROOT, "Chopin", "Etudes_op_10", "8"),
        "vn_midi": os.path.join(VN_BASE, "chopin_etude_10_8_by_isgn_z0.mid"),
        "performers": ["KyykhynenT09M", "Lou03M", "LuEric03M", "MiyashitaM05M", "Rizikov03M", "Tetzloff07M", "Zhou03M"],
    },
    {
        "id": "bps_23_1",
        "name": "Beethoven Sonata 23-1 (Appassionata)",
        "asap_dir": os.path.join(DATASET_ROOT, "Beethoven", "Piano_Sonatas", "23-1"),
        "vn_midi": os.path.join(VN_BASE, "bps_23_1_by_isgn_z0.mid"),
        "performers": ["Duepree05M", "Guo03M", "LiuC02M", "Mizumoto06M", "Teo02M", "Yi02M"],
    },
    {
        "id": "bps_7_1",
        "name": "Beethoven Sonata 7-1",
        "asap_dir": os.path.join(DATASET_ROOT, "Beethoven", "Piano_Sonatas", "7-1"),
        "vn_midi": os.path.join(VN_BASE, "bps_7_1_by_isgn_z0.mid"),
        "performers": ["Hebert01M", "Larionova04M", "LeeS02M", "Wang02M", "WangY01M", "ZhangE02M"],
    },
]

BRAINS = ["audio_self", "midi_aligned", "xml", "virtuoso"]


# ---------------------------------------------------------------------------
# Realistic tempo: no access to performance WAV length or GT annotations.
# In a real-time scenario, only the score file itself is available.
# ---------------------------------------------------------------------------
def _realistic_adjust_tempo(score, performance_audio, default_tempo=120):
    """Return default_tempo without reading the performance audio."""
    return default_tempo


matchmaker_misc.adjust_tempo_for_performance_audio = _realistic_adjust_tempo
import matchmaker.matchmaker as _mm_mod

_mm_mod.adjust_tempo_for_performance_audio = _realistic_adjust_tempo


def _get_xml_tempo(xml_path):
    """Extract the median tempo marking from a MusicXML file, converted
    to the beat unit used by partitura's onset_beat.

    MusicXML <sound tempo="X"/> is always in quarter-note BPM.
    But partitura's onset_beat uses the time-signature's beat_type:
      4/4  → onset_beat in quarter notes (no conversion)
      12/8 → onset_beat in eighth notes  (× 2)
      12/16 → onset_beat in 16th notes   (× 4)
    So we convert: bpm_beat_unit = bpm_quarter × (beat_type / 4).
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract tempo markings
    tempos = []
    for xpath in [
        './/{http://www.musicxml.org/ns/musicxml/3.1}sound[@tempo]',
        './/sound[@tempo]',
    ]:
        tempos = [float(n.attrib['tempo']) for n in tree.findall(xpath)]
        if tempos:
            break
    quarter_bpm = float(np.median(tempos)) if tempos else 120.0

    # Detect beat_type from time signature
    beat_type = 4  # default
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'
    bt_el = root.find(f'.//{ns}beat-type')
    if bt_el is not None and bt_el.text:
        beat_type = int(bt_el.text)

    return quarter_bpm * (beat_type / 4)



def _get_midi_embedded_bpm(midi_path):
    """Extract the embedded BPM from a MIDI file's first set_tempo event."""
    return 60e6 / next(
        (m.tempo for t in mido.MidiFile(midi_path).tracks
         for m in t if m.type == "set_tempo"),
        500000,
    )


# ---------------------------------------------------------------------------
# Gap Mask: MIDI-derived silence/rest detection for DTW robustness
# ---------------------------------------------------------------------------
def compute_gap_mask(midi_path, n_ref_frames, frame_rate, trim_seconds=0.0,
                     gap_threshold=0.5):
    """Compute boolean mask marking reference frames that fall in note gaps.

    Uses MIDI note on/off events to determine which frames have no
    sounding notes nearby.  During these "gap" frames the OLTW
    normalization bias causes overshoot; the mask lets the DTW force
    1:1 advancement instead.

    Args:
        midi_path: Path to reference MIDI (score MIDI for all brains).
        n_ref_frames: Number of reference feature frames.
        frame_rate: Feature frame rate (e.g. 30 fps).
        trim_seconds: Seconds trimmed from the start of reference audio.
        gap_threshold: A frame is marked as gap if no note onset/offset
            falls within ±gap_threshold seconds of its centre time.

    Returns:
        np.ndarray[bool] of shape (n_ref_frames,).  True = gap frame.
    """
    mid = mido.MidiFile(midi_path)
    default_tempo = next(
        (m.tempo for t in mid.tracks for m in t if m.type == "set_tempo"),
        500000,
    )

    # Collect all note sounding intervals [onset, offset]
    intervals = []
    for track in mid.tracks:
        abs_time = 0.0
        current_tempo = default_tempo
        active: dict = {}
        for msg in track:
            abs_time += mido.tick2second(msg.time, mid.ticks_per_beat, current_tempo)
            if msg.type == "set_tempo":
                current_tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_time
            elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    intervals.append((active.pop(msg.note), abs_time))
        # Close any still-open notes at end of track
        for note, onset in active.items():
            intervals.append((onset, abs_time))

    # Build gap mask
    is_gap = np.ones(n_ref_frames, dtype=bool)
    for onset, offset in intervals:
        # Adjust for silence trimming
        onset_adj = onset - trim_seconds
        offset_adj = offset - trim_seconds
        frame_start = max(0, int((onset_adj - gap_threshold) * frame_rate))
        frame_end = min(n_ref_frames, int((offset_adj + gap_threshold) * frame_rate) + 1)
        if frame_start < n_ref_frames and frame_end > 0:
            is_gap[frame_start:frame_end] = False

    return is_gap


# ---------------------------------------------------------------------------
# GT Loading
# ---------------------------------------------------------------------------
def _midi_times_to_beats(midi_path, times):
    """Convert playback times (seconds) in midi_score.mid to quarter-beat positions."""
    mid = mido.MidiFile(midi_path)
    ppq = mid.ticks_per_beat
    time_samples, tick_samples = [0.0], [0]
    abs_time, abs_tick = 0.0, 0
    current_tempo = 500000
    for msg in mid.tracks[0]:
        if msg.time > 0:
            abs_time += mido.tick2second(msg.time, ppq, current_tempo)
            abs_tick += msg.time
            time_samples.append(abs_time)
            tick_samples.append(abs_tick)
        if msg.type == "set_tempo":
            current_tempo = msg.tempo
    ticks = np.interp(times, time_samples, tick_samples)
    return ticks / ppq


def load_gt(piece_dir, performer_id):
    """Load GT (perf_times, gt_beats_in_xml_space) from ASAP JSON."""
    midi_score_path = os.path.join(piece_dir, "midi_score.mid")
    perf_midi_rel = os.path.relpath(
        os.path.join(piece_dir, f"{performer_id}.mid"),
        os.path.abspath(DATASET_ROOT),
    ).replace(os.sep, "/")

    with open(ASAP_JSON, "r") as f:
        asap = json.load(f)
    entry = asap.get(perf_midi_rel, {})
    perf_times = np.array(entry["performance_beats"], dtype=np.float64)
    score_times = np.array(entry["midi_score_beats"], dtype=np.float64)
    gt_beats = _midi_times_to_beats(midi_score_path, score_times)
    # One ASAP pair (chopin_etude_10_1/LuM02M) has a trailing extra beat entry
    # in the JSON (313 vs 312). Trim to the shorter of the two to keep
    # perf_times and gt_beats aligned; downstream indexing requires equal lengths.
    if len(perf_times) != len(gt_beats):
        n = min(len(perf_times), len(gt_beats))
        perf_times = perf_times[:n]
        gt_beats = gt_beats[:n]
    return perf_times, gt_beats


# ---------------------------------------------------------------------------
# Prior map building
# ---------------------------------------------------------------------------
def build_prior_from_midi(midi_path):
    """Dual-load MIDI to pair onset_beat (score) with onset_sec (performance)."""
    scr = partitura.load_score(midi_path)
    na_s = partitura.utils.ensure_notearray(scr)
    perf = partitura.load_performance_midi(midi_path)
    na_p = perf[0].note_array()
    s_sort = np.argsort(na_s["onset_div"])
    p_sort = np.argsort(na_p["onset_tick"])
    n = min(len(s_sort), len(p_sort))
    beats = na_s["onset_beat"][s_sort[:n]]
    times = na_p["onset_sec"][p_sort[:n]]
    _, u_idx = np.unique(beats, return_index=True)
    return beats[u_idx], times[u_idx]


# ---------------------------------------------------------------------------
# Direct beat mapping (brain_beat → XML_beat) via note-level pitch match
# ---------------------------------------------------------------------------
def build_direct_beat_mapping(brain_midi_path, xml_path):
    """Build a monotonic brain_beat → XML_beat mapping via greedy pitch match.
    Returns (brain_beats, xml_beats) arrays, or (None, None) if too few points.
    """
    brain_na = partitura.utils.ensure_notearray(partitura.load_score(brain_midi_path))
    xml_na = partitura.utils.ensure_notearray(partitura.load_score(xml_path))
    pairs = []
    for pitch in range(21, 109):
        bm = brain_na["pitch"] == pitch
        xm = xml_na["pitch"] == pitch
        if not np.any(bm) or not np.any(xm):
            continue
        bi = np.where(bm)[0]
        xi = np.where(xm)[0]
        bs = bi[np.argsort(brain_na["onset_beat"][bi])]
        xs = xi[np.argsort(xml_na["onset_beat"][xi])]
        for k in range(min(len(bs), len(xs))):
            pairs.append(
                (brain_na["onset_beat"][bs[k]], xml_na["onset_beat"][xs[k]])
            )
    if len(pairs) < 50:
        return None, None
    pairs.sort()
    sb = np.array([p[0] for p in pairs])
    tb = np.array([p[1] for p in pairs])
    _, ui = np.unique(sb, return_index=True)
    sb, tb = sb[ui], tb[ui]
    # Enforce monotonicity
    csb, ctb = [sb[0]], [tb[0]]
    for i in range(1, len(sb)):
        if tb[i] >= ctb[-1]:
            csb.append(sb[i])
            ctb.append(tb[i])
    csb, ctb = np.array(csb), np.array(ctb)
    if len(csb) < 50:
        return None, None
    return csb, ctb


# ---------------------------------------------------------------------------
# Score annotation time computation (brain-specific)
# ---------------------------------------------------------------------------
def _build_score_times_for_beats(score_part, beats, tempo):
    """Compute score audio time (seconds) for given beat positions.

    Uses beat * (60/tempo) directly.  The previous inv_beat_map /
    quarter_duration_map formula was incorrect for non-4/4 time
    signatures (12/8, 12/16) because quarter_duration_map returns
    divs-per-quarter, not divs-per-beat-unit.
    """
    return np.asarray(beats, dtype=np.float64) * (60.0 / tempo)


def _build_score_times_for_beats_variable(beats, tempo_map):
    """Compute score audio time (seconds) for given beat positions
    using a variable tempo map (from midi_score.mid set_tempo events).

    tempo_map: list of (beat_position, bpm) sorted by beat_position.
    Time is computed by integrating over piecewise-constant tempo segments.
    """
    beats = np.asarray(beats, dtype=np.float64)
    if len(tempo_map) == 0:
        return beats * 0.5  # fallback 120 BPM

    # Build cumulative time at each tempo change point
    t_beats = [t[0] for t in tempo_map]
    t_bpms = [t[1] for t in tempo_map]

    # Cumulative time at each tempo change boundary
    cum_time = [0.0]
    for i in range(1, len(t_beats)):
        dt = (t_beats[i] - t_beats[i - 1]) * (60.0 / t_bpms[i - 1])
        cum_time.append(cum_time[-1] + dt)

    # For each query beat, find which segment it belongs to and compute time
    result = np.zeros_like(beats)
    for i, b in enumerate(beats):
        # Find the segment: last tempo change <= b
        idx = np.searchsorted(t_beats, b, side='right') - 1
        idx = max(idx, 0)
        result[i] = cum_time[idx] + (b - t_beats[idx]) * (60.0 / t_bpms[idx])

    return result


def build_variable_tempo_map(midi_score_path, beat_type=4):
    """Extract tempo map from midi_score.mid as [(beat_in_onset_beat_units, bpm), ...].

    midi_score.mid uses standard MIDI ticks (quarter-note based).
    partitura's onset_beat uses beat_type units (half for 2/2, eighth for 12/8).
    We convert MIDI ticks to onset_beat coordinates and adjust BPM accordingly.

    Returns None if only 1 tempo event (constant tempo).
    """
    mid = mido.MidiFile(midi_score_path)
    tempo_events = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                quarter_bpm = mido.tempo2bpm(msg.tempo)
                # Convert tick to quarter-note beat
                quarter_beat = abs_tick / mid.ticks_per_beat
                # Convert to onset_beat units (beat_type based)
                onset_beat = quarter_beat * (beat_type / 4.0)
                # Convert BPM to beat_type BPM
                adjusted_bpm = quarter_bpm * (beat_type / 4.0)
                tempo_events.append((onset_beat, adjusted_bpm))

    if len(tempo_events) <= 1:
        return None  # Constant tempo, no benefit from variable

    tempo_events.sort(key=lambda x: x[0])
    return tempo_events


def generate_score_audio_variable(score_part, tempo_map, samplerate):
    """Generate score audio using variable tempo map.

    tempo_map: list of (onset_beat, bpm) from build_variable_tempo_map.
    """
    # Build per-note bpm_array using stepwise lookup
    onsets = score_part.note_array()["onset_beat"]
    t_beats = np.array([t[0] for t in tempo_map])
    t_bpms = np.array([t[1] for t in tempo_map])

    # Stepwise: each note gets BPM from the most recent tempo event
    indices = np.searchsorted(t_beats, onsets, side='right') - 1
    indices = np.clip(indices, 0, len(t_bpms) - 1)
    bpm_per_note = t_bpms[indices]

    bpm_array = np.column_stack([onsets, bpm_per_note])

    import os
    sf = os.environ.get('MATCHMAKER_SOUNDFONT', None)
    kwargs = dict(bpm=bpm_array, samplerate=samplerate)
    if sf is not None:
        kwargs["soundfont"] = sf
    score_audio = partitura.save_wav_fluidsynth(score_part, **kwargs)

    # Padding: time of first onset using variable tempo
    first_onset = onsets.min()
    first_time = _build_score_times_for_beats_variable(
        np.array([max(first_onset, 0.0)]), tempo_map
    )[0]
    padding_size = max(int(first_time * samplerate), 0)
    if padding_size > 0:
        score_audio = np.pad(score_audio, (padding_size, 0))

    # Truncation: end at last onset + buffer
    last_onset = onsets.max()
    last_time = _build_score_times_for_beats_variable(
        np.array([last_onset]), tempo_map
    )[0]
    last_time += 0.1  # buffer
    score_audio = score_audio[:int(last_time * samplerate)]

    return score_audio


def compute_score_annots(
    brain_type, mm, gt_perf_times, gt_beats, prior_b, prior_s, audio_dur
):
    """Convert GT beat annotations to score audio times for the given brain.

    Returns (score_annots, valid_mask):
        score_annots[i] = time in the reference audio where gt_beats[i] occurs.
        valid_mask[i] = True if this GT beat is within the brain's score range.
    """
    score_na = mm.score_part.note_array()
    score_beat_min = score_na["onset_beat"].min()
    score_beat_max = score_na["onset_beat"].max()

    if brain_type == "xml":
        # GT beats are in XML beat space — direct lookup
        valid = (gt_beats >= score_beat_min) & (gt_beats <= score_beat_max)
        score_annots = np.full(len(gt_beats), np.nan)
        score_annots[valid] = _build_score_times_for_beats(
            mm.score_part, gt_beats[valid], mm.tempo
        )
        return score_annots, valid

    elif brain_type == "audio_self":
        # Score IS the performance audio → score time = perf time
        return gt_perf_times.copy(), np.ones(len(gt_beats), dtype=bool)

    else:
        # virtuoso or midi_aligned: time round-trip
        # GT perf time → brain time (proportional) → brain beat → score audio time
        max_prior_s = prior_s[-1]
        max_perf_t = max(audio_dur, gt_perf_times[-1])
        scale = max_prior_s / max_perf_t if max_perf_t > 0 else 1.0
        brain_time = gt_perf_times * scale
        brain_beat = np.interp(brain_time, prior_s, prior_b)

        valid = (brain_beat >= score_beat_min) & (brain_beat <= score_beat_max)
        score_annots = np.full(len(gt_beats), np.nan)
        score_annots[valid] = _build_score_times_for_beats(
            mm.score_part, brain_beat[valid], mm.tempo
        )
        return score_annots, valid


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------
def run_single(piece, performer_id, brain_type, inject_intent=False,
               plan_path_override=None, force_rehearsal=False, algorithm='arzt',
               use_variable_tempo=False, use_gap_mask=True, norm_exponent=0.5,
               save_plan_from_wp=False, per_beat_csv=None,
               perf_audio_override=None, gt_override=None):
    """Run one (piece, performer, brain) evaluation. Returns dict of metrics.
    algorithm: 'arzt' or 'dixon'.

    If inject_intent=True, compute GT beat-level tempo ratios and inject
    into the PerformancePlan before running DTW (Experiment 2).
    If plan_path_override is set, use persistent plan path (Experiment 3).
    If force_rehearsal=True, keep plan unlocked for continued learning.
    If use_variable_tempo=True, synthesize reference using midi_score.mid tempo map.
    If use_gap_mask=False, disable gap mask (for ablation).
    """
    piece_dir = piece["asap_dir"]
    wav_path = perf_audio_override or os.path.join(piece_dir, f"{performer_id}.wav")
    perf_midi_path = os.path.join(piece_dir, f"{performer_id}.mid")
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")

    import librosa

    audio_dur = librosa.get_duration(path=wav_path)

    # Resolve brain path
    TN_VN_DIR = os.path.join(os.path.dirname(__file__), "datasets", "tn_vn_midi")
    PT_DIR = os.path.join(os.path.dirname(__file__), "datasets", "pt_midi")
    TN_PT_DIR = os.path.join(os.path.dirname(__file__), "datasets", "tn_pt_midi")
    if brain_type == "xml":
        brain_path = xml_path
    elif brain_type == "virtuoso":
        brain_path = piece["vn_midi"]
    elif brain_type == "tn_vn":
        brain_path = os.path.join(TN_VN_DIR, f"{piece['id']}_tn_vn.mid")
    elif brain_type == "pt":
        brain_path = os.path.join(PT_DIR, f"{piece['id']}_pt.mid")
    elif brain_type == "tn_pt":
        brain_path = os.path.join(TN_PT_DIR, f"{piece['id']}_tn_pt.mid")
    elif brain_type == "midi_aligned":
        brain_path = perf_midi_path
    elif brain_type == "audio_self":
        brain_path = wav_path
    else:
        raise ValueError(f"Unknown brain: {brain_type}")

    # Load GT (or use override for synthetic-gap experiments)
    if gt_override is not None:
        gt_perf_times, gt_beats = gt_override
    else:
        gt_perf_times, gt_beats = load_gt(piece_dir, performer_id)

    # Build beat mapping: brain_beat → XML_beat
    # VN: direct note-level mapping (avoids time domain errors)
    # midi_aligned/audio_self: time-based mapping (prior matches perf timing)
    # XML: identity (brain beat = XML beat)
    direct_map_b, direct_map_x = None, None
    prior_b, prior_s = np.array([0.0, 1.0]), np.array([0.0, audio_dur])

    if brain_type == "xml":
        pass  # pred_beat = val (identity)
    elif brain_type == "virtuoso":
        direct_map_b, direct_map_x = build_direct_beat_mapping(
            brain_path, xml_path
        )
        # Fallback: time-based mapping
        prior_b, prior_s = build_prior_from_midi(brain_path)
    elif brain_type in ("tn_vn", "pt", "tn_pt"):
        # TN-VN, PT, and TN-PT have XML beat structure → same mapping as XML
        pass
    elif brain_type in ("midi_aligned", "audio_self"):
        prior_b, prior_s = build_prior_from_midi(perf_midi_path)

    # Determine realistic tempo for each brain (no WAV length or GT access).
    if brain_type == "xml":
        initial_tempo = _get_xml_tempo(xml_path)
    elif brain_type == "virtuoso":
        initial_tempo = _get_midi_embedded_bpm(brain_path)
    elif brain_type in ("tn_vn", "pt", "tn_pt"):
        # TN-VN/PT/TN-PT use XML beat structure → use XML tempo
        initial_tempo = _get_xml_tempo(xml_path)
    elif brain_type == "midi_aligned":
        initial_tempo = _get_midi_embedded_bpm(brain_path)
    else:  # audio_self
        initial_tempo = 120

    # Set DEFAULT_TEMPO so Matchmaker uses our realistic tempo
    import matchmaker.matchmaker as _mm_mod
    _mm_mod.DEFAULT_TEMPO = initial_tempo

    # Run Matchmaker
    plan_path_is_temp = plan_path_override is None
    if plan_path_is_temp:
        fd, plan_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        if os.path.exists(plan_path):
            os.remove(plan_path)
    else:
        plan_path = plan_path_override

    # --- Feature caching: skip expensive Matchmaker init if cached ---
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "features")
    os.makedirs(CACHE_DIR, exist_ok=True)

    if brain_type in ("audio_self", "midi_aligned"):
        ref_cache_key = os.path.join(CACHE_DIR, f"{piece['id']}_{brain_type}_{performer_id}.npz")
    else:
        ref_cache_key = os.path.join(CACHE_DIR, f"{piece['id']}_{brain_type}.npz")

    # Variable tempo: build tempo map from midi_score.mid if requested
    var_tempo_map = None
    if use_variable_tempo and brain_type in ("xml", "tn_vn", "pt", "tn_pt"):
        midi_score_path = os.path.join(piece_dir, "midi_score.mid")
        if os.path.exists(midi_score_path):
            # Get beat_type from XML for coordinate conversion
            score_tmp = partitura.load_musicxml(xml_path)
            part_tmp = score_tmp[0] if hasattr(score_tmp, '__getitem__') else score_tmp
            bt = part_tmp.time_sigs[0].beat_type if part_tmp.time_sigs else 4
            var_tempo_map = build_variable_tempo_map(midi_score_path, beat_type=bt)

    try:
        mm = Matchmaker(
            score_file=brain_path,
            performance_file=wav_path,
            input_type="audio",
            wait=False,
            performance_plan_path=plan_path,
        )
        if force_rehearsal and mm.performance_plan is not None:
            mm.performance_plan.is_locked = False

        # Variable tempo: replace score audio & features with variable-tempo version
        if var_tempo_map is not None and brain_type != "audio_self":
            mm.score_audio = generate_score_audio_variable(
                mm.score_part, var_tempo_map, SAMPLE_RATE
            ).astype(np.float32)
            mm.reference_features = mm.processor(mm.score_audio)

        # SILENCE TRIMMING: trim padding before first note to prevent
        # DTW normalization bias overshoot.
        ref_trim_time = 0.0
        if brain_type != "audio_self":
            na_sp = mm.score_part.note_array()
            first_beat = na_sp["onset_beat"].min()
            if var_tempo_map is not None:
                ref_trim_time = _build_score_times_for_beats_variable(
                    np.array([max(first_beat, 0.0)]), var_tempo_map
                )[0]
            else:
                ref_trim_time = max(first_beat, 0) * (60.0 / mm.tempo)
            trim_samples = int(ref_trim_time * SAMPLE_RATE)
            if trim_samples > 0:
                mm.score_audio = mm.score_audio[trim_samples:]
                mm.reference_features = mm.processor(mm.score_audio)

        # Save reference features to cache (skip for variable tempo runs)
        if not os.path.exists(ref_cache_key) and var_tempo_map is None:
            np.savez_compressed(ref_cache_key,
                                features=mm.reference_features,
                                trim_time=ref_trim_time,
                                tempo=float(mm.tempo))

        # Compute gap mask from MIDI note events (for DTW silence robustness)
        from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArzt
        is_gap = None
        if use_gap_mask:
            if brain_type == "audio_self":
                gap_midi = perf_midi_path
            elif brain_type in ("xml", "tn_vn", "pt", "tn_pt"):
                gap_midi = os.path.join(piece_dir, "midi_score.mid")
            else:
                gap_midi = brain_path
            is_gap = compute_gap_mask(
                gap_midi, len(mm.reference_features), mm.frame_rate,
                trim_seconds=ref_trim_time,
            )
        if algorithm == 'dixon':
            from matchmaker.dp import OnlineTimeWarpingDixon
            mm.score_follower = OnlineTimeWarpingDixon(
                reference_features=mm.reference_features,
                queue=mm.stream.queue,
            )
        elif algorithm == 'dixonplus':
            from matchmaker.dp.oltw_dixon import OnlineTimeWarpingDixonPlus
            mm.score_follower = OnlineTimeWarpingDixonPlus(
                reference_features=mm.reference_features,
                queue=mm.stream.queue,
                performance_plan=mm.performance_plan,
                tempo_bpm=float(mm.tempo),
            )
        else:
            # 'arzt' or 'scorpion' — both use OnlineTimeWarpingArzt,
            # with different normalization methods
            mm.score_follower = OnlineTimeWarpingArzt(
                reference_features=mm.reference_features,
                queue=mm.stream.queue,
                frame_rate=mm.frame_rate,
                performance_plan=mm.performance_plan,
                tempo_bpm=float(mm.tempo),
                is_gap_frame=is_gap,
                normalization="scorpion" if algorithm == "scorpion" else "arzt",
                norm_exponent=norm_exponent,
            )

        # [Experiment 2] Inject GT tempo profile into PerformancePlan
        if inject_intent and mm.performance_plan is not None:
            gt_pt, gt_bt = load_gt(piece_dir, performer_id)
            # Compute per-beat IOI ratio (relative to constant-tempo expectation)
            if len(gt_pt) > 2:
                iois = np.diff(gt_pt)           # time between consecutive beats
                beat_diffs = np.diff(gt_bt)      # beat distance
                # Avoid division by zero
                beat_diffs = np.where(beat_diffs > 0, beat_diffs, 1.0)
                # Local tempo = beat_diff / IOI (beats per second)
                local_tempos = beat_diffs / iois
                median_tempo = np.median(local_tempos)
                # Tempo ratio: >1 = faster than median, <1 = slower
                tempo_ratios = local_tempos / median_tempo if median_tempo > 0 else np.ones_like(local_tempos)
                # Clip extreme values
                tempo_ratios = np.clip(tempo_ratios, 0.05, 5.0)

                # Map GT beats to reference beats (brain coordinate)
                if brain_type in ("midi_aligned", "audio_self"):
                    ref_beats = np.interp(gt_pt[:-1], prior_s, prior_b)
                elif brain_type == "virtuoso" and direct_map_b is not None:
                    xml_sp_tmp = partitura.load_score_as_part(xml_path)
                    bt_tmp = xml_sp_tmp.time_sigs[0].beat_type if xml_sp_tmp.time_sigs else 4
                    ref_beats = np.interp(gt_bt[:-1] * (bt_tmp / 4), direct_map_x, direct_map_b)
                else:
                    # xml, tn_vn, pt, tn_pt: scale gt_beats by beat_type
                    _bt = 4
                    if mm.score_part.time_sigs:
                        _bt = mm.score_part.time_sigs[0].beat_type
                    ref_beats = gt_bt[:-1] * (_bt / 4)

                # Inject into plan
                for rb, tr in zip(ref_beats, tempo_ratios):
                    idx = int(rb)
                    if 0 <= idx < len(mm.performance_plan.tempo_plan):
                        mm.performance_plan.tempo_plan[idx] = float(tr)
                        mm.performance_plan.locked_beats[idx] = True

        # Run DTW
        for _ in mm.run(verbose=False):
            pass

        wp = np.array(mm.score_follower.warping_path)
        tempo_for_eval = float(mm.tempo)
        ref_dur = len(mm.score_audio) / SAMPLE_RATE

        # [Experiment 3/Hybrid] Post-hoc plan update from warping path
        # force_rehearsal: Arzt rehearsal learning
        # save_plan_from_wp: Dixon→Arzt hybrid (save Dixon's plan for Arzt 2nd pass)
        if (force_rehearsal or save_plan_from_wp) and mm.performance_plan is not None:
            mm.performance_plan.update_from_warping_path(
                wp, mm.frame_rate, float(mm.tempo)
            )
            mm.performance_plan.save_plan()

        # ---- Compute score_annots: ref-audio time for each GT beat ----
        # This is the time in the REFERENCE AUDIO where each GT beat's
        # musical content plays. The warping path maps these to predicted
        # performance times.
        #
        # XML:          GT_beat → ref_time via _build_score_times_for_beats
        # VN:           GT_beat → VN_beat (direct map) → ref_time via LUT
        # midi/audio:   GT_perf_time → perf_beat (prior) → ref_time via LUT

        if brain_type == "audio_self":
            # Reference audio IS the performance WAV → score time = perf time directly
            score_annots = gt_perf_times.copy()
        else:
            # Build beat → ref_time LUT for this brain's score_part
            na_sp = mm.score_part.note_array()
            sp_bmin = max(na_sp["onset_beat"].min(), 0.0)  # clamp negative pickup beats
            sp_bmax = na_sp["onset_beat"].max()
            lut_b = np.linspace(sp_bmin, sp_bmax, 2000)
            if var_tempo_map is not None and brain_type in ("xml", "tn_vn", "pt", "tn_pt"):
                lut_t = _build_score_times_for_beats_variable(lut_b, var_tempo_map)
            else:
                lut_t = _build_score_times_for_beats(
                    mm.score_part, lut_b, tempo_for_eval
                )

            if brain_type in ("xml", "tn_vn", "pt", "tn_pt"):
                # gt_beats are in quarter-note units (from midi_score.mid,
                # always 4/4). XML/TN-VN/PT/TN-PT onset_beat uses the time-signature's
                # beat_type unit (eighth for 12/8, sixteenth for 12/16).
                # Scale gt_beats to match the beat unit.
                beat_type = 4
                if mm.score_part.time_sigs:
                    beat_type = mm.score_part.time_sigs[0].beat_type
                gt_beats_scaled = gt_beats * (beat_type / 4)
                score_annots = np.array([
                    float(np.interp(b, lut_b, lut_t))
                    for b in gt_beats_scaled
                ])
            elif brain_type == "virtuoso" and direct_map_b is not None:
                # GT_beat → VN_beat via direct note-level pitch mapping, then LUT.
                # beat_type scaling for non-4/4 time signatures.
                xml_sp = partitura.load_score_as_part(xml_path)
                bt = xml_sp.time_sigs[0].beat_type if xml_sp.time_sigs else 4
                gt_beats_scaled = gt_beats * (bt / 4)
                vn_beats_for_gt = np.interp(gt_beats_scaled, direct_map_x, direct_map_b)
                score_annots = np.array([
                    float(np.interp(b, lut_b, lut_t))
                    for b in vn_beats_for_gt
                ])
            elif brain_type == "midi_aligned":
                # Reference audio = synthesized from performer MIDI at constant tempo.
                # prior_b: performer-MIDI beat positions (different coordinate from gt_beats!)
                # prior_s: actual performance note times (same coordinate as gt_perf_times)
                # Correct: gt_perf_time → performer beat (via prior_s→prior_b) → ref time (via LUT)
                perf_beats_at_gt = np.interp(gt_perf_times, prior_s, prior_b)
                score_annots = np.array([
                    float(np.interp(b, lut_b, lut_t))
                    for b in perf_beats_at_gt
                ])
            elif brain_type == "virtuoso":
                # virtuoso fallback (no direct map available):
                # Use same XML-based approach as XML brain.
                # WARNING: direct_map was None, so pitch matching failed.
                # This path produces less accurate results.
                xml_sp_fallback = partitura.load_score_as_part(xml_path)
                bt_fb = xml_sp_fallback.time_sigs[0].beat_type if xml_sp_fallback.time_sigs else 4
                gt_beats_scaled_fb = gt_beats * (bt_fb / 4)
                score_annots = np.array([
                    float(np.interp(b, lut_b, lut_t))
                    for b in gt_beats_scaled_fb
                ])
            else:
                raise ValueError(f"Unhandled brain_type for score_annots: {brain_type}")

        # Adjust for trimmed initial silence (applies to all brains)
        score_annots -= ref_trim_time
        # Clip score_annots to valid ref range
        score_annots = np.clip(score_annots, 0, ref_dur)

        # ---- Predict performance times via warping path ----
        pred_perf = transfer_from_score_to_predicted_perf(
            wp, score_annots, mm.frame_rate
        )

        ok = ~np.isnan(pred_perf)
        total = len(gt_perf_times)
        errors_ms = np.abs(gt_perf_times[ok] - pred_perf[ok]) * 1000.0

        result = {
            "piece": piece["id"],
            "piece_name": piece["name"],
            "performer": performer_id,
            "brain": brain_type,
            "algorithm": algorithm,
            "variable_tempo": use_variable_tempo and var_tempo_map is not None,
            "gap_mask": use_gap_mask,
            "n_gt_total": total,
            "n_pred_ok": int(np.sum(ok)),
            "ref_audio_dur": round(ref_dur, 1),
            "perf_dur": round(audio_dur, 1),
            "tempo": round(tempo_for_eval, 1),
        }
        for tau in TOLERANCES:
            result[f"ar_{tau}"] = round(
                float(np.sum(errors_ms <= tau) / total * 100), 2
            )
        result["medae_ms"] = round(float(np.nanmedian(errors_ms)), 1)
        result["meanae_ms"] = round(float(np.nanmean(errors_ms)), 1)

        # Optional per-beat CSV dump for downstream section / causal analyses
        if per_beat_csv is not None:
            import csv as _csv
            os.makedirs(os.path.dirname(per_beat_csv) or ".", exist_ok=True)
            with open(per_beat_csv, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["beat_idx", "gt_beat", "gt_time_s", "pred_time_s", "err_ms", "ok"])
                for i in range(total):
                    gt_t = float(gt_perf_times[i])
                    pred_t = float(pred_perf[i]) if not np.isnan(pred_perf[i]) else float("nan")
                    err_ms = float(np.abs(gt_t - pred_t) * 1000.0) if ok[i] else float("nan")
                    w.writerow([i, float(gt_beats[i]), gt_t,
                                "" if np.isnan(pred_t) else pred_t,
                                "" if np.isnan(err_ms) else round(err_ms, 2),
                                int(ok[i])])

    finally:
        if plan_path_is_temp and os.path.exists(plan_path):
            os.remove(plan_path)
        # Clean up vivace memory
        if os.path.exists("vivace_memory.pkl"):
            os.remove("vivace_memory.pkl")

    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def run_all():
    """Run all 25 pairs × 4 brains. Returns DataFrame."""
    results = []
    total_runs = sum(len(p["performers"]) for p in PIECES) * len(BRAINS)
    run_idx = 0

    for piece in PIECES:
        for performer in piece["performers"]:
            for brain in BRAINS:
                run_idx += 1
                tag = f"[{run_idx}/{total_runs}] {piece['id']}/{performer}/{brain}"
                print(f"\n{'='*70}")
                print(f"  {tag}")
                print(f"{'='*70}")
                t0 = tmod.time()
                try:
                    r = run_single(piece, performer, brain)
                    elapsed = tmod.time() - t0
                    print(
                        f"  -> AR@500={r['ar_500']:.1f}%, "
                        f"MedAE={r['medae_ms']:.0f}ms, "
                        f"valid={r['n_pred_ok']}/{r['n_gt_total']}, "
                        f"{elapsed:.0f}s"
                    )
                    results.append(r)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"  -> FAILED: {e}")
                    results.append(
                        {
                            "piece": piece["id"],
                            "piece_name": piece["name"],
                            "performer": performer,
                            "brain": brain,
                            "error": str(e),
                        }
                    )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------
def run_stats(df):
    """Compute statistical tests. Returns dict of results."""
    stats_out = {}

    # Filter successful runs
    ok = df.dropna(subset=["ar_500"])

    # Pivot: one row per (piece, performer), columns per brain
    for tau in TOLERANCES:
        col = f"ar_{tau}"
        pivot = ok.pivot_table(
            index=["piece", "performer"], columns="brain", values=col
        )
        if "virtuoso" not in pivot.columns or "xml" not in pivot.columns:
            continue
        both = pivot.dropna(subset=["virtuoso", "xml"])
        vn_vals = both["virtuoso"].values
        xml_vals = both["xml"].values
        diff = vn_vals - xml_vals

        if len(diff) < 5:
            continue

        # Wilcoxon signed-rank (one-sided: VN > XML)
        try:
            stat, p_two = stats.wilcoxon(diff, alternative="two-sided")
            # one-sided: if median(diff) > 0, p_one = p_two / 2
            p_one = p_two / 2 if np.median(diff) > 0 else 1.0 - p_two / 2
        except Exception:
            stat, p_one = np.nan, np.nan

        # Effect size: rank-biserial r = Z / sqrt(N)
        n = len(diff)
        z = stats.norm.ppf(1 - p_one) if p_one < 1.0 else 0.0
        r_eff = z / np.sqrt(n) if n > 0 else 0.0

        # Bootstrap CI for mean difference
        rng = np.random.default_rng(42)
        boot_means = [
            np.mean(rng.choice(diff, size=n, replace=True)) for _ in range(10000)
        ]
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        # Count improvements
        n_vn_better = int(np.sum(diff > 0))
        n_xml_better = int(np.sum(diff < 0))
        n_tie = int(np.sum(diff == 0))

        stats_out[f"ar_{tau}"] = {
            "n_pairs": n,
            "mean_diff": round(float(np.mean(diff)), 2),
            "median_diff": round(float(np.median(diff)), 2),
            "ci_95_lo": round(float(ci_lo), 2),
            "ci_95_hi": round(float(ci_hi), 2),
            "wilcoxon_stat": round(float(stat), 2) if not np.isnan(stat) else None,
            "p_one_sided": round(float(p_one), 6) if not np.isnan(p_one) else None,
            "effect_r": round(float(r_eff), 3),
            "n_vn_better": n_vn_better,
            "n_xml_better": n_xml_better,
            "n_tie": n_tie,
        }

    # Holm-Bonferroni correction across tolerances
    p_vals = [
        stats_out[f"ar_{t}"]["p_one_sided"]
        for t in TOLERANCES
        if f"ar_{t}" in stats_out and stats_out[f"ar_{t}"]["p_one_sided"] is not None
    ]
    if p_vals:
        sorted_idx = np.argsort(p_vals)
        m = len(p_vals)
        corrected = [None] * m
        for rank, idx in enumerate(sorted_idx):
            corrected[idx] = min(1.0, p_vals[idx] * (m - rank))
        tol_keys = [
            f"ar_{t}"
            for t in TOLERANCES
            if f"ar_{t}" in stats_out
            and stats_out[f"ar_{t}"]["p_one_sided"] is not None
        ]
        for i, key in enumerate(tol_keys):
            stats_out[key]["p_holm"] = round(corrected[i], 6)

    # Friedman test across all 4 brains (AR@500)
    pivot_all = ok.pivot_table(
        index=["piece", "performer"], columns="brain", values="ar_500"
    )
    brains_present = [b for b in BRAINS if b in pivot_all.columns]
    complete = pivot_all.dropna(subset=brains_present)
    if len(complete) >= 5 and len(brains_present) >= 3:
        groups = [complete[b].values for b in brains_present]
        try:
            f_stat, f_p = stats.friedmanchisquare(*groups)
            stats_out["friedman_ar500"] = {
                "statistic": round(float(f_stat), 2),
                "p_value": round(float(f_p), 6),
                "n": len(complete),
                "brains": brains_present,
            }
        except Exception:
            pass

    return stats_out


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
def generate_outputs(df, stats_out):
    """Save CSV tables and plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Per-pair CSV
    csv_path = os.path.join(OUTPUT_DIR, "results_per_pair.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # 2. Aggregate CSV
    ok = df.dropna(subset=["ar_500"])
    agg_cols = [f"ar_{t}" for t in TOLERANCES] + ["medae_ms", "meanae_ms"]
    agg = ok.groupby("brain")[agg_cols].agg(["mean", "std", "count"])
    agg_path = os.path.join(OUTPUT_DIR, "results_aggregate.csv")
    agg.to_csv(agg_path)
    print(f"Saved: {agg_path}")

    # 3. Per-piece aggregate
    piece_agg = ok.groupby(["piece", "brain"])[agg_cols].agg(["mean", "std", "count"])
    piece_path = os.path.join(OUTPUT_DIR, "results_per_piece.csv")
    piece_agg.to_csv(piece_path)
    print(f"Saved: {piece_path}")

    # 4. Stats CSV
    stats_rows = []
    for key, val in stats_out.items():
        row = {"metric": key}
        row.update(val)
        stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(OUTPUT_DIR, "statistical_tests.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")

    # 5. Sanity check CSV
    sanity = ok[ok["brain"] == "audio_self"][
        ["piece", "performer", "ar_500", "ar_1000", "ar_2000", "medae_ms"]
    ]
    sanity_path = os.path.join(OUTPUT_DIR, "sanity_check.csv")
    sanity.to_csv(sanity_path, index=False)
    print(f"Saved: {sanity_path}")

    # --- Plots ---
    plt.style.use("seaborn-v0_8-whitegrid")
    brain_colors = {
        "audio_self": "#2ecc71",
        "midi_aligned": "#3498db",
        "virtuoso": "#e74c3c",
        "xml": "#95a5a6",
    }
    brain_labels = {
        "audio_self": "Audio Self (sanity)",
        "midi_aligned": "Perf. MIDI (upper bound)",
        "virtuoso": "VirtuosoNet (proposed)",
        "xml": "MusicXML (baseline)",
    }

    # Plot 1: AR Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    for brain in BRAINS:
        bdata = ok[ok["brain"] == brain]
        means = [bdata[f"ar_{t}"].mean() for t in TOLERANCES]
        stds = [bdata[f"ar_{t}"].std() for t in TOLERANCES]
        ax.errorbar(
            TOLERANCES,
            means,
            yerr=stds,
            marker="o",
            label=brain_labels.get(brain, brain),
            color=brain_colors.get(brain, "gray"),
            linewidth=2,
            capsize=4,
        )
    ax.set_xlabel("Tolerance (ms)", fontsize=12)
    ax.set_ylabel("Alignment Rate (%)", fontsize=12)
    ax.set_title("Alignment Rate vs Tolerance", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks(TOLERANCES)
    ax.set_xticklabels([str(t) for t in TOLERANCES])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_ar_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: 01_ar_curve.png")

    # Plot 2: Paired difference (VN - XML) at AR@500
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = ok.pivot_table(
        index=["piece", "performer"], columns="brain", values="ar_500"
    )
    if "virtuoso" in pivot.columns and "xml" in pivot.columns:
        both = pivot.dropna(subset=["virtuoso", "xml"])
        diff = both["virtuoso"] - both["xml"]
        pieces_for_color = [idx[0] for idx in diff.index]
        unique_pieces = sorted(set(pieces_for_color))
        piece_cmap = {p: plt.cm.tab10(i) for i, p in enumerate(unique_pieces)}
        colors = [piece_cmap[p] for p in pieces_for_color]
        ax.scatter(range(len(diff)), diff.values, c=colors, s=60, zorder=3)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_ylabel("AR@500ms Difference (VN - XML)", fontsize=12)
        ax.set_xlabel("Performer-Piece Pair", fontsize=12)
        ax.set_title(
            "Paired Difference: VirtuosoNet vs MusicXML", fontsize=14, fontweight="bold"
        )
        # Legend for pieces
        for p in unique_pieces:
            short = p.replace("chopin_", "").replace("beethoven_", "").replace("bach_", "")
            ax.scatter([], [], c=[piece_cmap[p]], label=short)
        ax.legend(fontsize=9, title="Piece")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_paired_difference.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: 02_paired_difference.png")

    # Plot 3: Per-piece grouped bars (AR@500)
    fig, ax = plt.subplots(figsize=(10, 5))
    piece_ids = [p["id"] for p in PIECES]
    x = np.arange(len(piece_ids))
    width = 0.2
    for i, brain in enumerate(BRAINS):
        means = []
        for pid in piece_ids:
            subset = ok[(ok["piece"] == pid) & (ok["brain"] == brain)]
            means.append(subset["ar_500"].mean() if len(subset) > 0 else 0)
        ax.bar(
            x + i * width,
            means,
            width,
            label=brain_labels.get(brain, brain),
            color=brain_colors.get(brain, "gray"),
        )
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(
        [p["name"] for p in PIECES], rotation=15, ha="right", fontsize=9
    )
    ax.set_ylabel("AR@500ms (%)", fontsize=12)
    ax.set_title("Per-Piece Alignment Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_per_piece_bars.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: 03_per_piece_bars.png")

    # Plot 4: Summary table printed to console
    print(f"\n{'='*80}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*80}")
    for brain in BRAINS:
        bdata = ok[ok["brain"] == brain]
        if len(bdata) == 0:
            continue
        ar500_m = bdata["ar_500"].mean()
        ar500_s = bdata["ar_500"].std()
        med_m = bdata["medae_ms"].mean()
        print(
            f"  {brain_labels.get(brain, brain):30s}  "
            f"AR@500={ar500_m:5.1f}±{ar500_s:4.1f}%  "
            f"MedAE={med_m:7.0f}ms  "
            f"(n={len(bdata)})"
        )

    print(f"\n{'='*80}")
    print("  STATISTICAL TESTS (VN vs XML)")
    print(f"{'='*80}")
    for tau in TOLERANCES:
        key = f"ar_{tau}"
        if key in stats_out:
            s = stats_out[key]
            sig = "***" if s.get("p_holm", 1) < 0.001 else (
                "**" if s.get("p_holm", 1) < 0.01 else (
                    "*" if s.get("p_holm", 1) < 0.05 else "n.s."
                )
            )
            print(
                f"  AR@{tau:4d}ms: diff={s['mean_diff']:+5.1f}% "
                f"[{s['ci_95_lo']:+5.1f}, {s['ci_95_hi']:+5.1f}], "
                f"p(Holm)={s.get('p_holm', 'N/A')}, r={s['effect_r']:.3f}, "
                f"VN>{s['n_vn_better']}/XML>{s['n_xml_better']}  {sig}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  VirtuosoNet-Augmented Score Following — Full Experiment")
    print(f"  {sum(len(p['performers']) for p in PIECES)} pairs × {len(BRAINS)} brains")
    print("=" * 80)

    t0 = tmod.time()
    df = run_all()
    elapsed = tmod.time() - t0
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")

    # Sanity check
    sanity = df[(df["brain"] == "audio_self") & df["ar_500"].notna()]
    failed = sanity[sanity["ar_500"] < 95]
    if len(failed) > 0:
        print("\n  ⚠ SANITY CHECK FAILURES (audio_self AR@500 < 95%):")
        for _, row in failed.iterrows():
            print(f"    {row['piece']}/{row['performer']}: AR@500={row['ar_500']:.1f}%")

    stats_out = run_stats(df)
    generate_outputs(df, stats_out)


def run_experiment2():
    """Experiment 2: Intent Injection — compare with/without GT tempo injection."""
    print("=" * 80)
    print("  Experiment 2: Intent Injection (GT Tempo Profile)")
    print("=" * 80)

    results = []
    for piece in PIECES:
        for performer in piece["performers"]:
            for brain in ["xml", "midi_aligned", "virtuoso"]:
                for intent in [False, True]:
                    tag = f"{piece['id']}/{performer}/{brain}/intent={intent}"
                    try:
                        r = run_single(piece, performer, brain, inject_intent=intent)
                        r["inject_intent"] = intent
                        print(
                            f"  {tag:60s} AR@500={r['ar_500']:5.1f}%"
                        )
                        results.append(r)
                    except Exception as e:
                        print(f"  {tag:60s} FAILED: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "exp2_intent_results.csv"), index=False)

    # Compare
    print("\n" + "=" * 70)
    print("  Experiment 2: Intent Injection Results")
    print("=" * 70)
    for brain in ["xml", "midi_aligned", "virtuoso"]:
        no_intent = df[(df["brain"] == brain) & (df["inject_intent"] == False)]["ar_500"]
        with_intent = df[(df["brain"] == brain) & (df["inject_intent"] == True)]["ar_500"]
        if len(no_intent) > 0 and len(with_intent) > 0:
            print(
                f"  {brain:15s}: without={no_intent.mean():5.1f}% → with={with_intent.mean():5.1f}%  "
                f"(diff={with_intent.mean() - no_intent.mean():+5.1f}%)"
            )


def run_experiment3():
    """Experiment 3: Rehearsal Learning — run same piece 3x, accumulate plan."""
    print("=" * 80)
    print("  Experiment 3: Rehearsal Learning (3 iterations)")
    print("=" * 80)

    N_ITER = 3
    results = []
    for piece in PIECES:
        for performer in piece["performers"]:
            for brain in ["xml", "midi_aligned"]:
                # Persistent plan path
                plan_path = os.path.join(
                    OUTPUT_DIR,
                    f"exp3_{piece['id']}_{performer}_{brain}_plan.pkl",
                )
                # Start fresh
                if os.path.exists(plan_path):
                    os.remove(plan_path)

                for iteration in range(N_ITER):
                    tag = f"{piece['id']}/{performer}/{brain}/iter={iteration}"
                    try:
                        r = run_single(
                            piece, performer, brain,
                            plan_path_override=plan_path,
                            force_rehearsal=True,
                        )
                        r["iteration"] = iteration
                        print(
                            f"  {tag:55s} AR@500={r['ar_500']:5.1f}%"
                        )
                        results.append(r)
                    except Exception as e:
                        print(f"  {tag:55s} FAILED: {e}")

                # Clean up plan
                if os.path.exists(plan_path):
                    os.remove(plan_path)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "exp3_rehearsal_results.csv"), index=False)

    # Summary
    print("\n" + "=" * 70)
    print("  Experiment 3: Rehearsal Learning Results")
    print("=" * 70)
    for brain in ["xml", "midi_aligned"]:
        for it in range(N_ITER):
            vals = df[(df["brain"] == brain) & (df["iteration"] == it)]["ar_500"]
            if len(vals) > 0:
                print(f"  {brain:15s} iter {it}: AR@500={vals.mean():5.1f}% ± {vals.std():4.1f}%")


def run_experiment_vartempo():
    """Variable Tempo ablation: constant vs variable on all 156 pairs, XML brain."""
    print("=" * 80)
    print("  Experiment VarTempo: Constant vs Variable Tempo Synthesis")
    print("=" * 80)

    results = []
    for piece in PIECES:
        for performer in piece["performers"]:
            for vt in [False, True]:
                tag = f"{piece['id']}/{performer}/vt={vt}"
                try:
                    r = run_single(piece, performer, "xml", use_variable_tempo=vt)
                    print(f"  {tag:55s} AR@500={r['ar_500']:5.1f}%", flush=True)
                    results.append(r)
                except Exception as e:
                    print(f"  {tag:55s} FAILED: {e}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "exp_vartempo_results.csv"), index=False)

    # Summary
    print("\n" + "=" * 70)
    for vt in [False, True]:
        vals = df[df["variable_tempo"] == vt]["ar_500"]
        if len(vals) > 0:
            label = "VarTempo" if vt else "Constant"
            print(f"  {label:12s}: AR@500={vals.mean():5.1f}% ± {vals.std():4.1f}%  (n={len(vals)})")


def run_experiment_gapmask():
    """Gap Mask ablation: on vs off on silence-heavy pieces."""
    print("=" * 80)
    print("  Experiment GapMask: Ablation (on vs off)")
    print("=" * 80)

    # Focus on pieces with known silence issues + control pieces
    # Pieces with many gaps: Ballade 1, Ballade 4, Barcarolle, Beethoven sonatas
    # Control: Etudes (continuous)
    results = []
    for piece in PIECES:
        for performer in piece["performers"]:
            for gm in [True, False]:
                tag = f"{piece['id']}/{performer}/gap_mask={gm}"
                try:
                    r = run_single(piece, performer, "xml", use_gap_mask=gm)
                    print(f"  {tag:55s} AR@500={r['ar_500']:5.1f}%")
                    results.append(r)
                except Exception as e:
                    print(f"  {tag:55s} FAILED: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "exp_gapmask_results.csv"), index=False)

    # Summary
    print("\n" + "=" * 70)
    for gm in [True, False]:
        vals = df[df["gap_mask"] == gm]["ar_500"]
        if len(vals) > 0:
            label = "MaskON" if gm else "MaskOFF"
            print(f"  {label:12s}: AR@500={vals.mean():5.1f}% ± {vals.std():4.1f}%")

    # Per-piece breakdown
    print("\n  Per-piece breakdown:")
    for pid in df["piece"].unique():
        on = df[(df["piece"] == pid) & (df["gap_mask"] == True)]["ar_500"]
        off = df[(df["piece"] == pid) & (df["gap_mask"] == False)]["ar_500"]
        if len(on) > 0 and len(off) > 0:
            diff = on.mean() - off.mean()
            print(f"    {pid:30s}: ON={on.mean():5.1f}%  OFF={off.mean():5.1f}%  diff={diff:+5.1f}%")


def run_experiment_locked_beats():
    """Locked Beats: compare intent+rehearsal with/without locked beats."""
    print("=" * 80)
    print("  Experiment LockedBeats: Intent Protection During Rehearsal")
    print("=" * 80)

    # Load exp2 results to find top intent-effective cases
    exp2_path = os.path.join(OUTPUT_DIR, "exp2_intent_results.csv")
    if not os.path.exists(exp2_path):
        print("  ERROR: exp2_intent_results.csv not found. Run exp2 first.")
        return

    exp2 = pd.read_csv(exp2_path)
    # Compute intent improvement per (piece, performer) pair
    diffs = []
    for (piece_id, perf), grp in exp2.groupby(["piece", "performer"]):
        no_intent = grp[grp["inject_intent"] == False]["ar_500"].values
        with_intent = grp[grp["inject_intent"] == True]["ar_500"].values
        if len(no_intent) > 0 and len(with_intent) > 0:
            diff = with_intent.mean() - no_intent.mean()
            diffs.append((piece_id, perf, diff))
    diffs.sort(key=lambda x: -x[2])
    top_cases = diffs[:5]
    print(f"  Top {len(top_cases)} intent-effective cases:")
    for pid, perf, d in top_cases:
        print(f"    {pid}/{perf}: intent diff={d:+.1f}%")

    # Find piece dict by id
    piece_by_id = {p["id"]: p for p in PIECES}

    results = []
    N_REHEARSAL = 2
    for pid, perf, _ in top_cases:
        if pid not in piece_by_id:
            continue
        piece = piece_by_id[pid]
        for use_lock in [True, False]:
            lock_label = "locked" if use_lock else "unlocked"
            plan_path = os.path.join(
                OUTPUT_DIR,
                f"exp_lock_{pid}_{perf}_{lock_label}.pkl",
            )
            if os.path.exists(plan_path):
                os.remove(plan_path)

            # Run 1: intent injection
            tag = f"{pid}/{perf}/{lock_label}/intent"
            try:
                r = run_single(piece, perf, "xml", inject_intent=True,
                              plan_path_override=plan_path)
                r["locked"] = use_lock
                r["phase"] = "intent"
                r["iteration"] = 0
                print(f"  {tag:55s} AR@500={r['ar_500']:5.1f}%")
                results.append(r)
            except Exception as e:
                print(f"  {tag:55s} FAILED: {e}")
                continue

            # If unlocked, clear locked_beats before rehearsal
            if not use_lock and os.path.exists(plan_path):
                import pickle
                with open(plan_path, "rb") as f:
                    plan_data = pickle.load(f)
                if isinstance(plan_data, tuple) and len(plan_data) >= 3:
                    # Clear locked_beats
                    plan_data = (plan_data[0], plan_data[1],
                                np.zeros_like(plan_data[2], dtype=bool))
                    with open(plan_path, "wb") as f:
                        pickle.dump(plan_data, f)

            # Rehearsal iterations
            for it in range(1, N_REHEARSAL + 1):
                tag = f"{pid}/{perf}/{lock_label}/rehearsal_{it}"
                try:
                    r = run_single(piece, perf, "xml",
                                  plan_path_override=plan_path,
                                  force_rehearsal=True)
                    r["locked"] = use_lock
                    r["phase"] = "rehearsal"
                    r["iteration"] = it
                    print(f"  {tag:55s} AR@500={r['ar_500']:5.1f}%")
                    results.append(r)
                except Exception as e:
                    print(f"  {tag:55s} FAILED: {e}")

            # Clean up
            if os.path.exists(plan_path):
                os.remove(plan_path)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "exp_locked_beats_results.csv"), index=False)

    # Summary
    print("\n" + "=" * 70)
    print("  Locked Beats Results (final rehearsal iteration):")
    for lock_val in [True, False]:
        final = df[(df["locked"] == lock_val) & (df["iteration"] == N_REHEARSAL)]
        if len(final) > 0:
            label = "Locked" if lock_val else "Unlocked"
            print(f"  {label:12s}: AR@500={final['ar_500'].mean():5.1f}%")


def run_experiment_hybrid():
    """Dixon→Arzt hybrid: Dixon 1st pass learns plan, Arzt 2nd pass uses it."""
    print("=" * 80)
    print("  Experiment Hybrid: Dixon 1st → Arzt 2nd (Rehearsal Learning)")
    print("=" * 80)

    results = []
    for piece in PIECES:
        for performer in piece["performers"]:
            for brain in ["xml", "midi_aligned"]:
                plan_path = os.path.join(
                    OUTPUT_DIR,
                    f"hybrid_{piece['id']}_{performer}_{brain}.pkl",
                )
                # Clean up any existing plan
                if os.path.exists(plan_path):
                    os.remove(plan_path)

                # Pass 1: Dixon (saves plan from warping path)
                tag1 = f"{piece['id']}/{performer}/{brain}/dixon_1st"
                try:
                    r1 = run_single(
                        piece, performer, brain, algorithm='dixon',
                        plan_path_override=plan_path,
                        save_plan_from_wp=True,
                    )
                    r1["hybrid_pass"] = "dixon_1st"
                    print(f"  {tag1:60s} AR@500={r1['ar_500']:5.1f}%", flush=True)
                    results.append(r1)
                except Exception as e:
                    print(f"  {tag1:60s} FAILED: {e}", flush=True)
                    if os.path.exists(plan_path):
                        os.remove(plan_path)
                    continue

                # Pass 2: Arzt with Dixon-learned plan (Concert Mode)
                tag2 = f"{piece['id']}/{performer}/{brain}/arzt_2nd"
                try:
                    r2 = run_single(
                        piece, performer, brain, algorithm='arzt',
                        plan_path_override=plan_path,
                        force_rehearsal=False,  # Concert Mode: use plan as-is
                    )
                    r2["hybrid_pass"] = "arzt_2nd"
                    print(f"  {tag2:60s} AR@500={r2['ar_500']:5.1f}%", flush=True)
                    results.append(r2)
                except Exception as e:
                    print(f"  {tag2:60s} FAILED: {e}", flush=True)

                # Clean up plan
                if os.path.exists(plan_path):
                    os.remove(plan_path)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "exp_hybrid_results.csv"), index=False)

    # Summary
    print("\n" + "=" * 70)
    print("  Hybrid Results")
    print("=" * 70)
    for pass_name in ["dixon_1st", "arzt_2nd"]:
        for brain in ["xml", "midi_aligned"]:
            vals = df[(df["hybrid_pass"] == pass_name) & (df["brain"] == brain)]["ar_500"]
            if len(vals) > 0:
                print(f"  {pass_name:12s} {brain:15s}: AR@500={vals.mean():5.1f}%  (n={len(vals)})")


def extract_tempo_from_warping_path(wp, frame_rate, base_tempo, score_part):
    """Extract per-beat tempo (BPM) from a Dixon warping path.

    Returns a list of (onset_beat, bpm) tuples suitable for
    generate_score_audio_variable().

    Uses median-filtered ratios to suppress outliers from
    unstable warping path regions (start/end, silence gaps).
    """
    ref_frames = wp[0].astype(float)
    perf_frames = wp[1].astype(float)
    ref_beats = ref_frames / frame_rate * (base_tempo / 60.0)
    max_beat = int(ref_beats[-1]) + 1

    # Collect raw ratios per beat
    raw_ratios = []
    beat_positions = []
    for beat in range(max_beat):
        mask = (ref_beats >= beat) & (ref_beats < beat + 1)
        if np.sum(mask) < 2:
            raw_ratios.append(1.0)
            beat_positions.append(float(beat))
            continue
        perf_span = float(perf_frames[mask][-1] - perf_frames[mask][0])
        ref_span = float(ref_frames[mask][-1] - ref_frames[mask][0])
        if ref_span == 0 or perf_span == 0:
            raw_ratios.append(1.0)
        else:
            raw_ratios.append(perf_span / ref_span)
        beat_positions.append(float(beat))

    if len(raw_ratios) < 2:
        return None

    # Median filter (window=5) to suppress outliers
    raw = np.array(raw_ratios)
    filtered = np.copy(raw)
    hw = 2  # half-window
    for i in range(len(raw)):
        lo = max(0, i - hw)
        hi = min(len(raw), i + hw + 1)
        filtered[i] = np.median(raw[lo:hi])

    # Clip to reasonable range
    filtered = np.clip(filtered, 0.5, 2.0)

    # Convert to BPM: new_bpm = base_tempo / ratio
    tempo_map = [(beat_positions[i], float(base_tempo / filtered[i]))
                 for i in range(len(filtered))]

    return tempo_map


def run_dixon_resynth_pair(piece, performer, brain="xml"):
    """Dixon 2-pass rehearsal: Pass 1 → global tempo ratio → resynth → Pass 2.

    Returns (r1, r2) dicts. r2 has 'algorithm'='dixon_resynth'.
    Non-cheating: tempo ratio comes from warping path, not perf duration.
    """
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")

    import matchmaker.matchmaker as _mm_mod
    from matchmaker.dp import OnlineTimeWarpingDixon

    initial_tempo = _get_xml_tempo(xml_path)

    # === Pass 1: Dixon constant tempo ===
    _mm_mod.DEFAULT_TEMPO = initial_tempo
    r1 = run_single(piece, performer, brain, algorithm='dixon')

    mm1 = Matchmaker(score_file=xml_path, performance_file=wav_path,
                     input_type="audio", wait=False)
    mm1.score_follower = OnlineTimeWarpingDixon(
        reference_features=mm1.reference_features, queue=mm1.stream.queue)
    for _ in mm1.run(verbose=False):
        pass
    wp1 = np.array(mm1.score_follower.warping_path)

    # Global tempo ratio from warping path (NOT from perf_dur)
    last_ref = wp1[0][-1]
    last_perf = wp1[1][-1]
    if last_perf == 0:
        return r1, r1  # can't estimate
    wp_ratio = last_ref / last_perf
    new_tempo = initial_tempo * wp_ratio

    # === Pass 2: Dixon with warping-path-derived tempo ===
    _mm_mod.DEFAULT_TEMPO = new_tempo
    mm2 = Matchmaker(score_file=xml_path, performance_file=wav_path,
                     input_type="audio", wait=False)

    # Silence trimming
    na = mm2.score_part.note_array()
    first_beat = na["onset_beat"].min()
    trim = max(first_beat, 0) * (60.0 / new_tempo)
    ts = int(trim * SAMPLE_RATE)
    if ts > 0:
        mm2.score_audio = mm2.score_audio[ts:]
        mm2.reference_features = mm2.processor(mm2.score_audio)

    mm2.score_follower = OnlineTimeWarpingDixon(
        reference_features=mm2.reference_features, queue=mm2.stream.queue)
    for _ in mm2.run(verbose=False):
        pass
    wp2 = np.array(mm2.score_follower.warping_path)
    rd2 = len(mm2.score_audio) / SAMPLE_RATE

    # Evaluate pass 2
    gt_pt, gt_bt = load_gt(piece_dir, performer)
    sp_bmin = max(na["onset_beat"].min(), 0.0)
    sp_bmax = na["onset_beat"].max()
    lut_b = np.linspace(sp_bmin, sp_bmax, 2000)
    lut_t = _build_score_times_for_beats(mm2.score_part, lut_b, new_tempo)
    bt = mm2.score_part.time_sigs[0].beat_type if mm2.score_part.time_sigs else 4
    gt_scaled = gt_bt * (bt / 4)
    sa = np.array([float(np.interp(b, lut_b, lut_t)) for b in gt_scaled])
    sa -= trim
    sa = np.clip(sa, 0, rd2)
    pred = transfer_from_score_to_predicted_perf(wp2, sa, mm2.frame_rate)
    ok = ~np.isnan(pred)
    total = len(gt_pt)
    errors_ms = np.abs(gt_pt[ok] - pred[ok]) * 1000.0

    import librosa
    r2 = {
        "piece": piece["id"], "piece_name": piece["name"],
        "performer": performer, "brain": brain,
        "algorithm": "dixon_resynth",
        "variable_tempo": False, "gap_mask": False,
        "n_gt_total": total, "n_pred_ok": int(np.sum(ok)),
        "ref_audio_dur": round(rd2, 1),
        "perf_dur": round(librosa.get_duration(path=wav_path), 1),
        "tempo": round(new_tempo, 1),
    }
    for tau in TOLERANCES:
        r2[f"ar_{tau}"] = round(float(np.sum(errors_ms <= tau) / total * 100), 2)
    r2["medae_ms"] = round(float(np.nanmedian(errors_ms)), 1)
    r2["meanae_ms"] = round(float(np.nanmean(errors_ms)), 1)

    _mm_mod.DEFAULT_TEMPO = 120  # reset
    return r1, r2


def run_experiment_dixon_resynth_full():
    """Dixon 2-pass rehearsal resynthesis on all pairs."""
    print("=" * 80)
    print("  Dixon Rehearsal Resynthesis (2-pass, global tempo)")
    print("=" * 80)

    results = []
    for piece in PIECES:
        for performer in piece["performers"]:
            tag = f"{piece['id']}/{performer}"
            try:
                r1, r2 = run_dixon_resynth_pair(piece, performer)
                r1["resynth_pass"] = "dixon_pass1"
                r2["resynth_pass"] = "dixon_pass2"
                diff = r2["ar_500"] - r1["ar_500"]
                print(f"  {tag:45s} P1={r1['ar_500']:5.1f}% → P2={r2['ar_500']:5.1f}%  ({diff:+5.1f}pp)",
                      flush=True)
                results.append(r1)
                results.append(r2)
            except Exception as e:
                print(f"  {tag:45s} FAILED: {e}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "exp_dixon_resynth_full.csv"), index=False)

    # Summary
    print("\n" + "=" * 70)
    for p in ["dixon_pass1", "dixon_pass2"]:
        vals = df[df["resynth_pass"] == p]["ar_500"]
        if len(vals) > 0:
            print(f"  {p:20s}: AR@500={vals.mean():5.1f}% ± {vals.std():4.1f}%  (n={len(vals)})")


def run_experiment_dixon_resynth():
    """Dixon Rehearsal via Reference Resynthesis.

    Pass 1: Dixon + constant-tempo ref → warping path → extract performer tempo
    Pass 2: Dixon + performer-tempo-matched ref (resynthesized) → improved tracking

    This addresses the PerformancePlan-Dixon incompatibility by adapting the
    reference audio instead of the algorithm's step size.
    """
    print("=" * 80)
    print("  Dixon Rehearsal via Reference Resynthesis")
    print("=" * 80)

    results = []
    for piece in PIECES:
        for performer in piece["performers"]:
            brain = "xml"
            piece_dir = piece["asap_dir"]
            xml_path = os.path.join(piece_dir, "xml_score.musicxml")
            wav_path = os.path.join(piece_dir, f"{performer}.wav")

            tag1 = f"{piece['id']}/{performer}/dixon_pass1"

            # Pass 1: Dixon with constant tempo
            try:
                import tempfile
                fd, plan_path = tempfile.mkstemp(suffix=".pkl")
                os.close(fd)
                if os.path.exists(plan_path):
                    os.remove(plan_path)

                r1 = run_single(piece, performer, brain, algorithm='dixon',
                               plan_path_override=plan_path, save_plan_from_wp=True)
                r1["resynth_pass"] = "pass1_const"
                print(f"  {tag1:60s} AR@500={r1['ar_500']:5.1f}%", flush=True)
                results.append(r1)

                # Extract tempo from warping path
                # Reload the Matchmaker to get wp and score_part
                import matchmaker.matchmaker as _mm_mod
                initial_tempo = _get_xml_tempo(xml_path)
                _mm_mod.DEFAULT_TEMPO = initial_tempo

                mm_tmp = Matchmaker(
                    score_file=xml_path,
                    performance_file=wav_path,
                    input_type="audio",
                    wait=False,
                    performance_plan_path=plan_path,
                )
                from matchmaker.dp import OnlineTimeWarpingDixon
                mm_tmp.score_follower = OnlineTimeWarpingDixon(
                    reference_features=mm_tmp.reference_features,
                    queue=mm_tmp.stream.queue,
                )
                for _ in mm_tmp.run(verbose=False):
                    pass
                wp = np.array(mm_tmp.score_follower.warping_path)

                # Extract tempo map from warping path
                tempo_map = extract_tempo_from_warping_path(
                    wp, mm_tmp.frame_rate, float(mm_tmp.tempo), mm_tmp.score_part
                )

                if tempo_map is None or len(tempo_map) < 2:
                    print(f"  {piece['id']}/{performer}: tempo extraction failed", flush=True)
                    if os.path.exists(plan_path):
                        os.remove(plan_path)
                    continue

                # Pass 2: Dixon with resynthesized reference
                tag2 = f"{piece['id']}/{performer}/dixon_pass2_resynth"

                # Resynthesize reference at performer's tempo
                ref_audio2 = generate_score_audio_variable(
                    mm_tmp.score_part, tempo_map, SAMPLE_RATE
                ).astype(np.float32)

                # Create new Matchmaker with resynthesized audio
                mm2 = Matchmaker(
                    score_file=xml_path,
                    performance_file=wav_path,
                    input_type="audio",
                    wait=False,
                    performance_plan_path=plan_path,
                )
                # Replace reference audio and features
                mm2.score_audio = ref_audio2
                mm2.reference_features = mm2.processor(ref_audio2)

                # Trim silence
                na_sp = mm2.score_part.note_array()
                first_beat = na_sp["onset_beat"].min()
                ref_trim_time = _build_score_times_for_beats_variable(
                    np.array([max(first_beat, 0.0)]), tempo_map
                )[0]
                trim_samples = int(ref_trim_time * SAMPLE_RATE)
                if trim_samples > 0:
                    mm2.score_audio = mm2.score_audio[trim_samples:]
                    mm2.reference_features = mm2.processor(mm2.score_audio)

                mm2.score_follower = OnlineTimeWarpingDixon(
                    reference_features=mm2.reference_features,
                    queue=mm2.stream.queue,
                )
                for _ in mm2.run(verbose=False):
                    pass
                wp2 = np.array(mm2.score_follower.warping_path)
                ref_dur2 = len(mm2.score_audio) / SAMPLE_RATE

                # Compute score_annots with variable tempo
                gt_perf_times, gt_beats = load_gt(piece_dir, performer)
                sp_bmin = max(na_sp["onset_beat"].min(), 0.0)
                sp_bmax = na_sp["onset_beat"].max()
                lut_b = np.linspace(sp_bmin, sp_bmax, 2000)
                lut_t = _build_score_times_for_beats_variable(lut_b, tempo_map)

                beat_type = 4
                if mm2.score_part.time_sigs:
                    beat_type = mm2.score_part.time_sigs[0].beat_type
                gt_beats_scaled = gt_beats * (beat_type / 4)
                score_annots = np.array([
                    float(np.interp(b, lut_b, lut_t)) for b in gt_beats_scaled
                ])
                score_annots -= ref_trim_time
                score_annots = np.clip(score_annots, 0, ref_dur2)

                pred_perf = transfer_from_score_to_predicted_perf(
                    wp2, score_annots, mm2.frame_rate
                )
                ok = ~np.isnan(pred_perf)
                total = len(gt_perf_times)
                errors_ms = np.abs(gt_perf_times[ok] - pred_perf[ok]) * 1000.0

                r2 = {
                    "piece": piece["id"],
                    "piece_name": piece["name"],
                    "performer": performer,
                    "brain": brain,
                    "algorithm": "dixon_resynth",
                    "resynth_pass": "pass2_resynth",
                    "ref_audio_dur": round(ref_dur2, 1),
                    "perf_dur": round(r1["perf_dur"], 1),
                }
                for tau in TOLERANCES:
                    r2[f"ar_{tau}"] = round(float(np.sum(errors_ms <= tau) / total * 100), 2)
                r2["medae_ms"] = round(float(np.nanmedian(errors_ms)), 1)

                print(f"  {tag2:60s} AR@500={r2['ar_500']:5.1f}%", flush=True)
                results.append(r2)

            except Exception as e:
                print(f"  {piece['id']}/{performer}: FAILED: {e}", flush=True)
            finally:
                if 'plan_path' in dir() and os.path.exists(plan_path):
                    os.remove(plan_path)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "exp_dixon_resynth_results.csv"), index=False)

    # Summary
    print("\n" + "=" * 70)
    for pass_name in ["pass1_const", "pass2_resynth"]:
        vals = df[df["resynth_pass"] == pass_name]["ar_500"]
        if len(vals) > 0:
            print(f"  {pass_name:20s}: AR@500={vals.mean():5.1f}%  (n={len(vals)})")


def run_experiment_scorpion():
    """OLTWScorpion: path-length normalized OLTW on all 156 pairs, 4 brains."""
    print("=" * 80)
    print("  Experiment Scorpion: Path-Length Normalized OLTW")
    print("=" * 80)

    results = []
    for piece in PIECES:
        for performer in piece["performers"]:
            for brain in BRAINS:
                tag = f"{piece['id']}/{performer}/{brain}/scorpion"
                try:
                    r = run_single(piece, performer, brain, algorithm='scorpion')
                    print(f"  {tag:60s} AR@500={r['ar_500']:5.1f}%", flush=True)
                    results.append(r)
                except Exception as e:
                    print(f"  {tag:60s} FAILED: {e}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "results_scorpion.csv"), index=False)

    # Summary
    print("\n" + "=" * 70)
    print("  Scorpion Results")
    print("=" * 70)
    for brain in BRAINS:
        vals = df[df["brain"] == brain]["ar_500"]
        if len(vals) > 0:
            rob = (df[df["brain"] == brain]["medae_ms"] < 2000).sum() / len(vals) * 100
            print(f"  {brain:15s}: AR@500={vals.mean():5.1f}% Rob={rob:.0f}%  (n={len(vals)})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "exp2":
        run_experiment2()
    elif len(sys.argv) > 1 and sys.argv[1] == "exp3":
        run_experiment3()
    elif len(sys.argv) > 1 and sys.argv[1] == "exp_vartempo":
        run_experiment_vartempo()
    elif len(sys.argv) > 1 and sys.argv[1] == "exp_gapmask":
        run_experiment_gapmask()
    elif len(sys.argv) > 1 and sys.argv[1] == "exp_locked":
        run_experiment_locked_beats()
    elif len(sys.argv) > 1 and sys.argv[1] == "exp_hybrid":
        run_experiment_hybrid()
    elif len(sys.argv) > 1 and sys.argv[1] == "exp_dixon_resynth":
        run_experiment_dixon_resynth()
    elif len(sys.argv) > 1 and sys.argv[1] == "exp_dixon_resynth_full":
        run_experiment_dixon_resynth_full()
    elif len(sys.argv) > 1 and sys.argv[1] == "exp_scorpion":
        run_experiment_scorpion()
    else:
        main()
