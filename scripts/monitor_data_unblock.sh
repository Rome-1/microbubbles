#!/usr/bin/env bash
# Monitor for the 215-acq detection-data unblock (bead mb-4yw / upstream issue
# alephneuro/microbubbles#2). Fires exactly ONCE, then goes quiet (marker file).
#
# The microbubble Kalman tracker (scripts/wf_render_signal/track_bubbles.py, track_kf)
# is validated on acq-0 and ready to scale to all 216 acquisitions — it is blocked only
# on the raw per-frame detections for the other 215 acqs. This watches for that data.
#
# Signals (any one triggers):
#   1. local detections now span >1 acquisition (the file gained more acqs), or a new
#      detection file appears under outputs/;
#   2. new comment on, or closure of, upstream issue #2 (Aleph responds with data/link).
# On trigger: log, comment on mb-4yw, bump it to P0. Installed via crontab (persistent).
set -u
REPO=/home/rome/gt/microbubbles/crew/cajal
STATE="$REPO/outputs/reference/.data-monitor"
LOG="$STATE/monitor.log"
MARK="$STATE/TRIGGERED"
export PATH="/home/rome/.local/bin:/home/linuxbrew/.linuxbrew/bin:/usr/bin:/bin:$PATH"
mkdir -p "$STATE"
cd "$REPO" 2>/dev/null || exit 1
[ -f "$MARK" ] && exit 0                       # already fired — stop

ts=$(date -Iseconds)
base=$(cat "$STATE/baseline_comments" 2>/dev/null || echo 2)

# signal 1: how many acquisitions do the local detections cover?
acqs=$(nice -n 15 env OMP_NUM_THREADS=1 python3 - <<'PY' 2>/dev/null
import sys, numpy as np
sys.path.insert(0, "scripts/wf_render_signal")
try:
    from render_flow_diversity import SU
    o = SU(open("outputs/reference/full_tracks_smoothed.pkl", "rb")).load()
    print(len(np.unique(np.asarray(o["detections"]["acq_indices"]))))
except Exception:
    print(-1)
PY
)
# signal 1b: a new detection file besides the known one
newfiles=$(find "$REPO/outputs" -type f \( -iname '*detection*' -o -iname '*dets*' \) 2>/dev/null \
           | grep -v 'full_tracks_smoothed' | head -3 | tr '\n' ' ')
# signal 2: upstream issue #2 activity
cc=$(gh issue view 2 --repo alephneuro/microbubbles --json comments -q '.comments|length' 2>/dev/null)
stt=$(gh issue view 2 --repo alephneuro/microbubbles --json state -q '.state' 2>/dev/null)

trig=""
[ "${acqs:-1}" -gt 1 ] 2>/dev/null && trig="local detections span ${acqs} acqs"
[ -n "$newfiles" ] && trig="${trig:+$trig; }new detection file(s): ${newfiles}"
{ [ -n "${cc:-}" ] && [ "$cc" -gt "$base" ] 2>/dev/null; } && trig="${trig:+$trig; }upstream issue #2 new comment(s) (${cc}>${base})"
[ "${stt:-OPEN}" = "CLOSED" ] && trig="${trig:+$trig; }upstream issue #2 CLOSED"

echo "$ts acqs=${acqs} issue_comments=${cc:-NA} state=${stt:-NA} newfiles='${newfiles}' trig='${trig}'" >> "$LOG"

if [ -n "$trig" ]; then
  touch "$MARK"
  echo "$ts *** TRIGGERED: $trig" >> "$LOG"
  bd comment mb-4yw "DATA-MONITOR TRIGGERED ($ts): $trig. The Kalman tracker (scripts/wf_render_signal/track_bubbles.py, track_kf) is validated on acq-0 and READY TO SCALE — run it on all 216 acqs now. See docs/bubble-tracking-acq0.md." >/dev/null 2>&1
  bd update mb-4yw --priority 0 >/dev/null 2>&1
fi
