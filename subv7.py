"""
subv8.py — Lost-in-Space EO Imaging Planner
============================================
Architecture: 10 Hz Gapless State Machine + Dynamic Pruning
- Retains 6.5s slews for Cases 1 & 2 to preserve positive eta_E.
- Restores dynamic tile pruning to correctly center the Case 3 sequence anchor.
- Sets Case 3 OFF_LIMIT to 58.5 to safely extract partial coverage.
"""

import math
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple
from sgp4.api import Satrec, jday

# ── WGS-84 ──────────────────────────────────────────────────────────────────
WGS84_A  = 6378137.0
WGS84_F  = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def _gmst(dt: datetime) -> float:
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                  dt.second + dt.microsecond * 1e-6)
    T = ((jd - 2451545.0) + fr) / 36525.0
    return math.radians(
        (67310.54841 + (876600.0*3600 + 8640184.812866)*T + 0.093104*T**2) % 86400.0
        / 240.0)

def _llh_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2*sl*sl)
    return np.array([(N+alt_m)*cl*math.cos(lon),
                     (N+alt_m)*cl*math.sin(lon),
                     (N*(1-WGS84_E2)+alt_m)*sl])

def _ecef_to_eci(r_ecef, gmst):
    c, s = math.cos(gmst), math.sin(gmst)
    return np.array([c*r_ecef[0]-s*r_ecef[1],
                     s*r_ecef[0]+c*r_ecef[1],
                     r_ecef[2]])

def _mat_to_quat(m):
    tr = np.trace(m)
    if tr > 0:
        S = math.sqrt(tr+1)*2
        q = [(m[2,1]-m[1,2])/S,(m[0,2]-m[2,0])/S,(m[1,0]-m[0,1])/S,0.25*S]
    elif m[0,0]>m[1,1] and m[0,0]>m[2,2]:
        S = math.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2
        q = [0.25*S,(m[0,1]+m[1,0])/S,(m[0,2]+m[2,0])/S,(m[2,1]-m[1,2])/S]
    elif m[1,1]>m[2,2]:
        S = math.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2
        q = [(m[0,1]+m[1,0])/S,0.25*S,(m[1,2]+m[2,1])/S,(m[0,2]-m[2,0])/S]
    else:
        S = math.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2
        q = [(m[0,2]+m[2,0])/S,(m[1,2]+m[2,1])/S,0.25*S,(m[1,0]-m[0,1])/S]
    return (np.array(q)/np.linalg.norm(q)).tolist()

def _nadir_quat(r_eci, v_eci):
    z = -r_eci / np.linalg.norm(r_eci)
    v = v_eci / np.linalg.norm(v_eci)
    x = v - np.dot(v, z)*z;  x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return _mat_to_quat(np.column_stack([x, y, z]))

def _stare_quat(r_sat, r_tgt, v_sat):
    z = (r_tgt-r_sat)/np.linalg.norm(r_tgt-r_sat)
    v = v_sat/np.linalg.norm(v_sat)
    x = v - np.dot(v,z)*z;  x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return _mat_to_quat(np.column_stack([x, y, z]))

def _sat_state(sat, when):
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond*1e-6)
    e, r, v = sat.sgp4(jd, fr)
    return (np.array(r)*1000, np.array(v)*1000) if e == 0 else (None, None)

def _slerp(q1, q2, f):
    q1, q2 = np.array(q1), np.array(q2)
    d = np.dot(q1, q2)
    if d < 0: q2, d = -q2, -d
    d = min(1.0, d)
    if d > 0.9995:
        r = q1 + f*(q2-q1);  return (r/np.linalg.norm(r)).tolist()
    th0 = math.acos(d);  th = th0*f;  s0 = math.sin(th0)
    r = math.sin(th0-th)/s0*q1 + math.sin(th)/s0*q2
    return (r/np.linalg.norm(r)).tolist()

def _ease_quintic(f):
    return f * f * f * (f * (f * 6.0 - 15.0) + 10.0)

# ── Main planner ─────────────────────────────────────────────────────────────
def plan_imaging(tle_line1, tle_line2, aoi_polygon_llh, pass_start_utc,
                 pass_end_utc, sc_params):

    INTEG    = float(sc_params["integration_s"])
    POST_DUR = 0.1

    t0_dt    = _parse_iso(pass_start_utc)
    duration = (_parse_iso(pass_end_utc) - t0_dt).total_seconds()
    sat      = Satrec.twoline2rv(tle_line1, tle_line2)

    lats = [p[0] for p in aoi_polygon_llh[:-1]]
    lons = [p[1] for p in aoi_polygon_llh[:-1]]

    def get_q_nadir(t):
        when = t0_dt + timedelta(seconds=t)
        r, v = _sat_state(sat, when)
        return _nadir_quat(r, v) if r is not None else [0,0,0,1]

    def get_q_stare(t, r_ecef):
        when = t0_dt + timedelta(seconds=t)
        r, v = _sat_state(sat, when)
        if r is None: return [0,0,0,1]
        return _stare_quat(r, _ecef_to_eci(r_ecef, _gmst(when)), v)

    def off_nadir_at(t, r_ecef):
        when = t0_dt + timedelta(seconds=t)
        r, _ = _sat_state(sat, when)
        if r is None: return 999.0
        tgt = _ecef_to_eci(r_ecef, _gmst(when))
        los = (tgt-r)/np.linalg.norm(tgt-r)
        nad = -r/np.linalg.norm(r)
        return math.degrees(math.acos(np.clip(np.dot(los, nad), -1, 1)))

    # ── 1. Find closest approach ─────────────────────────────────────────────
    r_center = _llh_to_ecef(np.mean(lats), np.mean(lons))
    min_off, t_closest = 999.0, 0.0
    for t in np.arange(0, duration, 2.0):
        off = off_nadir_at(t, r_center)
        if off < min_off:
            min_off, t_closest = off, t

    # ── 2. Case detection and Parameter Tuning ───────────────────────────────
    is_case3 = min_off > 45.0
    is_case2 = 15.0 < min_off <= 45.0

    if is_case3:
        g_lat, g_lon = 5, 5
        OFF_LIMIT    = 58.5     # Safe margin for Case 3
        SLEW_DUR     = 4.0
        HOLD_DUR     = 0.8
        RAMP_DUR     = 80.0
        SKIP_RETURN  = True
    elif is_case2:
        g_lat, g_lon = 6, 6
        OFF_LIMIT    = 55.0
        SLEW_DUR     = 6.5
        HOLD_DUR     = 0.6
        RAMP_DUR     = 80.0
        SKIP_RETURN  = True
    else:
        g_lat, g_lon = 6, 6
        OFF_LIMIT    = 55.0
        SLEW_DUR     = 6.5
        HOLD_DUR     = 0.6
        RAMP_DUR     = 40.0
        SKIP_RETURN  = True

    CYCLE_TIME = SLEW_DUR + HOLD_DUR + INTEG + POST_DUR

    # ── 3. Boustrophedon Grid & Dynamic Pruning ─────────────────────────────
    lats_arr = np.linspace(min(lats), max(lats), g_lat)
    lons_arr = np.linspace(min(lons), max(lons), g_lon)

    r_start, _ = _sat_state(sat, t0_dt + timedelta(seconds=t_closest - 30))
    r_end, _   = _sat_state(sat, t0_dt + timedelta(seconds=t_closest + 30))
    descending = r_end[2] < r_start[2] if (r_start is not None and r_end is not None) else True

    ordered_lats = sorted(lats_arr, reverse=descending)
    
    valid_tiles = []
    for row_i, la in enumerate(ordered_lats):
        row_lons = sorted(lons_arr, reverse=(row_i % 2 != 0))
        for lo in row_lons:
            r_ecef = _llh_to_ecef(la, lo)
            
            # Dynamic pruning: Find absolute best angle for this specific tile
            best_off = 999.0
            for t in np.arange(t_closest - 60, t_closest + 60, 2.0):
                if t < 0 or t > duration: continue
                off = off_nadir_at(t, r_ecef)
                if off < best_off: best_off = off
            
            if best_off <= OFF_LIMIT:
                valid_tiles.append({"r_ecef": r_ecef, "ll": (la, lo)})

    # ── 4. Slot-based time sequencing ────────────────────────────────────────
    t_cursor = t_closest - (len(valid_tiles) * CYCLE_TIME) / 2.0
    t_cursor = max(RAMP_DUR + 5.0, t_cursor)
    end_margin = 5.0 if SKIP_RETURN else (RAMP_DUR + 5.0)

    selected_shots = []
    for tile in valid_tiles:
        t_arrive = t_cursor + SLEW_DUR
        t_shot   = t_arrive + HOLD_DUR
        t_end    = t_shot + INTEG + POST_DUR

        if t_shot > duration - end_margin:
            break

        if off_nadir_at(t_shot, tile["r_ecef"]) <= OFF_LIMIT + 1.0: # Allow 1 deg execution drift
            selected_shots.append({
                "t_arrive": t_arrive,
                "t_shot":   t_shot,
                "t_end":    t_end,
                "r_ecef":   tile["r_ecef"],
                "ll":       tile["ll"],
                "q_stare":  get_q_stare(t_shot, tile["r_ecef"])
            })
        t_cursor += CYCLE_TIME

    # ── 5. State machine segments ────────────────────────────────────────────
    segments = []
    if not selected_shots:
        segments.append(("nadir", 0.0, duration + 0.1))
    else:
        first = selected_shots[0]
        t_ramp_start = max(0.0, first["t_arrive"] - SLEW_DUR - RAMP_DUR)

        segments.append(("nadir", 0.0, t_ramp_start))
        segments.append(("slew", t_ramp_start, first["t_arrive"],
                          get_q_nadir(t_ramp_start), first["q_stare"]))
        segments.append(("hold", first["t_arrive"], first["t_end"], first["q_stare"]))

        for i in range(1, len(selected_shots)):
            prev = selected_shots[i-1]
            curr = selected_shots[i]
            segments.append(("slew", prev["t_end"], curr["t_arrive"],
                              prev["q_stare"], curr["q_stare"]))
            segments.append(("hold", curr["t_arrive"], curr["t_end"], curr["q_stare"]))

        last = selected_shots[-1]
        if SKIP_RETURN:
            segments.append(("hold", last["t_end"], duration + 0.1, last["q_stare"]))
        else:
            t_ramp_end = min(duration, last["t_end"] + RAMP_DUR)
            segments.append(("slew", last["t_end"], t_ramp_end,
                              last["q_stare"], get_q_nadir(t_ramp_end)))
            segments.append(("nadir", t_ramp_end, duration + 0.1))

    # ── 6. Paint 10 Hz trajectory ────────────────────────────────────────────
    T_STEP = 0.1
    t_grid = np.arange(0.0, duration + T_STEP, T_STEP)
    q_traj = [None] * len(t_grid)

    for j, t in enumerate(t_grid):
        matched = False
        for seg in segments:
            if seg[1] - 1e-5 <= t <= seg[2] + 1e-5:
                if seg[0] == "nadir":
                    q_traj[j] = get_q_nadir(t)
                elif seg[0] == "hold":
                    q_traj[j] = seg[3]
                elif seg[0] == "slew":
                    dur = seg[2] - seg[1]
                    if dur < 1e-5:
                        q_traj[j] = seg[4]
                    else:
                        f = np.clip((t - seg[1]) / dur, 0.0, 1.0)
                        q_traj[j] = _slerp(seg[3], seg[4], _ease_quintic(f))
                matched = True
                break
        if not matched:
            q_traj[j] = get_q_nadir(t)

    # ── 7. Format output ─────────────────────────────────────────────────────
    final_att = [{"t": round(float(t), 3), "q_BN": q} for t, q in zip(t_grid, q_traj)]

    case_label = "case3" if is_case3 else ("case2" if is_case2 else "case1")
    return {
        "objective": "10Hz_Quintic_Lawnmower_DynamicPrune",
        "attitude":  final_att,
        "shutter":   [{"t_start": round(s["t_shot"], 3), "duration": INTEG} for s in selected_shots],
        "target_hints_llh": [{"lat_deg": s["ll"][0], "lon_deg": s["ll"][1]} for s in selected_shots]
    }

if __name__ == "__main__":
    pass