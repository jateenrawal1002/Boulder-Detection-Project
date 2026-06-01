import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- CONFIG ----------
location = r"C:\Users\Admin\Desktop\Internship Documents\Open CV Documents\Boulder detection\Boulder Detection\Boulder 29\Front View\data_20251013_150036.csv"
tid = 0          # choose ID; if not present, script will choose the first available
win = 7          # desired odd smoothing window (will shrink if needed)
sigma = 3.5      # outlier threshold for vy

# ---------- LOAD & NORMALIZE COLUMNS ----------
df = pd.read_csv(location, sep=";")
df.columns = df.columns.str.strip()

# unify velocity magnitude column name if needed
rename_map = {}
for col in list(df.columns):
    if col.lower() == "speed(m/s)":
        rename_map[col] = "speed(m/s)"
    if col.lower() == "velocity(m/s)":
        rename_map[col] = "speed(m/s)"
df = df.rename(columns=rename_map)

# type coercion (only if columns exist)
if "ID" in df.columns: df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
if "Time(s)" in df.columns: df["Time(s)"] = pd.to_numeric(df["Time(s)"], errors="coerce")
for c in ["vx(m/s)", "vy(m/s)", "Speed(m/s)"]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- PICK/VALIDATE ID ----------
available_ids = df["ID"].dropna().unique()
if available_ids.size == 0:
    raise SystemExit("No IDs found in CSV. Check the file/content.")

if tid not in available_ids:
    print(f"[info] Requested ID {tid} not in file. Using first available ID: {int(available_ids[0])}")
    tid = int(available_ids[0])

d = df[df["ID"] == tid].sort_values("Time(s)").copy()
d = d.dropna(subset=["Time(s)"])  # ensure we have times

if d.empty:
    raise SystemExit(f"Selected ID {tid} has no rows.")

# ensure needed columns exist
need_cols = ["vx(m/s)", "vy(m/s)"]
for c in need_cols:
    if c not in d.columns:
        raise SystemExit(f"Column '{c}' not found in CSV. Columns are: {list(df.columns)}")

# ---------- EXTRACT ARRAYS ----------
t  = d["Time(s)"].values
vx = d["vx(m/s)"].values
vy = d["vy(m/s)"].values
spd = d["Speed(m/s)"].values if "Speed(m/s)" in d.columns else np.sqrt(vx**2 + vy**2)

# guard: if too few points
if len(t) < 3:
    raise SystemExit(f"Not enough samples for smoothing/fit (len={len(t)}).")

# ---------- OUTLIER MASK (vy); keep at least 3 points ----------
def mask_spikes(y, sigma=3.5):
    m = np.nanmedian(y); s = np.nanstd(y) + 1e-12
    return np.abs(y - m) < sigma * s

mask = mask_spikes(vy, sigma)
if mask.sum() < 3:
    print("[warn] Outlier filter removed too many points; disabling spike removal.")
else:
    t, vx, vy, spd = t[mask], vx[mask], vy[mask], spd[mask]

# ---------- MOVING AVERAGE (centered) ----------
def moving_average_centered(y, k):
    # require odd k and k <= len(y)
    k = int(k)
    if k % 2 == 0: k += 1
    k = max(1, min(k, len(y) if len(y)%2==1 else len(y)-1))
    if k < 1: k = 1
    if k == 1: return y.copy()
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(ypad, ker, mode="valid")

# auto-shrink window if needed
win_eff = win
if win_eff > len(t):
    win_eff = len(t) if len(t)%2==1 else len(t)-1
    if win_eff < 1: win_eff = 1
    print(f"[info] Smoothing window reduced to {win_eff} due to short signal.")

vx_s  = moving_average_centered(vx, win_eff)
vy_s  = moving_average_centered(vy, win_eff)
spd_s = moving_average_centered(spd, win_eff)

# ---------- LINEAR FIT vy(t) ----------
if len(t) >= 2 and len(vy_s) == len(t):
    a, b = np.polyfit(t, vy_s, 1)  # vy ≈ a*t + b
    vy_fit = a * t + b
    print(f"ID {tid}: vy slope = {a:.3f} m/s^2  ->  g_est ≈ {-a:.3f} m/s^2")

else:
    vy_fit = None
    print("[warn] Not enough aligned points for linear fit.")

# ---------- PLOTS ----------
plt.figure(figsize=(10,5))
plt.plot(t, vy, '.', alpha=0.35, label="vy raw")
plt.plot(t, vy_s, '-', lw=2, label=f"vy smoothed (win={win_eff})")
if vy_fit is not None:
    plt.plot(t, vy_fit, '--', lw=2, label=f"linear fit (slope={a:.2f})")
plt.xlabel("Time (s)"); plt.ylabel("Vertical velocity vy (m/s)")
plt.title(f"ID {tid} — vy vs time (smoothed & fitted)")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, vx, '.', alpha=0.35, label="vx raw")
plt.plot(t, vx_s, '-', lw=2, label=f"vx smoothed (win={win_eff})")
plt.xlabel("Time (s)"); plt.ylabel("vx (m/s)")
plt.title(f"ID {tid} — vx vs time")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, spd, '.', alpha=0.35, label="speed raw")
plt.plot(t, spd_s, '-', lw=2, label=f"speed smoothed (win={win_eff})")
plt.xlabel("Time (s)"); plt.ylabel("|v| (m/s)")
plt.title(f"ID {tid} — speed vs time")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()
