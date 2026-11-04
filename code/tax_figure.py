"""Social-post figure: before -> after the published weight edit, % correct, per task (dumbbell chart)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = sys.argv[1]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
GAIN, LOSS, SAME = "#2a78d6", "#e34948", "#898781"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

# (kind, label, before %, after %, effect); exact counts in comments
ROWS = [
    ("head", "What the edit was for"),
    ("row", "MATH-500 (original post)", 7.9, 48.1, "gain"),
    ("head", "Where it pays"),
    ("row", "Copying 30–60 lines of code exactly", 100.0, 25 / 38 * 100, "loss"),   # 38/38 -> 25/38
    ("row", "Continuing a simple template", 17 / 46 * 100, 4 / 46 * 100, "loss"),       # 17/46 -> 4/46
]
COLOR = {"gain": GAIN, "loss": LOSS, "same": SAME}

fig = plt.figure(figsize=(8, 4.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax = fig.add_axes([0.42, 0.17, 0.44, 0.56])
ax.set_facecolor(SURFACE)

y, ys = 0.0, []
for r in ROWS:
    if r[0] == "head":
        y += 0.35 if ys else 0.0
        ys.append(y)
        y += 0.75
    else:
        ys.append(y)
        y += 1.0
ymax = y - 0.5

for gx in (0, 25, 50, 75, 100):
    ax.axvline(gx, color=GRID, lw=0.8, zorder=0)
    ax.text(gx, ymax + 0.35, f"{gx}%", ha="center", va="top", fontsize=10, color=MUTED)

for r, yy in zip(ROWS, ys):
    if r[0] == "head":
        ax.text(-4, yy, r[1].upper(), ha="right", va="center", fontsize=9.5, color=MUTED,
                fontweight="bold", clip_on=False)
        continue
    _, label, b, a, eff = r
    c = COLOR[eff]
    fmt = (lambda v: f"{v:.1f}%") if label.startswith("MATH") else (lambda v: f"{round(v):.0f}%")
    ax.text(-4, yy, label, ha="right", va="center", fontsize=13, color=INK, clip_on=False)
    if eff == "same":
        # unaffected: one marker at the after value, numbers to its left
        ax.scatter([a], [yy], s=66, facecolor=c, edgecolor=SURFACE, linewidth=2, zorder=4)
        txt = fmt(a) if round(a) == round(b) else f"{fmt(b)} → {fmt(a)}"
        ax.text(a - 3, yy, txt, ha="right", va="center", fontsize=8.5, color=INK2)
    else:
        ax.annotate("", xy=(a, yy), xytext=(b, yy),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=3, shrinkA=8, shrinkB=9, mutation_scale=15),
                    zorder=2)
        ax.scatter([b], [yy], s=95, facecolor=SURFACE, edgecolor=INK2, linewidth=1.8, zorder=3)
        ax.scatter([a], [yy], s=105, facecolor=c, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(b, yy - 0.36, fmt(b), ha="center", va="bottom", fontsize=11, color=INK2)
        ax.text(a, yy - 0.36, fmt(a), ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")
    d = a - b
    dtxt = "no change" if eff == "same" and abs(d) < 0.5 else f"{d:+.0f} pts".replace("-", "−")
    ax.text(107, yy, dtxt, ha="left", va="center", fontsize=13, clip_on=False,
            color=INK if eff != "same" else INK2, fontweight="bold" if eff != "same" else "normal")

ax.set_xlim(-1, 101)
ax.set_ylim(ymax + 0.2, -0.6)
ax.axis("off")

fig.text(0.04, 0.925, "Weight-space programming has a tax", fontsize=21, fontweight="bold", color=INK)
fig.text(0.04, 0.855, "Qwen3.5-0.8B before and after the published weight edit, % correct",
         fontsize=12.5, color=INK2)
handles = [Line2D([], [], ls="", marker="o", ms=7, mfc=SURFACE, mec=INK2, mew=1.4, label="Before the edit"),
           Line2D([], [], ls="", marker="o", ms=7.5, mfc=INK2, mec=SURFACE, mew=1.5, label="After the edit")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.83), frameon=False, fontsize=11,
           labelcolor=INK2, handletextpad=0.3, columnspacing=1.2, ncol=2)
fig.text(0.04, 0.058, "MATH-500 as reported by Takehi et al. (2026). Other tasks: our tests on the released weights "
         "(greedy decoding, thinking off, exact match).", fontsize=8.5, color=MUTED)
fig.text(0.04, 0.025, "From “No Free Lunch: The Tax of Weight-Space Programming”, Jinhong Lin, 2026",
         fontsize=8.5, color=MUTED)
fig.savefig(OUT, facecolor=SURFACE)
print("saved", OUT)
