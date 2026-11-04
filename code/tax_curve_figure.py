"""Social-post figure 2: as the edit's strength grows, math accuracy rises while copying accuracy falls."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, RED = "#2a78d6", "#e34948"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

c = [0, 1, 2, 3, 4, 5, 6]
series = [  # name, values (% correct), color, dash, end-label dy (points)
    ("Math (GSM8K)", [10, 24, 33, 48, 56, 65, 55], BLUE, "-", -2),
    ("Copying long code", [100, 100, 100, 100, 35 / 38 * 100, 32 / 38 * 100, 25 / 38 * 100], RED, "-", 10),
    ("Continuing a template", [17 / 46 * 100, 16 / 46 * 100, 15 / 46 * 100, 11 / 46 * 100, 9 / 46 * 100,
                               8 / 46 * 100, 4 / 46 * 100], RED, "--", 0),
]

fig = plt.figure(figsize=(8, 4.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax = fig.add_axes([0.085, 0.25, 0.62, 0.55])
ax.set_facecolor(SURFACE)
for v in (0, 25, 50, 75, 100):
    ax.axhline(v, color=GRID, lw=0.8, zorder=0)
for name, vals, col, ls, dy in series:
    ax.plot(c, vals, color=col, lw=3, ls=ls, zorder=2, solid_capstyle="round")
    ax.scatter(c, vals, s=42, color=col, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.annotate(f"{round(vals[0])}%", (c[0], vals[0]), xytext=(9, 9 if name != "Math (GSM8K)" else -16),
                textcoords="offset points", fontsize=11, color=INK, fontweight="bold")
    ax.annotate(f"{name}  {round(vals[-1])}%", (c[-1], vals[-1]), xytext=(12, dy), textcoords="offset points",
                va="center", fontsize=12.5, color=INK, annotation_clip=False)
ax.set_xlim(-0.2, 6.2)
ax.set_ylim(0, 108)
ax.set_xticks(c)
ax.set_xticklabels(["0\noriginal", "1", "2", "3", "4", "5", "6\nreleased"], fontsize=11, color=INK2)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.5, color=MUTED)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("Strength of the weight edit", fontsize=11.5, color=INK2, labelpad=6)

fig.text(0.04, 0.925, "Every strength that improves math also hurts copying",
         fontsize=19, fontweight="bold", color=INK)
fig.text(0.04, 0.862, "Qwen3.5-0.8B, % correct at each strength of the same weight edit", fontsize=12, color=INK2)
fig.text(0.04, 0.055, "Math: 100 GSM8K training problems, thinking on. Copying: 38 long functions; templates: 46 prompts "
         "(greedy, thinking off, exact match).", fontsize=8.5, color=MUTED)
fig.text(0.04, 0.022, "From “No Free Lunch: The Tax of Weight-Space Programming”, Jinhong Lin, 2026",
         fontsize=8.5, color=MUTED)
fig.savefig(OUT, facecolor=SURFACE)
print("saved", OUT)
