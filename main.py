"""
Planet simulation: an N-body model of the inner + outer solar system.

Units: distance in AU, time in years, mass in solar masses.
In these units the gravitational constant is G = 4*pi^2.

Integrator: velocity Verlet (symplectic), so orbits stay stable over long runs.

Run:   python planet_sim.py            -> live animation
       python planet_sim.py --save     -> writes planet_sim.gif
Needs: numpy, matplotlib  (pip install numpy matplotlib)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 4 * np.pi**2          # AU^3 / (Msun * yr^2)
SOFTENING = 1e-4          # avoids division blow-ups at tiny separations

# name, mass (Msun), orbital radius (AU), color, marker size
BODIES = [
    ("Sun",     1.0,        0.0,   "#FDB813", 18),
    ("Mercury", 1.66e-7,    0.387, "#9E9E9E", 3),
    ("Venus",   2.45e-6,    0.723, "#E8CDA2", 5),
    ("Earth",   3.00e-6,    1.000, "#3E7BFA", 5),
    ("Mars",    3.23e-7,    1.524, "#C1440E", 4),
    ("Jupiter", 9.55e-4,    5.203, "#D8A26B", 11),
    ("Saturn",  2.86e-4,    9.537, "#E3C987", 9),
    ("Uranus",  4.37e-5,   19.191, "#7FDBFF", 8),
    ("Neptune", 5.15e-5,   30.069, "#4B70DD", 8),
]
N_INNER = 5   # Sun..Mars, shown in the zoomed inset


def initial_conditions(bodies, seed=1):
    """Place each planet on a circular orbit at a random phase."""
    rng = np.random.default_rng(seed)
    n = len(bodies)
    mass = np.array([b[1] for b in bodies])
    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    for i, (_, _, r, _, _) in enumerate(bodies):
        if r == 0:
            continue
        theta = rng.uniform(0, 2 * np.pi)
        v = np.sqrt(G * mass[0] / r)                 # circular speed
        pos[i] = r * np.array([np.cos(theta), np.sin(theta)])
        vel[i] = v * np.array([-np.sin(theta), np.cos(theta)])
    # shift to centre-of-mass frame so the system doesn't drift
        pos -= (mass[:, None] * pos).sum(0) / mass.sum()
        vel -= (mass[:, None] * vel).sum(0) / mass.sum()
    return mass, pos, vel


def accelerations(pos, mass):
    """Pairwise Newtonian gravity, vectorised."""
    diff = pos[None, :, :] - pos[:, None, :]          # r_j - r_i
    dist2 = (diff**2).sum(-1) + SOFTENING**2
    inv_d3 = dist2**-1.5
    np.fill_diagonal(inv_d3, 0.0)
    return G * (diff * (mass[None, :, None] * inv_d3[:, :, None])).sum(1)


def total_energy(pos, vel, mass):
    ke = 0.5 * (mass * (vel**2).sum(1)).sum()
    diff = pos[None] - pos[:, None]
    d = np.sqrt((diff**2).sum(-1) + SOFTENING**2)
    pair = G * mass[:, None] * mass[None, :] / d   # n×n table: G·mᵢ·mⱼ / rᵢⱼ
    np.fill_diagonal(pair, 0)                      # remove each body paired with itself
    pe = -pair.sum() / 2                           # every pair was counted twice
    return ke + pe

class Simulation:
    def __init__(self, bodies, dt=0.001):
        self.mass, self.pos, self.vel = initial_conditions(bodies)
        self.dt = dt
        self.t = 0.0
        self.acc = accelerations(self.pos, self.mass)
        self.e0 = total_energy(self.pos, self.vel, self.mass)

    def step(self, n=1):
        dt = self.dt
        for _ in range(n):
            self.vel += 0.5 * dt * self.acc
            self.pos += dt * self.vel
            self.acc = accelerations(self.pos, self.mass)
            self.vel += 0.5 * dt * self.acc
            self.t += dt

    def energy_drift(self):
        return abs((total_energy(self.pos, self.vel, self.mass) - self.e0) / self.e0)


def run(save=False):
    sim = Simulation(BODIES, dt=0.001)
    # simulation speed: integration steps per frame (each step = 0.001 yr).
    # 10 -> 0.01 yr per frame, i.e. Earth takes ~100 frames (~2 s) per orbit.
    # In the live window: Up/+ = faster, Down/- = slower, Space = pause.
    speed = {"steps": 10, "paused": False}
    trail_len = 600

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    lim = 32
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")

    # faint reference orbits so the far-out planets are easy to follow
    t = np.linspace(0, 2 * np.pi, 300)
    for (_, _, r, color, _) in BODIES[1:]:
        ax.plot(r * np.cos(t), r * np.sin(t), color=color, lw=0.4, alpha=0.18)

    label_kw = dict(fontsize=9, fontweight="bold", va="bottom", ha="left", zorder=4)
    dots, trails, labels, history = [], [], [], []
    for (name, _, _, color, size) in BODIES:
        (d,) = ax.plot([], [], "o", color=color, ms=size, zorder=3)
        (tr,) = ax.plot([], [], "-", color=color, lw=0.8, alpha=0.5)
        labels.append(ax.text(0, 0, name, color=color, **label_kw))
        dots.append(d)
        trails.append(tr)
        history.append([])
    # inner planets are too crowded at this scale: label them in the inset only
    for lbl in labels[1:N_INNER]:
        lbl.set_visible(False)
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="white",
                   va="top", family="monospace", fontsize=10)

    # inset showing the inner planets up close
    inset = fig.add_axes([0.66, 0.63, 0.32, 0.32], facecolor="#0b0b16")
    inset.set_xlim(-1.8, 1.8)
    inset.set_ylim(-1.8, 1.8)
    inset.set_aspect("equal")
    inset.set_xticks([]); inset.set_yticks([])
    for s in inset.spines.values():
        s.set_color("#555")
    inset.set_title("inner planets (zoomed)", color="#bbb", fontsize=9)
    in_dots = [inset.plot([], [], "o", color=c, ms=max(s * 0.6, 2))[0]
               for (_, _, _, c, s) in BODIES[:N_INNER]]
    in_trails = [inset.plot([], [], "-", color=c, lw=0.6, alpha=0.5)[0]
                 for (_, _, _, c, _) in BODIES[:N_INNER]]
    in_labels = [inset.text(0, 0, n, color=c, fontsize=7, fontweight="bold",
                            va="bottom", ha="left", zorder=4)
                 for (n, _, _, c, _) in BODIES[:N_INNER]]

    def on_key(event):
        if event.key in ("up", "+", "="):
            speed["steps"] = min(speed["steps"] * 2, 640)
        elif event.key in ("down", "-"):
            speed["steps"] = max(speed["steps"] // 2, 1)
        elif event.key == " ":
            speed["paused"] = not speed["paused"]
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        if not speed["paused"]:
            sim.step(speed["steps"])
        for i, p in enumerate(sim.pos):
            history[i].append(p.copy())
            if len(history[i]) > trail_len:
                history[i].pop(0)
            h = np.array(history[i])
            dots[i].set_data([p[0]], [p[1]])
            trails[i].set_data(h[:, 0], h[:, 1])
            labels[i].set_position((p[0] + 0.6, p[1] + 0.6))
            if i < N_INNER:
                in_dots[i].set_data([p[0]], [p[1]])
                in_trails[i].set_data(h[-40:, 0], h[-40:, 1])
                in_labels[i].set_position((p[0] + 0.07, p[1] + 0.07))
        info.set_text(f"t = {sim.t:6.2f} yr\n"
                      f"speed = {speed['steps'] * 0.001:.3f} yr/frame"
                      f"{'  [paused]' if speed['paused'] else ''}\n"
                      f"Up/Down: speed   Space: pause")
        return dots + trails + labels + in_dots + in_trails + in_labels + [info]

    if save:
        anim = FuncAnimation(fig, update, frames=300, interval=30, blit=False)
        anim.save("planet_sim.gif", writer="pillow", fps=30, dpi=80)
        print(f"saved planet_sim.gif  (t={sim.t:.1f} yr, "
              f"energy drift {sim.energy_drift():.1e})")
    else:
        anim = FuncAnimation(fig, update, interval=20, blit=False,
                             cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    run(save="--save" in sys.argv)