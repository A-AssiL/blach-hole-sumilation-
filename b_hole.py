"""
GPU Schwarzschild black-hole ray tracer (PyTorch).
Backward ray tracing: one photon per pixel, integrating the real
null geodesic  d2u/dphi2 = -u + 3 M u^2   in each photon's orbital plane.
Units: G = c = 1  ->  r_s = 2M.
"""
import torch, numpy as np, imageio.v2 as imageio, math

device = "cuda" if torch.cuda.is_available() else "cpu"
dt     = torch.float32
print("Running on:", device)


import os

print("Script file:", os.path.abspath(__file__))
print("Working directory:", os.getcwd())

# ----------------------- black hole + scene -----------------------
M      = 1.0
r_s    = 2.0 * M          # event horizon
R_IN   = 3.0 * M          # inner edge of accretion disk (ISCO-ish)
R_OUT  = 12.0 * M         # outer edge
R_ESC  = 60.0             # ray considered escaped beyond this

W, H   = 1280, 720        # render resolution (lower if low on VRAM)
FOV    = math.radians(55)
N_STEP = 550              # geodesic integration steps
DPHI   = 0.02             # step in phi (radians)

# ----------------------- procedural starfield ---------------------
def make_background(hh=1024, ww=2048):
    torch.manual_seed(7)
    bg = torch.zeros(hh, ww, 3)
    # faint nebula gradient
    yv = torch.linspace(0, 1, hh)[:, None]
    bg += torch.stack([0.02+0.03*yv.repeat(1, ww),
                       0.01+0.02*yv.repeat(1, ww),
                       0.05+0.08*yv.repeat(1, ww)], -1)
    # random stars
    idx = torch.randint(0, hh*ww, (12000,))
    bright = torch.rand(12000)**8 * 3.0
    flat = bg.view(-1, 3)
    flat[idx] += bright[:, None]
    return bg.clamp(0, 1).to(device)

BG = make_background()

def sample_bg(dirs):
    # equirectangular lookup by ray direction
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    u = (torch.atan2(z, x) / (2*math.pi) + 0.5) % 1.0
    v = torch.acos(y.clamp(-1, 1)) / math.pi
    H0, W0 = BG.shape[:2]
    iu = (u * (W0 - 1)).long()
    iv = (v * (H0 - 1)).long()
    return BG[iv, iu]

# ----------------------- disk color (by temperature) --------------
def disk_color(r, doppler):
    t = ((R_OUT - r) / (R_OUT - R_IN)).clamp(0, 1)      # hotter inward
    # warm blackbody-ish ramp: deep orange -> white
    col = torch.stack([0.9*t + 0.1,
                       0.5*t**1.5,
                       0.15*t**3], -1)
    bright = (1.5 / r.clamp(min=R_IN)).unsqueeze(-1)
    return (col * bright * doppler.unsqueeze(-1)).clamp(0, 4)

# ----------------------- camera ray setup -------------------------
def camera_rays(cam_pos):
    forward = -cam_pos / cam_pos.norm()
    up0 = torch.tensor([0., 1., 0.], device=device)
    right = torch.cross(forward, up0); right /= right.norm()
    up = torch.cross(right, forward)
    j, i = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device), indexing="ij")
    aspect = W / H
    px = (2*(i+0.5)/W - 1) * aspect * math.tan(FOV/2)
    py = (1 - 2*(j+0.5)/H) * math.tan(FOV/2)
    d = (px[..., None]*right + py[..., None]*up + forward)
    return (d / d.norm(dim=-1, keepdim=True)).reshape(-1, 3)

# ----------------------- the GPU ray tracer -----------------------
def render(cam_pos):
    cam_pos = cam_pos.to(device)
    d = camera_rays(cam_pos)                    # (N,3) ray dirs
    N = d.shape[0]

    r0 = cam_pos.norm()
    e1 = (cam_pos / r0).expand(N, 3)            # radial basis (to camera)
    cr = (d * e1).sum(-1)                        # radial comp of ray
    dperp = d - cr[:, None]*e1
    cp = dperp.norm(dim=-1).clamp(min=1e-6)
    e2 = dperp / cp[:, None]                     # in-plane perpendicular basis

    u  = torch.full((N,), 1.0/r0, device=device)     # u = 1/r
    w  = -u * cr / cp                                 # du/dphi initial
    phi = torch.zeros(N, device=device)

    color  = torch.zeros(N, 3, device=device)
    active = torch.ones(N, dtype=torch.bool, device=device)

    def position(u, phi):
        r = 1.0/u
        return r[:, None]*(torch.cos(phi)[:, None]*e1 +
                           torch.sin(phi)[:, None]*e2), r

    pos_prev, _ = position(u, phi)

    def deriv(u, w):
        return w, (-u + 3.0*M*u*u)

    for _ in range(N_STEP):
        # ---- RK4 step of the geodesic equation ----
        k1u, k1w = deriv(u, w)
        k2u, k2w = deriv(u + 0.5*DPHI*k1u, w + 0.5*DPHI*k1w)
        k3u, k3w = deriv(u + 0.5*DPHI*k2u, w + 0.5*DPHI*k2w)
        k4u, k4w = deriv(u + DPHI*k3u,     w + DPHI*k3w)
        u = u + DPHI/6*(k1u + 2*k2u + 2*k3u + k4u)
        w = w + DPHI/6*(k1w + 2*k2w + 2*k3w + k4w)
        phi = phi + DPHI

        pos, r = position(u, phi)

        # ---- horizon capture ----
        hit_h = active & (r <= r_s)
        color[hit_h] = 0.0
        active &= ~hit_h

        # ---- accretion-disk crossing (equatorial plane y=0) ----
        zc, zp = pos[:, 1], pos_prev[:, 1]
        crossed = active & (zc*zp < 0)
        if crossed.any():
            t = (zp / (zp - zc)).clamp(0, 1)
            pcross = pos_prev + t[:, None]*(pos - pos_prev)
            rc = pcross.norm(dim=-1)
            on_disk = crossed & (rc >= R_IN) & (rc <= R_OUT)
            if on_disk.any():
                # simple Doppler beaming: approaching side brighter
                vel_dir = pcross[on_disk, 2]      # z-component ~ orbital motion
                doppler = (1.0 + 0.6*torch.sign(vel_dir)).clamp(0.4, 1.8)
                color[on_disk] = disk_color(rc[on_disk], doppler)
                active &= ~on_disk

        # ---- escape to background ----
        esc = active & (r > R_ESC)
        if esc.any():
            dirs = (pos[esc] - pos_prev[esc])
            dirs = dirs / dirs.norm(dim=-1, keepdim=True)
            color[esc] = sample_bg(dirs)
            active &= ~esc

        pos_prev = pos
        if not active.any():
            break

    # remaining active rays -> background
    if active.any():
        dirs = (pos - pos_prev)[active]
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        color[active] = sample_bg(dirs)

    img = color.reshape(H, W, 3)
    img = (img / (1.0 + img))            # tone map (Reinhard)
    img = img.clamp(0, 1) ** (1/2.2)     # gamma
    return (img.cpu().numpy()*255).astype(np.uint8)

# ----------------------- single image -----------------------------
cam = torch.tensor([0.0, 3.0, 22.0])     # slightly above the disk plane
frame = render(cam)
imageio.imwrite("blackhole_gpu.png", frame)
print("PNG path:", os.path.abspath("blackhole_gpu.png"))
print("PNG exists:", os.path.exists("blackhole_gpu.png"))
print("Saved blackhole_gpu.png")

# ----------------------- orbit video (the '4D' part) --------------
def make_video(fname="blackhole_gpu.mp4", n=180, radius=22.0, elev=3.0, fps=30):
    writer = imageio.get_writer(fname, fps=fps)
    for k in range(n):
        a = 2*math.pi*k/n
        cam = torch.tensor([radius*math.sin(a), elev, radius*math.cos(a)])
        writer.append_data(render(cam))
        if k % 20 == 0:
            print(f"frame {k}/{n}")
    writer.close()
    print("MP4 path:", os.path.abspath(fname))
    print("MP4 exists:", os.path.exists(fname))
    print("Saved", fname)

make_video()   # comment out if you only want the still image








import os

print("Current directory:", os.getcwd())
print("PNG:", os.path.abspath("blackhole_gpu.png"))
print("MP4:", os.path.abspath("blackhole_gpu.mp4"))


