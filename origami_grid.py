#!python3.12

"""
Origami Triangle Grid — Python/pygame rewrite
Uses exact rational arithmetic (fractions.Fraction) for all grid coordinates.
No floating-point proximity thresholds — vertex identity is exact tuple equality.

Requirements: pip install pygame
Run: python origami_grid.py
"""

import pygame
import sys
import math
import numpy as np
from fractions import Fraction
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Exact arithmetic helpers
# ---------------------------------------------------------------------------

# Rational point: (Fraction, Fraction)
RatPt = tuple  # (Fraction, Fraction)

# We keep a single exact sqrt(3) as Fraction approximation? No —
# instead we represent diagonal lines symbolically. The three line families are:
#   A: horizontal,  dy=0,  dx=1
#   B: slope +√3,  dy/dx = √3   (i.e. 60° from horizontal)
#   C: slope -√3,  dy/dx = −√3  (i.e. 120° from horizontal)
#
# All intersection points between these lines have coordinates of the form
#   x = p/N,  y = q/N   where p,q are integers and N = number of divisions.
# So we store coords as Fraction(p, N) — exact, no sqrt(3) needed.


def rat(p, q=1):
    return Fraction(p, q)


def vec_angle_float(dx_f: float, dy_f: float) -> float:
    """Float angle in [0,360) for direction comparisons."""
    return (math.atan2(dy_f, dx_f) * 180 / math.pi + 360) % 360


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


# ---------------------------------------------------------------------------
# Grid geometry (exact)
# ---------------------------------------------------------------------------

@dataclass
class LineDesc:
    family: str          # 'A', 'B', or 'C'
    key: str             # unique string id
    # Direction vector as floats (for angle comparisons only — not for arithmetic)
    dx_f: float
    dy_f: float
    # Anchor point as Fraction pair (for exact intersection)
    px: Fraction
    py: Fraction


@dataclass
class Crease:
    a: RatPt             # exact start
    b: RatPt             # exact end
    line_key: str
    active: bool = False


@dataclass
class Vertex:
    pt: RatPt            # exact rational point
    line_key: str
    seg_idx: int         # index into line.seg_keys


@dataclass
class LineData:
    desc: LineDesc
    seg_keys: list       # ordered list of crease segment keys
    verts: list          # ordered list of RatPt (len = len(seg_keys)+1)


# ---------------------------------------------------------------------------
# Grid state (module-level, rebuilt on N change)
# ---------------------------------------------------------------------------

N = 128
SCREEN_W = 1280
SCREEN_H = 720

# Camera: cam_x/cam_y = screen pixel position of grid origin (0,0)
# cam_scale = pixels per normalised grid unit (grid goes from 0..1 in both axes)
cam_x: float = 0.0
cam_y: float = 0.0
cam_scale: float = 700.0

PAN_SPEED = 300.0   # pixels per second for WASD

def init_camera(w, h):
    """Start zoomed so ~48 divisions are visible, centred on the grid."""
    global cam_x, cam_y, cam_scale, SCREEN_W, SCREEN_H
    SCREEN_W, SCREEN_H = w, h
    VISIBLE_DIVS = 48
    cam_scale = min(w, h) * 0.9 * (N / VISIBLE_DIVS)
    # Centre the middle of the grid on screen
    cam_x = w / 2.0 - cam_scale * 0.5
    cam_y = h / 2.0 - cam_scale * 0.5

def update_screen_size(w, h):
    global SCREEN_W, SCREEN_H
    SCREEN_W, SCREEN_H = w, h

lines: dict[str, LineData] = {}
creases: dict[str, Crease] = {}
segs_by_vertex: dict[RatPt, list[str]] = defaultdict(list)

# Interaction state
selected_line: Optional[str] = None
hovered_line: Optional[str] = None
hovered_vertex: Optional[Vertex] = None
hovered_blocked: bool = False
picked_vertex: Optional[Vertex] = None   # for manual side popup

vertices: dict[str, Vertex] = {}         # v_key -> Vertex
ray_boundaries: dict[str, set] = defaultdict(set)   # line_key -> set of vert indices
dead_vertices: set = set()               # set of RatPt

guided_mode: bool = False
guided_vertices: dict[str, Vertex] = {}
guided_hovered: Optional[Vertex] = None
guided_ray_origin: Optional[RatPt] = None
guided_ray_lk: Optional[str] = None
guided_ray_angle: Optional[float] = None
guided_queue: list = []

popup_active: bool = False
popup_rect: Optional[pygame.Rect] = None
popup_buttons: list = []   # list of (rect, label)

info_text: str = "Hover a line to highlight it, click to select."

# Undo stack — each entry is a delta snapshot of mutable fold state
undo_stack: list = []
MAX_UNDO = 64
_undo_frame: Optional[dict] = None   # points to the top snap while an action is in progress

# Vertex status cache — recomputed once per state change, not per frame
_vertex_blocked_cache: dict = {}   # vk -> bool (True = blocked)
_vertex_status_dirty: bool = False

# Grid center marker — the (kb, kc) vertex closest to normalised (0.5, 0.5)
_center_vertex: Optional[tuple] = None

# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def to_px(pt: RatPt) -> tuple:
    """Convert normalised grid coord to screen pixel coord."""
    return (cam_x + float(pt[0]) * cam_scale, cam_y + float(pt[1]) * cam_scale)


def from_px(sx: float, sy: float) -> tuple:
    """Convert screen pixel coord to float normalised grid coord."""
    return ((sx - cam_x) / cam_scale, (sy - cam_y) / cam_scale)


# ---------------------------------------------------------------------------
# Exact line intersection
# ---------------------------------------------------------------------------

def x_lines_exact(l1: LineDesc, l2: LineDesc) -> Optional[RatPt]:
    """
    Intersect two lines exactly. Lines are described by anchor (px,py) and
    direction. For family B (slope √3): dy/dx = √3.
    We parameterize: P = anchor + t * dir
    For horizontal lines: y = py, so the point has y=py exactly.
    For B lines: y - py = √3*(x - px)  =>  y = √3*x + (py - √3*px)
    For C lines: y - py = -√3*(x - px) =>  y = -√3*x + (py + √3*px)

    We handle the three family combinations:
      A∩B, A∩C, B∩C
    All intersections on an N-division grid have exact rational coordinates
    with denominators dividing 2N (for B/C families) or N (for A).
    """
    fa, fb = l1.family, l2.family

    # Ensure canonical order
    if (fa, fb) in [('B', 'A'), ('C', 'A'), ('C', 'B')]:
        return x_lines_exact(l2, l1)

    px1, py1 = l1.px, l1.py
    px2, py2 = l2.px, l2.py

    if fa == 'A' and fb == 'B':
        # l1: y = py1 (horizontal)
        # l2: y = √3*x + (py2 - √3*px2)  =>  y - py2 = √3*(x - px2)
        # √3*x = y - py2 + √3*px2  =>  x = (y - py2)/√3 + px2
        # But we need exact. The grid guarantees:
        # x = px2 + (py1 - py2)/√3
        # For this to be rational, (py1-py2) must be divisible by √3 in grid units.
        # In our grid, B-lines have py = k * RH * √3 where RH = 1/N (normalised).
        # Actually in normalised coords: B lines have slope √3 and
        # anchor py = c where c is a multiple of √3/N.
        # Let's use the symbolic approach:
        # py2 = m * √3 / N  for some integer m (stored as Fraction(m,N) * √3)
        # We store B/C anchors as their "rational part" divided by √3.
        # See build() where we set b_offset / c_offset.
        # This function is called with precomputed rational anchors where:
        # for B-family: the line is  y - √3*x = py_rational   (py stored = py - √3*px at x=0)
        #   actually anchor is at x=0, y=py_rational
        # So B-line: y = √3*x + py_rational
        # A∩B: y = py1,  y = √3*x + py2  =>  x = (py1 - py2) / √3
        # This x is rational only if (py1-py2) is a rational multiple of √3.
        # In our normalised grid py2 for B-lines = k*√3/N, py1 = j/N
        # => x = (j/N - k*√3/N) / √3 = j/(N*√3) - k/N
        # That's irrational unless j=0. So we need a different representation.
        #
        # REVISED APPROACH: store B-line anchor as (bx_rat, by_rat) where
        # by_rat = py - √3 * px  (the y-intercept at x=0, which IS rational for our grid)
        # Wait, let me think more carefully.
        #
        # In the JS code, B-lines have (dx=1, dy=√3) and are parameterized by
        # c = py (y-intercept at x=0). The y-intercept c = n * stepD where stepD = √3/N (normalised).
        # So c = n*√3/N. That's irrational.
        #
        # Key insight: the actual intersection coordinates ARE rational in units of 1/N.
        # A∩B:  y=j/N,  y=√3*x+n*√3/N  =>  j/N = √3*(x + n/N)  =>  x = j/(N*√3) - n/N
        # This is irrational in general. The coordinates are NOT rational — they involve 1/√3.
        #
        # So pure rational arithmetic won't work directly. Instead we use a
        # MIXED EXACT representation: coords of the form  a/N + b/(N) * (1/√3)
        # That's messy. Better approach:
        #
        # Use integer grid indices. Every intersection on the triangular grid can be
        # addressed by two integers (i,j) in the oblique coordinate system.
        # See build_exact() below.
        pass

    return None  # placeholder — see build_exact()


# ---------------------------------------------------------------------------
# EXACT BUILD using integer oblique coordinates
# ---------------------------------------------------------------------------
#
# The triangular grid has three families of parallel lines.
# We use a coordinate system (u, v) where:
#   u = column index along family A (horizontal)
#   v = row index  along family B (diagonal)
# Every grid vertex is uniquely identified by (u, v) integers.
#
# In screen pixels (with grid size Q = GRID_PX):
#   x_px = PAD + (u + v/2) * cell_w
#   y_px = PAD + v * row_h
# where:
#   row_h = Q / N    (pixels per row)
#   cell_w = row_h / √3 * 2 ... actually for equilateral triangles:
#   row_h = Q/N, cell_w = row_h (each small equilateral triangle has height row_h)
#   Hmm, let me re-derive.
#
# For an N-division square grid of equilateral triangles:
# The JS uses:
#   RH = Q/N   (row height)
#   SIDE = RH*2/√3  (side length of small triangle)
#   stepD = SIDE*√3 = 2*RH  (step along diagonal direction)
#   Family A: horizontal lines at y = k*RH, k=0..N
#   Family B: lines with slope √3 (60°)
#   Family C: lines with slope -√3 (120°)
#
# Intersection of A(row=r) and B(step=s):
#   y = r*RH
#   y = √3*x + s*stepD = √3*x + 2*s*RH
#   => √3*x = r*RH - 2*s*RH = (r-2s)*RH
#   => x = (r-2s)*RH/√3 = (r-2s)*SIDE/2
#
# So x = (r-2s) * SIDE/2. Still involves SIDE which involves 1/√3.
#
# For pixel coordinates: x_px = PAD + x * (GRID_PX/... no.
# Actually in the JS, Q = GRID_PX and coordinates ARE in pixels.
# So x_px = (r-2s)*RH/√3 = (r-2s)*Q/(N*√3)  — irrational.
#
# The coordinates ARE irrational in Cartesian space. But we can represent them
# exactly in the oblique (u,v) lattice. Every vertex is:
#   (u, v)  integers,  where u ∈ [0,N], v ∈ [0,N] (roughly)
# with pixel positions:
#   x_px = PAD + (u - v * 0.5) * col_w     (NOT using √3 directly)
#   y_px = PAD + v * row_h
# where row_h = GRID_PX/N and col_w = GRID_PX/N (wait, that's not right either).
#
# Let me just use a clean oblique system. Define:
#   For the triangular grid on a square, there are three families:
#   A-lines: horizontal (slope 0)
#   B-lines: slope +√3 (NE direction)  
#   C-lines: slope -√3 (NW direction)
#
# Every intersection vertex can be indexed by (a, b) where:
#   a = which A-line (horizontal row index, 0..N)
#   b = which B-line
# Then the C-line through (a,b) is determined.
#
# Pixel coordinates of vertex (a, b):
#   The A-line at row a has y_px = PAD + a * row_h
#   The B-line b has the equation: y = √3*(x - x0_b) for some x0_b
#   At row a: x_px = PAD + ... 
#
# I'll use a direct formula. In normalised coords [0,1]^2:
# row_h_norm = 1/N
# For B-lines through integer multiples:
# The B-line spacing is 2/N (in y-intercept units since slope=√3 and stepD=2/N in norm coords)
# B-line b (b ∈ integers, ranging to cover the square): y = √3*x + b*(2/N) ... wait
# stepD = 2*RH = 2/N in normalised (Q=1) coords. b ranges so the line covers [0,1]^2.
#
# A(row r) ∩ B(line b):
#   y = r/N
#   y = √3*x + b*(2/N)
#   => √3*x = r/N - 2b/N = (r-2b)/N
#   => x = (r-2b)/(N*√3)
#
# This is genuinely irrational. The key insight for exact computation:
# We don't need Fraction — we need a number system that handles 1/√3.
# Represent each coordinate as (p + q/√3) / N  = p/N + q/(N√3)
# where p,q are integers. Addition and comparison work exactly on (p,q) pairs.
#
# SIMPLER: represent each vertex as an (a,b) integer pair in oblique coords.
# Pixel rendering: convert (a,b) to float pixels only at draw time.
# All geometry (topology, adjacency, ordering) works on (a,b) integers.
#
# Let's define:
#   Vertex (a, b) lives at the intersection of A-line a and B-line b.
#   (a = horizontal row index 0..N, b = B-line index)
# The C-line through (a,b) is c = a + b (easily derived).
# 
# A-line a: contains all vertices (a, b) for varying b.
# B-line b: contains all vertices (a, b) for varying a.
# C-line c: contains all vertices (a, b) where a + b = c, i.e. a-b = a-(c-a) hmm.
#   Let c = a + b. C-line c contains vertices (a, c-a) for varying a.
#   Wait: C-lines have slope -√3. At row a, x decreases as a increases.
#   Let me verify: A(r)∩C(c): y=r/N, y=-√3*x+c*(2/N)
#   => √3*x = c*(2/N)-r/N = (2c-r)/N => x=(2c-r)/(N√3)
#   A(r)∩B(b): x=(r-2b)/(N√3) [derived above, with b measured differently]
#   Hmm, let me just pick a concrete convention.
#
# FINAL CONVENTION (matching JS grid):
# =====================================
# N divisions. row_h = 1.0/N (normalised). step = 2.0/N.
# A-lines: y = a/N,  a = 0..N
# B-lines: y = √3*x + b_offset,  b_offset = k*(2/N), k ∈ Z
# C-lines: y = -√3*x + c_offset, c_offset = k*(2/N), k ∈ Z
#
# Vertex index: Let's use (a, k_b) where:
#   a = row (A-line index, 0..N)
#   k_b = B-line index k (integer, some range)
# Then k_c = a - k_b is the C-line index (derived).
# Wait: A(a) ∩ B(k_b):
#   y = a/N,  y = √3*x + k_b*(2/N)
#   a/N = √3*x + 2*k_b/N => √3*x = (a-2*k_b)/N => x = (a-2*k_b)/(N*√3)
# A(a) ∩ C(k_c):
#   y = a/N,  y = -√3*x + k_c*(2/N)
#   a/N = -√3*x + 2*k_c/N => √3*x = (2*k_c-a)/N => x = (2*k_c-a)/(N*√3)
# For these to be the SAME point: (a-2*k_b) = -(2*k_c-a) => -2*k_b = -2*k_c => k_c = k_b?
# That can't be right. Let me try B∩C:
# B(k_b) ∩ C(k_c):
#   y = √3*x + 2*k_b/N
#   y = -√3*x + 2*k_c/N
#   2√3*x = 2*(k_c-k_b)/N => x = (k_c-k_b)/(N*√3)
#   y = √3*(k_c-k_b)/(N*√3) + 2*k_b/N = (k_c-k_b)/N + 2*k_b/N = (k_c+k_b)/N
#
# So B(k_b) ∩ C(k_c) has y = (k_b+k_c)/N. This is at A-line a = k_b+k_c.
# So every vertex is determined by any two of {a, k_b, k_c} with a = k_b + k_c.
# We can index vertices as (k_b, k_c) with a = k_b + k_c.
#
# Pixel coordinates of vertex (k_b, k_c):
#   a = k_b + k_c
#   x_float = (k_c - k_b) / (N * √3)  [in normalised [0,1] coords]
#   y_float = a / N = (k_b + k_c) / N
#   x_px = PAD + x_float * GRID_PX
#   y_px = PAD + y_float * GRID_PX
#
# Lines:
#   A-line a contains vertices (k_b, k_c) with k_b + k_c = a
#   B-line b contains vertices (b, k_c) for varying k_c  [k_b fixed = b]
#   C-line c contains vertices (k_b, c) for varying k_b  [k_c fixed = c]
#
# This gives us EXACT INTEGER coordinates for all topology.
# Float conversion only happens at draw time.

S3 = math.sqrt(3)

def vertex_px(kb: int, kc: int) -> tuple:
    """Convert oblique (kb,kc) vertex to screen pixel coordinates using camera."""
    x_norm = (kc - kb) / (N * S3)
    y_norm = (kb + kc) / N
    return (cam_x + x_norm * cam_scale, cam_y + y_norm * cam_scale)


def screen_to_norm(sx: float, sy: float) -> tuple:
    """Convert screen pixel → normalised grid coords (for hit-testing)."""
    return ((sx - cam_x) / cam_scale, (sy - cam_y) / cam_scale)


def seg_key(va: tuple, vb: tuple) -> tuple:
    """Canonical segment key from two vertex (kb,kc) tuples."""
    return (min(va, vb), max(va, vb))


def v_key(line_key: str, seg_idx: int) -> str:
    return f"{line_key}::{seg_idx}"


def line_family(lk: str) -> str:
    return lk[0]   # 'A', 'B', or 'C'


# ---------------------------------------------------------------------------
# Grid build
# ---------------------------------------------------------------------------

def build():
    global lines, creases, segs_by_vertex

    lines = {}
    creases = {}
    segs_by_vertex = defaultdict(list)
    reset_state()

    # Vertex (kb, kc) is inside the unit square when:
    #   0 <= (kc - kb) / (N * S3) <= 1   =>  0 <= kc - kb <= floor(N * S3)
    #   0 <= (kb + kc) / N        <= 1   =>  0 <= kb + kc <= N
    # We enumerate by a = kb + kc (0..N) and d = kc - kb (0..floor(N*S3)):
    #   kc = (a + d) / 2,  kb = (a - d) / 2  — must both be integers => a and d same parity

    d_max = int(N * S3)  # floor(N * sqrt(3))

    all_verts = set()
    for a in range(0, N + 1):
        for d in range(0, d_max + 1):
            if (a + d) % 2 != 0:
                continue  # kb and kc must be integers
            kb = (a - d) // 2
            kc = (a + d) // 2
            all_verts.add((kb, kc))

    def add_to_sv(vpt, sk):
        lst = segs_by_vertex[vpt]
        if sk not in lst:
            lst.append(sk)

    def register_line(lk, family, sorted_verts):
        seg_keys_list = []
        for i in range(len(sorted_verts) - 1):
            va = sorted_verts[i]
            vb = sorted_verts[i + 1]
            sk = seg_key(va, vb)
            if sk not in creases:
                creases[sk] = Crease(a=va, b=vb, line_key=lk, active=False)
            seg_keys_list.append(sk)
            add_to_sv(va, sk)
            add_to_sv(vb, sk)
        lines[lk] = LineData(
            desc=None,
            seg_keys=seg_keys_list,
            verts=sorted_verts
        )

    # Group by A-line (a = kb+kc), B-line (kb), C-line (kc)
    from collections import defaultdict as _dd
    a_groups = _dd(list)
    b_groups = _dd(list)
    c_groups = _dd(list)
    for (kb, kc) in all_verts:
        a_groups[kb + kc].append((kb, kc))
        b_groups[kb].append((kb, kc))
        c_groups[kc].append((kb, kc))

    for a, vlist in sorted(a_groups.items()):
        vlist.sort(key=lambda v: v[1] - v[0])
        if len(vlist) >= 2:
            register_line(f'A:{a}', 'A', vlist)

    for b, vlist in sorted(b_groups.items()):
        vlist.sort(key=lambda v: v[1])
        if len(vlist) >= 2:
            register_line(f'B:{b}', 'B', vlist)

    for c, vlist in sorted(c_groups.items()):
        vlist.sort(key=lambda v: v[0])
        if len(vlist) >= 2:
            register_line(f'C:{c}', 'C', vlist)

    build_accel()
    _compute_center_vertex()



# ---------------------------------------------------------------------------
# Numpy acceleration: precomputed normalised coords + spatial grid
# ---------------------------------------------------------------------------
#
# crease_norm: float32 array shape (M, 4) = [ax_n, ay_n, bx_n, by_n] per crease
#   where ax_n = (kc-kb)/(N*S3), ay_n = (kb+kc)/N  (normalised 0..1)
# crease_keys: list of seg_keys in same order as crease_norm rows
# crease_line_keys: list of line_keys in same order
#
# Spatial grid: normalised space [0,1]^2 divided into GRID_CELLS x GRID_CELLS.
# Each cell stores list of crease indices (into crease_norm) whose midpoint falls in it.

SPATIAL_CELLS = 64   # 64x64 grid over normalised space

crease_norm: np.ndarray = None      # (M, 4) float32
crease_keys: list = []
crease_line_keys: list = []
spatial_grid: list = []             # SPATIAL_CELLS*SPATIAL_CELLS lists of int indices
sk_to_idx: dict = {}                # seg_key -> index into crease_norm
active_crease_indices: set = set()  # indices of currently active creases

# Cached background surface (inactive grid lines) — redrawn only when camera changes
_bg_surface: 'pygame.Surface | None' = None
_bg_cam_state: tuple = (None, None, None, None, None)  # (cam_x, cam_y, cam_scale, sw, sh)


def invalidate_bg():
    global _bg_surface, _vertex_status_dirty
    _bg_surface = None
    _vertex_status_dirty = True


def _refresh_vertex_statuses():
    """Recompute blocked/free for every active vertex once per state change."""
    global _vertex_blocked_cache, _vertex_status_dirty
    _vertex_blocked_cache = {}
    for vk, v in list(vertices.items()):   # list() because vertex_status may delete entries
        status = vertex_status(v)
        if vk in vertices:                 # vertex_status may have removed it
            _vertex_blocked_cache[vk] = (status == 'blocked')
    _vertex_status_dirty = False


def _norm_xy(kb: int, kc: int):
    return (kc - kb) / (N * S3), (kb + kc) / N


def build_accel():
    """Build numpy arrays and spatial grid from current creases dict."""
    global crease_norm, crease_keys, crease_line_keys, spatial_grid, sk_to_idx, active_crease_indices

    n = len(creases)
    arr = np.empty((n, 4), dtype=np.float32)
    ckeys = []
    clkeys = []
    idx_map = {}

    for i, (sk, c) in enumerate(creases.items()):
        ax, ay = _norm_xy(c.a[0], c.a[1])
        bx, by = _norm_xy(c.b[0], c.b[1])
        arr[i, 0] = ax
        arr[i, 1] = ay
        arr[i, 2] = bx
        arr[i, 3] = by
        ckeys.append(sk)
        clkeys.append(c.line_key)
        idx_map[sk] = i

    crease_norm = arr
    crease_keys = ckeys
    crease_line_keys = clkeys
    sk_to_idx = idx_map
    active_crease_indices = set()

    # Build spatial grid on midpoints
    C = SPATIAL_CELLS
    grid = [[] for _ in range(C * C)]
    mx_arr = (arr[:, 0] + arr[:, 2]) * 0.5
    my_arr = (arr[:, 1] + arr[:, 3]) * 0.5
    gx = np.clip((mx_arr * C).astype(np.int32), 0, C - 1)
    gy = np.clip((my_arr * C).astype(np.int32), 0, C - 1)
    for i in range(n):
        grid[gy[i] * C + gx[i]].append(i)
    spatial_grid = grid


def _compute_center_vertex():
    """Find the (kb, kc) grid vertex closest to normalised centre (0.5, 0.5).

    y_norm = (kb+kc)/N = 0.5  →  a = N//2
    x_norm = (kc-kb)/(N·√3) = 0.5  →  d ≈ N·√3/2
    Integer parity: (a+d) must be even so kb=(a-d)/2 and kc=(a+d)/2 are integers.
    """
    global _center_vertex
    a = N // 2
    d = round(N * S3 / 2)
    if (a + d) % 2 != 0:
        d += 1
    kb = (a - d) // 2
    kc = (a + d) // 2
    _center_vertex = (kb, kc)


def set_crease_active(sk, value: bool):
    """Set a crease active flag and keep active_crease_indices in sync.
    If an undo frame is open, records the previous value the first time
    this segment is touched (so pop_undo can replay only the diff)."""
    c = creases.get(sk)
    if c is None:
        return
    if _undo_frame is not None and sk not in _undo_frame['crease_delta']:
        _undo_frame['crease_delta'][sk] = c.active   # save old value once
    c.active = value
    idx = sk_to_idx.get(sk)
    if idx is None:
        return
    if value:
        active_crease_indices.add(idx)
    else:
        active_crease_indices.discard(idx)


def screen_to_crease_idx(sx: float, sy: float, threshold_px: float = 10.0) -> Optional[int]:
    """Find index into crease_norm of the nearest crease to screen point (sx,sy).
    Uses spatial grid in normalised space then exact distance check."""
    if crease_norm is None:
        return None
    # Convert screen pos to normalised
    nx = (sx - cam_x) / cam_scale
    ny = (sy - cam_y) / cam_scale
    # Threshold in normalised space
    thresh_n = threshold_px / cam_scale

    C = SPATIAL_CELLS
    # Which cells to check: a radius of thresh_n around (nx,ny)
    cell_radius = max(1, int(thresh_n * C) + 1)
    cx0 = int(nx * C)
    cy0 = int(ny * C)

    best_i = -1
    best_d2 = thresh_n * thresh_n

    for cy in range(max(0, cy0 - cell_radius), min(C, cy0 + cell_radius + 1)):
        for cx in range(max(0, cx0 - cell_radius), min(C, cx0 + cell_radius + 1)):
            for i in spatial_grid[cy * C + cx]:
                ax, ay, bx, by = crease_norm[i]
                dx, dy = bx - ax, by - ay
                l2 = dx * dx + dy * dy
                if l2 < 1e-12:
                    d2 = (nx - ax) ** 2 + (ny - ay) ** 2
                else:
                    t = ((nx - ax) * dx + (ny - ay) * dy) / l2
                    t = max(0.0, min(1.0, t))
                    ex, ey = ax + t * dx, ay + t * dy
                    d2 = (nx - ex) ** 2 + (ny - ey) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
    return best_i if best_i >= 0 else None



def line_direction_float(lk: str) -> tuple:
    """Return (dx_f, dy_f) float direction of line family A/B/C.
    For A: (1,0); B: (1,√3) normalised; C: (1,-√3) normalised."""
    fam = line_family(lk)
    if fam == 'A':
        return (1.0, 0.0)
    elif fam == 'B':
        # Sort order in B-lines is by increasing kc (= increasing y = going down-right)
        return (1.0, S3)
    else:  # C
        # Sort order in C-lines is by increasing kb (= increasing y = going down-left)
        return (-1.0, S3)


def line_fwd_angle(lk: str) -> float:
    dx, dy = line_direction_float(lk)
    return vec_angle_float(dx, dy)


# ---------------------------------------------------------------------------
# Oblique direction for vertex comparisons
# ---------------------------------------------------------------------------

def vert_position_on_line(lk: str, vpt: tuple) -> int:
    """Return the sort key of vpt on line lk (its index in verts list)."""
    ld = lines.get(lk)
    if ld is None:
        return -1
    try:
        return ld.verts.index(vpt)
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def reset_state():
    global selected_line, hovered_line, hovered_vertex, hovered_blocked
    global picked_vertex, vertices, ray_boundaries, dead_vertices
    global guided_mode, guided_vertices, guided_hovered
    global guided_ray_origin, guided_ray_lk, guided_ray_angle, guided_queue
    global popup_active, info_text, active_crease_indices, _undo_frame
    global _vertex_blocked_cache, _vertex_status_dirty
    _undo_frame = None
    _vertex_blocked_cache = {}
    _vertex_status_dirty = False

    selected_line = None
    hovered_line = None
    hovered_vertex = None
    hovered_blocked = False
    picked_vertex = None
    vertices = {}
    ray_boundaries = defaultdict(set)
    dead_vertices = set()
    guided_mode = False
    guided_vertices = {}
    guided_hovered = None
    guided_ray_origin = None
    guided_ray_lk = None
    guided_ray_angle = None
    guided_queue = []
    popup_active = False
    active_crease_indices = set()
    info_text = "Hover a line to highlight it, click to select."


def clear_guided_mode():
    global guided_mode, guided_vertices, guided_hovered
    global guided_ray_origin, guided_ray_lk, guided_ray_angle
    guided_mode = False
    guided_vertices = {}
    guided_hovered = None
    guided_ray_origin = None
    guided_ray_lk = None
    guided_ray_angle = None


def push_undo():
    """Record a delta (only what changed) onto the undo stack.

    Complexity: O(k) where k = segments touched this action, instead of O(N²).
    We open a 'pending' frame before the action runs (see begin_undo_frame /
    commit_undo_frame), so callers still just call push_undo() at the top of
    each action and the diff is built automatically.
    """
    global undo_stack, _undo_frame
    # Snapshot the lightweight non-crease state (these are all small sets)
    snap = {
        'crease_delta':   {},           # populated by set_crease_active during action
        'vertices':       dict(vertices),
        'ray_boundaries': {k: set(v) for k, v in ray_boundaries.items()},
        'dead_vertices':  set(dead_vertices),
        'selected_line':  selected_line,
        'guided_mode':        guided_mode,
        'guided_vertices':    dict(guided_vertices),
        'guided_ray_origin':  guided_ray_origin,
        'guided_ray_lk':      guided_ray_lk,
        'guided_ray_angle':   guided_ray_angle,
        'guided_queue':       list(guided_queue),
    }
    undo_stack.append(snap)
    if len(undo_stack) > MAX_UNDO:
        undo_stack.pop(0)
    # Mark that set_crease_active should record into this frame
    _undo_frame = snap


def pop_undo():
    """Replay the delta stored in the most recent undo frame."""
    global vertices, ray_boundaries, dead_vertices, selected_line
    global guided_mode, guided_vertices, guided_hovered
    global guided_ray_origin, guided_ray_lk, guided_ray_angle, guided_queue
    global hovered_vertex, hovered_blocked, info_text, _undo_frame
    if not undo_stack:
        info_text = "Nothing to undo."
        return
    _undo_frame = None   # stop recording while we replay
    snap = undo_stack.pop()
    # Restore only the creases that changed (O(k) not O(N²))
    for sk, was_active in snap['crease_delta'].items():
        if sk in creases:
            set_crease_active(sk, was_active)
    # Restore vertex/boundary/dead sets (these are O(V) ≤ O(N))
    vertices.clear()
    vertices.update(snap['vertices'])
    ray_boundaries.clear()
    ray_boundaries.update({k: set(v) for k, v in snap['ray_boundaries'].items()})
    dead_vertices.clear()
    dead_vertices.update(snap['dead_vertices'])
    # Restore selection and guided state
    selected_line = snap['selected_line']
    guided_mode = snap['guided_mode']
    guided_vertices = snap['guided_vertices']
    guided_hovered = None
    guided_ray_origin = snap['guided_ray_origin']
    guided_ray_lk = snap['guided_ray_lk']
    guided_ray_angle = snap['guided_ray_angle']
    guided_queue = snap['guided_queue']
    hovered_vertex = None
    hovered_blocked = False
    info_text = "Undo. Hover a vertex to continue." if selected_line else "Hover a line to highlight it, click to select."
    invalidate_bg()


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def compute_ray_range(start_vpt: tuple, lk: str, ray_angle: float):
    """Return (start_idx, end_idx, boundary_vert_idx) for a ray from start_vpt along lk."""
    ld = lines.get(lk)
    if ld is None:
        return None
    try:
        si = ld.verts.index(start_vpt)
    except ValueError:
        return None
    fwd = line_fwd_angle(lk)
    if angle_diff(ray_angle, fwd) < 90:
        return {'start_idx': si, 'end_idx': len(ld.seg_keys) - 1, 'boundary_vert_idx': si}
    else:
        return {'end_idx': si - 1, 'start_idx': 0, 'boundary_vert_idx': si}


def get_ray_seg_set(lk: str, range_info) -> set:
    ld = lines.get(lk)
    if ld is None or range_info is None:
        return set()
    return set(ld.seg_keys[range_info['start_idx']:range_info['end_idx'] + 1])


def add_vertices_for_range(lk: str, from_seg: int, to_seg: int):
    ld = lines.get(lk)
    if ld is None:
        return
    for i in range(from_seg, to_seg):
        if i + 1 < len(ld.verts):
            vpt = ld.verts[i + 1]
            k = v_key(lk, i + 1)
            if k not in vertices:
                vertices[k] = Vertex(pt=vpt, line_key=lk, seg_idx=i + 1)


def add_ray_boundary(lk: str, vert_idx: int):
    ray_boundaries[lk].add(vert_idx)


def apply_double_bound_deletion():
    for lk, bounds in ray_boundaries.items():
        if len(bounds) < 2:
            continue
        sorted_b = sorted(bounds)
        lo, hi = sorted_b[0], sorted_b[-1]
        for i in range(lo, hi):
            vk = v_key(lk, i + 1)
            if vk in vertices:
                del vertices[vk]


def trace_and_remove(lk: str, trim_vert_idx: int, direction: str):
    ld = lines.get(lk)
    if ld is None:
        return
    segs = ld.seg_keys
    if direction == 'left':
        for i in range(trim_vert_idx - 1, -1, -1):
            c = creases.get(segs[i])
            if not c or not c.active:
                break
            set_crease_active(segs[i], False)
            for vk in [v_key(lk, i + 1), v_key(lk, i)]:
                vertices.pop(vk, None)
    else:
        for i in range(trim_vert_idx, len(segs)):
            c = creases.get(segs[i])
            if not c or not c.active:
                break
            set_crease_active(segs[i], False)
            for vk in [v_key(lk, i + 1), v_key(lk, i)]:
                vertices.pop(vk, None)
    vertices.pop(v_key(lk, trim_vert_idx), None)


def get_neighbors_from_point(vpt: tuple, unset_angle: float, exclude_lk: str) -> list:
    result = []
    for sk in segs_by_vertex.get(vpt, []):
        c = creases.get(sk)
        if c is None or c.line_key == exclude_lk:
            continue
        if c.a == vpt:
            sdx, sdy = direction_to_float(c.b[0] - c.a[0], c.b[1] - c.a[1], c.line_key)
        else:
            sdx, sdy = direction_to_float(c.a[0] - c.b[0], c.a[1] - c.b[1], c.line_key)
        seg_angle = vec_angle_float(sdx, sdy)
        if angle_diff(seg_angle, unset_angle) < 90:
            result.append({'line_key': c.line_key, 'ray_angle': seg_angle})
    return result


def direction_to_float(dkb, dkc, lk):
    """Convert oblique delta (dkb, dkc) to approximate float (dx,dy) for angle use."""
    # pixel delta: dx = (dkc-dkb)/(N*√3)*GRID_PX, dy = (dkb+dkc)/N*GRID_PX
    dx = (dkc - dkb) / (N * S3)
    dy = (dkb + dkc) / N
    return (dx, dy)


def ray_intersects_active(segs: set, exclude_lks: set) -> bool:
    for sk in segs:
        c = creases.get(sk)
        if c is None:
            continue
        for vpt in [c.a, c.b]:
            for nsk in segs_by_vertex.get(vpt, []):
                nc = creases.get(nsk)
                if nc and nc.active and nc.line_key not in exclude_lks:
                    return True
    return False


def find_ray_intersection_vert_idx(origin_pt, ray_lk, ray_angle, exclude_lks) -> int:
    ld = lines.get(ray_lk)
    if ld is None:
        return -1
    rng = compute_ray_range(origin_pt, ray_lk, ray_angle)
    if rng is None:
        return -1
    fwd = line_fwd_angle(ray_lk)
    go_fwd = angle_diff(ray_angle, fwd) < 90
    segs = ld.seg_keys
    if go_fwd:
        for i in range(rng['start_idx'], rng['end_idx'] + 1):
            c = creases.get(segs[i])
            if c is None:
                continue
            ep = c.b
            for nsk in segs_by_vertex.get(ep, []):
                nc = creases.get(nsk)
                if nc and nc.active and nc.line_key not in exclude_lks:
                    return i + 1
    else:
        for i in range(rng['end_idx'], rng['start_idx'] - 1, -1):
            c = creases.get(segs[i])
            if c is None:
                continue
            ep = c.a
            for nsk in segs_by_vertex.get(ep, []):
                nc = creases.get(nsk)
                if nc and nc.active and nc.line_key not in exclude_lks:
                    return i
    return -1


def find_intersecting_line_vertex(origin_pt, ray_lk, ray_angle, exclude_lks):
    ld = lines.get(ray_lk)
    if ld is None:
        return None
    rng = compute_ray_range(origin_pt, ray_lk, ray_angle)
    if rng is None:
        return None
    fwd = line_fwd_angle(ray_lk)
    go_fwd = angle_diff(ray_angle, fwd) < 90
    segs = ld.seg_keys
    if go_fwd:
        for i in range(rng['start_idx'], rng['end_idx'] + 1):
            c = creases.get(segs[i])
            if c is None:
                continue
            ep = c.b
            for nsk in segs_by_vertex.get(ep, []):
                nc = creases.get(nsk)
                if nc and nc.active and nc.line_key not in exclude_lks:
                    cross_ld = lines.get(nc.line_key)
                    if cross_ld is None:
                        continue
                    try:
                        j = cross_ld.verts.index(ep)
                        return Vertex(pt=ep, line_key=nc.line_key, seg_idx=j)
                    except ValueError:
                        pass
    else:
        for i in range(rng['end_idx'], rng['start_idx'] - 1, -1):
            c = creases.get(segs[i])
            if c is None:
                continue
            ep = c.a
            for nsk in segs_by_vertex.get(ep, []):
                nc = creases.get(nsk)
                if nc and nc.active and nc.line_key not in exclude_lks:
                    cross_ld = lines.get(nc.line_key)
                    if cross_ld is None:
                        continue
                    try:
                        j = cross_ld.verts.index(ep)
                        return Vertex(pt=ep, line_key=nc.line_key, seg_idx=j)
                    except ValueError:
                        pass
    return None


def activate_ray(start_vpt, lk, ray_angle):
    ld = lines.get(lk)
    if ld is None:
        return
    rng = compute_ray_range(start_vpt, lk, ray_angle)
    if rng is None:
        return
    si, ei, bvi = rng['start_idx'], rng['end_idx'], rng['boundary_vert_idx']
    for i in range(si, ei + 1):
        sk = ld.seg_keys[i]
        if sk in creases:
            set_crease_active(sk, True)
    add_vertices_for_range(lk, si, ei)
    fwd = line_fwd_angle(lk)
    edge_vert_idx = len(ld.seg_keys) if angle_diff(ray_angle, fwd) < 90 else 0
    if 0 <= edge_vert_idx < len(ld.verts):
        ek = v_key(lk, edge_vert_idx)
        if ek not in vertices:
            vertices[ek] = Vertex(pt=ld.verts[edge_vert_idx], line_key=lk, seg_idx=edge_vert_idx)
    add_ray_boundary(lk, bvi)
    apply_double_bound_deletion()


def activate_partial_ray_segments(lk, ray_angle, start_vi, stop_vi):
    ld = lines.get(lk)
    if ld is None:
        return
    fwd = line_fwd_angle(lk)
    go_fwd = angle_diff(ray_angle, fwd) < 90
    segs = ld.seg_keys
    if go_fwd:
        for i in range(start_vi, stop_vi):
            sk = segs[i]
            if sk in creases:
                set_crease_active(sk, True)
    else:
        for i in range(stop_vi, start_vi):
            sk = segs[i]
            if sk in creases:
                set_crease_active(sk, True)


def activate_60_neighbors(boundary_vpt, unset_side, trimmed_lk):
    fwd = line_fwd_angle(trimmed_lk)
    unset_angle = (fwd + 180) % 360 if unset_side == 'left' else fwd
    neighbors = get_neighbors_from_point(boundary_vpt, unset_angle, trimmed_lk)
    for n in neighbors:
        activate_ray(boundary_vpt, n['line_key'], n['ray_angle'])


def is_edge_vertex(v: Vertex) -> bool:
    ld = lines.get(v.line_key)
    if ld is None:
        return False
    return v.seg_idx == 0 or v.seg_idx == len(ld.seg_keys)


def auto_detect_side(v: Vertex) -> Optional[str]:
    ld = lines.get(v.line_key)
    if ld is None:
        return None
    # v.seg_idx is a vertex index. Segs to its left: [0..seg_idx-1]. Segs to its right: [seg_idx..].
    first = v.seg_idx
    while first > 0 and (c := creases.get(ld.seg_keys[first - 1])) and c.active:
        first -= 1
    last = v.seg_idx
    while last < len(ld.seg_keys) and (c := creases.get(ld.seg_keys[last])) and c.active:
        last += 1
    last -= 1  # last active seg index
    has_left = first > 0
    has_right = last < len(ld.seg_keys) - 1
    if has_left and has_right:
        if (v.seg_idx - first) <= (last - v.seg_idx + 1):
            return 'right'
        else:
            return 'left'
    if has_left:
        return 'right'
    if has_right:
        return 'left'
    return None


def is_doubly_bounded(lk: str, seg_idx: int) -> bool:
    bounds = ray_boundaries.get(lk, set())
    if len(bounds) < 2:
        return False
    return any(b < seg_idx for b in bounds) and any(b > seg_idx for b in bounds)


def point_is_doubly_bounded(vpt: tuple, exclude_lks: set) -> bool:
    if vpt in dead_vertices:
        return True
    for nsk in segs_by_vertex.get(vpt, []):
        nc = creases.get(nsk)
        if nc is None:
            continue
        cnld = lines.get(nc.line_key)
        if cnld is None:
            continue
        try:
            j = cnld.verts.index(vpt)
            if is_doubly_bounded(nc.line_key, j):
                return True
        except ValueError:
            pass
    return False


def ray_would_intersect(v: Vertex, side: str) -> bool:
    ld = lines.get(v.line_key)
    if ld is None:
        return False
    fwd = line_fwd_angle(v.line_key)
    unset_angle = (fwd + 180) % 360 if side == 'left' else fwd
    all_candidates = set()
    neighbor_lks = {v.line_key}
    for sk in segs_by_vertex.get(v.pt, []):
        c = creases.get(sk)
        if c is None or c.line_key == v.line_key:
            continue
        if c.a == v.pt:
            sdx, sdy = direction_to_float(c.b[0] - c.a[0], c.b[1] - c.a[1], c.line_key)
        else:
            sdx, sdy = direction_to_float(c.a[0] - c.b[0], c.a[1] - c.b[1], c.line_key)
        seg_angle = vec_angle_float(sdx, sdy)
        if angle_diff(seg_angle, unset_angle) < 90:
            neighbor_lks.add(c.line_key)
            rng = compute_ray_range(v.pt, c.line_key, seg_angle)
            all_candidates.update(get_ray_seg_set(c.line_key, rng))
    return ray_intersects_active(all_candidates, neighbor_lks)


def guided_range_has_active_segments(origin_pt, ray_lk, ray_angle, exclude_lks) -> bool:
    ld = lines.get(ray_lk)
    if ld is None:
        return False
    rng = compute_ray_range(origin_pt, ray_lk, ray_angle)
    if rng is None:
        return False
    bvi = rng['boundary_vert_idx']
    fwd = line_fwd_angle(ray_lk)
    go_fwd = angle_diff(ray_angle, fwd) < 90
    intersect_vi = find_ray_intersection_vert_idx(origin_pt, ray_lk, ray_angle, exclude_lks)
    if intersect_vi < 0:
        return False
    segs = ld.seg_keys
    if go_fwd:
        for i in range(bvi, intersect_vi):
            c = creases.get(segs[i])
            if c and c.active:
                return True
    else:
        for i in range(intersect_vi, bvi):
            c = creases.get(segs[i])
            if c and c.active:
                return True
    return False


def guided_vertex_would_hit_boundary(v: Vertex, origin_pt, ray_lk, ray_angle) -> bool:
    ld = lines.get(ray_lk)
    if ld is None:
        return False
    fwd = line_fwd_angle(ray_lk)
    go_forward = angle_diff(ray_angle, fwd) < 90
    unset_angle = fwd if go_forward else (fwd + 180) % 360
    exclude_lks = {ray_lk}
    candidates = []
    for sk in segs_by_vertex.get(v.pt, []):
        c = creases.get(sk)
        if c is None or c.line_key == ray_lk:
            continue
        if c.a == v.pt:
            sdx, sdy = direction_to_float(c.b[0] - c.a[0], c.b[1] - c.a[1], c.line_key)
        else:
            sdx, sdy = direction_to_float(c.a[0] - c.b[0], c.a[1] - c.b[1], c.line_key)
        seg_angle = vec_angle_float(sdx, sdy)
        if angle_diff(seg_angle, unset_angle) < 90:
            candidates.append({'line_key': c.line_key, 'ray_angle': seg_angle})
    for n in candidates:
        exclude_lks.add(n['line_key'])
    ray_family = line_family(ray_lk)
    for n in candidates:
        nld = lines.get(n['line_key'])
        if nld is None:
            continue
        rng = compute_ray_range(v.pt, n['line_key'], n['ray_angle'])
        if rng is None:
            continue
        nfwd = line_fwd_angle(n['line_key'])
        go_fwd = angle_diff(n['ray_angle'], nfwd) < 90
        segs = nld.seg_keys
        if go_fwd:
            for i in range(rng['start_idx'], rng['end_idx'] + 1):
                cr = creases.get(segs[i])
                if cr is None:
                    continue
                ep = cr.b
                if ep in dead_vertices:
                    return True
                own_bounds = ray_boundaries.get(n['line_key'], set())
                if (i + 1) in own_bounds:
                    return True
                if point_is_doubly_bounded(ep, exclude_lks):
                    return True
                for nsk in segs_by_vertex.get(ep, []):
                    nc = creases.get(nsk)
                    if nc and nc.active and nc.line_key not in exclude_lks:
                        if line_family(nc.line_key) == ray_family:
                            return True
                        return False
        else:
            for i in range(rng['end_idx'], rng['start_idx'] - 1, -1):
                cr = creases.get(segs[i])
                if cr is None:
                    continue
                ep = cr.a
                if ep in dead_vertices:
                    return True
                own_bounds = ray_boundaries.get(n['line_key'], set())
                if i in own_bounds:
                    return True
                if point_is_doubly_bounded(ep, exclude_lks):
                    return True
                for nsk in segs_by_vertex.get(ep, []):
                    nc = creases.get(nsk)
                    if nc and nc.active and nc.line_key not in exclude_lks:
                        if line_family(nc.line_key) == ray_family:
                            return True
                        return False
    return False


def simulate_guided_vertices(origin_pt, blocked_neighbor, exclude_lks) -> int:
    ray_lk = blocked_neighbor['line_key']
    ray_angle = blocked_neighbor['ray_angle']
    ld = lines.get(ray_lk)
    if ld is None:
        return 0
    rng = compute_ray_range(origin_pt, ray_lk, ray_angle)
    if rng is None:
        return 0
    bvi = rng['boundary_vert_idx']
    fwd = line_fwd_angle(ray_lk)
    go_fwd = angle_diff(ray_angle, fwd) < 90
    intersect_vi = find_ray_intersection_vert_idx(origin_pt, ray_lk, ray_angle, exclude_lks)
    if intersect_vi < 0:
        return 0
    count = 0
    if go_fwd:
        for i in range(bvi + 1, intersect_vi):
            if i < len(ld.verts):
                vpt = ld.verts[i]
                v = Vertex(pt=vpt, line_key=ray_lk, seg_idx=i)
                if not guided_vertex_would_hit_boundary(v, origin_pt, ray_lk, ray_angle):
                    count += 1
    else:
        for i in range(intersect_vi + 1, bvi):
            if i < len(ld.verts):
                vpt = ld.verts[i]
                v = Vertex(pt=vpt, line_key=ray_lk, seg_idx=i)
                if not guided_vertex_would_hit_boundary(v, origin_pt, ray_lk, ray_angle):
                    count += 1
    return count


def vertex_status(v: Vertex) -> str:
    global dead_vertices
    auto = auto_detect_side(v)
    if auto:
        if not ray_would_intersect(v, auto):
            return 'free'
    else:
        lb = ray_would_intersect(v, 'left')
        rb = ray_would_intersect(v, 'right')
        if not lb and not rb:
            return 'free'
    ld = lines.get(v.line_key)
    if ld is None:
        return 'free'
    fwd = line_fwd_angle(v.line_key)
    for side in ['left', 'right']:
        unset_angle = (fwd + 180) % 360 if side == 'left' else fwd
        neighbors = get_neighbors_from_point(v.pt, unset_angle, v.line_key)
        exclude_lks = {v.line_key}
        for n in neighbors:
            exclude_lks.add(n['line_key'])
        for n in neighbors:
            rng = compute_ray_range(v.pt, n['line_key'], n['ray_angle'])
            segs = get_ray_seg_set(n['line_key'], rng)
            if not ray_intersects_active(segs, exclude_lks):
                continue
            if simulate_guided_vertices(v.pt, n, exclude_lks) > 0:
                return 'blocked'
    vk = v_key(v.line_key, v.seg_idx)
    if vk in vertices:
        del vertices[vk]
    dead_vertices.add(v.pt)
    return 'free'


def enter_guided_mode(origin_pt, blocked_neighbor, exclude_lks):
    global guided_mode, guided_vertices, guided_hovered
    global guided_ray_origin, guided_ray_lk, guided_ray_angle
    ray_lk = blocked_neighbor['line_key']
    ray_angle = blocked_neighbor['ray_angle']
    ld = lines.get(ray_lk)
    if ld is None:
        return
    rng = compute_ray_range(origin_pt, ray_lk, ray_angle)
    if rng is None:
        return
    bvi = rng['boundary_vert_idx']
    fwd = line_fwd_angle(ray_lk)
    go_fwd = angle_diff(ray_angle, fwd) < 90
    intersect_vi = find_ray_intersection_vert_idx(origin_pt, ray_lk, ray_angle, exclude_lks)
    if intersect_vi < 0:
        return
    new_guided = {}
    if go_fwd:
        for i in range(bvi + 1, intersect_vi):
            if i < len(ld.verts):
                vpt = ld.verts[i]
                v = Vertex(pt=vpt, line_key=ray_lk, seg_idx=i)
                if not guided_vertex_would_hit_boundary(v, origin_pt, ray_lk, ray_angle):
                    new_guided[v_key(ray_lk, i)] = v
    else:
        for i in range(intersect_vi + 1, bvi):
            if i < len(ld.verts):
                vpt = ld.verts[i]
                v = Vertex(pt=vpt, line_key=ray_lk, seg_idx=i)
                if not guided_vertex_would_hit_boundary(v, origin_pt, ray_lk, ray_angle):
                    new_guided[v_key(ray_lk, i)] = v
    if new_guided:
        guided_vertices = new_guided
        guided_mode = True
        guided_hovered = None
        guided_ray_origin = origin_pt
        guided_ray_lk = ray_lk
        guided_ray_angle = ray_angle


def enqueue_guided_mode(origin_pt, blocked_neighbor, exclude_lks):
    guided_queue.append((origin_pt, blocked_neighbor, exclude_lks))


def drain_guided_queue() -> bool:
    while guided_queue:
        origin_pt, blocked_neighbor, exclude_lks = guided_queue.pop(0)
        suppress = guided_range_has_active_segments(
            origin_pt, blocked_neighbor['line_key'],
            blocked_neighbor['ray_angle'], exclude_lks
        )
        if not suppress:
            enter_guided_mode(origin_pt, blocked_neighbor, exclude_lks)
            if guided_mode:
                return True
    return False


def on_guided_vertex_selected(gv: Vertex):
    global guided_mode
    push_undo()
    ray_origin = guided_ray_origin
    ray_lk = guided_ray_lk
    ray_angle = guided_ray_angle
    clear_guided_mode()
    ld = lines.get(ray_lk)
    if ld is None:
        return
    rng = compute_ray_range(ray_origin, ray_lk, ray_angle)
    if rng is None:
        return
    bvi = rng['boundary_vert_idx']
    fwd = line_fwd_angle(ray_lk)
    go_fwd = angle_diff(ray_angle, fwd) < 90
    segs = ld.seg_keys
    if go_fwd:
        for i in range(bvi, gv.seg_idx):
            sk = segs[i]
            if sk in creases:
                set_crease_active(sk, True)
    else:
        for i in range(gv.seg_idx, bvi):
            sk = segs[i]
            if sk in creases:
                set_crease_active(sk, True)
    vpoint = gv.pt
    unset_angle = fwd if go_fwd else (fwd + 180) % 360
    exclude_lks = {ray_lk}
    candidates = []
    for sk in segs_by_vertex.get(vpoint, []):
        c = creases.get(sk)
        if c is None or c.line_key == ray_lk:
            continue
        if c.a == vpoint:
            sdx, sdy = direction_to_float(c.b[0] - c.a[0], c.b[1] - c.a[1], c.line_key)
        else:
            sdx, sdy = direction_to_float(c.a[0] - c.b[0], c.a[1] - c.b[1], c.line_key)
        seg_angle = vec_angle_float(sdx, sdy)
        if angle_diff(seg_angle, unset_angle) < 90:
            candidates.append({'line_key': c.line_key, 'ray_angle': seg_angle})
    for n in candidates:
        exclude_lks.add(n['line_key'])
    auto_trigger_v = None
    for n in candidates:
        ray_rng = compute_ray_range(vpoint, n['line_key'], n['ray_angle'])
        segs_set = get_ray_seg_set(n['line_key'], ray_rng)
        if not ray_intersects_active(segs_set, exclude_lks):
            activate_ray(vpoint, n['line_key'], n['ray_angle'])
        else:
            intersect_vi = find_ray_intersection_vert_idx(vpoint, n['line_key'], n['ray_angle'], exclude_lks)
            if intersect_vi >= 0:
                activate_partial_ray_segments(n['line_key'], n['ray_angle'], ray_rng['boundary_vert_idx'], intersect_vi)
                cross_v = find_intersecting_line_vertex(vpoint, n['line_key'], n['ray_angle'], exclude_lks)
                if cross_v:
                    auto_trigger_v = cross_v
    if auto_trigger_v:
        on_blocked_vertex(auto_trigger_v)
    if not guided_mode:
        drain_guided_queue()
    invalidate_bg()


def do_trim(v: Vertex, side: str):
    global hovered_vertex, hovered_blocked, info_text
    push_undo()
    trace_and_remove(v.line_key, v.seg_idx, side)
    add_ray_boundary(v.line_key, v.seg_idx)
    apply_double_bound_deletion()
    if not is_edge_vertex(v):
        activate_60_neighbors(v.pt, side, v.line_key)
    hovered_vertex = None
    hovered_blocked = False
    invalidate_bg()
    info_text = "Trimmed. Hover a vertex to trim further, or press C to clear."


def on_blocked_vertex(v: Vertex):
    global hovered_vertex, hovered_blocked
    push_undo()
    side = auto_detect_side(v)
    if side is None:
        return
    trace_and_remove(v.line_key, v.seg_idx, side)
    add_ray_boundary(v.line_key, v.seg_idx)
    apply_double_bound_deletion()
    ld = lines.get(v.line_key)
    if ld is None:
        return
    fwd = line_fwd_angle(v.line_key)
    unset_angle = (fwd + 180) % 360 if side == 'left' else fwd
    neighbors = get_neighbors_from_point(v.pt, unset_angle, v.line_key)
    exclude_lks = {v.line_key}
    for n in neighbors:
        exclude_lks.add(n['line_key'])
    for n in neighbors:
        rng = compute_ray_range(v.pt, n['line_key'], n['ray_angle'])
        segs = get_ray_seg_set(n['line_key'], rng)
        if not ray_intersects_active(segs, exclude_lks):
            activate_ray(v.pt, n['line_key'], n['ray_angle'])
        else:
            enqueue_guided_mode(v.pt, n, exclude_lks)
    hovered_vertex = None
    hovered_blocked = False
    invalidate_bg()
    if not guided_mode:
        drain_guided_queue()


# ---------------------------------------------------------------------------
# Hit testing
# ---------------------------------------------------------------------------

def seg_dist_px(mx, my, va, vb) -> float:
    ax, ay = vertex_px(va[0], va[1])
    bx, by = vertex_px(vb[0], vb[1])
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-10:
        return math.hypot(mx - ax, my - ay)
    t = max(0, min(1, ((mx - ax) * dx + (my - ay) * dy) / l2))
    return math.hypot(mx - (ax + t * dx), my - (ay + t * dy))


def find_nearest_line(mx, my) -> Optional[str]:
    idx = screen_to_crease_idx(mx, my, threshold_px=10.0)
    if idx is None:
        return None
    return crease_line_keys[idx]


def find_nearest_vertex(mx, my):
    best_v = None
    best_vk = None
    bd = 14.0
    for vk, v in vertices.items():
        px_, py_ = vertex_px(v.pt[0], v.pt[1])
        d = math.hypot(mx - px_, my - py_)
        if d < bd:
            bd = d
            best_v = v
            best_vk = vk
    if best_v is None:
        return (None, False)
    return (best_v, _vertex_blocked_cache.get(best_vk, False))


def find_nearest_guided_vertex(mx, my) -> Optional[Vertex]:
    best = None
    bd = 14.0
    for v in guided_vertices.values():
        px_, py_ = vertex_px(v.pt[0], v.pt[1])
        d = math.hypot(mx - px_, my - py_)
        if d < bd:
            bd = d
            best = v
    return best


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

COLOR_BG = (250, 250, 248)
COLOR_INACTIVE = (40, 38, 32, 46)    # with alpha — handled manually
COLOR_ACTIVE = (192, 57, 43)
COLOR_HOVERED_LINE = (230, 126, 34)
COLOR_VERTEX_FREE = (192, 57, 43)
COLOR_VERTEX_FREE_HOV = (231, 76, 60)
COLOR_VERTEX_BLOCKED = (36, 113, 163)
COLOR_VERTEX_BLOCKED_HOV = (26, 82, 118)
COLOR_GUIDED = (142, 68, 173)
COLOR_GUIDED_HOV = (108, 52, 131)
COLOR_WHITE = (255, 255, 255)
COLOR_PANEL = (240, 239, 235)
COLOR_TEXT = (26, 26, 24)
COLOR_TEXT2 = (100, 100, 95)
COLOR_BORDER = (200, 198, 195)


_BG_PAD = 1   # render bg at (1 + 2*PAD) x screen size so panning has slack

def _rebuild_bg(sw, sh):
    """Render inactive grid onto an oversized surface centred on current cam.
    Panning blits a shifted sub-rect without re-rendering."""
    global _bg_surface, _bg_cam_state
    bw = sw * (1 + 2 * _BG_PAD)
    bh = sh * (1 + 2 * _BG_PAD)
    ox = sw * _BG_PAD   # bg pixel [ox,oy] maps to screen pixel [0,0]
    oy = sh * _BG_PAD
    bg_cam_x = cam_x + ox
    bg_cam_y = cam_y + oy

    surf = pygame.Surface((bw, bh))
    surf.fill(COLOR_BG)
    if crease_norm is not None and len(crease_norm) > 0:
        pts = np.empty_like(crease_norm)
        pts[:, 0] = bg_cam_x + crease_norm[:, 0] * cam_scale
        pts[:, 1] = bg_cam_y + crease_norm[:, 1] * cam_scale
        pts[:, 2] = bg_cam_x + crease_norm[:, 2] * cam_scale
        pts[:, 3] = bg_cam_y + crease_norm[:, 3] * cam_scale
        margin = 4
        visible = ~(
            (np.maximum(pts[:, 0], pts[:, 2]) < -margin) |
            (np.minimum(pts[:, 0], pts[:, 2]) > bw + margin) |
            (np.maximum(pts[:, 1], pts[:, 3]) < -margin) |
            (np.minimum(pts[:, 1], pts[:, 3]) > bh + margin)
        )
        for idx in np.where(visible)[0]:
            c_obj = creases[crease_keys[idx]]
            if not c_obj.active:
                pygame.draw.line(surf, (180, 178, 170),
                                 (float(pts[idx, 0]), float(pts[idx, 1])),
                                 (float(pts[idx, 2]), float(pts[idx, 3])), 1)
    _bg_surface = surf
    _bg_cam_state = (cam_x, cam_y, cam_scale, sw, sh, ox, oy)


def _needs_bg_rebuild(sw, sh):
    """True if camera has drifted enough that the oversized bg no longer covers the screen."""
    if _bg_surface is None or len(_bg_cam_state) < 7:
        return True
    stored_cx, stored_cy, stored_scale, _, _, ox, oy = _bg_cam_state
    if stored_scale != cam_scale:
        return True   # zoom changed — always rebuild
    dx = int(cam_x - stored_cx)
    dy = int(cam_y - stored_cy)
    bx = ox - dx
    by = oy - dy
    bw, bh = _bg_surface.get_size()
    return bx < 0 or by < 0 or bx + sw > bw or by + sh > bh


def _bg_blit_src_rect(sw, sh):
    """Return pygame.Rect into _bg_surface that maps to the current screen view."""
    stored_cx, stored_cy, _, _, _, ox, oy = _bg_cam_state
    dx = int(cam_x - stored_cx)
    dy = int(cam_y - stored_cy)
    return pygame.Rect(ox - dx, oy - dy, sw, sh)


def draw(screen: pygame.Surface, font_sm, font_xs):
    global _bg_surface, _bg_cam_state
    sw, sh = SCREEN_W, SCREEN_H
    has_sel = selected_line is not None

    # --- Background: cached inactive grid (oversized, panning shifts blit rect) ---
    if _needs_bg_rebuild(sw, sh):
        _rebuild_bg(sw, sh)
    src_rect = _bg_blit_src_rect(sw, sh)
    screen.blit(_bg_surface, (0, 0), src_rect)

    # --- Active lines drawn on top (iterate only active set, not all creases) ---
    for i in active_crease_indices:
        n = crease_norm[i]
        ax = cam_x + float(n[0]) * cam_scale
        ay = cam_y + float(n[1]) * cam_scale
        bx = cam_x + float(n[2]) * cam_scale
        by = cam_y + float(n[3]) * cam_scale
        if max(ax, bx) < -4 or min(ax, bx) > sw + 4 or max(ay, by) < -4 or min(ay, by) > sh + 4:
            continue
        pygame.draw.line(screen, COLOR_ACTIVE, (ax, ay), (bx, by), 2)

    # --- Hovered line (only when no line selected) ---
    if not has_sel and hovered_line is not None:
        for sk, c in creases.items():
            if c.line_key == hovered_line:
                i = sk_to_idx.get(sk)
                if i is None:
                    continue
                n = crease_norm[i]
                ax = cam_x + float(n[0]) * cam_scale
                ay = cam_y + float(n[1]) * cam_scale
                bx = cam_x + float(n[2]) * cam_scale
                by = cam_y + float(n[3]) * cam_scale
                if max(ax, bx) < -4 or min(ax, bx) > sw + 4 or max(ay, by) < -4 or min(ay, by) > sh + 4:
                    continue
                pygame.draw.line(screen, COLOR_HOVERED_LINE, (ax, ay), (bx, by), 2)

    # --- Vertices ---
    if has_sel:
        if guided_mode:
            for v in guided_vertices.values():
                px_, py_ = vertex_px(v.pt[0], v.pt[1])
                is_hv = guided_hovered and v.pt == guided_hovered.pt
                r = 6 if is_hv else 4
                pygame.draw.circle(screen, COLOR_GUIDED_HOV if is_hv else COLOR_GUIDED, (int(px_), int(py_)), r)
                if is_hv:
                    pygame.draw.circle(screen, COLOR_WHITE, (int(px_), int(py_)), r, 2)
        else:
            if _vertex_status_dirty:
                _refresh_vertex_statuses()
            for vk, v in vertices.items():
                blocked = _vertex_blocked_cache.get(vk, False)
                is_hv = hovered_vertex and v.pt == hovered_vertex.pt and v.line_key == hovered_vertex.line_key
                r = 6 if is_hv else 4
                col = (COLOR_VERTEX_BLOCKED_HOV if is_hv else COLOR_VERTEX_BLOCKED) if blocked else (COLOR_VERTEX_FREE_HOV if is_hv else COLOR_VERTEX_FREE)
                px_, py_ = vertex_px(v.pt[0], v.pt[1])
                pygame.draw.circle(screen, col, (int(px_), int(py_)), r)
                if is_hv:
                    pygame.draw.circle(screen, COLOR_WHITE, (int(px_), int(py_)), r, 2)

    # --- Grid centre marker ---
    if _center_vertex is not None:
        cx, cy = vertex_px(_center_vertex[0], _center_vertex[1])
        pygame.draw.circle(screen, (0, 0, 0), (int(cx), int(cy)), 5)

    # --- Popup ---
    if popup_active and popup_buttons:
        pygame.draw.rect(screen, COLOR_PANEL, popup_rect, border_radius=8)
        pygame.draw.rect(screen, COLOR_BORDER, popup_rect, 1, border_radius=8)
        for btn_rect, label in popup_buttons:
            col = (220, 218, 212) if btn_rect.collidepoint(pygame.mouse.get_pos()) else COLOR_PANEL
            pygame.draw.rect(screen, col, btn_rect, border_radius=6)
            pygame.draw.rect(screen, COLOR_BORDER, btn_rect, 1, border_radius=6)
            txt = font_sm.render(label, True, COLOR_TEXT)
            screen.blit(txt, (btn_rect.x + 8, btn_rect.y + 6))

    draw_ui(screen, font_sm, font_xs)


def draw_ui(screen, font_sm, font_xs):
    pass


# ---------------------------------------------------------------------------
# Popup
# ---------------------------------------------------------------------------

def show_popup(mx, my):
    global popup_active, popup_rect, popup_buttons
    bw, bh, gap, pad = 160, 28, 6, 10
    total_h = pad * 2 + bh * 3 + gap * 2
    ww, wh = pygame.display.get_surface().get_size()
    rx = min(mx + 8, ww - bw - pad * 2 - 4)
    ry = min(my + 8, wh - total_h - 4)
    popup_rect = pygame.Rect(rx, ry, bw + pad * 2, total_h)
    popup_buttons = [
        (pygame.Rect(rx + pad, ry + pad, bw, bh), "Unset left side"),
        (pygame.Rect(rx + pad, ry + pad + bh + gap, bw, bh), "Unset right side"),
        (pygame.Rect(rx + pad, ry + pad + (bh + gap) * 2, bw, bh), "Cancel"),
    ]
    popup_active = True


def close_popup():
    global popup_active, picked_vertex
    popup_active = False
    picked_vertex = None


def handle_popup_click(mx, my) -> bool:
    """Returns True if click was consumed by popup."""
    global popup_active
    if not popup_active:
        return False
    for btn_rect, label in popup_buttons:
        if btn_rect.collidepoint(mx, my):
            v = picked_vertex   # save before close_popup clears it
            close_popup()
            if label == "Unset left side" and v:
                do_trim(v, 'left')
            elif label == "Unset right side" and v:
                do_trim(v, 'right')
            return True
    # Click outside popup — close it
    close_popup()
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def set_n(new_n):
    global N
    N = new_n
    build()


def main():
    global N, hovered_line, hovered_vertex, hovered_blocked
    global selected_line, guided_hovered, info_text, picked_vertex
    global cam_x, cam_y, cam_scale, SCREEN_W, SCREEN_H

    pygame.init()
    info = pygame.display.Info()
    sw, sh = info.current_w, info.current_h
    screen = pygame.display.set_mode((sw, sh), pygame.RESIZABLE)
    SCREEN_W, SCREEN_H = sw, sh
    pygame.display.set_caption("Origami Triangle Grid")
    clock = pygame.time.Clock()

    font_sm = pygame.font.SysFont("Arial", 13)
    font_xs = pygame.font.SysFont("Arial", 11)

    init_camera(sw, sh)
    build()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                SCREEN_W, SCREEN_H = event.w, event.h
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            elif event.type == pygame.MOUSEWHEEL:
                # Zoom toward/away from mouse position
                zoom_factor = 1.1 if event.y > 0 else (1.0 / 1.1)
                new_scale = cam_x_before = cam_scale * zoom_factor
                # Clamp scale: min shows full grid in ~200px, max is very zoomed
                new_scale = max(100.0, min(new_scale, cam_scale * 10 if zoom_factor > 1 else cam_scale))
                new_scale = max(100.0, min(50000.0, cam_scale * zoom_factor))
                # Zoom toward mouse: keep mouse position fixed in grid space
                cam_x = mx - (mx - cam_x) * (new_scale / cam_scale)
                cam_y = my - (my - cam_y) * (new_scale / cam_scale)
                cam_scale = new_scale

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    for sk in list(active_crease_indices):
                        set_crease_active(crease_keys[sk], False)
                    reset_state()
                    undo_stack.clear()
                    init_camera(SCREEN_W, SCREEN_H)
                    invalidate_bg()
                elif event.key == pygame.K_c:
                    for sk in list(active_crease_indices):
                        set_crease_active(crease_keys[sk], False)
                    reset_state()
                    undo_stack.clear()
                    invalidate_bg()
                elif event.key == pygame.K_z and (event.mod & pygame.KMOD_CTRL):
                    close_popup()
                    pop_undo()
                elif event.key == pygame.K_ESCAPE:
                    close_popup()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if popup_active:
                    handle_popup_click(mx, my)
                elif guided_mode:
                    gv = find_nearest_guided_vertex(mx, my)
                    if gv:
                        on_guided_vertex_selected(gv)
                        info_text = ("Select a point along the blocked ray."
                                     if guided_mode else
                                     "Hover a vertex on any active line to trim.")
                elif selected_line:
                    v, blocked = find_nearest_vertex(mx, my)
                    if v:
                        if blocked:
                            on_blocked_vertex(v)
                            info_text = ("Select a point along the blocked ray."
                                         if guided_mode else
                                         "Intersection vertex handled. Hover a vertex to continue.")
                        else:
                            auto = auto_detect_side(v)
                            if auto:
                                do_trim(v, auto)
                            else:
                                lb = ray_would_intersect(v, 'left')
                                rb = ray_would_intersect(v, 'right')
                                if lb and not rb:
                                    do_trim(v, 'right')
                                elif rb and not lb:
                                    do_trim(v, 'left')
                                else:
                                    picked_vertex = v
                                    show_popup(mx, my)
                else:
                    lk = find_nearest_line(mx, my)
                    if lk:
                        push_undo()
                        selected_line = lk
                        hovered_line = None
                        ld = lines[lk]
                        for sk in ld.seg_keys:
                            set_crease_active(sk, True)
                        add_vertices_for_range(lk, 0, len(ld.seg_keys))
                        invalidate_bg()
                        info_text = "Hover a vertex on any active line to trim."
                        pygame.display.set_caption("Origami Triangle Grid")

        # WASD pan (held keys, dt-based)
        keys = pygame.key.get_pressed()
        pan = PAN_SPEED * dt
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            cam_x += pan
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            cam_x -= pan
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            cam_y += pan
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            cam_y -= pan

        # Hover logic (outside event loop for smooth update)
        if not popup_active:
            if guided_mode:
                prev = guided_hovered
                guided_hovered = find_nearest_guided_vertex(mx, my)
                if guided_hovered != prev:
                    info_text = ("Click to place fold here."
                                 if guided_hovered else
                                 "Select a point along the blocked ray to place the fold.")
            elif selected_line:
                v, blocked = find_nearest_vertex(mx, my)
                hovered_vertex = v
                hovered_blocked = blocked
            else:
                prev_lk = hovered_line
                hovered_line = find_nearest_line(mx, my)
                if hovered_line != prev_lk:
                    info_text = ("Click to select this fold line."
                                 if hovered_line else
                                 "Hover a line to highlight it, click to select.")

        draw(screen, font_sm, font_xs)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()