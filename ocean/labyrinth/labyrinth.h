/* Labyrinth — tilt-a-ball marble maze Ocean env.
 *
 * This header wires the pure physics (labyrinth_physics.h) and the maze
 * generator (labyrinth_maze.h) into the Ocean RL env API:
 *   LabyrinthEnv   — the struct with observations/actions/rewards/log
 *   c_reset        — start an episode (fresh maze + ball at start)
 *   c_step         — apply one action, advance physics, write reward/obs
 *   c_render       — raylib 3D view (board + tilt + pits + ball + path)
 *   c_close        — close the window
 *
 * binding.c is thin — just macros + my_init/my_log.
 * labyrinth.c (standalone) uses this same header for the game loop.
 * test_labyrinth.c pulls in labyrinth_physics.h directly (no raylib link).
 */

#ifndef LABYRINTH_H
#define LABYRINTH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include "labyrinth_physics.h"
#include "labyrinth_maze.h"

// ===== Render constants =====

#define BOARD_CENTER_Y 0.06f // lift board above the ground grid
#define WALL_THICKNESS WALL_THICKNESS_M
#define WALL_HEIGHT 0.012f
#define PI_F 3.14159265358979323846f

// Slab-top mesh resolution. ~0.6mm cells at 500×400 → sub-pixel staircase.
#define SLAB_GRID_NX 500
#define SLAB_GRID_NZ 400

// Y-offsets above the slab top for layered surfaces (in meters, 0.1mm each).
// Order (low to high): grid → rim ring → goal disk ≈ path line.
#define Z_ABOVE_SLAB_RIM_RING  1.0e-4f
#define Z_ABOVE_SLAB_GOAL_DISK 1.5e-4f
#define Z_ABOVE_SLAB_PATH_LINE 1.5e-4f

// ===== Render helpers =====
// All static inline so this header can be included in multiple translation
// units safely (standalone main + binding).

static inline Vector3 ball_world_position(const Labyrinth* env) {
    // Physics ball_z is above the slab top plane; shift slab so it's centered
    // around y_local = 0 then translate up by BOARD_CENTER_Y after tilt.
    Vector3 local = {
        env->ball_x - 0.5f * BOARD_W,
        0.5f * BOARD_THICKNESS + env->ball_z,
        env->ball_y - 0.5f * BOARD_H,
    };
    Matrix tilt = MatrixRotateXYZ((Vector3){env->tilt_y, 0.0f, -env->tilt_x});
    Vector3 rotated = Vector3Transform(local, tilt);
    rotated.y += BOARD_CENTER_Y;
    return rotated;
}

static inline Matrix board_tilt_matrix(const Labyrinth* env) {
    return MatrixRotateXYZ((Vector3){env->tilt_y, 0.0f, -env->tilt_x});
}

// Draw a wall as a capsule (rectangle body + cylinder end caps) matching the
// physics collision shape. Must be called inside the board's tilted matrix.
static inline void draw_wall_local(const Wall* w) {
    float cx = 0.5f * (w->x1 + w->x2) - 0.5f * BOARD_W;
    float cz = 0.5f * (w->y1 + w->y2) - 0.5f * BOARD_H;
    float dx = w->x2 - w->x1;
    float dz = w->y2 - w->y1;
    float length = sqrtf(dx * dx + dz * dz);
    float angle = atan2f(dz, dx);
    float cy = 0.5f * BOARD_THICKNESS + 0.5f * WALL_HEIGHT;
    float cap_radius = 0.5f * WALL_THICKNESS;
    Color wood = (Color){180, 140, 80, 255};
    rlPushMatrix();
    rlTranslatef(cx, cy, cz);
    rlRotatef(-angle * 180.0f / PI_F, 0.0f, 1.0f, 0.0f);
    DrawCubeV((Vector3){0, 0, 0}, (Vector3){length, WALL_HEIGHT, WALL_THICKNESS}, wood);
    float half_len = 0.5f * length;
    DrawCylinder((Vector3){+half_len, -0.5f * WALL_HEIGHT, 0.0f}, cap_radius, cap_radius,
                 WALL_HEIGHT, 12, wood);
    DrawCylinder((Vector3){-half_len, -0.5f * WALL_HEIGHT, 0.0f}, cap_radius, cap_radius,
                 WALL_HEIGHT, 12, wood);
    rlPopMatrix();
}

// Vector3 wrapper around the physics header's labyrinth_rotate_by_quat.
static inline Vector3 rotate_by_quat(float qw, float qx, float qy, float qz, Vector3 v) {
    Vector3 r;
    labyrinth_rotate_by_quat(qw, qx, qy, qz, v.x, v.y, v.z, &r.x, &r.y, &r.z);
    return r;
}

// Draw the ball as a white sphere with 6 colored markers on its surface so
// rolling is visible. Each marker is a filled triangle fan tangent to the
// sphere, with the body-frame axis direction rotated by the ball's orientation.
static inline void draw_ball(const Labyrinth* env) {
    Vector3 ball_pos = ball_world_position(env);
    DrawSphere(ball_pos, BALL_RADIUS, (Color){230, 230, 230, 255});
    DrawSphereWires(ball_pos, BALL_RADIUS, 8, 8, (Color){180, 180, 180, 128});
    float marker_radius = BALL_RADIUS * 0.35f;
    float surface_offset = BALL_RADIUS * 1.001f;
    Vector3 body_axes[6] = {
        {+1, 0, 0}, {-1, 0, 0}, {0, +1, 0}, {0, -1, 0}, {0, 0, +1}, {0, 0, -1},
    };
    Color marker_colors[6] = {
        {230, 70, 70, 255},  {130, 30, 30, 255},
        {70, 230, 70, 255},  {30, 130, 30, 255},
        {70, 120, 230, 255}, {30, 60, 130, 255},
    };
    float qw = env->orient_w, qx = env->orient_x, qy = env->orient_y, qz = env->orient_z;
    for (int i = 0; i < 6; i++) {
        Vector3 body_unit = body_axes[i];
        Vector3 n = rotate_by_quat(qw, qx, qy, qz, body_unit);
        Vector3 center = {ball_pos.x + n.x * surface_offset, ball_pos.y + n.y * surface_offset,
                          ball_pos.z + n.z * surface_offset};
        // Gram-Schmidt: build a tangent basis at the marker center.
        Vector3 seed = (fabsf(n.y) < 0.9f) ? (Vector3){0, 1, 0} : (Vector3){1, 0, 0};
        Vector3 u = {seed.x - n.x * (n.x * seed.x + n.y * seed.y + n.z * seed.z),
                     seed.y - n.y * (n.x * seed.x + n.y * seed.y + n.z * seed.z),
                     seed.z - n.z * (n.x * seed.x + n.y * seed.y + n.z * seed.z)};
        float u_len = sqrtf(u.x * u.x + u.y * u.y + u.z * u.z);
        u.x /= u_len; u.y /= u_len; u.z /= u_len;
        Vector3 v = {n.y * u.z - n.z * u.y, n.z * u.x - n.x * u.z, n.x * u.y - n.y * u.x};
        const int segments = 16;
        Color color = marker_colors[i];
        for (int k = 0; k < segments; k++) {
            float a0 = (2.0f * PI_F * (float)k) / (float)segments;
            float a1 = (2.0f * PI_F * (float)(k + 1)) / (float)segments;
            Vector3 p0 = {center.x + marker_radius * (cosf(a0) * u.x + sinf(a0) * v.x),
                          center.y + marker_radius * (cosf(a0) * u.y + sinf(a0) * v.y),
                          center.z + marker_radius * (cosf(a0) * u.z + sinf(a0) * v.z)};
            Vector3 p1 = {center.x + marker_radius * (cosf(a1) * u.x + sinf(a1) * v.x),
                          center.y + marker_radius * (cosf(a1) * u.y + sinf(a1) * v.y),
                          center.z + marker_radius * (cosf(a1) * u.z + sinf(a1) * v.z)};
            DrawTriangle3D(center, p0, p1, color);
        }
    }
}

// Shared between slab grid mesh and per-hole rim ring so the seam is invisible.
static const Color SLAB_TOP_COLOR = (Color){70, 80, 90, 255};

// Returns 1 if any part of the (board-frame) cell rectangle is within
// (hole_r + extra) of any hole center.
static inline int cell_overlaps_any_hole(const Labyrinth* env, float bx0, float by0,
                                         float bx1, float by1, float extra) {
    for (int i = 0; i < env->num_holes; i++) {
        const Hole* h = &env->holes[i];
        float cx = fmaxf(bx0, fminf(h->cx, bx1));
        float cy = fmaxf(by0, fminf(h->cy, by1));
        float dx = h->cx - cx;
        float dy = h->cy - cy;
        float r = h->radius + extra;
        if (dx * dx + dy * dy < r * r)
            return 1;
    }
    return 0;
}

// Smooth annular ring just above the slab top, in slab color, from r=hole_r
// out to r=hole_r+outer_extra. Hides the residual staircase from the grid
// mesh. Drops any 1-slice segment whose midpoint sits inside another hole's
// pit (prevents crescent artifacts on overlapping barrier holes).
static inline void draw_hole_rim_ring(const Labyrinth* env, int hole_idx, float outer_extra) {
    const Hole* h = &env->holes[hole_idx];
    const int segments = 64;
    float hx = h->cx - 0.5f * BOARD_W;
    float hz = h->cy - 0.5f * BOARD_H;
    float y = 0.5f * BOARD_THICKNESS + Z_ABOVE_SLAB_RIM_RING;
    float r_in = h->radius;
    float r_out = h->radius + outer_extra;
    float r_mid = 0.5f * (r_in + r_out);
    for (int s = 0; s < segments; s++) {
        float a0 = (2.0f * PI_F * (float)s) / (float)segments;
        float a1 = (2.0f * PI_F * (float)(s + 1)) / (float)segments;
        float a_mid = 0.5f * (a0 + a1);
        float mx = h->cx + r_mid * cosf(a_mid);
        float mz = h->cy + r_mid * sinf(a_mid);
        int covered = 0;
        for (int o = 0; o < env->num_holes; o++) {
            if (o == hole_idx)
                continue;
            const Hole* h2 = &env->holes[o];
            float dx = mx - h2->cx;
            float dz = mz - h2->cy;
            if (dx * dx + dz * dz < h2->radius * h2->radius) {
                covered = 1;
                break;
            }
        }
        if (covered)
            continue;
        float c0 = cosf(a0), s0 = sinf(a0);
        float c1 = cosf(a1), s1 = sinf(a1);
        Vector3 i0 = {hx + r_in * c0, y, hz + r_in * s0};
        Vector3 i1 = {hx + r_in * c1, y, hz + r_in * s1};
        Vector3 o0 = {hx + r_out * c0, y, hz + r_out * s0};
        Vector3 o1 = {hx + r_out * c1, y, hz + r_out * s1};
        DrawTriangle3D(i0, o1, o0, SLAB_TOP_COLOR);
        DrawTriangle3D(i0, i1, o1, SLAB_TOP_COLOR);
    }
}

// Slab top as a fine grid, skipping any cell whose rectangle overlaps a hole.
static inline void draw_slab_top_with_holes(const Labyrinth* env) {
    const int nx = SLAB_GRID_NX;
    const int nz = SLAB_GRID_NZ;
    float y_top = 0.5f * BOARD_THICKNESS;
    float cell_w = BOARD_W / nx;
    float cell_h = BOARD_H / nz;
    for (int iz = 0; iz < nz; iz++) {
        for (int ix = 0; ix < nx; ix++) {
            float bx0 = ix * cell_w;
            float bx1 = bx0 + cell_w;
            float by0 = iz * cell_h;
            float by1 = by0 + cell_h;
            if (cell_overlaps_any_hole(env, bx0, by0, bx1, by1, 0.0f))
                continue;
            float lx0 = bx0 - 0.5f * BOARD_W;
            float lx1 = bx1 - 0.5f * BOARD_W;
            float lz0 = by0 - 0.5f * BOARD_H;
            float lz1 = by1 - 0.5f * BOARD_H;
            Vector3 p00 = {lx0, y_top, lz0};
            Vector3 p10 = {lx1, y_top, lz0};
            Vector3 p11 = {lx1, y_top, lz1};
            Vector3 p01 = {lx0, y_top, lz1};
            DrawTriangle3D(p00, p11, p10, SLAB_TOP_COLOR);
            DrawTriangle3D(p00, p01, p11, SLAB_TOP_COLOR);
        }
    }
}

// 4 vertical strips around the slab perimeter. Bottom face omitted (not
// visible from a near-top-down camera and would z-fight with hole floors).
static inline void draw_slab_sides_and_bottom(void) {
    Color side_color = (Color){55, 65, 75, 255};
    float y_top = 0.5f * BOARD_THICKNESS;
    float y_bot = -0.5f * BOARD_THICKNESS;
    float xmin = -0.5f * BOARD_W;
    float xmax = 0.5f * BOARD_W;
    float zmin = -0.5f * BOARD_H;
    float zmax = 0.5f * BOARD_H;
    DrawTriangle3D((Vector3){xmin, y_bot, zmin}, (Vector3){xmax, y_bot, zmin},
                   (Vector3){xmax, y_top, zmin}, side_color);
    DrawTriangle3D((Vector3){xmin, y_bot, zmin}, (Vector3){xmax, y_top, zmin},
                   (Vector3){xmin, y_top, zmin}, side_color);
    DrawTriangle3D((Vector3){xmin, y_bot, zmax}, (Vector3){xmax, y_top, zmax},
                   (Vector3){xmax, y_bot, zmax}, side_color);
    DrawTriangle3D((Vector3){xmin, y_bot, zmax}, (Vector3){xmin, y_top, zmax},
                   (Vector3){xmax, y_top, zmax}, side_color);
    DrawTriangle3D((Vector3){xmin, y_bot, zmin}, (Vector3){xmin, y_top, zmax},
                   (Vector3){xmin, y_bot, zmax}, side_color);
    DrawTriangle3D((Vector3){xmin, y_bot, zmin}, (Vector3){xmin, y_top, zmin},
                   (Vector3){xmin, y_top, zmax}, side_color);
    DrawTriangle3D((Vector3){xmax, y_bot, zmin}, (Vector3){xmax, y_bot, zmax},
                   (Vector3){xmax, y_top, zmax}, side_color);
    DrawTriangle3D((Vector3){xmax, y_bot, zmin}, (Vector3){xmax, y_top, zmax},
                   (Vector3){xmax, y_top, zmin}, side_color);
}

// Inside walls + bottom disk of one hole. Wall normals point inward (toward
// the hole axis) so they're visible from a camera looking down into the pit.
static inline void draw_hole_pit(const Hole* h) {
    Color wall_color = (Color){22, 25, 30, 255};
    Color bottom_color = (Color){8, 10, 14, 255};
    const int segments = 24;
    float hx = h->cx - 0.5f * BOARD_W;
    float hz = h->cy - 0.5f * BOARD_H;
    float y_top = 0.5f * BOARD_THICKNESS;
    float y_bot = -0.5f * BOARD_THICKNESS;
    for (int s = 0; s < segments; s++) {
        float a0 = (2.0f * PI_F * (float)s) / (float)segments;
        float a1 = (2.0f * PI_F * (float)(s + 1)) / (float)segments;
        Vector3 t0 = {hx + h->radius * cosf(a0), y_top, hz + h->radius * sinf(a0)};
        Vector3 t1 = {hx + h->radius * cosf(a1), y_top, hz + h->radius * sinf(a1)};
        Vector3 b0 = {hx + h->radius * cosf(a0), y_bot, hz + h->radius * sinf(a0)};
        Vector3 b1 = {hx + h->radius * cosf(a1), y_bot, hz + h->radius * sinf(a1)};
        DrawTriangle3D(t0, b0, b1, wall_color);
        DrawTriangle3D(t0, b1, t1, wall_color);
    }
    Vector3 center = {hx, y_bot, hz};
    for (int s = 0; s < segments; s++) {
        float a0 = (2.0f * PI_F * (float)s) / (float)segments;
        float a1 = (2.0f * PI_F * (float)(s + 1)) / (float)segments;
        Vector3 p0 = {hx + h->radius * cosf(a0), y_bot, hz + h->radius * sinf(a0)};
        Vector3 p1 = {hx + h->radius * cosf(a1), y_bot, hz + h->radius * sinf(a1)};
        DrawTriangle3D(center, p1, p0, bottom_color);
    }
}

// Solution path as a thin translucent strip on top of the slab.
static inline void draw_path_line(const Labyrinth* env) {
    if (env->num_path_points < 2)
        return;
    Color path_color = (Color){220, 200, 90, 180};
    float y = 0.5f * BOARD_THICKNESS + Z_ABOVE_SLAB_PATH_LINE;
    float half_width = 0.0015f;
    for (int i = 0; i < env->num_path_points - 1; i++) {
        float x1 = env->path_points[i][0] - 0.5f * BOARD_W;
        float z1 = env->path_points[i][1] - 0.5f * BOARD_H;
        float x2 = env->path_points[i + 1][0] - 0.5f * BOARD_W;
        float z2 = env->path_points[i + 1][1] - 0.5f * BOARD_H;
        float dx = x2 - x1, dz = z2 - z1;
        float len = sqrtf(dx * dx + dz * dz);
        if (len < 1.0e-6f)
            continue;
        float nx = -(dz / len) * half_width;
        float nz = (dx / len) * half_width;
        Vector3 p00 = {x1 - nx, y, z1 - nz};
        Vector3 p01 = {x1 + nx, y, z1 + nz};
        Vector3 p10 = {x2 - nx, y, z2 - nz};
        Vector3 p11 = {x2 + nx, y, z2 + nz};
        DrawTriangle3D(p00, p11, p10, path_color);
        DrawTriangle3D(p00, p01, p11, path_color);
    }
}

// Composes the full board + its contents. Caller sets up camera/background.
static inline void draw_board(const Labyrinth* env) {
    Matrix tilt = board_tilt_matrix(env);
    rlPushMatrix();
    rlTranslatef(0.0f, BOARD_CENTER_Y, 0.0f);
    float m[16] = {tilt.m0, tilt.m1, tilt.m2,  tilt.m3,  tilt.m4,  tilt.m5,  tilt.m6,  tilt.m7,
                   tilt.m8, tilt.m9, tilt.m10, tilt.m11, tilt.m12, tilt.m13, tilt.m14, tilt.m15};
    rlMultMatrixf(m);

    const float slab_grid_cell_diag = 1.41421356f * (BOARD_W / (float)SLAB_GRID_NX);
    draw_slab_sides_and_bottom();
    draw_slab_top_with_holes(env);
    for (int i = 0; i < env->num_holes; i++) {
        draw_hole_rim_ring(env, i, slab_grid_cell_diag * 0.85f);
    }
    draw_path_line(env);
    DrawCubeWiresV((Vector3){0, 0, 0}, (Vector3){BOARD_W, BOARD_THICKNESS, BOARD_H},
                   (Color){0, 187, 187, 255});

    // Border frame: outer walls pushed OUTWARD so visible inner face aligns
    // with the physics collision line (otherwise ball sinks into half the box).
    float cy_wall = 0.5f * BOARD_THICKNESS + 0.5f * WALL_HEIGHT;
    float half_t = 0.5f * WALL_THICKNESS;
    DrawCubeV((Vector3){0, cy_wall, -0.5f * BOARD_H - half_t},
              (Vector3){BOARD_W + WALL_THICKNESS * 2, WALL_HEIGHT, WALL_THICKNESS},
              (Color){180, 140, 80, 255});
    DrawCubeV((Vector3){0, cy_wall, 0.5f * BOARD_H + half_t},
              (Vector3){BOARD_W + WALL_THICKNESS * 2, WALL_HEIGHT, WALL_THICKNESS},
              (Color){180, 140, 80, 255});
    DrawCubeV((Vector3){-0.5f * BOARD_W - half_t, cy_wall, 0},
              (Vector3){WALL_THICKNESS, WALL_HEIGHT, BOARD_H}, (Color){180, 140, 80, 255});
    DrawCubeV((Vector3){0.5f * BOARD_W + half_t, cy_wall, 0},
              (Vector3){WALL_THICKNESS, WALL_HEIGHT, BOARD_H}, (Color){180, 140, 80, 255});

    // Interior walls (capsules centred on the physics line).
    for (int i = 4; i < env->num_walls; i++) {
        draw_wall_local(&env->walls[i]);
    }
    // 3D hole pits.
    for (int i = 0; i < env->num_holes; i++) {
        draw_hole_pit(&env->holes[i]);
    }
    // Goal disk (green pad) above the slab top.
    if (env->goal_radius > 0.0f) {
        const int segments = 48;
        float gx = env->goal_cx - 0.5f * BOARD_W;
        float gz = env->goal_cy - 0.5f * BOARD_H;
        float y = 0.5f * BOARD_THICKNESS + Z_ABOVE_SLAB_GOAL_DISK;
        Vector3 center = {gx, y, gz};
        Color goal_color = (Color){80, 220, 110, 255};
        for (int s = 0; s < segments; s++) {
            float a0 = (2.0f * PI_F * (float)s) / (float)segments;
            float a1 = (2.0f * PI_F * (float)(s + 1)) / (float)segments;
            Vector3 p0 = {gx + env->goal_radius * cosf(a0), y,
                          gz + env->goal_radius * sinf(a0)};
            Vector3 p1 = {gx + env->goal_radius * cosf(a1), y,
                          gz + env->goal_radius * sinf(a1)};
            DrawTriangle3D(center, p1, p0, goal_color);
        }
    }
    rlPopMatrix();
}

// ===== RL env API =====

// ---- Action space ----
// Discrete: 0=no-op, 1/2=tilt_x +/-, 3/4=tilt_y +/-.
#define LABYRINTH_NUM_ACTIONS 5
// How much tilt changes per single +/- action (radians).
#define LABYRINTH_TILT_STEP 0.02f

// ---- Observation space: local vision + scalars ----
// Board is rasterized into a coarse grid at reset(); each step the env copies
// a square window centered on the ball into the observation buffer.
#define LABYRINTH_VIEW_CELL_M 0.006f                                    // 6mm per cell
#define LABYRINTH_GRID_W ((int)(BOARD_W / LABYRINTH_VIEW_CELL_M + 0.5f)) // 50
#define LABYRINTH_GRID_H ((int)(BOARD_H / LABYRINTH_VIEW_CELL_M + 0.5f)) // 40
#define LABYRINTH_VISION 7                                              // half-window
#define LABYRINTH_VIEW_DIM (2 * LABYRINTH_VISION + 1)                   // 15
#define LABYRINTH_VIEW_SIZE (LABYRINTH_VIEW_DIM * LABYRINTH_VIEW_DIM)   // 225

// Cell codes. Priority (high→low) when multiple features land on the same
// cell: HOLE > WALL > GOAL > PATH > EMPTY. HOLE is most important for
// survival, PATH is just a hint.
#define LABYRINTH_CELL_EMPTY 0
#define LABYRINTH_CELL_WALL  1
#define LABYRINTH_CELL_HOLE  2
#define LABYRINTH_CELL_GOAL  3
#define LABYRINTH_CELL_PATH  4

// 8 scalar features appended after the grid:
// [vx, vy, vz, tilt_x, tilt_y, goal_dx, goal_dy, bfs_dist_norm]
// bfs_dist_norm is the ball cell's BFS distance-to-goal through the maze,
// normalized to [0, 1]. Encodes "how much maze is left", not "how far off the
// designer's ideal polyline" — the PATH cells in the local grid do that.
#define LABYRINTH_SCALAR_FEATURES 8

// Potential-based reward shaping (Ng, Harada & Russell 1999):
//   F(s, s') = γ_shaping · φ(s') − φ(s),   φ(terminal) ≡ 0
// with potential
//   φ(s) = LABYRINTH_POTENTIAL_SCALE · (1 − bfs_dist_norm(s))
// so φ = k at the goal cell, 0 at the worst-reachable cell, ≥ 0 everywhere.
//
// Why this shape:
//  - Sitting still pays F = (γ−1)·φ(s) ≤ 0 every step (worse the closer to
//    goal the agent is) — kills the "go to a corner and time out" exploit.
//  - Backtracking A→B→A nets negative: the discount makes the return trip
//    cost more than the forward trip pays — kills oscillation farming.
//  - Discounted cumulative shaping over an episode telescopes to −φ(s_0)
//    (policy-invariant offset) once we force φ(terminal)=0; the sparse
//    ±1 on goal/fall is still what the optimizer actually trades for.
//
// γ_shaping must match the training discount; bump together if either changes.
#define LABYRINTH_SHAPING_GAMMA 0.995f
#define LABYRINTH_POTENTIAL_SCALE 3.0f

// Flat per-step cost. Tiebreaker against shaping plateaus where φ doesn't
// change much between cells; also a small extra push to terminate quickly.
#define LABYRINTH_STEP_PENALTY 0.002f

// Small bonus for being on a PATH-tagged cell (the BFS-optimal corridor).
// Nudges the policy toward "find the highway, then follow it" — it's small
// enough that staying still on path still loses net reward (step penalty +
// shaping's stall term > bonus), so it doesn't create a new stall attractor.
#define LABYRINTH_ON_PATH_BONUS 0.005f

// Total observation size — must stay in sync with binding.c's OBS_SIZE.
#define LABYRINTH_OBS_SIZE (LABYRINTH_VIEW_SIZE + LABYRINTH_SCALAR_FEATURES)

// Default episode length cap (c_steps, not physics steps).
#define LABYRINTH_MAX_STEPS 2000

// Default physics substeps per agent decision. Each c_step advances the
// physics by `physics_substeps` × PHYSICS_DT seconds. 4 gives a ~50 Hz agent
// rate on top of the 200 Hz physics — human-ish reaction cadence and 4× more
// simulated time per step, so a 2000-step episode = 40 s of real time.
#define LABYRINTH_PHYSICS_SUBSTEPS 4

// Curriculum defaults. `difficulty_start` is the floor (0=trivial, 1=full maze).
// `curriculum_episodes` = 0 means no ramping — stay at `difficulty_start` the
// whole run. >0 means each env ramps from `difficulty_start` up to 1.0 over its
// own first N episodes, then stays at 1.0. Default keeps current behavior.
#define LABYRINTH_DIFFICULTY_START 1.0f
#define LABYRINTH_CURRICULUM_EPISODES 0

typedef struct Log {
    float perf;           // normalized score in [0, 1]: 1.0 if reached goal
    float score;          // unnormalized: +1 reach, -1 fall, 0 timeout
    float episode_return; // sum of step rewards over episode
    float episode_length;
    float reached_goal;   // counter: how many episodes ended at goal
    float fell_in_hole;   // counter: how many ended in a hole
    float n;              // required last field: how many episodes logged
} Log;

typedef struct Client {
    Camera3D camera;
    int window_w;
    int window_h;
} Client;

typedef struct LabyrinthEnv {
    // Required by vecenv.h binding framework. Buffer pointers are written by
    // the framework (point into a pre-allocated agent-slot of a larger tensor)
    // when the env is constructed from Python; allocate_LabyrinthEnv is only
    // used for the standalone binary.
    Log log;
    float* observations;
    float* actions;       // discrete actions arrive as floats (cast to int)
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;     // framework-initialized RNG seed

    // Env-local state:
    Client* client;
    Labyrinth phys;       // underlying physics state (walls/holes/ball)
    int tick;
    int max_steps;
    int physics_substeps; // physics steps per c_step; bigger = slower action rate
    uint32_t seed;        // seed supplied by kwargs
    uint32_t maze_seed;   // current episode's maze seed (rotates each reset)
    float episode_return;

    // Rasterized static board layout — rebuilt once per episode at reset(),
    // then just indexed each step when building the observation window.
    unsigned char grid[LABYRINTH_GRID_W * LABYRINTH_GRID_H];

    // BFS distance-to-goal through the maze, per grid cell. Walls and holes
    // are impassable in the BFS, so distance respects obstacles. Rebuilt at
    // reset(). Max distance across the map is cached for normalization.
    unsigned short dist_to_goal[LABYRINTH_GRID_W * LABYRINTH_GRID_H];
    int max_dist;                // largest reachable BFS distance (for normalization)
    float prev_dist_norm;        // last step's bfs_dist_norm — used to compute
                                 // F = γ·φ(s') − φ(s) on the next step.

    // Curriculum state. `episode_count` is this env's completed-episode counter.
    // At each c_reset the difficulty passed to the maze generator is:
    //   d = difficulty_start + (1 − difficulty_start) · min(1, ep / curr_eps)
    // (or just difficulty_start if curriculum_episodes == 0).
    int episode_count;
    float difficulty_start;
    int curriculum_episodes;
} LabyrinthEnv;

// Allocate per-agent buffers. Only needed for the standalone binary — the
// Python/vecenv path wires buffer pointers into a pre-allocated tensor.
static inline LabyrinthEnv* allocate_LabyrinthEnv(LabyrinthEnv* env) {
    env->observations = (float*)calloc(LABYRINTH_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
    return env;
}

static inline void free_allocated(LabyrinthEnv* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

// One-time setup — called after kwargs unpacked.
static inline void init(LabyrinthEnv* env) {
    memset(&env->log, 0, sizeof(Log));
    env->tick = 0;
    if (env->max_steps <= 0)
        env->max_steps = LABYRINTH_MAX_STEPS;
    if (env->physics_substeps <= 0)
        env->physics_substeps = LABYRINTH_PHYSICS_SUBSTEPS;
    if (env->seed == 0)
        env->seed = 12345u;
    env->maze_seed = env->seed;
    env->episode_return = 0.0f;
    env->client = NULL;
    env->episode_count = 0;
    if (env->difficulty_start < 0.0f || env->difficulty_start > 1.0f)
        env->difficulty_start = LABYRINTH_DIFFICULTY_START;
    if (env->curriculum_episodes < 0)
        env->curriculum_episodes = LABYRINTH_CURRICULUM_EPISODES;
}

// Log-aggregation helper called on terminal transitions.
static inline void add_log(LabyrinthEnv* env) {
    int won = env->phys.reached_goal;
    int lost = env->phys.fell_in_hole;
    env->log.perf += won ? 1.0f : 0.0f;
    env->log.score += won ? 1.0f : (lost ? -1.0f : 0.0f);
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.reached_goal += (float)won;
    env->log.fell_in_hole += (float)lost;
    env->log.n += 1.0f;
}

// ---- Local-vision grid rasterization ----
// Sample the static maze features (walls, holes, goal, path) into a board-
// sized grid once per episode at reset(). Each observation step then just
// copies a vision-sized window out of this grid.

// Test whether a point (px, py) lies inside any wall's collision capsule.
static inline int point_in_any_wall(const Labyrinth* p, float px, float py) {
    for (int i = 0; i < p->num_walls; i++) {
        const Wall* w = &p->walls[i];
        float qx, qy;
        closest_point_on_segment(w->x1, w->y1, w->x2, w->y2, px, py, &qx, &qy);
        float dx = px - qx, dy = py - qy;
        if (dx * dx + dy * dy < COLLISION_RADIUS * COLLISION_RADIUS)
            return 1;
    }
    return 0;
}

// Test whether a point lies within thickness `r` of any path polyline segment.
static inline int point_on_path(const Labyrinth* p, float px, float py, float r) {
    for (int i = 0; i + 1 < p->num_path_points; i++) {
        float qx, qy;
        closest_point_on_segment(p->path_points[i][0], p->path_points[i][1],
                                 p->path_points[i + 1][0], p->path_points[i + 1][1],
                                 px, py, &qx, &qy);
        float dx = px - qx, dy = py - qy;
        if (dx * dx + dy * dy < r * r)
            return 1;
    }
    return 0;
}

static inline void build_grid(LabyrinthEnv* env) {
    const Labyrinth* p = &env->phys;
    const float cell = LABYRINTH_VIEW_CELL_M;
    // Path cells are marked within this radius of the BFS-optimal polyline.
    // Wider than one view-cell so the path reads as a CORRIDOR in the agent's
    // 15×15 local view rather than a thin hairline it can easily drift off of.
    // ~2.5× view-cell = 15mm = roughly half a maze-cell width.
    const float path_r = 2.5f * cell;
    for (int gy = 0; gy < LABYRINTH_GRID_H; gy++) {
        for (int gx = 0; gx < LABYRINTH_GRID_W; gx++) {
            float cx = (gx + 0.5f) * cell;
            float cy = (gy + 0.5f) * cell;
            unsigned char v = LABYRINTH_CELL_EMPTY;
            // Priority: hole > wall > goal > path > empty.
            if (xy_in_any_hole(p, cx, cy)) {
                v = LABYRINTH_CELL_HOLE;
            } else if (point_in_any_wall(p, cx, cy)) {
                v = LABYRINTH_CELL_WALL;
            } else if (p->goal_radius > 0.0f) {
                float dx = cx - p->goal_cx, dy = cy - p->goal_cy;
                if (dx * dx + dy * dy < p->goal_radius * p->goal_radius)
                    v = LABYRINTH_CELL_GOAL;
            }
            if (v == LABYRINTH_CELL_EMPTY && point_on_path(p, cx, cy, path_r))
                v = LABYRINTH_CELL_PATH;
            env->grid[gy * LABYRINTH_GRID_W + gx] = v;
        }
    }
}

// BFS distance-to-goal per grid cell. Walls AND holes are impassable, so
// the distance field is a true "how far is goal through the maze" — any
// route the ball can actually travel respects it. Rebuilt at reset().
static inline void build_distance_field(LabyrinthEnv* env) {
    const int N = LABYRINTH_GRID_W * LABYRINTH_GRID_H;
    const unsigned short INF = 0xFFFFu;
    for (int i = 0; i < N; i++)
        env->dist_to_goal[i] = INF;

    // Seed: any cell tagged GOAL in the rasterized grid. Fallback: the
    // goal-disk center mapped to its cell (in case the goal is too small to
    // cover any grid cell's center).
    int queue[LABYRINTH_GRID_W * LABYRINTH_GRID_H];
    int qh = 0, qt = 0;
    for (int i = 0; i < N; i++) {
        if (env->grid[i] == LABYRINTH_CELL_GOAL) {
            env->dist_to_goal[i] = 0;
            queue[qt++] = i;
        }
    }
    if (qt == 0 && env->phys.goal_radius > 0.0f) {
        int gx = (int)(env->phys.goal_cx / LABYRINTH_VIEW_CELL_M);
        int gy = (int)(env->phys.goal_cy / LABYRINTH_VIEW_CELL_M);
        if (gx >= 0 && gx < LABYRINTH_GRID_W && gy >= 0 && gy < LABYRINTH_GRID_H) {
            int idx = gy * LABYRINTH_GRID_W + gx;
            env->dist_to_goal[idx] = 0;
            queue[qt++] = idx;
        }
    }

    int max_d = 0;
    while (qh < qt) {
        int idx = queue[qh++];
        int x = idx % LABYRINTH_GRID_W;
        int y = idx / LABYRINTH_GRID_W;
        int d = env->dist_to_goal[idx];
        if (d > max_d)
            max_d = d;
        int neigh[4] = {idx - 1, idx + 1, idx - LABYRINTH_GRID_W, idx + LABYRINTH_GRID_W};
        int valid[4] = {x > 0, x < LABYRINTH_GRID_W - 1, y > 0, y < LABYRINTH_GRID_H - 1};
        for (int k = 0; k < 4; k++) {
            if (!valid[k])
                continue;
            int n = neigh[k];
            unsigned char c = env->grid[n];
            // Walls and holes block BFS — the ball can't actually traverse
            // through them alive, so distance through them isn't meaningful.
            if (c == LABYRINTH_CELL_WALL || c == LABYRINTH_CELL_HOLE)
                continue;
            if (env->dist_to_goal[n] != INF)
                continue;
            env->dist_to_goal[n] = (unsigned short)(d + 1);
            queue[qt++] = n;
        }
    }
    env->max_dist = (max_d > 0) ? max_d : 1;
}

// BFS distance-to-goal at an arbitrary view-grid cell, normalized to [0, 1].
// Off-grid or unreachable (wall/hole) cells return 1.0.
static inline float labyrinth_dist_norm_at(const LabyrinthEnv* env, int gx, int gy) {
    if (gx < 0 || gx >= LABYRINTH_GRID_W || gy < 0 || gy >= LABYRINTH_GRID_H)
        return 1.0f;
    unsigned short d = env->dist_to_goal[gy * LABYRINTH_GRID_W + gx];
    if (d == 0xFFFFu)
        return 1.0f;
    float n = (float)d / (float)env->max_dist;
    if (n > 1.0f) n = 1.0f;
    return n;
}

// Look up the ball's current BFS-distance-to-goal, normalized to [0, 1].
// 0 = at the goal, 1 = maximum-distance reachable cell. Off-board positions
// clamp to the nearest grid cell (so the BFS field is always defined).
static inline float labyrinth_dist_to_goal_norm(const LabyrinthEnv* env) {
    int gx = (int)(env->phys.ball_x / LABYRINTH_VIEW_CELL_M);
    int gy = (int)(env->phys.ball_y / LABYRINTH_VIEW_CELL_M);
    if (gx < 0) gx = 0;
    if (gx >= LABYRINTH_GRID_W) gx = LABYRINTH_GRID_W - 1;
    if (gy < 0) gy = 0;
    if (gy >= LABYRINTH_GRID_H) gy = LABYRINTH_GRID_H - 1;
    return labyrinth_dist_norm_at(env, gx, gy);
}

// Build the observation buffer: a 15x15 local window centered on the ball,
// plus 8 scalar features (last one is progress-along-path).
static inline void compute_observations(LabyrinthEnv* env) {
    const Labyrinth* p = &env->phys;
    // Ball's grid-cell center. Window spans [center-VISION, center+VISION].
    int gx_center = (int)(p->ball_x / LABYRINTH_VIEW_CELL_M);
    int gy_center = (int)(p->ball_y / LABYRINTH_VIEW_CELL_M);
    float* o = env->observations;
    for (int r = 0; r < LABYRINTH_VIEW_DIM; r++) {
        for (int c = 0; c < LABYRINTH_VIEW_DIM; c++) {
            int gx = gx_center + (c - LABYRINTH_VISION);
            int gy = gy_center + (r - LABYRINTH_VISION);
            unsigned char v;
            if (gx < 0 || gx >= LABYRINTH_GRID_W || gy < 0 || gy >= LABYRINTH_GRID_H)
                v = LABYRINTH_CELL_WALL; // off-board reads as wall
            else
                v = env->grid[gy * LABYRINTH_GRID_W + gx];
            o[r * LABYRINTH_VIEW_DIM + c] = (float)v;
        }
    }
    // Scalar features.
    float inv_vmax = 1.0f / 2.0f; // normalize velocities roughly to [-1, 1]
    float inv_t = 1.0f / MAX_TILT_RAD;
    float inv_w = 1.0f / BOARD_W;
    float inv_h = 1.0f / BOARD_H;
    o[LABYRINTH_VIEW_SIZE + 0] = p->ball_vx * inv_vmax;
    o[LABYRINTH_VIEW_SIZE + 1] = p->ball_vy * inv_vmax;
    o[LABYRINTH_VIEW_SIZE + 2] = p->ball_vz * inv_vmax;
    o[LABYRINTH_VIEW_SIZE + 3] = p->tilt_x * inv_t;
    o[LABYRINTH_VIEW_SIZE + 4] = p->tilt_y * inv_t;
    o[LABYRINTH_VIEW_SIZE + 5] = (p->goal_cx - p->ball_x) * inv_w;
    o[LABYRINTH_VIEW_SIZE + 6] = (p->goal_cy - p->ball_y) * inv_h;
    o[LABYRINTH_VIEW_SIZE + 7] = labyrinth_dist_to_goal_norm(env);
}

// Start a new episode: rotate the maze seed, regenerate maze, write obs.
static inline void c_reset(LabyrinthEnv* env) {
    env->tick = 0;
    env->episode_return = 0.0f;
    env->maze_seed += 1u;
    // Per-env curriculum: linear ramp in episode count from difficulty_start
    // up to full difficulty (1.0) over curriculum_episodes episodes.
    float difficulty = env->difficulty_start;
    if (env->curriculum_episodes > 0) {
        float progress = (float)env->episode_count / (float)env->curriculum_episodes;
        if (progress > 1.0f) progress = 1.0f;
        difficulty = env->difficulty_start + (1.0f - env->difficulty_start) * progress;
    }
    labyrinth_reset(&env->phys);
    labyrinth_load_curriculum_maze(&env->phys, env->maze_seed, difficulty);
    build_grid(env);
    build_distance_field(env);
    env->prev_dist_norm = labyrinth_dist_to_goal_norm(env);
    compute_observations(env);
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    env->episode_count += 1;
}

// Apply action, advance physics, compute reward/terminal/obs.
static inline void c_step(LabyrinthEnv* env) {
    // Read discrete action → tilt delta. Actions arrive as floats even when
    // the space is discrete — cast to an integer.
    int a = (int)env->actions[0];
    float dt_tilt = LABYRINTH_TILT_STEP;
    switch (a) {
        case 1: env->phys.tilt_x += dt_tilt; break;
        case 2: env->phys.tilt_x -= dt_tilt; break;
        case 3: env->phys.tilt_y += dt_tilt; break;
        case 4: env->phys.tilt_y -= dt_tilt; break;
        default: break; // 0 = no-op
    }
    if (env->phys.tilt_x > MAX_TILT_RAD)  env->phys.tilt_x = MAX_TILT_RAD;
    if (env->phys.tilt_x < -MAX_TILT_RAD) env->phys.tilt_x = -MAX_TILT_RAD;
    if (env->phys.tilt_y > MAX_TILT_RAD)  env->phys.tilt_y = MAX_TILT_RAD;
    if (env->phys.tilt_y < -MAX_TILT_RAD) env->phys.tilt_y = -MAX_TILT_RAD;

    // Run multiple physics substeps per agent decision. Early-exit on
    // terminal so we don't keep simulating after the episode ends.
    for (int i = 0; i < env->physics_substeps; i++) {
        labyrinth_step(&env->phys);
        if (env->phys.reached_goal || env->phys.fell_in_hole)
            break;
    }
    env->tick += 1;

    // Potential-based reward shaping (see LABYRINTH_SHAPING_GAMMA block above
    // for the derivation). Force φ(terminal)=0 so the discounted cumulative
    // shaping telescopes to a policy-invariant −φ(s_0).
    int terminal = 0;
    if (env->phys.reached_goal || env->phys.fell_in_hole || env->tick >= env->max_steps)
        terminal = 1;
    float cur_dist_norm = labyrinth_dist_to_goal_norm(env);
    float phi_cur  = terminal ? 0.0f
                              : LABYRINTH_POTENTIAL_SCALE * (1.0f - cur_dist_norm);
    float phi_prev = LABYRINTH_POTENTIAL_SCALE * (1.0f - env->prev_dist_norm);
    float shaping = LABYRINTH_SHAPING_GAMMA * phi_cur - phi_prev;
    env->prev_dist_norm = cur_dist_norm;

    float r = shaping - LABYRINTH_STEP_PENALTY;
    // On-path bonus: if the ball's current view-grid cell is tagged PATH,
    // pay a small per-step reward.
    int gx_cell = (int)(env->phys.ball_x / LABYRINTH_VIEW_CELL_M);
    int gy_cell = (int)(env->phys.ball_y / LABYRINTH_VIEW_CELL_M);
    if (gx_cell >= 0 && gx_cell < LABYRINTH_GRID_W &&
        gy_cell >= 0 && gy_cell < LABYRINTH_GRID_H &&
        env->grid[gy_cell * LABYRINTH_GRID_W + gx_cell] == LABYRINTH_CELL_PATH) {
        r += LABYRINTH_ON_PATH_BONUS;
    }
    if (env->phys.reached_goal)
        r += 1.0f;
    else if (env->phys.fell_in_hole)
        r += -1.0f;
    env->rewards[0] = r;
    env->terminals[0] = (float)terminal;
    env->episode_return += r;

    if (terminal) {
        add_log(env);
        c_reset(env);
    } else {
        compute_observations(env);
    }
}

// Raylib render. Lazy-inits the window on first call.
static inline void c_render(LabyrinthEnv* env) {
    if (env->client == NULL) {
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->window_w = 1024;
        env->client->window_h = 720;
        InitWindow(env->client->window_w, env->client->window_h, "PufferLib Labyrinth (3D)");
        SetTargetFPS(60);
        env->client->camera = (Camera3D){
            .position = (Vector3){0.0f, 0.50f, 0.10f},
            .target = (Vector3){0.0f, BOARD_CENTER_Y, 0.0f},
            .up = (Vector3){0.0f, 1.0f, 0.0f},
            .fovy = 40.0f,
            .projection = CAMERA_PERSPECTIVE,
        };
    }
    if (IsKeyDown(KEY_ESCAPE))
        exit(0);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    BeginMode3D(env->client->camera);
    DrawGrid(10, 0.1f);
    draw_board(&env->phys);
    draw_ball(&env->phys);
    EndMode3D();
    DrawFPS(env->client->window_w - 90, 10);
    EndDrawing();
}

static inline void c_close(LabyrinthEnv* env) {
    if (env->client != NULL) {
        if (IsWindowReady())
            CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

#endif
