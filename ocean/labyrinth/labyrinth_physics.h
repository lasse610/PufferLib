/* Labyrinth physics: tilt-a-ball marble maze.
 *
 * This header is the pure 3D rigid-body simulation — no raylib, no RL env
 * plumbing. labyrinth.h wraps this as the Ocean-style RL env; labyrinth.c is
 * the standalone demo; labyrinth_maze.h builds levels on top of this; the
 * test suite drives this header directly.
 */

#ifndef LABYRINTH_PHYSICS_H
#define LABYRINTH_PHYSICS_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ===== Constants =====

#define BOARD_W 0.30f
#define BOARD_H 0.24f
// Visible slab thickness. Used by physics fall animation and the renderer.
// Made deeper than ball diameter so the ball fits fully inside a hole pit.
#define BOARD_THICKNESS 0.018f
#define BALL_RADIUS 0.006f

#define GRAVITY 9.81f
// Rolling acceleration factor for a uniform solid sphere: a = (5/7) g sin(theta).
// Energy splits between translation and rotation (I = 2/5 mr^2).
#define ROLLING_GRAVITY_FACTOR (5.0f / 7.0f)
// Rolling friction is weak — marbles coast a long way on a smooth board.
#define FRICTION_COEF 0.001f
#define MAX_TILT_RAD 0.20f
#define PHYSICS_DT (1.0f / 200.0f)

#define MAX_WALLS 96
#define WALL_THICKNESS_M 0.004f
#define WALL_HALF_THICKNESS (0.5f * WALL_THICKNESS_M)
#define COLLISION_RADIUS (BALL_RADIUS + WALL_HALF_THICKNESS)
#define RESTITUTION 0.3f
#define COLLISION_SLOP 1.0e-5f
#define STICK_THRESHOLD 0.02f
#define COLLISION_ITERATIONS 4

#define MAX_HOLES 128
#define MAX_PATH_POINTS 128

// Hole rim restitution is much softer than walls — the rim is thin and lossy.
#define HOLE_RIM_RESTITUTION 0.15f

// ===== Types =====

typedef struct {
    float x1, y1;
    float x2, y2;
} Wall;

typedef struct {
    float cx, cy;
    float radius;
} Hole;

typedef struct {
    // Board-local position. (x, y) lie in the slab plane (range [0,BOARD_W] x
    // [0,BOARD_H]); z is the height above the slab top. z = BALL_RADIUS means
    // ball sitting on the slab. z < 0 means the ball is in a pit. The pit
    // floor sits at z = -BOARD_THICKNESS.
    float ball_x, ball_y, ball_z;
    float ball_vx, ball_vy, ball_vz;
    // Tilt of the board around its local x and y axes (radians). Used only to
    // rotate gravity into the board frame — geometry stays axis-aligned.
    float tilt_x, tilt_y;
    // 3D orientation of the ball (unit quaternion, w+xyz). Tracks how the
    // ball has rolled over time so we can render its spin.
    float orient_w, orient_x, orient_y, orient_z;
    Wall walls[MAX_WALLS];
    int num_walls;
    Hole holes[MAX_HOLES];
    int num_holes;
    // Solution path from start to goal (cell centers in board-local coords),
    // produced by the maze generator. Useful for rendering a route hint and
    // for RL reward shaping (project ball xy onto polyline, reward progress).
    float path_points[MAX_PATH_POINTS][2];
    int num_path_points;
    // Goal disk on the slab top. radius == 0 means "no goal configured".
    float goal_cx, goal_cy, goal_radius;
    int tick;
    // Derived flags (recomputed each step). They don't influence physics —
    // only renderer / HUD / RL terminal signal.
    int fell_in_hole; // ball center below slab top AND inside some hole
    int reached_goal; // ball on slab AND inside the goal disk
} Labyrinth;

// ===== Wall management =====

static inline void labyrinth_clear_walls(Labyrinth* env) {
    env->num_walls = 0;
}

static inline void labyrinth_add_wall(Labyrinth* env, float x1, float y1, float x2, float y2) {
    if (env->num_walls >= MAX_WALLS)
        return;
    Wall* w = &env->walls[env->num_walls++];
    w->x1 = x1;
    w->y1 = y1;
    w->x2 = x2;
    w->y2 = y2;
}

static inline void labyrinth_add_border(Labyrinth* env) {
    labyrinth_add_wall(env, 0, 0, BOARD_W, 0);
    labyrinth_add_wall(env, BOARD_W, 0, BOARD_W, BOARD_H);
    labyrinth_add_wall(env, BOARD_W, BOARD_H, 0, BOARD_H);
    labyrinth_add_wall(env, 0, BOARD_H, 0, 0);
}

// ===== Holes =====

static inline void labyrinth_clear_holes(Labyrinth* env) {
    env->num_holes = 0;
}

static inline void labyrinth_add_hole(Labyrinth* env, float cx, float cy, float radius) {
    if (env->num_holes >= MAX_HOLES)
        return;
    Hole* h = &env->holes[env->num_holes++];
    h->cx = cx;
    h->cy = cy;
    h->radius = radius;
}

// ===== Collision =====

static inline void closest_point_on_segment(float p1x, float p1y, float p2x, float p2y, float cx,
                                            float cy, float* qx, float* qy) {
    float dx = p2x - p1x;
    float dy = p2y - p1y;
    float len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-12f) {
        *qx = p1x;
        *qy = p1y;
        return;
    }
    float t = ((cx - p1x) * dx + (cy - p1y) * dy) / len_sq;
    if (t < 0.0f)
        t = 0.0f;
    if (t > 1.0f)
        t = 1.0f;
    *qx = p1x + t * dx;
    *qy = p1y + t * dy;
}

// Returns 1 if the (board-frame) point (x, y) is inside any hole disk.
static inline int xy_in_any_hole(const Labyrinth* env, float x, float y) {
    for (int i = 0; i < env->num_holes; i++) {
        const Hole* h = &env->holes[i];
        float dx = x - h->cx;
        float dy = y - h->cy;
        if (dx * dx + dy * dy < h->radius * h->radius)
            return 1;
    }
    return 0;
}

// Resolve a single sphere-vs-circle contact. The circle lies in the plane z=0,
// centered at (cx, cy), radius R. Pushes the ball out along the contact normal
// with restitution and dampens velocity along that normal.
static inline void resolve_rim_contact(Labyrinth* env, float cx, float cy, float R) {
    float dx = env->ball_x - cx;
    float dy = env->ball_y - cy;
    float d_xy = sqrtf(dx * dx + dy * dy);
    // Ball center inside the hole disk means the ball is past the rim on its
    // way into the pit. The rim doesn't catch it from below — let it fall.
    if (d_xy <= R)
        return;
    float rx, ry; // closest rim point in xy
    if (d_xy > 1.0e-6f) {
        rx = cx + R * (dx / d_xy);
        ry = cy + R * (dy / d_xy);
    } else {
        return;
    }
    float ex = env->ball_x - rx;
    float ey = env->ball_y - ry;
    float ez = env->ball_z;
    float dist_sq = ex * ex + ey * ey + ez * ez;
    if (dist_sq >= BALL_RADIUS * BALL_RADIUS)
        return;
    float dist = sqrtf(dist_sq);
    float nx, ny, nz;
    if (dist > 1.0e-6f) {
        nx = ex / dist;
        ny = ey / dist;
        nz = ez / dist;
    } else {
        nx = 0.0f;
        ny = 0.0f;
        nz = 1.0f;
    }
    float pen = BALL_RADIUS - dist + COLLISION_SLOP;
    env->ball_x += nx * pen;
    env->ball_y += ny * pen;
    env->ball_z += nz * pen;
    float vn = env->ball_vx * nx + env->ball_vy * ny + env->ball_vz * nz;
    if (vn >= 0.0f)
        return;
    if (-vn < STICK_THRESHOLD) {
        env->ball_vx -= vn * nx;
        env->ball_vy -= vn * ny;
        env->ball_vz -= vn * nz;
    } else {
        float j = -(1.0f + HOLE_RIM_RESTITUTION) * vn;
        env->ball_vx += j * nx;
        env->ball_vy += j * ny;
        env->ball_vz += j * nz;
    }
}

static inline void resolve_wall_collision(Labyrinth* env, const Wall* w) {
    float qx, qy;
    closest_point_on_segment(w->x1, w->y1, w->x2, w->y2, env->ball_x, env->ball_y, &qx, &qy);
    float dx = env->ball_x - qx;
    float dy = env->ball_y - qy;
    float dist_sq = dx * dx + dy * dy;
    if (dist_sq >= COLLISION_RADIUS * COLLISION_RADIUS)
        return;

    float dist = sqrtf(dist_sq);
    float nx, ny;
    if (dist > 1e-6f) {
        nx = dx / dist;
        ny = dy / dist;
    } else {
        // Ball center exactly on the wall: pick a normal perpendicular to the wall.
        float wx = w->x2 - w->x1;
        float wy = w->y2 - w->y1;
        float wlen = sqrtf(wx * wx + wy * wy);
        if (wlen > 1e-6f) {
            nx = -wy / wlen;
            ny = wx / wlen;
        } else {
            nx = 1.0f;
            ny = 0.0f;
        }
    }

    float penetration = COLLISION_RADIUS - dist + COLLISION_SLOP;
    env->ball_x += nx * penetration;
    env->ball_y += ny * penetration;

    float vn = env->ball_vx * nx + env->ball_vy * ny;
    if (vn >= 0.0f)
        return; // moving away from wall

    if (-vn < STICK_THRESHOLD) {
        // Low-energy contact: kill the normal component so the ball settles.
        env->ball_vx -= vn * nx;
        env->ball_vy -= vn * ny;
    } else {
        float j = -(1.0f + RESTITUTION) * vn;
        env->ball_vx += j * nx;
        env->ball_vy += j * ny;
    }
}

// ===== Public API =====

static inline void labyrinth_reset(Labyrinth* env) {
    env->ball_x = BOARD_W * 0.5f;
    env->ball_y = BOARD_H * 0.5f;
    env->ball_z = BALL_RADIUS; // sitting on the slab top (z = 0 plane)
    env->ball_vx = 0;
    env->ball_vy = 0;
    env->ball_vz = 0;
    env->tilt_x = 0;
    env->tilt_y = 0;
    env->orient_w = 1.0f;
    env->orient_x = 0.0f;
    env->orient_y = 0.0f;
    env->orient_z = 0.0f;
    env->goal_cx = 0;
    env->goal_cy = 0;
    env->goal_radius = 0;
    env->num_path_points = 0;
    env->tick = 0;
    env->fell_in_hole = 0;
    env->reached_goal = 0;
    labyrinth_clear_walls(env);
    labyrinth_clear_holes(env);
    labyrinth_add_border(env);
}

static inline void labyrinth_set_goal(Labyrinth* env, float cx, float cy, float radius) {
    env->goal_cx = cx;
    env->goal_cy = cy;
    env->goal_radius = radius;
}

// Move the ball to (x, y) on the slab top with zero velocity. Useful for the
// maze loader to drop the ball at a fixed start cell after reset().
static inline void labyrinth_place_ball(Labyrinth* env, float x, float y) {
    env->ball_x = x;
    env->ball_y = y;
    env->ball_z = BALL_RADIUS;
    env->ball_vx = 0;
    env->ball_vy = 0;
    env->ball_vz = 0;
}


// Rotate a body-frame vector by a unit quaternion, writing the world-frame
// result through output pointers. Direct expansion of q * (0, v) * q^-1
// (sandwich product for a pure-vector quaternion) — no heap, no raylib.
// Shared with the renderer (rendering rolled-ball markers) and the test suite
// (verifying marker trajectories).
static inline void labyrinth_rotate_by_quat(float qw, float qx, float qy, float qz,
                                            float bx, float by, float bz,
                                            float* ox, float* oy, float* oz) {
    float tx = 2.0f * (qy * bz - qz * by);
    float ty = 2.0f * (qz * bx - qx * bz);
    float tz = 2.0f * (qx * by - qy * bx);
    *ox = bx + qw * tx + (qy * tz - qz * ty);
    *oy = by + qw * ty + (qz * tx - qx * tz);
    *oz = bz + qw * tz + (qx * ty - qy * tx);
}

// Integrate the ball orientation for one physics step.
// The ball rolls without slipping, so angular velocity magnitude is |v|/r
// and its axis is perpendicular to the velocity direction (in the board plane).
// We rotate around the world axis (ball_vy, 0, -ball_vx)/|v| — perpendicular
// to motion, lying on the board surface (right-hand rule: ball moving in +x
// spins around -z, so the top marker moves toward +x, matching real rolling).
static inline void integrate_ball_orientation(Labyrinth* env) {
    float speed = sqrtf(env->ball_vx * env->ball_vx + env->ball_vy * env->ball_vy);
    if (speed < 1e-6f)
        return;

    float omega = speed / BALL_RADIUS; // rad / s
    float angle = omega * PHYSICS_DT;  // radians rotated this step

    // Rolling-without-slipping axis derivation (right-hand rule):
    //   ball rolls in +x, contact point at world (0,-r,0) is stationary
    //   ⇒ ω = (0, 0, -v/r).
    //   ball rolls in +z (board y) ⇒ ω = (+v/r, 0, 0).
    // test_marker_trajectory in test_labyrinth.c pins the sign.
    float inv_speed = 1.0f / speed;
    float ax = env->ball_vy * inv_speed;
    float ay = 0.0f;
    float az = -env->ball_vx * inv_speed;

    // Quaternion for rotation by `angle` around unit axis (ax, ay, az)
    float half = 0.5f * angle;
    float s = sinf(half);
    float dq_w = cosf(half);
    float dq_x = ax * s;
    float dq_y = ay * s;
    float dq_z = az * s;

    // Compose: new_orient = dq * old_orient (Hamilton product)
    float ow = env->orient_w, ox = env->orient_x, oy = env->orient_y, oz = env->orient_z;
    env->orient_w = dq_w * ow - dq_x * ox - dq_y * oy - dq_z * oz;
    env->orient_x = dq_w * ox + dq_x * ow + dq_y * oz - dq_z * oy;
    env->orient_y = dq_w * oy - dq_x * oz + dq_y * ow + dq_z * ox;
    env->orient_z = dq_w * oz + dq_x * oy - dq_y * ox + dq_z * ow;

    // Normalize periodically to fight drift.
    float n_sq = env->orient_w * env->orient_w + env->orient_x * env->orient_x +
                 env->orient_y * env->orient_y + env->orient_z * env->orient_z;
    float inv_n = 1.0f / sqrtf(n_sq);
    env->orient_w *= inv_n;
    env->orient_x *= inv_n;
    env->orient_y *= inv_n;
    env->orient_z *= inv_n;
}

// Resolve sphere-vs-horizontal-plane contact for an upward-facing surface at
// height `surface_z`. The surface only supports the ball where xy is on solid
// material (slab top: outside any hole; pit floor: inside any hole) — the
// caller passes `support_when_in_hole` to switch between those cases.
static inline void resolve_horizontal_surface(Labyrinth* env, float surface_z,
                                              int support_when_in_hole) {
    float min_z = surface_z + BALL_RADIUS;
    if (env->ball_z >= min_z)
        return;
    int in_hole = xy_in_any_hole(env, env->ball_x, env->ball_y);
    if (in_hole != support_when_in_hole)
        return;
    env->ball_z = min_z + COLLISION_SLOP;
    if (env->ball_vz < 0.0f) {
        if (-env->ball_vz < STICK_THRESHOLD)
            env->ball_vz = 0.0f;
        else
            env->ball_vz = -RESTITUTION * env->ball_vz;
    }
}

// Push the ball inward when it presses on a pit's cylindrical wall.
// Only acts while the ball is below the slab top AND inside this hole's disk —
// otherwise iterating over all holes would teleport a ball that's fallen into
// hole A toward hole B's center.
static inline void resolve_pit_cylinder_contact(Labyrinth* env, const Hole* h) {
    if (env->ball_z >= 0.0f)
        return;
    float dx = env->ball_x - h->cx;
    float dy = env->ball_y - h->cy;
    float d_sq = dx * dx + dy * dy;
    if (d_sq >= h->radius * h->radius)
        return; // ball center isn't in this hole's disk
    float d = sqrtf(d_sq);
    float allowed = h->radius - BALL_RADIUS;
    if (d <= allowed)
        return;
    float nx, ny;
    if (d > 1.0e-6f) {
        // Normal points inward (toward axis), opposite the radial direction.
        nx = -dx / d;
        ny = -dy / d;
    } else {
        nx = 1.0f;
        ny = 0.0f;
    }
    float pen = d - allowed + COLLISION_SLOP;
    env->ball_x += nx * pen;
    env->ball_y += ny * pen;
    float vn = env->ball_vx * nx + env->ball_vy * ny;
    if (vn >= 0.0f)
        return;
    if (-vn < STICK_THRESHOLD) {
        env->ball_vx -= vn * nx;
        env->ball_vy -= vn * ny;
    } else {
        float j = -(1.0f + RESTITUTION) * vn;
        env->ball_vx += j * nx;
        env->ball_vy += j * ny;
    }
}

static inline void labyrinth_step(Labyrinth* env) {
    // 1. Gravity in board-local frame. Tilt rotates the board; the ball's
    //    coordinates ARE in the board frame, so we rotate gravity instead of
    //    rotating geometry. Small-angle decomposition matches the previous
    //    tilt-as-acceleration formula at first order, but with a real -z term.
    float gx_raw = GRAVITY * sinf(env->tilt_x);
    float gy_raw = GRAVITY * sinf(env->tilt_y);
    float gz_raw = -GRAVITY * cosf(env->tilt_x) * cosf(env->tilt_y);

    // 2. When the ball is in contact with the slab top (or pit floor), it rolls
    //    rather than slides — the tangential acceleration is reduced by a factor
    //    of 5/7 for a uniform solid sphere. Detect contact by current z; the
    //    contact normal is +z, so tangential = (gx, gy), normal = gz.
    int on_slab = (env->ball_z <= BALL_RADIUS + 1.0e-4f) &&
                  !xy_in_any_hole(env, env->ball_x, env->ball_y);
    int on_floor = (env->ball_z <= -BOARD_THICKNESS + BALL_RADIUS + 1.0e-4f) &&
                   xy_in_any_hole(env, env->ball_x, env->ball_y);
    int rolling = on_slab || on_floor;
    float tan_factor = rolling ? ROLLING_GRAVITY_FACTOR : 1.0f;

    env->ball_vx += tan_factor * gx_raw * PHYSICS_DT;
    env->ball_vy += tan_factor * gy_raw * PHYSICS_DT;
    env->ball_vz += gz_raw * PHYSICS_DT;

    // Friction acts on the in-plane velocity only when in rolling contact.
    if (rolling) {
        env->ball_vx *= (1.0f - FRICTION_COEF);
        env->ball_vy *= (1.0f - FRICTION_COEF);
    }

    // 3. Integrate.
    float prev_x = env->ball_x;
    float prev_y = env->ball_y;
    int was_in_hole = xy_in_any_hole(env, prev_x, prev_y);
    env->ball_x += env->ball_vx * PHYSICS_DT;
    env->ball_y += env->ball_vy * PHYSICS_DT;
    env->ball_z += env->ball_vz * PHYSICS_DT;

    // 3b. Continuous collision against hole disks. At full tilt the ball
    //     covers several hole diameters per step; without CCD it'll "cheat"
    //     by leaping over holes. Find the earliest t∈(0,1] along the xy
    //     segment where ball center enters any hole, and snap ball to just
    //     inside that entry point. The slab-support check in step 4 will then
    //     see the ball over a hole and let it fall.
    {
        float dx = env->ball_x - prev_x;
        float dy = env->ball_y - prev_y;
        float seg_len_sq = dx * dx + dy * dy;
        if (seg_len_sq > 1.0e-12f) {
            float best_t = 2.0f;
            for (int i = 0; i < env->num_holes; i++) {
                const Hole* h = &env->holes[i];
                float ex = prev_x - h->cx;
                float ey = prev_y - h->cy;
                float C = ex * ex + ey * ey - h->radius * h->radius;
                if (C < 0.0f)
                    continue; // prev was already inside this hole
                float B = 2.0f * (ex * dx + ey * dy);
                float disc = B * B - 4.0f * seg_len_sq * C;
                if (disc < 0.0f)
                    continue;
                float t0 = (-B - sqrtf(disc)) / (2.0f * seg_len_sq);
                if (t0 > 0.0f && t0 <= 1.0f && t0 < best_t)
                    best_t = t0;
            }
            if (best_t <= 1.0f) {
                // Nudge just past the entry so the xy-in-hole test trips.
                float t_snap = best_t + 0.001f;
                if (t_snap > 1.0f)
                    t_snap = 1.0f;
                env->ball_x = prev_x + t_snap * dx;
                env->ball_y = prev_y + t_snap * dy;
            }
        }
    }

    // 4. Resolve contacts. Iterated like the wall solver: rim contact may push
    //    the ball into a pit wall, which then pushes back, etc.
    for (int iter = 0; iter < COLLISION_ITERATIONS; iter++) {
        resolve_horizontal_surface(env, 0.0f, 0);              // slab top (xy on solid slab)
        resolve_horizontal_surface(env, -BOARD_THICKNESS, 1);  // pit floor (xy in some hole)
        for (int w = 0; w < env->num_walls; w++) {
            resolve_wall_collision(env, &env->walls[w]);
        }
        for (int i = 0; i < env->num_holes; i++) {
            const Hole* h = &env->holes[i];
            resolve_rim_contact(env, h->cx, h->cy, h->radius);
            resolve_pit_cylinder_contact(env, h);
        }
    }

    integrate_ball_orientation(env);

    // 5. If the ball just crossed into a hole this step, kill its horizontal
    //    velocity. Without this, a ball moving faster than ~hole_diameter per
    //    step will exit the hole laterally before gravity can pull it below
    //    the slab — the "cheat by going fast" tunneling exploit.
    int now_in_hole = xy_in_any_hole(env, env->ball_x, env->ball_y);
    if (now_in_hole && !was_in_hole) {
        env->ball_vx = 0.0f;
        env->ball_vy = 0.0f;
    }

    // 6. Derive terminal flags for the renderer / RL signal.
    env->fell_in_hole =
        (env->ball_z < 0.0f) && xy_in_any_hole(env, env->ball_x, env->ball_y);
    env->reached_goal = 0;
    // Goal counts only while the ball is at or above the slab top plane —
    // a fallen ball at the bottom of a pit at the goal xy doesn't count.
    if (env->goal_radius > 0.0f && env->ball_z >= 0.0f) {
        float dx = env->ball_x - env->goal_cx;
        float dy = env->ball_y - env->goal_cy;
        if (dx * dx + dy * dy < env->goal_radius * env->goal_radius)
            env->reached_goal = 1;
    }

    env->tick++;
}

#endif
