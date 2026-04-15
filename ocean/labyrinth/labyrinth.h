/* Labyrinth: tilt-a-ball maze.
 * Header-only per the Ocean env convention. See labyrinth.c for the standalone game.
 *
 * Levels (we are at 2):
 *   1  tilt + gravity + friction on an empty board
 *   2  walls + bounce
 *   3  proper rolling (angular velocity coupled to translation)
 *   4  holes (terminal on fall)
 *   5  real maze layout
 */

#ifndef LABYRINTH_H
#define LABYRINTH_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ===== Constants =====

#define BOARD_W 0.30f
#define BOARD_H 0.24f
#define BALL_RADIUS 0.006f

#define GRAVITY 9.81f
// Rolling acceleration factor for a uniform solid sphere: a = (5/7) g sin(theta).
// Energy splits between translation and rotation (I = 2/5 mr^2).
#define ROLLING_GRAVITY_FACTOR (5.0f / 7.0f)
// Rolling friction is weak — marbles coast a long way on a smooth board.
#define FRICTION_COEF 0.001f
#define MAX_TILT_RAD 0.20f
#define PHYSICS_DT (1.0f / 200.0f)

#define MAX_WALLS 32
#define WALL_THICKNESS_M 0.004f
#define WALL_HALF_THICKNESS (0.5f * WALL_THICKNESS_M)
#define COLLISION_RADIUS (BALL_RADIUS + WALL_HALF_THICKNESS)
#define RESTITUTION 0.3f
#define COLLISION_SLOP 1.0e-5f
#define STICK_THRESHOLD 0.02f
#define COLLISION_ITERATIONS 4

// ===== Types =====

typedef struct {
    float x1, y1;
    float x2, y2;
} Wall;

typedef struct {
    float ball_x, ball_y;
    float ball_vx, ball_vy;
    float tilt_x, tilt_y;
    // 3D orientation of the ball (unit quaternion, w+xyz). Tracks how the
    // ball has rolled over time so we can render its spin. Physics stays 2D —
    // angular velocity is derived from the motion direction.
    float orient_w, orient_x, orient_y, orient_z;
    Wall walls[MAX_WALLS];
    int num_walls;
    int tick;
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
    env->ball_vx = 0;
    env->ball_vy = 0;
    env->tilt_x = 0;
    env->tilt_y = 0;
    env->orient_w = 1.0f;
    env->orient_x = 0.0f;
    env->orient_y = 0.0f;
    env->orient_z = 0.0f;
    env->tick = 0;
    labyrinth_clear_walls(env);
    labyrinth_add_border(env);
}

static inline void labyrinth_add_demo_walls(Labyrinth* env) {
    labyrinth_add_wall(env, 0.06f, 0.08f, 0.14f, 0.08f);
    labyrinth_add_wall(env, 0.20f, 0.05f, 0.20f, 0.15f);
    labyrinth_add_wall(env, 0.08f, 0.18f, 0.18f, 0.21f);
}

// Integrate the ball orientation for one physics step.
// The ball rolls without slipping, so angular velocity magnitude is |v|/r
// and its axis is perpendicular to the velocity direction (in the board plane).
// We rotate around the world axis (-vy, 0, vx) — perpendicular to motion, lying
// on the board surface.
static inline void integrate_ball_orientation(Labyrinth* env) {
    float speed = sqrtf(env->ball_vx * env->ball_vx + env->ball_vy * env->ball_vy);
    if (speed < 1e-6f)
        return;

    float omega = speed / BALL_RADIUS; // rad / s
    float angle = omega * PHYSICS_DT;  // radians rotated this step

    // Rolling-without-slipping axis derivation (right-hand rule):
    // Ball rolls in +x, contact point at world (0,-r,0) is stationary,
    // => ω = (0, 0, -v/r). Similarly motion in +z (board y) => ω = (+v/r, 0, 0).
    // NOTE: the sign is under active investigation via tests — see
    // test_marker_trajectory in test_labyrinth.c.
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

static inline void labyrinth_step(Labyrinth* env) {
    // Rolling acceleration from tilt (reduced from pure sliding by I = 2/5 mr^2).
    env->ball_vx += ROLLING_GRAVITY_FACTOR * GRAVITY * sinf(env->tilt_x) * PHYSICS_DT;
    env->ball_vy += ROLLING_GRAVITY_FACTOR * GRAVITY * sinf(env->tilt_y) * PHYSICS_DT;

    env->ball_vx *= (1.0f - FRICTION_COEF);
    env->ball_vy *= (1.0f - FRICTION_COEF);

    env->ball_x += env->ball_vx * PHYSICS_DT;
    env->ball_y += env->ball_vy * PHYSICS_DT;

    for (int iter = 0; iter < COLLISION_ITERATIONS; iter++) {
        for (int w = 0; w < env->num_walls; w++) {
            resolve_wall_collision(env, &env->walls[w]);
        }
    }

    integrate_ball_orientation(env);

    env->tick++;
}

#endif
