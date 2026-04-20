/* Unit tests for labyrinth physics. No raylib. Compile with:
 *   clang -O0 -g -std=c11 \
 *     -fsanitize=address,undefined,bounds,pointer-overflow \
 *     ocean/labyrinth/test_labyrinth.c -o test_labyrinth -lm && ./test_labyrinth
 */

#include "labyrinth_physics.h"
#include "labyrinth_maze.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_failed = 0;

static void report(const char* name, int ok, const char* msg) {
    tests_run++;
    if (ok) {
        printf("  \x1b[32mPASS\x1b[0m %s\n", name);
    } else {
        tests_failed++;
        printf("  \x1b[31mFAIL\x1b[0m %s — %s\n", name, msg);
    }
}

#define EXPECT(cond, name, msg) report((name), (int)(cond), (msg))

#define EXPECT_NEAR(actual, expected, tol, name)                                                   \
    do {                                                                                           \
        float _a = (float)(actual);                                                                \
        float _e = (float)(expected);                                                              \
        float _t = (float)(tol);                                                                   \
        char _msg[128];                                                                            \
        snprintf(_msg, sizeof(_msg), "expected %.6f ± %.6f, got %.6f", _e, _t, _a);                \
        report((name), fabsf(_a - _e) <= _t, _msg);                                                \
    } while (0)

// ---- Quaternion tests ----

static void test_quaternion_stays_unit(void) {
    printf("quaternion stays unit length\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    env.tilt_x = 0.1f;
    env.tilt_y = 0.05f;
    for (int i = 0; i < 2000; i++)
        labyrinth_step(&env);
    float n = sqrtf(env.orient_w * env.orient_w + env.orient_x * env.orient_x +
                    env.orient_y * env.orient_y + env.orient_z * env.orient_z);
    EXPECT_NEAR(n, 1.0f, 1e-5f, "|q| ≈ 1 after 2000 steps");
}

// ---- Rolling direction tests ----

// Rotate a body-frame point by the env's current orientation quaternion.
// Thin wrapper around labyrinth_rotate_by_quat so the test code reads clean.
static void world_offset_of_marker(const Labyrinth* env, float bx, float by, float bz, float* ox,
                                   float* oy, float* oz) {
    labyrinth_rotate_by_quat(env->orient_w, env->orient_x, env->orient_y, env->orient_z,
                             bx, by, bz, ox, oy, oz);
}

// Track the body-frame marker (0, r, 0) — the "top" of the ball at t=0 — over
// a full rolling revolution in +x direction, and verify it sweeps the correct way.
//
// For a ball rolling in +x (right on screen), the top marker must sweep:
//   start at +y → move toward +x → pass through -y (contact point) → -x → back to +y.
// This is the right-hand-rule rotation around the -z axis.
// If rotation went the other way (around +z), the top marker would sweep
// toward -x first, which would look like "rolling in reverse" on screen.
static void test_marker_trajectory(void) {
    printf("marker trajectory on a ball rolling in +x\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_clear_walls(&env); // remove borders so the ball can roll freely
    env.ball_x = BOARD_W * 0.5f;
    env.ball_y = BOARD_H * 0.5f;
    env.ball_vx = 0.5f;
    env.ball_vy = 0.0f;
    env.tilt_x = 0.0f;
    env.tilt_y = 0.0f;

    // One full revolution in +x direction should cover 2·π·r = ~0.038 m.
    // At 0.5 m/s this takes ~0.075 s = ~15 physics steps.
    // Step 4x (~quarter rev each) and check the marker's quadrant:
    //
    //   After ~1/4 rev: marker should be mostly at +x (front of ball)    → ox > 0
    //   After ~1/2 rev: marker should be mostly at -y (below ball)       → oy < 0
    //   After ~3/4 rev: marker should be mostly at -x (behind the ball)  → ox < 0
    //   After ~full rev: marker should be back near +y                   → oy > 0
    //
    // If the rotation is reversed the sequence flips: first move -x, then +y, etc.

    float steps_per_rev = (2.0f * 3.14159265358979f * BALL_RADIUS / env.ball_vx) / PHYSICS_DT;
    int quarter = (int)(steps_per_rev * 0.25f);
    if (quarter < 1)
        quarter = 1;

    // Start
    float ox0, oy0, oz0;
    world_offset_of_marker(&env, 0.0f, BALL_RADIUS, 0.0f, &ox0, &oy0, &oz0);
    EXPECT_NEAR(ox0, 0.0f, 1e-5f, "t=0: marker starts at ox ≈ 0");
    EXPECT_NEAR(oy0, BALL_RADIUS, 1e-5f, "t=0: marker starts at oy ≈ +r (top)");

    // 1/4 revolution
    for (int i = 0; i < quarter; i++)
        labyrinth_step(&env);
    float ox1, oy1, oz1;
    world_offset_of_marker(&env, 0.0f, BALL_RADIUS, 0.0f, &ox1, &oy1, &oz1);
    EXPECT(ox1 > 0.5f * BALL_RADIUS, "1/4 rev: marker is at +x (front of rolling direction)",
           "expected marker to have moved toward +x — if it moved toward -x, "
           "physics is spinning the ball in reverse");

    // 1/2 revolution
    for (int i = 0; i < quarter; i++)
        labyrinth_step(&env);
    float ox2, oy2, oz2;
    world_offset_of_marker(&env, 0.0f, BALL_RADIUS, 0.0f, &ox2, &oy2, &oz2);
    EXPECT(oy2 < -0.5f * BALL_RADIUS, "1/2 rev: marker is at -y (under the ball, contact point)",
           "expected marker to be below ball center (contact side)");

    // 3/4 revolution
    for (int i = 0; i < quarter; i++)
        labyrinth_step(&env);
    float ox3, oy3, oz3;
    world_offset_of_marker(&env, 0.0f, BALL_RADIUS, 0.0f, &ox3, &oy3, &oz3);
    EXPECT(ox3 < -0.5f * BALL_RADIUS, "3/4 rev: marker is at -x (behind rolling direction)",
           "expected marker to be behind the ball, in -x");
}

// ---- Analytical acceleration test ----

static void test_rolling_acceleration(void) {
    printf("rolling acceleration matches analytic (with friction model)\n");
    // Reimplement the exact discrete physics (no walls, no orientation) and
    // compare to the env's step output. This catches algorithmic regressions
    // without being fragile to friction-vs-continuous-analytic mismatches.
    Labyrinth env = {0};
    labyrinth_reset(&env);
    env.ball_x = BOARD_W * 0.5f;
    env.ball_y = BOARD_H * 0.5f;
    env.tilt_x = 0.1f;
    env.tilt_y = 0.0f;
    labyrinth_clear_walls(&env);

    // Same state, but we integrate by hand as a reference.
    float ref_vx = 0.0f, ref_vy = 0.0f;
    float ref_x = BOARD_W * 0.5f, ref_y = BOARD_H * 0.5f;
    float accel_x = ROLLING_GRAVITY_FACTOR * GRAVITY * sinf(env.tilt_x);
    float accel_y = ROLLING_GRAVITY_FACTOR * GRAVITY * sinf(env.tilt_y);

    int steps = 50;
    for (int i = 0; i < steps; i++) {
        labyrinth_step(&env);
        ref_vx += accel_x * PHYSICS_DT;
        ref_vy += accel_y * PHYSICS_DT;
        ref_vx *= (1.0f - FRICTION_COEF);
        ref_vy *= (1.0f - FRICTION_COEF);
        ref_x += ref_vx * PHYSICS_DT;
        ref_y += ref_vy * PHYSICS_DT;
    }

    EXPECT_NEAR(env.ball_vx, ref_vx, 1e-5f, "vx matches hand-integrated reference");
    EXPECT_NEAR(env.ball_x, ref_x, 1e-5f, "x matches hand-integrated reference");
    // Sanity check that the hand-integrated values are close to the continuous analytic
    // (they should differ by only a few percent due to friction).
    float t = steps * PHYSICS_DT;
    float continuous_vx = accel_x * t;
    EXPECT(ref_vx > 0.9f * continuous_vx && ref_vx < 1.01f * continuous_vx,
           "discrete v is close to continuous analytic", "friction effect outside expected range");
}

// ---- Wall collision tests ----

static void test_wall_collision_reflects(void) {
    printf("wall collision reflects normal velocity, preserves tangent\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    // Place ball near right edge (the right wall is at x=BOARD_W in the border),
    // moving directly into it.
    env.ball_x = BOARD_W - BALL_RADIUS * 2.0f;
    env.ball_y = BOARD_H * 0.5f;
    env.ball_vx = 0.5f;
    env.ball_vy = 0.1f; // tangential — should survive
    env.tilt_x = 0.0f;
    env.tilt_y = 0.0f;

    float v_tangent_before = env.ball_vy;
    // Run until collision happens (typically 1-2 steps)
    for (int i = 0; i < 10; i++)
        labyrinth_step(&env);

    // Normal component should reverse (with restitution); tangent mostly preserved.
    EXPECT(env.ball_vx < 0.0f, "vx reverses after hitting right wall",
           "expected vx < 0 after collision");
    EXPECT(fabsf(env.ball_vy - v_tangent_before) < fabsf(v_tangent_before) * 0.2f,
           "tangent component preserved (within 20% due to friction)",
           "tangent velocity changed too much");
}

// ---- Stick threshold test ----

static void test_stick_threshold(void) {
    printf("low-energy contact sticks instead of bouncing\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    env.ball_x = BOARD_W - BALL_RADIUS - 0.001f; // just inside right wall
    env.ball_y = BOARD_H * 0.5f;
    env.ball_vx = 0.001f; // very slow into the wall (< STICK_THRESHOLD)
    env.ball_vy = 0.0f;
    env.tilt_x = 0.0f;
    env.tilt_y = 0.0f;

    for (int i = 0; i < 5; i++)
        labyrinth_step(&env);
    // After contact the normal velocity should be zero (sticking), not reflected.
    EXPECT(fabsf(env.ball_vx) < 1e-4f, "slow contact parks the ball (|vx| near zero)",
           "ball bounced when it should have parked");
}

// ---- Closest-point-on-segment geometry ----

static void test_closest_point_on_segment(void) {
    printf("closest_point_on_segment geometry\n");
    float qx, qy;
    // Horizontal segment from (0,0) to (1,0). Closest point to (0.5, 0.3) is (0.5, 0).
    closest_point_on_segment(0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 0.3f, &qx, &qy);
    EXPECT_NEAR(qx, 0.5f, 1e-6f, "interior: qx = 0.5");
    EXPECT_NEAR(qy, 0.0f, 1e-6f, "interior: qy = 0.0");

    // Point off the left end — should clamp to p1.
    closest_point_on_segment(0.0f, 0.0f, 1.0f, 0.0f, -0.5f, 0.2f, &qx, &qy);
    EXPECT_NEAR(qx, 0.0f, 1e-6f, "past left end: qx = 0.0");
    EXPECT_NEAR(qy, 0.0f, 1e-6f, "past left end: qy = 0.0");

    // Point off the right end — should clamp to p2.
    closest_point_on_segment(0.0f, 0.0f, 1.0f, 0.0f, 1.8f, -0.2f, &qx, &qy);
    EXPECT_NEAR(qx, 1.0f, 1e-6f, "past right end: qx = 1.0");
    EXPECT_NEAR(qy, 0.0f, 1e-6f, "past right end: qy = 0.0");

    // Degenerate segment (two identical endpoints).
    closest_point_on_segment(0.5f, 0.5f, 0.5f, 0.5f, 0.2f, 0.1f, &qx, &qy);
    EXPECT_NEAR(qx, 0.5f, 1e-6f, "degenerate: qx = p1.x");
    EXPECT_NEAR(qy, 0.5f, 1e-6f, "degenerate: qy = p1.y");
}

// ---- Reset state ----

// ---- Hole tests ----

static void test_hole_center_triggers_fall(void) {
    printf("ball over hole center drops into the pit within a few steps\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_clear_walls(&env);
    labyrinth_add_hole(&env, 0.15f, 0.12f, 1.3f * BALL_RADIUS);
    env.ball_x = 0.15f;
    env.ball_y = 0.12f;
    env.ball_vx = 0.0f;
    env.ball_vy = 0.0f;
    env.tilt_x = 0.0f;
    env.tilt_y = 0.0f;
    // ball needs ~sqrt(2*BALL_RADIUS/g) ≈ 35 ms to drop its center past z=0
    for (int i = 0; i < 20; i++)
        labyrinth_step(&env);
    EXPECT(env.ball_z < 0.0f, "ball_z dropped below slab top", "ball didn't fall");
    EXPECT(env.fell_in_hole == 1, "fell_in_hole derived flag set", "flag not set");
}

static void test_ball_overhangs_rim_without_falling(void) {
    printf("ball can overhang the hole rim without falling (center still outside)\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_clear_walls(&env);
    float hole_r = 1.3f * BALL_RADIUS;
    labyrinth_add_hole(&env, 0.15f, 0.12f, hole_r);
    // Ball edge overlaps the hole but center stays outside: center_dist ∈ (hole_r, hole_r + r).
    env.ball_x = 0.15f + hole_r + 0.2f * BALL_RADIUS;
    env.ball_y = 0.12f;
    env.ball_vx = 0.0f;
    env.ball_vy = 0.0f;
    env.tilt_x = 0.0f;
    env.tilt_y = 0.0f;
    labyrinth_step(&env);
    EXPECT(env.fell_in_hole == 0, "overhang does not fall when center is outside hole radius",
           "unexpected fall while center was outside hole radius");
}

static void test_hole_far_away_no_effect(void) {
    printf("hole does not affect a distant ball\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_clear_walls(&env);
    labyrinth_add_hole(&env, 0.25f, 0.20f, 1.3f * BALL_RADIUS);
    env.ball_x = 0.05f;
    env.ball_y = 0.05f;
    env.ball_vx = 0.1f;
    env.ball_vy = 0.0f;
    env.tilt_x = 0.0f;
    env.tilt_y = 0.0f;

    float vx_before = env.ball_vx;
    labyrinth_step(&env);
    // Only friction should have touched vx. Hole should be irrelevant.
    float expected_vx = vx_before * (1.0f - FRICTION_COEF);
    EXPECT_NEAR(env.ball_vx, expected_vx, 1e-6f, "distant hole leaves vx unchanged");
    EXPECT(env.fell_in_hole == 0, "distant hole does not trigger fall", "unexpected fall");
}

static void test_fall_settles_at_pit_floor(void) {
    printf("fall accelerates under gravity and settles at the pit floor\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_clear_walls(&env);
    labyrinth_add_hole(&env, 0.15f, 0.12f, 1.3f * BALL_RADIUS);
    env.ball_x = 0.15f; // dead center — ball falls straight down
    env.ball_y = 0.12f;
    env.tilt_x = 0.0f;
    env.tilt_y = 0.0f;

    // Run long enough for the ball to reach the floor (should be < 50ms).
    for (int i = 0; i < 200; i++)
        labyrinth_step(&env);
    float floor_z = -BOARD_THICKNESS + BALL_RADIUS;
    EXPECT_NEAR(env.ball_z, floor_z, 1e-3f, "ball_z settles at pit floor");
    EXPECT(env.ball_vz <= 0.05f && env.ball_vz >= -0.05f,
           "vertical velocity is small at rest", "ball still bouncing on floor");
    EXPECT(env.fell_in_hole == 1, "fell_in_hole derived flag set", "flag not set");
}

static void test_fast_ball_does_not_tunnel_over_hole(void) {
    printf("ball moving faster than hole-per-step still falls in (CCD)\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_clear_walls(&env);
    labyrinth_add_hole(&env, 0.15f, 0.12f, 1.3f * BALL_RADIUS);
    // Park the ball to the left of the hole, pointed at it at a speed that
    // moves the ball well past the hole in a single step.
    env.ball_x = 0.10f;
    env.ball_y = 0.12f;
    env.ball_vx = 8.0f; // ~40mm per step at PHYSICS_DT=1/200, hole is at x=0.15
    env.ball_vy = 0.0f;
    env.tilt_x = 0.0f;
    env.tilt_y = 0.0f;
    // Let the ball cross the hole. Without CCD it would sail right past; with
    // CCD it should trip xy_in_any_hole on the step whose xy segment crosses
    // the hole disk.
    int caught = 0;
    for (int i = 0; i < 10; i++) {
        labyrinth_step(&env);
        if (env.fell_in_hole) {
            caught = 1;
            break;
        }
    }
    EXPECT(caught == 1, "fast ball caught by hole CCD", "ball tunneled over the hole");
}

static void test_no_clip_through_slab(void) {
    printf("ball never clips through the slab top during a tip into a hole\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_clear_walls(&env);
    float hole_r = 1.3f * BALL_RADIUS;
    labyrinth_add_hole(&env, 0.15f, 0.12f, hole_r);
    // Place the ball near the rim, drifting slowly inward.
    env.ball_x = 0.15f + hole_r - 0.0005f;
    env.ball_y = 0.12f;
    env.ball_vx = -0.05f;
    env.ball_vy = 0.0f;
    env.tilt_x = 0.0f;
    env.tilt_y = 0.0f;

    // Step until ball lands on pit floor or 0.5s elapses. At every moment the
    // contact point xy is over the slab, ball_z must be >= BALL_RADIUS minus a
    // small tolerance — otherwise the ball is poking through the slab top.
    for (int i = 0; i < 100; i++) {
        labyrinth_step(&env);
        // Sample the contact point xy (directly below ball center).
        float cx = env.ball_x;
        float cy = env.ball_y;
        float dx = cx - 0.15f;
        float dy = cy - 0.12f;
        int over_slab = (dx * dx + dy * dy) >= (hole_r * hole_r);
        if (over_slab && env.ball_z < BALL_RADIUS - 1e-3f) {
            EXPECT(0, "ball clipped through slab top", "ball_z < BALL_RADIUS over slab");
            return;
        }
    }
    EXPECT(1, "no slab clip during tip", "");
}

static void test_goal_disabled_by_default(void) {
    printf("goal_radius == 0 disables the goal mechanic\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_step(&env);
    EXPECT(env.reached_goal == 0, "reached_goal stays 0 with no goal", "unexpected goal flag");
}

static void test_goal_reached_when_ball_on_pad(void) {
    printf("ball on the goal pad sets reached_goal\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_clear_walls(&env);
    labyrinth_set_goal(&env, 0.20f, 0.10f, 0.018f);
    labyrinth_place_ball(&env, 0.20f, 0.10f);
    labyrinth_step(&env);
    EXPECT(env.reached_goal == 1, "reached_goal set when ball is on goal", "flag not set");
}

static void test_goal_not_reached_when_ball_falls(void) {
    printf("falling into a hole at the goal location does not set reached_goal\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_clear_walls(&env);
    labyrinth_add_hole(&env, 0.20f, 0.10f, 1.3f * BALL_RADIUS);
    labyrinth_set_goal(&env, 0.20f, 0.10f, 0.018f);
    labyrinth_place_ball(&env, 0.20f, 0.10f);
    for (int i = 0; i < 50; i++)
        labyrinth_step(&env);
    EXPECT(env.fell_in_hole == 1, "ball fell in hole", "fall not triggered");
    EXPECT(env.reached_goal == 0, "reached_goal NOT set while ball is in pit",
           "goal flag set despite fall");
}

// ---- Procedural maze tests ----

// BFS from start to goal cell over the maze the generator just produced.
// Returns 1 if the goal cell is reachable through the maze's open passages
// (i.e., not blocked by a wall). We approximate "open passage" by sampling
// the midpoint between two cell centers and checking it isn't inside a wall
// capsule. This treats holes as passable — the path exists, dodging is the
// player's problem.
static int maze_goal_reachable(const Labyrinth* env, int rows, int cols) {
    float cw = BOARD_W / (float)cols;
    float ch = BOARD_H / (float)rows;
    char visited[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM] = {0};
    int queue[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
    int qh = 0, qt = 0;
    int start_c = (int)(env->ball_x / cw);
    int start_r = (int)(env->ball_y / ch);
    int goal_c = (int)(env->goal_cx / cw);
    int goal_r = (int)(env->goal_cy / ch);
    int start_idx = start_r * cols + start_c;
    int goal_idx = goal_r * cols + goal_c;
    queue[qt++] = start_idx;
    visited[start_idx] = 1;
    while (qh < qt) {
        int idx = queue[qh++];
        if (idx == goal_idx)
            return 1;
        int r = idx / cols, c = idx % cols;
        // For each cardinal neighbor, check if midpoint between cell centers
        // is inside any wall capsule (within ball radius + wall half-thickness).
        int candidates[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int k = 0; k < 4; k++) {
            int nr = r + candidates[k][0], nc = c + candidates[k][1];
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols)
                continue;
            int nidx = nr * cols + nc;
            if (visited[nidx])
                continue;
            float mx = ((c + nc + 1) * 0.5f) * cw; // shared edge midpoint x
            float my = ((r + nr + 1) * 0.5f) * ch; // shared edge midpoint y
            int blocked = 0;
            for (int w = 0; w < env->num_walls; w++) {
                float qx, qy;
                closest_point_on_segment(env->walls[w].x1, env->walls[w].y1, env->walls[w].x2,
                                         env->walls[w].y2, mx, my, &qx, &qy);
                float ddx = mx - qx, ddy = my - qy;
                if (ddx * ddx + ddy * ddy < COLLISION_RADIUS * COLLISION_RADIUS) {
                    blocked = 1;
                    break;
                }
            }
            if (!blocked) {
                visited[nidx] = 1;
                queue[qt++] = nidx;
            }
        }
    }
    return 0;
}

static void test_grid_maze_solvable_for_many_seeds(void) {
    printf("generated mazes are always solvable from start to goal cell\n");
    int unsolvable = 0;
    for (uint32_t seed = 1; seed <= 50; seed++) {
        Labyrinth env = {0};
        labyrinth_reset(&env);
        labyrinth_load_random_maze(&env, seed);
        if (!maze_goal_reachable(&env, 5, 6))
            unsolvable++;
    }
    EXPECT(unsolvable == 0, "all 50 seeds produce a reachable goal", "some seeds produced unreachable mazes");
}

static void test_grid_maze_deterministic(void) {
    printf("same seed → identical wall and hole layout\n");
    Labyrinth a = {0}, b = {0};
    labyrinth_reset(&a);
    labyrinth_reset(&b);
    labyrinth_load_random_maze(&a, 7777);
    labyrinth_load_random_maze(&b, 7777);
    int ok = (a.num_walls == b.num_walls) && (a.num_holes == b.num_holes);
    for (int i = 0; ok && i < a.num_walls; i++) {
        ok = (a.walls[i].x1 == b.walls[i].x1) && (a.walls[i].y1 == b.walls[i].y1) &&
             (a.walls[i].x2 == b.walls[i].x2) && (a.walls[i].y2 == b.walls[i].y2);
    }
    for (int i = 0; ok && i < a.num_holes; i++) {
        ok = (a.holes[i].cx == b.holes[i].cx) && (a.holes[i].cy == b.holes[i].cy);
    }
    EXPECT(ok, "two mazes from seed 7777 are identical", "mazes differ");
}

// The renderer merges nearby hole pairs with a dark capsule. If any barrier
// hole pair falls just OUTSIDE that threshold, the slab strip between the
// two pits stays visible ("weird line" artifact). This test asserts:
//   1. The renderer's threshold covers the maximum possible barrier-pair
//      distance in the default 6x7 grid (colinear same-barrier pair of
//      L/2 for max(cw, ch)).
//   2. Across 100 seeds, no hole pair falls in the danger zone between
//      the threshold and 1.3x — i.e., no pair is close enough to LOOK
//      connected but fail to bridge.
static void test_grid_maze_start_and_goal_are_safe(void) {
    printf("the ball's start position and the goal disk are clear of holes\n");
    int total_unsafe_starts = 0, total_blocked_goals = 0;
    for (uint32_t seed = 1; seed <= 50; seed++) {
        Labyrinth env = {0};
        labyrinth_reset(&env);
        labyrinth_load_random_maze(&env, seed);
        // ball starts at (env.ball_x, env.ball_y); must not be inside any hole.
        for (int i = 0; i < env.num_holes; i++) {
            float dx = env.ball_x - env.holes[i].cx;
            float dy = env.ball_y - env.holes[i].cy;
            if (dx * dx + dy * dy < env.holes[i].radius * env.holes[i].radius) {
                total_unsafe_starts++;
                break;
            }
        }
        // goal center must not be inside any hole.
        for (int i = 0; i < env.num_holes; i++) {
            float dx = env.goal_cx - env.holes[i].cx;
            float dy = env.goal_cy - env.holes[i].cy;
            if (dx * dx + dy * dy < env.holes[i].radius * env.holes[i].radius) {
                total_blocked_goals++;
                break;
            }
        }
    }
    EXPECT(total_unsafe_starts == 0, "ball start is never inside a hole",
           "ball spawned inside a hole on some seeds");
    EXPECT(total_blocked_goals == 0, "goal center is never inside a hole",
           "goal sits on a hole on some seeds");
}

static void test_reset_clears_fall_state(void) {
    printf("reset clears fell_in_hole and holes\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_add_hole(&env, 0.1f, 0.1f, 1.3f * BALL_RADIUS);
    env.fell_in_hole = 1;
    labyrinth_reset(&env);
    EXPECT(env.fell_in_hole == 0, "reset clears fell_in_hole", "flag still set");
    EXPECT(env.num_holes == 0, "reset clears holes", "holes still present");
}

static void test_reset_is_idempotent(void) {
    printf("reset is idempotent and returns a clean state\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    // Perturb everything
    env.ball_x = 0.999f;
    env.ball_y = 0.888f;
    env.ball_vx = 10.0f;
    env.ball_vy = -10.0f;
    env.orient_w = 0.0f;
    env.orient_x = 1.0f; // invalid quaternion
    env.tick = 12345;
    // Reset
    labyrinth_reset(&env);
    EXPECT_NEAR(env.ball_x, BOARD_W * 0.5f, 1e-6f, "ball_x centered");
    EXPECT_NEAR(env.ball_y, BOARD_H * 0.5f, 1e-6f, "ball_y centered");
    EXPECT_NEAR(env.ball_z, BALL_RADIUS, 1e-6f, "ball_z sits on slab top");
    EXPECT_NEAR(env.ball_vx, 0.0f, 1e-6f, "ball_vx zeroed");
    EXPECT_NEAR(env.ball_vy, 0.0f, 1e-6f, "ball_vy zeroed");
    EXPECT_NEAR(env.ball_vz, 0.0f, 1e-6f, "ball_vz zeroed");
    EXPECT_NEAR(env.orient_w, 1.0f, 1e-6f, "orient identity");
    EXPECT(env.tick == 0, "tick zeroed", "tick not zero");
    EXPECT(env.num_walls == 4, "reset leaves the 4 border walls", "unexpected wall count");
}

// ---- Path / hole observation helper tests ----

// Build a Labyrinth with the given polyline as its path (border walls stay
// from labyrinth_reset; no holes). Path is in the env's world frame.
static void make_env_with_path(Labyrinth* env, const float (*pts)[2], int n) {
    labyrinth_reset(env);
    env->num_path_points = n;
    for (int i = 0; i < n && i < MAX_PATH_POINTS; i++) {
        env->path_points[i][0] = pts[i][0];
        env->path_points[i][1] = pts[i][1];
    }
}

static void test_path_lengths_simple(void) {
    printf("labyrinth_path_lengths: cumulative lengths for known polyline\n");
    Labyrinth env = {0};
    // Two segments: (0,0)→(0.10,0)=0.10, (0.10,0)→(0.10,0.05)=0.05. Total=0.15.
    float pts[3][2] = {{0.00f, 0.00f}, {0.10f, 0.00f}, {0.10f, 0.05f}};
    make_env_with_path(&env, pts, 3);
    float len_from[MAX_PATH_POINTS];
    float total = -1.0f;
    labyrinth_path_lengths(&env, len_from, &total);
    EXPECT_NEAR(total, 0.15f, 1e-6f, "total path length");
    EXPECT_NEAR(len_from[0], 0.15f, 1e-6f, "len_from[0] == total");
    EXPECT_NEAR(len_from[1], 0.05f, 1e-6f, "len_from[1] (after first seg)");
    EXPECT_NEAR(len_from[2], 0.00f, 1e-6f, "len_from[end] == 0");
}

static void test_path_lengths_degenerate(void) {
    printf("labyrinth_path_lengths: degenerate paths (0 / 1 points)\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    env.num_path_points = 0;
    float len_from[MAX_PATH_POINTS];
    float total = 999.0f;
    labyrinth_path_lengths(&env, len_from, &total);
    EXPECT_NEAR(total, 0.0f, 1e-9f, "0-point path: total=0");
    EXPECT_NEAR(len_from[0], 0.0f, 1e-9f, "0-point path: len_from[0]=0");
    env.num_path_points = 1;
    env.path_points[0][0] = 0.05f;
    env.path_points[0][1] = 0.07f;
    total = 999.0f;
    labyrinth_path_lengths(&env, len_from, &total);
    EXPECT_NEAR(total, 0.0f, 1e-9f, "1-point path: total=0");
}

static void test_nearest_path_point_on_path(void) {
    printf("labyrinth_nearest_path_point: ball on path → offset ≈ 0\n");
    Labyrinth env = {0};
    float pts[3][2] = {{0.00f, 0.10f}, {0.20f, 0.10f}, {0.20f, 0.20f}};
    make_env_with_path(&env, pts, 3);
    float ox, oy; int seg; float t;
    // Midpoint of first segment.
    labyrinth_nearest_path_point(&env, 0.10f, 0.10f, &ox, &oy, &seg, &t);
    EXPECT_NEAR(ox, 0.10f, 1e-6f, "on-path: out_x");
    EXPECT_NEAR(oy, 0.10f, 1e-6f, "on-path: out_y");
    EXPECT(seg == 0, "on-path: seg=0", "wrong segment");
    EXPECT_NEAR(t, 0.5f, 1e-6f, "on-path: t=0.5");
}

static void test_nearest_path_point_off_path(void) {
    printf("labyrinth_nearest_path_point: ball off path → correct projection\n");
    Labyrinth env = {0};
    float pts[2][2] = {{0.00f, 0.10f}, {0.20f, 0.10f}};
    make_env_with_path(&env, pts, 2);
    float ox, oy; int seg; float t;
    // Ball at (0.05, 0.13) — projects perpendicularly onto y=0.10 line.
    labyrinth_nearest_path_point(&env, 0.05f, 0.13f, &ox, &oy, &seg, &t);
    EXPECT_NEAR(ox, 0.05f, 1e-6f, "off-path: out_x");
    EXPECT_NEAR(oy, 0.10f, 1e-6f, "off-path: out_y");
    EXPECT(seg == 0, "off-path: seg=0", "wrong segment");
    EXPECT_NEAR(t, 0.25f, 1e-6f, "off-path: t=0.25");
}

static void test_nearest_path_point_clamp_to_endpoint(void) {
    printf("labyrinth_nearest_path_point: query past segment end clamps to endpoint\n");
    Labyrinth env = {0};
    float pts[2][2] = {{0.00f, 0.10f}, {0.20f, 0.10f}};
    make_env_with_path(&env, pts, 2);
    float ox, oy; int seg; float t;
    // Ball way past the right end.
    labyrinth_nearest_path_point(&env, 0.50f, 0.10f, &ox, &oy, &seg, &t);
    EXPECT_NEAR(ox, 0.20f, 1e-6f, "clamp-end: out_x snaps to endpoint");
    EXPECT_NEAR(oy, 0.10f, 1e-6f, "clamp-end: out_y");
    EXPECT_NEAR(t, 1.0f, 1e-6f, "clamp-end: t=1");
    // Ball way before the left end.
    labyrinth_nearest_path_point(&env, -0.30f, 0.10f, &ox, &oy, &seg, &t);
    EXPECT_NEAR(ox, 0.00f, 1e-6f, "clamp-start: out_x snaps to endpoint");
    EXPECT_NEAR(t, 0.0f, 1e-6f, "clamp-start: t=0");
}

static void test_nearest_path_point_multi_segment(void) {
    printf("labyrinth_nearest_path_point: multi-segment picks closest segment\n");
    Labyrinth env = {0};
    // L-shape: horizontal then vertical.
    float pts[3][2] = {{0.00f, 0.10f}, {0.20f, 0.10f}, {0.20f, 0.20f}};
    make_env_with_path(&env, pts, 3);
    float ox, oy; int seg; float t;
    // Ball clearly closest to the second (vertical) segment.
    labyrinth_nearest_path_point(&env, 0.22f, 0.15f, &ox, &oy, &seg, &t);
    EXPECT(seg == 1, "multi-seg: picks vertical seg", "wrong segment chosen");
    EXPECT_NEAR(ox, 0.20f, 1e-6f, "multi-seg: projects onto x=0.20");
    EXPECT_NEAR(oy, 0.15f, 1e-6f, "multi-seg: out_y");
    EXPECT_NEAR(t, 0.5f, 1e-6f, "multi-seg: t=0.5 along vertical");
}

static void test_nearest_path_point_empty_path(void) {
    printf("labyrinth_nearest_path_point: empty path falls back to goal\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    env.num_path_points = 0;
    env.goal_cx = 0.123f; env.goal_cy = 0.045f;
    float ox, oy; int seg; float t;
    labyrinth_nearest_path_point(&env, 0.5f, 0.5f, &ox, &oy, &seg, &t);
    EXPECT_NEAR(ox, 0.123f, 1e-9f, "empty path: out_x = goal_cx");
    EXPECT_NEAR(oy, 0.045f, 1e-9f, "empty path: out_y = goal_cy");
}

static void test_path_tangent_unit_and_direction(void) {
    printf("labyrinth_path_tangent: unit length and correct direction\n");
    Labyrinth env = {0};
    // Diagonal segment, then horizontal.
    float pts[3][2] = {{0.00f, 0.00f}, {0.03f, 0.04f}, {0.13f, 0.04f}};
    make_env_with_path(&env, pts, 3);
    float tx, ty;
    labyrinth_path_tangent(&env, 0, &tx, &ty);
    // (0.03, 0.04) length 0.05 → unit (0.6, 0.8).
    EXPECT_NEAR(tx, 0.6f, 1e-6f, "diag tangent x");
    EXPECT_NEAR(ty, 0.8f, 1e-6f, "diag tangent y");
    EXPECT_NEAR(sqrtf(tx * tx + ty * ty), 1.0f, 1e-6f, "diag tangent unit length");
    labyrinth_path_tangent(&env, 1, &tx, &ty);
    EXPECT_NEAR(tx, 1.0f, 1e-6f, "horiz tangent x");
    EXPECT_NEAR(ty, 0.0f, 1e-6f, "horiz tangent y");
}

static void test_path_tangent_out_of_range(void) {
    printf("labyrinth_path_tangent: out-of-range seg → zero vector\n");
    Labyrinth env = {0};
    float pts[2][2] = {{0.00f, 0.00f}, {0.10f, 0.00f}};
    make_env_with_path(&env, pts, 2);
    float tx = 99.0f, ty = 99.0f;
    labyrinth_path_tangent(&env, -1, &tx, &ty);
    EXPECT_NEAR(tx, 0.0f, 1e-9f, "negative seg: tx=0");
    EXPECT_NEAR(ty, 0.0f, 1e-9f, "negative seg: ty=0");
    tx = 99.0f; ty = 99.0f;
    labyrinth_path_tangent(&env, 5, &tx, &ty); // beyond last seg
    EXPECT_NEAR(tx, 0.0f, 1e-9f, "past-end seg: tx=0");
    EXPECT_NEAR(ty, 0.0f, 1e-9f, "past-end seg: ty=0");
}

static void test_nearest_hole_no_holes(void) {
    printf("labyrinth_nearest_hole: zero holes → diag sentinel\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    env.num_holes = 0;
    float dx = 99.0f, dy = 99.0f, d = 99.0f;
    labyrinth_nearest_hole(&env, 0.05f, 0.05f, &dx, &dy, &d);
    EXPECT_NEAR(dx, 0.0f, 1e-9f, "no holes: dx=0");
    EXPECT_NEAR(dy, 0.0f, 1e-9f, "no holes: dy=0");
    float diag = sqrtf(BOARD_W * BOARD_W + BOARD_H * BOARD_H);
    EXPECT_NEAR(d, diag, 1e-6f, "no holes: dist=board_diag");
}

static void test_nearest_hole_single(void) {
    printf("labyrinth_nearest_hole: single hole → exact offset\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_add_hole(&env, 0.10f, 0.20f, 1.3f * BALL_RADIUS);
    float dx, dy, d;
    labyrinth_nearest_hole(&env, 0.07f, 0.16f, &dx, &dy, &d);
    EXPECT_NEAR(dx, 0.03f, 1e-6f, "single hole: dx=hole-ball");
    EXPECT_NEAR(dy, 0.04f, 1e-6f, "single hole: dy=hole-ball");
    EXPECT_NEAR(d, 0.05f, 1e-6f, "single hole: dist=5cm (3-4-5)");
}

static void test_nearest_hole_picks_closest(void) {
    printf("labyrinth_nearest_hole: multiple holes → picks closest\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_add_hole(&env, 0.05f, 0.05f, 1.3f * BALL_RADIUS); // far
    labyrinth_add_hole(&env, 0.21f, 0.20f, 1.3f * BALL_RADIUS); // closest
    labyrinth_add_hole(&env, 0.28f, 0.05f, 1.3f * BALL_RADIUS); // far
    float dx, dy, d;
    labyrinth_nearest_hole(&env, 0.20f, 0.20f, &dx, &dy, &d);
    EXPECT_NEAR(dx, 0.01f, 1e-6f, "multi-hole: dx of closest");
    EXPECT_NEAR(dy, 0.00f, 1e-6f, "multi-hole: dy of closest");
    EXPECT_NEAR(d, 0.01f, 1e-6f, "multi-hole: dist of closest");
}

// Integration: replicate compute_observations() against a known geometry and
// verify all 13 values are correct and within their declared ranges.
static void test_full_obs_known_geometry(void) {
    printf("compute_observations: 13-dim obs on known geometry\n");
    Labyrinth env = {0};
    // Two-segment L-path. Ball starts at (0.05, 0.10) — exactly on first seg.
    float pts[3][2] = {{0.00f, 0.10f}, {0.20f, 0.10f}, {0.20f, 0.20f}};
    make_env_with_path(&env, pts, 3);
    env.ball_x = 0.05f; env.ball_y = 0.10f;
    env.ball_vx = 0.5f; env.ball_vy = -0.4f; env.ball_vz = 0.0f;
    env.tilt_x = 0.5f * MAX_TILT_RAD;
    env.tilt_y = -0.25f * MAX_TILT_RAD;
    labyrinth_add_hole(&env, 0.08f, 0.14f, 1.3f * BALL_RADIUS);

    float len_from[MAX_PATH_POINTS]; float total;
    labyrinth_path_lengths(&env, len_from, &total);
    // total = 0.20 (horiz) + 0.10 (vert) = 0.30
    EXPECT_NEAR(total, 0.30f, 1e-6f, "obs setup: total path len = 0.30");

    // Replicate compute_observations() body.
    float o[13];
    const float inv_vmax = 1.0f / 2.0f;
    const float inv_t = 1.0f / MAX_TILT_RAD;
    const float inv_w = 1.0f / BOARD_W;
    const float inv_h = 1.0f / BOARD_H;
    const float diag = sqrtf(BOARD_W * BOARD_W + BOARD_H * BOARD_H);
    o[0] = env.ball_vx * inv_vmax;
    o[1] = env.ball_vy * inv_vmax;
    o[2] = env.ball_vz * inv_vmax;
    o[3] = env.tilt_x * inv_t;
    o[4] = env.tilt_y * inv_t;
    float ppx, ppy, ptx, pty; int seg; float tt;
    labyrinth_nearest_path_point(&env, env.ball_x, env.ball_y, &ppx, &ppy, &seg, &tt);
    labyrinth_path_tangent(&env, seg, &ptx, &pty);
    o[5] = (ppx - env.ball_x) * inv_w;
    o[6] = (ppy - env.ball_y) * inv_h;
    o[7] = ptx; o[8] = pty;
    float dist_remaining = (1.0f - tt) * (len_from[seg] - len_from[seg + 1]) + len_from[seg + 1];
    o[9] = (total > 1e-9f) ? (dist_remaining / total) : 0.0f;
    float hdx, hdy, hd;
    labyrinth_nearest_hole(&env, env.ball_x, env.ball_y, &hdx, &hdy, &hd);
    o[10] = hdx * inv_w; o[11] = hdy * inv_h; o[12] = hd / diag;

    // Ball v scaled (0.25, -0.20, 0).
    EXPECT_NEAR(o[0], 0.25f, 1e-6f, "obs[0] = vx/2");
    EXPECT_NEAR(o[1], -0.20f, 1e-6f, "obs[1] = vy/2");
    EXPECT_NEAR(o[2], 0.0f, 1e-6f, "obs[2] = vz/2");
    EXPECT_NEAR(o[3], 0.5f, 1e-6f, "obs[3] = tilt_x/MAX");
    EXPECT_NEAR(o[4], -0.25f, 1e-6f, "obs[4] = tilt_y/MAX");
    // Ball ON segment 0 → offset (0,0).
    EXPECT_NEAR(o[5], 0.0f, 1e-6f, "obs[5] path offset x ≈ 0");
    EXPECT_NEAR(o[6], 0.0f, 1e-6f, "obs[6] path offset y ≈ 0");
    EXPECT_NEAR(o[7], 1.0f, 1e-6f, "obs[7] tangent x = +1");
    EXPECT_NEAR(o[8], 0.0f, 1e-6f, "obs[8] tangent y = 0");
    // dist remaining at (0.05, 0.10) along seg 0 (t=0.25): 0.15 + 0.10 = 0.25 / 0.30
    EXPECT_NEAR(o[9], 0.25f / 0.30f, 1e-5f, "obs[9] dist_along normalized");
    // Hole at (0.08, 0.14), ball (0.05, 0.10): offset (0.03, 0.04), dist 0.05.
    EXPECT_NEAR(o[10], 0.03f / BOARD_W, 1e-6f, "obs[10] hole offset x");
    EXPECT_NEAR(o[11], 0.04f / BOARD_H, 1e-6f, "obs[11] hole offset y");
    EXPECT_NEAR(o[12], 0.05f / diag, 1e-6f, "obs[12] hole dist normalized");

    // Range bounds — the policy assumes obs in roughly [-1, 1].
    int all_in_range = 1;
    for (int i = 0; i < 13; i++) {
        if (o[i] < -1.5f || o[i] > 1.5f) { all_in_range = 0; break; }
    }
    EXPECT(all_in_range, "all 13 obs values in [-1.5, 1.5]", "out-of-range value");
}

// Procedural-maze integration: build a real curriculum maze, cache lengths,
// and confirm the obs computation does not crash and produces in-range values
// across many seeds and difficulties.
static void test_obs_on_procedural_maze(void) {
    printf("compute_observations: in-range values across many procedural mazes\n");
    int total = 0, in_range = 0;
    for (uint32_t seed = 1; seed <= 60; seed++) {
        for (int di = 0; di <= 4; di++) {
            float difficulty = di * 0.25f;
            Labyrinth env = {0};
            labyrinth_reset(&env);
            labyrinth_load_curriculum_maze(&env, seed, difficulty);
            if (env.num_path_points < 2) continue;
            float len_from[MAX_PATH_POINTS]; float tot;
            labyrinth_path_lengths(&env, len_from, &tot);
            float ppx, ppy, ptx, pty; int seg; float tt;
            labyrinth_nearest_path_point(&env, env.ball_x, env.ball_y, &ppx, &ppy, &seg, &tt);
            labyrinth_path_tangent(&env, seg, &ptx, &pty);
            float hdx, hdy, hd;
            labyrinth_nearest_hole(&env, env.ball_x, env.ball_y, &hdx, &hdy, &hd);
            float diag = sqrtf(BOARD_W * BOARD_W + BOARD_H * BOARD_H);
            float vals[8] = {
                (ppx - env.ball_x) / BOARD_W,
                (ppy - env.ball_y) / BOARD_H,
                ptx, pty,
                (1.0f - tt) * (len_from[seg] - len_from[seg + 1]) + len_from[seg + 1],
                hdx / BOARD_W, hdy / BOARD_H, hd / diag,
            };
            // Normalize the dist_along for the range check.
            vals[4] = (tot > 1e-9f) ? vals[4] / tot : 0.0f;
            int ok = 1;
            for (int k = 0; k < 8; k++) {
                if (vals[k] < -1.01f || vals[k] > 1.01f) { ok = 0; break; }
            }
            if (ok) in_range++;
            total++;
        }
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "%d / %d (seed,diff) configs in range", in_range, total);
    EXPECT(in_range == total && total > 0, "procedural mazes all in [-1,1]", msg);
}

// ---- minimal-obs helper tests ----

static void test_next_path_idx_start_of_path(void) {
    printf("labyrinth_next_path_idx: ball near start → idx 1\n");
    Labyrinth env = {0};
    float pts[4][2] = {{0.00f, 0.10f}, {0.10f, 0.10f}, {0.10f, 0.20f}, {0.20f, 0.20f}};
    make_env_with_path(&env, pts, 4);
    // Ball near the start point (path_points[0]).
    int idx = labyrinth_next_path_idx(&env, 0.01f, 0.10f);
    EXPECT(idx == 1, "start → next=1", "wrong next idx");
}

static void test_next_path_idx_mid_segment(void) {
    printf("labyrinth_next_path_idx: ball mid-segment → idx of segment end\n");
    Labyrinth env = {0};
    float pts[4][2] = {{0.00f, 0.10f}, {0.10f, 0.10f}, {0.10f, 0.20f}, {0.20f, 0.20f}};
    make_env_with_path(&env, pts, 4);
    // Ball in the middle of segment 0 (0,0.1)→(0.1,0.1).
    int idx = labyrinth_next_path_idx(&env, 0.05f, 0.10f);
    EXPECT(idx == 1, "mid-seg-0 → next=1", "wrong next idx");
    // Ball in the middle of segment 1 (0.1,0.1)→(0.1,0.2).
    idx = labyrinth_next_path_idx(&env, 0.10f, 0.15f);
    EXPECT(idx == 2, "mid-seg-1 → next=2", "wrong next idx");
    // Ball in the middle of segment 2 (0.1,0.2)→(0.2,0.2).
    idx = labyrinth_next_path_idx(&env, 0.15f, 0.20f);
    EXPECT(idx == 3, "mid-seg-2 → next=3", "wrong next idx");
}

static void test_next_path_idx_at_last_waypoint(void) {
    printf("labyrinth_next_path_idx: ball at last waypoint → idx clamped to last\n");
    Labyrinth env = {0};
    float pts[4][2] = {{0.00f, 0.10f}, {0.10f, 0.10f}, {0.10f, 0.20f}, {0.20f, 0.20f}};
    make_env_with_path(&env, pts, 4);
    int idx = labyrinth_next_path_idx(&env, 0.20f, 0.20f);
    EXPECT(idx == 3, "at last waypoint → idx=3 (clamped)", "wrong");
}

static void test_next_path_idx_degenerate(void) {
    printf("labyrinth_next_path_idx: empty/single-point path → idx 0\n");
    Labyrinth env = {0};
    labyrinth_reset(&env);
    env.num_path_points = 0;
    EXPECT(labyrinth_next_path_idx(&env, 0.5f, 0.5f) == 0, "0-point: idx=0", "wrong");
    env.num_path_points = 1;
    env.path_points[0][0] = 0.1f; env.path_points[0][1] = 0.1f;
    EXPECT(labyrinth_next_path_idx(&env, 0.5f, 0.5f) == 0, "1-point: idx=0", "wrong");
}

// Integration test: reproduce the 10-dim obs on a known geometry.
static void test_minimal_obs_known_geometry(void) {
    printf("minimal obs: 10 scalars on known path\n");
    Labyrinth env = {0};
    // Path: start (0,0.1) → corner (0.1,0.1) → corner (0.1,0.2) → goal (0.2,0.2)
    float pts[4][2] = {{0.00f, 0.10f}, {0.10f, 0.10f}, {0.10f, 0.20f}, {0.20f, 0.20f}};
    make_env_with_path(&env, pts, 4);
    env.ball_x = 0.05f; env.ball_y = 0.10f;  // mid-segment 0
    env.ball_vx = 0.4f; env.ball_vy = -0.2f;
    env.tilt_x = 0.5f * MAX_TILT_RAD;
    env.tilt_y = -0.25f * MAX_TILT_RAD;
    env.goal_cx = 0.20f; env.goal_cy = 0.20f;

    // Replicate compute_observations logic.
    float obs[10];
    const float inv_vmax = 1.0f / 2.0f;
    const float inv_t = 1.0f / MAX_TILT_RAD;
    const float inv_w = 1.0f / BOARD_W;
    const float inv_h = 1.0f / BOARD_H;
    obs[0] = env.ball_vx * inv_vmax;
    obs[1] = env.ball_vy * inv_vmax;
    obs[2] = env.tilt_x * inv_t;
    obs[3] = env.tilt_y * inv_t;
    int next_idx = labyrinth_next_path_idx(&env, env.ball_x, env.ball_y);
    int after_idx = next_idx + 1;
    if (after_idx >= env.num_path_points) after_idx = env.num_path_points - 1;
    obs[4] = (env.path_points[next_idx][0]  - env.ball_x) * inv_w;
    obs[5] = (env.path_points[next_idx][1]  - env.ball_y) * inv_h;
    obs[6] = (env.path_points[after_idx][0] - env.ball_x) * inv_w;
    obs[7] = (env.path_points[after_idx][1] - env.ball_y) * inv_h;
    obs[8] = (env.goal_cx - env.ball_x) * inv_w;
    obs[9] = (env.goal_cy - env.ball_y) * inv_h;

    EXPECT_NEAR(obs[0], 0.2f, 1e-6f, "obs[0] = vx/2");
    EXPECT_NEAR(obs[1], -0.1f, 1e-6f, "obs[1] = vy/2");
    EXPECT_NEAR(obs[2], 0.5f, 1e-6f, "obs[2] = tilt_x/MAX");
    EXPECT_NEAR(obs[3], -0.25f, 1e-6f, "obs[3] = tilt_y/MAX");
    // next waypoint = path_points[1] = (0.1, 0.1). offset from (0.05, 0.10) = (0.05, 0).
    EXPECT_NEAR(obs[4], 0.05f / BOARD_W, 1e-6f, "obs[4] next.dx");
    EXPECT_NEAR(obs[5], 0.00f / BOARD_H, 1e-6f, "obs[5] next.dy");
    // after waypoint = path_points[2] = (0.1, 0.2). offset = (0.05, 0.10).
    EXPECT_NEAR(obs[6], 0.05f / BOARD_W, 1e-6f, "obs[6] after.dx");
    EXPECT_NEAR(obs[7], 0.10f / BOARD_H, 1e-6f, "obs[7] after.dy");
    // goal = (0.20, 0.20). offset = (0.15, 0.10).
    EXPECT_NEAR(obs[8], 0.15f / BOARD_W, 1e-6f, "obs[8] goal.dx");
    EXPECT_NEAR(obs[9], 0.10f / BOARD_H, 1e-6f, "obs[9] goal.dy");

    int all_in_range = 1;
    for (int i = 0; i < 10; i++) {
        if (obs[i] < -1.5f || obs[i] > 1.5f) { all_in_range = 0; break; }
    }
    EXPECT(all_in_range, "all 10 obs in [-1.5, 1.5]", "out of range");
}

static void test_minimal_obs_advances_with_ball(void) {
    printf("minimal obs: next-waypoint advances as ball crosses segments\n");
    Labyrinth env = {0};
    float pts[4][2] = {{0.00f, 0.10f}, {0.10f, 0.10f}, {0.10f, 0.20f}, {0.20f, 0.20f}};
    make_env_with_path(&env, pts, 4);
    // At start: next=1
    EXPECT(labyrinth_next_path_idx(&env, 0.01f, 0.10f) == 1, "start: next=1", "");
    // Just past segment 0: next=2
    EXPECT(labyrinth_next_path_idx(&env, 0.10f, 0.12f) == 2, "past seg0: next=2", "");
    // Just past segment 1: next=3
    EXPECT(labyrinth_next_path_idx(&env, 0.12f, 0.20f) == 3, "past seg1: next=3", "");
    // At goal: still clamped to 3
    EXPECT(labyrinth_next_path_idx(&env, 0.20f, 0.20f) == 3, "at goal: next=3", "");
}

int main(void) {
    printf("labyrinth physics tests\n");
    printf("=======================\n");
    test_quaternion_stays_unit();
    test_marker_trajectory();
    test_rolling_acceleration();
    test_wall_collision_reflects();
    test_stick_threshold();
    test_closest_point_on_segment();
    test_hole_center_triggers_fall();
    test_ball_overhangs_rim_without_falling();
    test_hole_far_away_no_effect();
    test_fall_settles_at_pit_floor();
    test_fast_ball_does_not_tunnel_over_hole();
    test_no_clip_through_slab();
    test_goal_disabled_by_default();
    test_goal_reached_when_ball_on_pad();
    test_goal_not_reached_when_ball_falls();
    test_grid_maze_solvable_for_many_seeds();
    test_grid_maze_deterministic();
    test_grid_maze_start_and_goal_are_safe();
    test_reset_clears_fall_state();
    test_reset_is_idempotent();
    test_path_lengths_simple();
    test_path_lengths_degenerate();
    test_nearest_path_point_on_path();
    test_nearest_path_point_off_path();
    test_nearest_path_point_clamp_to_endpoint();
    test_nearest_path_point_multi_segment();
    test_nearest_path_point_empty_path();
    test_path_tangent_unit_and_direction();
    test_path_tangent_out_of_range();
    test_nearest_hole_no_holes();
    test_nearest_hole_single();
    test_nearest_hole_picks_closest();
    test_full_obs_known_geometry();
    test_obs_on_procedural_maze();
    test_next_path_idx_start_of_path();
    test_next_path_idx_mid_segment();
    test_next_path_idx_at_last_waypoint();
    test_next_path_idx_degenerate();
    test_minimal_obs_known_geometry();
    test_minimal_obs_advances_with_ball();
    printf("\n%d / %d tests passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
