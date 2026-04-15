/* Unit tests for labyrinth physics. No raylib. Compile with:
 *   clang -O0 -g -std=c11 \
 *     -fsanitize=address,undefined,bounds,pointer-overflow \
 *     ocean/labyrinth/test_labyrinth.c -o test_labyrinth -lm && ./test_labyrinth
 */

#include "labyrinth.h"
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
    for (int i = 0; i < 2000; i++) labyrinth_step(&env);
    float n = sqrtf(env.orient_w * env.orient_w + env.orient_x * env.orient_x +
                    env.orient_y * env.orient_y + env.orient_z * env.orient_z);
    EXPECT_NEAR(n, 1.0f, 1e-5f, "|q| ≈ 1 after 2000 steps");
}

// ---- Rolling direction tests ----

// Apply ball orientation quaternion to a body-frame point, giving the
// marker's world-frame offset from ball center. Mirrors rotate_by_quat in
// labyrinth.c, so a sign error there won't hide a physics sign error here.
static void world_offset_of_marker(const Labyrinth* env, float bx, float by, float bz, float* ox,
                                   float* oy, float* oz) {
    float qw = env->orient_w, qx = env->orient_x, qy = env->orient_y, qz = env->orient_z;
    float tx = 2.0f * (qy * bz - qz * by);
    float ty = 2.0f * (qz * bx - qx * bz);
    float tz = 2.0f * (qx * by - qy * bx);
    *ox = bx + qw * tx + (qy * tz - qz * ty);
    *oy = by + qw * ty + (qz * tx - qx * tz);
    *oz = bz + qw * tz + (qx * ty - qy * tx);
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
    if (quarter < 1) quarter = 1;

    // Start
    float ox0, oy0, oz0;
    world_offset_of_marker(&env, 0.0f, BALL_RADIUS, 0.0f, &ox0, &oy0, &oz0);
    EXPECT_NEAR(ox0, 0.0f, 1e-5f, "t=0: marker starts at ox ≈ 0");
    EXPECT_NEAR(oy0, BALL_RADIUS, 1e-5f, "t=0: marker starts at oy ≈ +r (top)");

    // 1/4 revolution
    for (int i = 0; i < quarter; i++) labyrinth_step(&env);
    float ox1, oy1, oz1;
    world_offset_of_marker(&env, 0.0f, BALL_RADIUS, 0.0f, &ox1, &oy1, &oz1);
    EXPECT(ox1 > 0.5f * BALL_RADIUS,
           "1/4 rev: marker is at +x (front of rolling direction)",
           "expected marker to have moved toward +x — if it moved toward -x, "
           "physics is spinning the ball in reverse");

    // 1/2 revolution
    for (int i = 0; i < quarter; i++) labyrinth_step(&env);
    float ox2, oy2, oz2;
    world_offset_of_marker(&env, 0.0f, BALL_RADIUS, 0.0f, &ox2, &oy2, &oz2);
    EXPECT(oy2 < -0.5f * BALL_RADIUS,
           "1/2 rev: marker is at -y (under the ball, contact point)",
           "expected marker to be below ball center (contact side)");

    // 3/4 revolution
    for (int i = 0; i < quarter; i++) labyrinth_step(&env);
    float ox3, oy3, oz3;
    world_offset_of_marker(&env, 0.0f, BALL_RADIUS, 0.0f, &ox3, &oy3, &oz3);
    EXPECT(ox3 < -0.5f * BALL_RADIUS,
           "3/4 rev: marker is at -x (behind rolling direction)",
           "expected marker to be behind the ball, in -x");
}

// Keep the earlier, weaker test as a quick sanity check.
static void test_rolling_top_moves_with_velocity(void) {
    printf("rolling direction (top of ball moves with translation)\n");
    // Give the ball pure +x velocity. After one step, a body-frame point
    // originally at +y (top) should have moved in +x.
    Labyrinth env = {0};
    labyrinth_reset(&env);
    env.ball_vx = 1.0f;
    env.ball_vy = 0.0f;
    env.tilt_x = 0.0f;
    env.tilt_y = 0.0f;

    // Body-frame top point (before step)
    float bx = 0.0f, by = BALL_RADIUS, bz = 0.0f;
    (void)bx; (void)bz;

    // Run one step (ignore translation, we just want orientation change)
    float x0 = env.ball_x, y0 = env.ball_y;
    labyrinth_step(&env);
    (void)x0; (void)y0;

    // Apply orientation to (0, r, 0): v' = q * v * q^-1
    float qw = env.orient_w, qx = env.orient_x, qy = env.orient_y, qz = env.orient_z;
    // Sandwich: (0, p) rotated by quaternion (w, x, y, z) gives coordinates:
    float vx_world = 2.0f * (qx * qy * by - qz * qw * by + 0 + 0);
    // Forget the full formula — just rotate using explicit sandwich helper.
    // Hamilton-product form: p' = q * p * q^-1
    // For p = (0, 0, r, 0) (pure-vector quaternion) that works out to:
    //   p'x = 2(qx*qy - qz*qw)*r      -- wait this doesn't match my other helper
    // To avoid bugs, re-derive using the explicit rotate_point formula:
    //   t = 2 * (q_vec × v)
    //   v' = v + q_w * t + (q_vec × t)
    float tx = 2.0f * (qy * bz - qz * by);
    float ty = 2.0f * (qz * bx - qx * bz);
    float tz = 2.0f * (qx * by - qy * bx);
    float rx = bx + qw * tx + (qy * tz - qz * ty);
    float ry = by + qw * ty + (qz * tx - qx * tz);
    float rz = bz + qw * tz + (qx * ty - qy * tx);
    (void)vx_world;
    (void)ry;
    (void)rz;

    // After rotating (0, r, 0) by one step's worth of rolling in +x,
    // the rotated point's x component should be positive (top moves with motion).
    EXPECT(rx > 0.0f, "top of ball moves in +x when rolling +x",
           "expected rotated +Y body point to have positive world X");
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
           "discrete v is close to continuous analytic",
           "friction effect outside expected range");
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
    for (int i = 0; i < 10; i++) labyrinth_step(&env);

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

    for (int i = 0; i < 5; i++) labyrinth_step(&env);
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
    EXPECT_NEAR(env.ball_vx, 0.0f, 1e-6f, "ball_vx zeroed");
    EXPECT_NEAR(env.ball_vy, 0.0f, 1e-6f, "ball_vy zeroed");
    EXPECT_NEAR(env.orient_w, 1.0f, 1e-6f, "orient identity");
    EXPECT(env.tick == 0, "tick zeroed", "tick not zero");
    EXPECT(env.num_walls == 4, "reset leaves the 4 border walls", "unexpected wall count");
}

int main(void) {
    printf("labyrinth physics tests\n");
    printf("=======================\n");
    test_quaternion_stays_unit();
    test_marker_trajectory();
    test_rolling_top_moves_with_velocity();
    test_rolling_acceleration();
    test_wall_collision_reflects();
    test_stick_threshold();
    test_closest_point_on_segment();
    test_reset_is_idempotent();
    printf("\n%d / %d tests passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
