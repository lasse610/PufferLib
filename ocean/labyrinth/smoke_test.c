/* Labyrinth env smoke test.
 *
 * Runs the Ocean RL env API (LabyrinthEnv) headlessly for N steps with random
 * actions, then reports:
 *   - observation range sanity (grid codes in [0,4], scalars finite)
 *   - reward distribution & NaN/Inf detection
 *   - episode stats (how many terminals, goal vs fall ratio)
 *
 * Catches dumb wiring bugs before we burn GPU time on training.
 *
 * Compile (same flags as standalone, since labyrinth.h pulls in raylib
 * headers — we never call render, so no window opens):
 *   ./build.sh labyrinth --fast
 *   # Then compile this separately:
 *   clang -O2 -std=c11 -I. -Iocean/labyrinth -I./raylib-5.5_macos/include \
 *     -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
 *     -DPLATFORM_DESKTOP \
 *     ocean/labyrinth/smoke_test.c \
 *     ./raylib-5.5_macos/lib/libraylib.a \
 *     -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL \
 *     -lm -lpthread -lomp -o smoke_test
 *   ./smoke_test
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "labyrinth.h"

static int is_finite_f(float x) {
    return !(isnan(x) || isinf(x));
}

int main(void) {
    const int N_STEPS = 5000;

    LabyrinthEnv env = {0};
    env.num_agents = 1;
    env.seed = 12345u;
    env.max_steps = 2000;
    env.physics_substeps = 4;
    init(&env);
    allocate_LabyrinthEnv(&env);
    c_reset(&env);

    // Track stats across the run. c_step internally calls c_reset on
    // terminals, so env.terminals[0] is always 0 by the time we read it after
    // c_step returns. Instead we detect terminals by watching env.log.n
    // (increments once per add_log call) and classify by comparing reward
    // sign in that frame.
    int goals = 0, falls = 0, timeouts = 0;
    float total_reward = 0.0f;
    float min_reward = 0.0f, max_reward = 0.0f;
    int bad_obs = 0;
    int nonfinite_obs = 0;
    int nonfinite_reward = 0;
    int nonfinite_terminal = 0;
    float scalar_abs_max[LABYRINTH_SCALAR_FEATURES] = {0};
    unsigned int rng = 42u;
    float last_log_n = env.log.n;
    float last_goals_log = env.log.reached_goal;
    float last_falls_log = env.log.fell_in_hole;

    for (int t = 0; t < N_STEPS; t++) {
        rng = rng * 1664525u + 1013904223u;
        // Continuous: 2 random tilts in [-1, 1].
        env.actions[0] = ((float)(rng & 0xFFFF) / 32768.0f) - 1.0f;
        env.actions[1] = ((float)((rng >> 16) & 0xFFFF) / 32768.0f) - 1.0f;

        c_step(&env);

        float r = env.rewards[0];
        if (!is_finite_f(r)) nonfinite_reward++;
        else {
            total_reward += r;
            if (r < min_reward) min_reward = r;
            if (r > max_reward) max_reward = r;
        }
        if (!is_finite_f(env.terminals[0])) nonfinite_terminal++;

        // Terminal detection via log counter increments (rewards[0] and
        // terminals[0] have been cleared by c_reset inside c_step, but the
        // log fields are bumped BEFORE that reset).
        if (env.log.n > last_log_n) {
            int was_goal = env.log.reached_goal > last_goals_log;
            int was_fall = env.log.fell_in_hole > last_falls_log;
            if (was_goal)      goals++;
            else if (was_fall) falls++;
            else               timeouts++;
            last_log_n = env.log.n;
            last_goals_log = env.log.reached_goal;
            last_falls_log = env.log.fell_in_hole;
        }

        for (int i = 0; i < LABYRINTH_VIEW_SIZE; i++) {
            float v = env.observations[i];
            if (!is_finite_f(v)) { nonfinite_obs++; continue; }
            int iv = (int)v;
            if (iv < 0 || iv > 4) bad_obs++;
        }
        for (int i = 0; i < LABYRINTH_SCALAR_FEATURES; i++) {
            float v = env.observations[LABYRINTH_VIEW_SIZE + i];
            if (!is_finite_f(v)) { nonfinite_obs++; continue; }
            float av = fabsf(v);
            if (av > scalar_abs_max[i]) scalar_abs_max[i] = av;
        }
    }
    int num_episodes = (int)env.log.n;

    printf("\n=== Labyrinth env smoke test ===\n");
    printf("Steps run:       %d\n", N_STEPS);
    printf("Episodes ended:  %d\n", num_episodes);
    printf("  reached goal:  %d\n", goals);
    printf("  fell in hole:  %d\n", falls);
    printf("  timeouts:      %d\n", timeouts);
    printf("\n");
    printf("Total reward:    %.4f\n", total_reward);
    printf("Reward range:    [%.4f, %.4f]\n", min_reward, max_reward);
    printf("Avg reward/step: %.6f\n", total_reward / N_STEPS);
    printf("\n");
    printf("Observation grid codes out of [0,4]: %d (of %d)\n",
           bad_obs, N_STEPS * LABYRINTH_VIEW_SIZE);
    printf("Non-finite observations:    %d\n", nonfinite_obs);
    printf("Non-finite rewards:         %d\n", nonfinite_reward);
    printf("Non-finite terminals:       %d\n", nonfinite_terminal);
    printf("\n");
    printf("Scalar obs abs-max (should stay near 1.0 for normalization):\n");
    const char* names[LABYRINTH_SCALAR_FEATURES] = {
        "vx", "vy", "vz", "tilt_x", "tilt_y", "goal_dx", "goal_dy", "bfs_dist_norm",
    };
    for (int i = 0; i < LABYRINTH_SCALAR_FEATURES; i++) {
        printf("  %s = %.3f\n", names[i], scalar_abs_max[i]);
    }
    printf("\n");
    printf("Log aggregate (from add_log across episodes):\n");
    printf("  perf:           %.4f\n", env.log.perf);
    printf("  score:          %.4f\n", env.log.score);
    printf("  episode_return: %.4f\n", env.log.episode_return);
    printf("  episode_length: %.1f\n", env.log.episode_length);
    printf("  reached_goal:   %.1f\n", env.log.reached_goal);
    printf("  fell_in_hole:   %.1f\n", env.log.fell_in_hole);
    printf("  n:              %.1f\n", env.log.n);

    // Decide pass/fail.
    int pass = 1;
    if (nonfinite_obs || nonfinite_reward || nonfinite_terminal) pass = 0;
    if (bad_obs) pass = 0;
    if (num_episodes == 0) {
        // Random policy should reach SOME terminal in 5000 steps (max 1500/ep).
        printf("\nWARN: no episodes terminated — max_steps too high or termination broken?\n");
    }

    printf("\n%s\n", pass ? "PASS" : "FAIL");
    c_close(&env);
    free_allocated(&env);
    return pass ? 0 : 1;
}
