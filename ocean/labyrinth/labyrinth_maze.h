/* Labyrinth maze generation.
 *
 * Three ways to populate an empty Labyrinth env (after labyrinth_reset):
 *   1. labyrinth_load_serpentine_maze(env)        — hand-authored reference level
 *   2. labyrinth_load_random_maze(env, seed)      — default 6x7 procedural maze
 *   3. labyrinth_generate_grid_maze(env, ...)     — parametric procedural maze
 *
 * Header-only. Pulls in labyrinth_physics.h for the types.
 */

#ifndef LABYRINTH_MAZE_H
#define LABYRINTH_MAZE_H

#include "labyrinth_physics.h"

// Hole radius a bit larger than the ball so the rim is catchable but the ball
// reliably falls through once the center crosses.
#define DEMO_HOLE_RADIUS (1.3f * BALL_RADIUS)

// ===== Hand-authored "serpentine" level =====

// Two horizontal walls split the board into three channels with alternating
// gaps (right, left). Ball starts bottom-left, goal sits top-right. Holes are
// scattered along each channel as obstacles.
//
//   +--------------------------+
//   |                  [GOAL]  |  <- top channel    (gap on left)
//   |  ._______________________|
//   |  ●           ●           |  <- middle channel (gap on right)
//   |______________________.   |
//   |  ●    ●     ●            |  <- bottom channel (gap on right)
//   |  S                       |
//   +--------------------------+
//
// Call AFTER labyrinth_reset(). Reset adds the border walls.
static inline void labyrinth_load_serpentine_maze(Labyrinth* env) {
    // Bottom divider: blocks x=[0, 0.22], leaves a gap at x=[0.22, BOARD_W].
    labyrinth_add_wall(env, 0.00f, 0.10f, 0.22f, 0.10f);
    // Top divider: blocks x=[0.08, BOARD_W], leaves a gap at x=[0, 0.08].
    labyrinth_add_wall(env, 0.08f, 0.16f, BOARD_W, 0.16f);

    // Obstacle holes along the path.
    labyrinth_add_hole(env, 0.09f, 0.05f, DEMO_HOLE_RADIUS); // bottom channel
    labyrinth_add_hole(env, 0.17f, 0.05f, DEMO_HOLE_RADIUS);
    labyrinth_add_hole(env, 0.20f, 0.13f, DEMO_HOLE_RADIUS); // middle channel
    labyrinth_add_hole(env, 0.13f, 0.13f, DEMO_HOLE_RADIUS);
    labyrinth_add_hole(env, 0.18f, 0.20f, DEMO_HOLE_RADIUS); // top channel

    labyrinth_set_goal(env, 0.275f, 0.205f, 0.018f);
    labyrinth_place_ball(env, 0.025f, 0.040f);
}

// ===== Procedural maze =====

// Max rows or cols the generator supports. Sized to fit the stack arrays in
// labyrinth_generate_grid_maze without heap allocation.
#define LAB_MAZE_MAX_DIM 12

// Cheap deterministic xorshift32. Same seed → same maze.
// `seed | 1` below guarantees a non-zero state (xorshift gets stuck at 0).
static inline uint32_t lab_rng_next(uint32_t* state) {
    uint32_t x = *state ? *state : 0xCAFEBABEu;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Build a DFS-spanning-tree maze on a `rows x cols` grid sized to fit the board.
// For each cell pair the DFS didn't connect, we place either a wall or a hole
// "barrier" (a line of holes the ball can't cross). `barrier_prob` (0..1) sets
// the chance that any given non-connection becomes a barrier rather than a wall.
//
// `num_obstacle_holes` adds extra single holes scattered on the unique
// start→goal path, giving direct in-corridor dodging challenge.
//
// Call AFTER labyrinth_reset(). Reset adds the border walls.
static inline void labyrinth_generate_grid_maze(Labyrinth* env, uint32_t seed, int rows, int cols,
                                                float barrier_prob, int num_obstacle_holes) {
    if (rows < 2 || cols < 2 || rows > LAB_MAZE_MAX_DIM || cols > LAB_MAZE_MAX_DIM)
        return;

    const int N = rows * cols;
    char visited[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
    char conn_h[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM]; // (r,c) ↔ (r,c+1) connected?
    char conn_v[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM]; // (r,c) ↔ (r+1,c) connected?
    int  stack[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
    memset(visited, 0, (size_t)N);
    memset(conn_h, 0, (size_t)N);
    memset(conn_v, 0, (size_t)N);
    uint32_t rng = seed | 1u;

    // ---- 1. DFS spanning tree ----
    int top = 0;
    stack[top++] = 0;
    visited[0] = 1;
    while (top > 0) {
        int idx = stack[top - 1];
        int r = idx / cols, c = idx % cols;
        int neigh[4][2]; // (dr, dc) for each unvisited neighbour
        int n_count = 0;
        if (r > 0 && !visited[(r - 1) * cols + c]) { neigh[n_count][0] = -1; neigh[n_count][1] = 0; n_count++; }
        if (r < rows - 1 && !visited[(r + 1) * cols + c]) { neigh[n_count][0] = 1; neigh[n_count][1] = 0; n_count++; }
        if (c > 0 && !visited[r * cols + (c - 1)]) { neigh[n_count][0] = 0; neigh[n_count][1] = -1; n_count++; }
        if (c < cols - 1 && !visited[r * cols + (c + 1)]) { neigh[n_count][0] = 0; neigh[n_count][1] = 1; n_count++; }
        if (n_count == 0) {
            top--;
            continue;
        }
        int pick = (int)(lab_rng_next(&rng) % (uint32_t)n_count);
        int dr = neigh[pick][0], dc = neigh[pick][1];
        int nr = r + dr, nc = c + dc;
        if (dr == -1)      conn_v[(r - 1) * cols + c] = 1;
        else if (dr == 1)  conn_v[r * cols + c] = 1;
        else if (dc == -1) conn_h[r * cols + (c - 1)] = 1;
        else               conn_h[r * cols + c] = 1;
        visited[nr * cols + nc] = 1;
        stack[top++] = nr * cols + nc;
    }

    // ---- 2. Decide wall vs barrier for each non-connected adjacency ----
    // Two passes because the placement in pass 3 needs to know what's at each
    // corner to decide per-end hole margins.
    char is_barrier_h[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
    char is_barrier_v[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
    memset(is_barrier_h, 0, sizeof(is_barrier_h));
    memset(is_barrier_v, 0, sizeof(is_barrier_v));
    uint32_t prob_threshold = (uint32_t)(barrier_prob * (float)0xFFFFFFFFu);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols - 1; c++) {
            if (!conn_h[r * cols + c])
                is_barrier_h[r * cols + c] = (lab_rng_next(&rng) < prob_threshold) ? 1 : 0;
        }
    }
    for (int r = 0; r < rows - 1; r++) {
        for (int c = 0; c < cols; c++) {
            if (!conn_v[r * cols + c])
                is_barrier_v[r * cols + c] = (lab_rng_next(&rng) < prob_threshold) ? 1 : 0;
        }
    }

    // ---- 2b. Demote "both ends T-junction" barriers to walls ----
    // With asymmetric per-end margins, this is the only case where we cannot
    // keep both the same-barrier pair AND the corner-diagonal pairs from
    // visually overlapping. Drop these specific barriers back to walls.
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols - 1; c++) {
            if (!is_barrier_h[r * cols + c])
                continue;
            int perp_top = (r > 0) && (is_barrier_v[(r - 1) * cols + c] ||
                                       is_barrier_v[(r - 1) * cols + (c + 1)]);
            int perp_bot = (r < rows - 1) && (is_barrier_v[r * cols + c] ||
                                              is_barrier_v[r * cols + (c + 1)]);
            if (perp_top && perp_bot)
                is_barrier_h[r * cols + c] = 0;
        }
    }
    for (int r = 0; r < rows - 1; r++) {
        for (int c = 0; c < cols; c++) {
            if (!is_barrier_v[r * cols + c])
                continue;
            int perp_left = (c > 0) && (is_barrier_h[r * cols + (c - 1)] ||
                                        is_barrier_h[(r + 1) * cols + (c - 1)]);
            int perp_right = (c < cols - 1) && (is_barrier_h[r * cols + c] ||
                                                is_barrier_h[(r + 1) * cols + c]);
            if (perp_left && perp_right)
                is_barrier_v[r * cols + c] = 0;
        }
    }

    // ---- 3. Place walls and barriers ----
    // Each barrier has 2 holes, one near each end of the boundary. The margin
    // from the corner depends on whether a perpendicular barrier meets there:
    //   Free end:        L/4 per-axis — gives uniform spacing both within the
    //                    barrier AND across adjacent colinear barriers
    //                    (intra = L - 2·L/4 = L/2 = inter = 2·L/4).
    //   T-junction end:  END_T, pulled inward so the corner-diagonal pair of
    //                    holes (our end + the perpendicular barrier's end) is
    //                    sqrt(2)·END_T apart — ~2.8mm edge vs 1.4mm at L/4.
    float cw = BOARD_W / (float)cols;
    float ch = BOARD_H / (float)rows;
    const float END_T = 0.013f;

    // Vertical barriers (wall spans y; non-connection stored in is_barrier_h).
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols - 1; c++) {
            if (conn_h[r * cols + c])
                continue;
            float x = (c + 1) * cw;
            if (!is_barrier_h[r * cols + c]) {
                labyrinth_add_wall(env, x, r * ch, x, (r + 1) * ch);
                continue;
            }
            int perp_top = (r > 0) && (is_barrier_v[(r - 1) * cols + c] ||
                                       is_barrier_v[(r - 1) * cols + (c + 1)]);
            int perp_bot = (r < rows - 1) && (is_barrier_v[r * cols + c] ||
                                              is_barrier_v[r * cols + (c + 1)]);
            float e_top = perp_top ? END_T : 0.25f * ch;
            float e_bot = perp_bot ? END_T : 0.25f * ch;
            labyrinth_add_hole(env, x, r * ch + e_top, DEMO_HOLE_RADIUS);
            labyrinth_add_hole(env, x, (r + 1) * ch - e_bot, DEMO_HOLE_RADIUS);
        }
    }
    // Horizontal barriers (wall spans x; non-connection stored in is_barrier_v).
    for (int r = 0; r < rows - 1; r++) {
        for (int c = 0; c < cols; c++) {
            if (conn_v[r * cols + c])
                continue;
            float y = (r + 1) * ch;
            if (!is_barrier_v[r * cols + c]) {
                labyrinth_add_wall(env, c * cw, y, (c + 1) * cw, y);
                continue;
            }
            int perp_left = (c > 0) && (is_barrier_h[r * cols + (c - 1)] ||
                                        is_barrier_h[(r + 1) * cols + (c - 1)]);
            int perp_right = (c < cols - 1) && (is_barrier_h[r * cols + c] ||
                                                is_barrier_h[(r + 1) * cols + c]);
            float e_left = perp_left ? END_T : 0.25f * cw;
            float e_right = perp_right ? END_T : 0.25f * cw;
            labyrinth_add_hole(env, c * cw + e_left, y, DEMO_HOLE_RADIUS);
            labyrinth_add_hole(env, (c + 1) * cw - e_right, y, DEMO_HOLE_RADIUS);
        }
    }

    // ---- 4. Pick start + goal as random non-adjacent cells (not corners) ----
    // Corners are ram-friendly (two border walls decelerate the ball for you),
    // which lets the agent skip the "slow down and park" skill we actually
    // want it to learn. Random placement with Manhattan distance ≥ 3 forces
    // real deceleration and generalizes the policy over starting positions.
    int start_idx = 0, goal_idx = 0;
    for (int attempt = 0; attempt < 32; attempt++) {
        start_idx = (int)(lab_rng_next(&rng) % (uint32_t)N);
        goal_idx  = (int)(lab_rng_next(&rng) % (uint32_t)N);
        int sr = start_idx / cols, sc = start_idx % cols;
        int gr = goal_idx  / cols, gc = goal_idx  % cols;
        if (start_idx != goal_idx && (abs(sr - gr) + abs(sc - gc)) >= 3)
            break;
    }
    int start_row = start_idx / cols, start_col = start_idx % cols;
    int goal_row  = goal_idx  / cols, goal_col  = goal_idx  % cols;
    // Goal radius bumped to 3× ball radius (~18mm disc, vs 1.3× = 7.8mm).
    // Easier parking; lets the agent succeed from a wider arrival window.
    float goal_radius_m = 3.0f * BALL_RADIUS;
    labyrinth_set_goal(env, (goal_col + 0.5f) * cw, (goal_row + 0.5f) * ch, goal_radius_m);
    labyrinth_place_ball(env, (start_col + 0.5f) * cw, (start_row + 0.5f) * ch);

    // ---- 5. BFS along spanning tree for unique shortest path ----
    int parent[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
    int dist[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
    int queue[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
    for (int i = 0; i < N; i++) { parent[i] = -1; dist[i] = -1; }
    int qh = 0, qt = 0;
    queue[qt++] = start_idx;
    dist[start_idx] = 0;
    while (qh < qt) {
        int idx = queue[qh++];
        int r = idx / cols, c = idx % cols;
        int candidates[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int k = 0; k < 4; k++) {
            int dr = candidates[k][0], dc = candidates[k][1];
            int nr = r + dr, nc = c + dc;
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols)
                continue;
            int connected =
                (dr == -1) ? conn_v[(r - 1) * cols + c] :
                (dr == 1)  ? conn_v[r * cols + c] :
                (dc == -1) ? conn_h[r * cols + (c - 1)] :
                             conn_h[r * cols + c];
            int nidx = nr * cols + nc;
            if (connected && dist[nidx] == -1) {
                dist[nidx] = dist[idx] + 1;
                parent[nidx] = idx;
                queue[qt++] = nidx;
            }
        }
    }

    // ---- 6. Extract path (goal → start → reverse) and store in env ----
    int rev_path[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
    int rev_len = 0;
    for (int idx = goal_idx; idx != -1 && rev_len < LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM;
         idx = parent[idx]) {
        rev_path[rev_len++] = idx;
        if (idx == start_idx)
            break;
    }
    env->num_path_points = 0;
    for (int k = rev_len - 1; k >= 0 && env->num_path_points < MAX_PATH_POINTS; k--) {
        int idx = rev_path[k];
        int pr = idx / cols, pc = idx % cols;
        env->path_points[env->num_path_points][0] = (pc + 0.5f) * cw;
        env->path_points[env->num_path_points][1] = (pr + 0.5f) * ch;
        env->num_path_points++;
    }

    // ---- 7. Sprinkle obstacle holes on interior path cells ----
    int path[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
    int path_len = 0;
    for (int idx = parent[goal_idx]; idx >= 0 && idx != start_idx; idx = parent[idx]) {
        path[path_len++] = idx;
    }
    if (num_obstacle_holes > path_len) num_obstacle_holes = path_len;
    for (int i = 0; i < num_obstacle_holes; i++) {
        // Partial Fisher-Yates: pick a random path cell we haven't used yet.
        int j = i + (int)(lab_rng_next(&rng) % (uint32_t)(path_len - i));
        int tmp = path[i]; path[i] = path[j]; path[j] = tmp;
        int idx = path[i];
        int r = idx / cols, c = idx % cols;
        labyrinth_add_hole(env, (c + 0.5f) * cw, (r + 0.5f) * ch, DEMO_HOLE_RADIUS);
    }
}

// Default random maze: 6x7 cells (cw≈0.043, ch=0.04) — tighter Brio-style
// corridors. Half the walls become 2-hole barriers. No extra obstacles.
static inline void labyrinth_load_random_maze(Labyrinth* env, uint32_t seed) {
    labyrinth_generate_grid_maze(env, seed, 6, 7, 0.5f, 0);
}

// ===== Path / hole observation helpers =====
// Fill `len_from[i]` with the path length from path_points[i] to the goal.
// Sets `*total` = len_from[0]. If path has < 2 points, all values are 0.
static inline void labyrinth_path_lengths(const Labyrinth* env,
        float len_from[MAX_PATH_POINTS], float* total) {
    int n = env->num_path_points;
    for (int i = 0; i < MAX_PATH_POINTS; i++) len_from[i] = 0.0f;
    if (n < 2) { *total = 0.0f; return; }
    len_from[n - 1] = 0.0f;
    for (int i = n - 2; i >= 0; i--) {
        float dx = env->path_points[i + 1][0] - env->path_points[i][0];
        float dy = env->path_points[i + 1][1] - env->path_points[i][1];
        len_from[i] = len_from[i + 1] + sqrtf(dx * dx + dy * dy);
    }
    *total = len_from[0];
}

// Closest point on the path polyline to (px, py). Returns segment index and
// parametric t [0,1] along that segment. Falls back to goal coords when path
// has < 2 points.
static inline void labyrinth_nearest_path_point(const Labyrinth* env,
        float px, float py, float* out_x, float* out_y, int* seg, float* t_out) {
    if (env->num_path_points < 2) {
        *out_x = env->goal_cx; *out_y = env->goal_cy;
        *seg = 0; *t_out = 0.0f;
        return;
    }
    float best_d2 = 1e30f;
    float best_x = env->path_points[0][0], best_y = env->path_points[0][1];
    int   best_i = 0;
    float best_t = 0.0f;
    for (int i = 0; i + 1 < env->num_path_points; i++) {
        float ax = env->path_points[i][0], ay = env->path_points[i][1];
        float bx = env->path_points[i + 1][0], by = env->path_points[i + 1][1];
        float qx, qy;
        closest_point_on_segment(ax, ay, bx, by, px, py, &qx, &qy);
        float dx = px - qx, dy = py - qy;
        float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best_x = qx; best_y = qy;
            best_i = i;
            float seg_dx = bx - ax, seg_dy = by - ay;
            float seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy;
            if (seg_len2 < 1e-12f) {
                best_t = 0.0f;
            } else {
                best_t = ((qx - ax) * seg_dx + (qy - ay) * seg_dy) / seg_len2;
                if (best_t < 0.0f) best_t = 0.0f;
                if (best_t > 1.0f) best_t = 1.0f;
            }
        }
    }
    *out_x = best_x; *out_y = best_y;
    *seg = best_i; *t_out = best_t;
}

// Unit tangent of the path at the given segment (pointing toward goal).
static inline void labyrinth_path_tangent(const Labyrinth* env, int seg,
        float* tx, float* ty) {
    if (seg < 0 || seg + 1 >= env->num_path_points) { *tx = 0.0f; *ty = 0.0f; return; }
    float dx = env->path_points[seg + 1][0] - env->path_points[seg][0];
    float dy = env->path_points[seg + 1][1] - env->path_points[seg][1];
    float n = sqrtf(dx * dx + dy * dy);
    if (n < 1e-9f) { *tx = 0.0f; *ty = 0.0f; return; }
    *tx = dx / n; *ty = dy / n;
}

// Nearest hole to (px, py). Sets dx/dy as offset (hole_center - ball) and dist
// as the euclidean distance. With no holes, returns dx=dy=0, dist=board_diag.
static inline void labyrinth_nearest_hole(const Labyrinth* env, float px, float py,
        float* out_dx, float* out_dy, float* out_dist) {
    float diag = sqrtf(BOARD_W * BOARD_W + BOARD_H * BOARD_H);
    if (env->num_holes == 0) {
        *out_dx = 0.0f; *out_dy = 0.0f; *out_dist = diag;
        return;
    }
    float best_d2 = 1e30f;
    float best_dx = 0.0f, best_dy = 0.0f;
    for (int i = 0; i < env->num_holes; i++) {
        float dx = env->holes[i].cx - px;
        float dy = env->holes[i].cy - py;
        float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; best_dx = dx; best_dy = dy; }
    }
    *out_dx = best_dx; *out_dy = best_dy; *out_dist = sqrtf(best_d2);
}

// Curriculum maze: smaller 4×5 layout, only barrier_prob ramps with difficulty.
// Two-phase curriculum:
//   d ∈ [0, 1] → spawn-position curriculum, barrier_prob fixed at 0.5,
//                non-path cells empty. Ball spawns closer to goal at low d.
//   d ∈ [1, 2] → spawn at original start. Non-path cells become obstacles:
//                a hole is placed at the center of each non-BFS-path cell
//                with probability (d - 1). At d=2, every off-path cell has
//                a hole. The agent must follow the path strictly — anywhere
//                else is deadly.
static inline void labyrinth_load_curriculum_maze(Labyrinth* env, uint32_t seed, float difficulty) {
    if (difficulty < 0.0f) difficulty = 0.0f;
    if (difficulty > 2.0f) difficulty = 2.0f;
    const int rows = 4, cols = 5;
    labyrinth_generate_grid_maze(env, seed, rows, cols, 0.5f, 0);

    // Phase 2: fill non-path cells with center-holes proportional to (d - 1).
    if (difficulty > 1.0f) {
        float fill_frac = difficulty - 1.0f;  // 0..1
        float cw = BOARD_W / (float)cols;
        float ch = BOARD_H / (float)rows;
        // Mark cells that are on the BFS solution path (we want them empty).
        char on_path[LAB_MAZE_MAX_DIM * LAB_MAZE_MAX_DIM];
        memset(on_path, 0, sizeof(on_path));
        for (int i = 0; i < env->num_path_points; i++) {
            int c = (int)(env->path_points[i][0] / cw);
            int r = (int)(env->path_points[i][1] / ch);
            if (c >= 0 && c < cols && r >= 0 && r < rows)
                on_path[r * cols + c] = 1;
        }
        uint32_t rng = (seed ^ 0xDEADBEEFu) | 1u;
        // 16-bit precision for the fill probability — avoids float→uint32
        // overflow at fill_frac=1.0.
        uint32_t threshold = (uint32_t)(fill_frac * 65536.0f);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (on_path[r * cols + c]) continue;
                if ((lab_rng_next(&rng) & 0xFFFFu) < threshold) {
                    // 2 holes per filled cell, vertically aligned at column
                    // center, spaced like a barrier (1/4 from each cell edge).
                    float cx = (c + 0.5f) * cw;
                    float cy_top = r * ch + 0.25f * ch;
                    float cy_bot = (r + 1) * ch - 0.25f * ch;
                    labyrinth_add_hole(env, cx, cy_top, DEMO_HOLE_RADIUS);
                    labyrinth_add_hole(env, cx, cy_bot, DEMO_HOLE_RADIUS);
                }
            }
        }
    }

    // Phase 1: spawn-position curriculum. For d>=1, spawn at original start.
    float spawn_d = (difficulty < 1.0f) ? difficulty : 1.0f;
    if (env->num_path_points >= 2) {
        int last_idx = env->num_path_points - 2;  // one step before the goal cell
        if (last_idx < 0) last_idx = 0;
        int spawn_idx = (int)((1.0f - spawn_d) * (float)last_idx + 0.5f);
        if (spawn_idx < 0) spawn_idx = 0;
        if (spawn_idx > last_idx) spawn_idx = last_idx;
        labyrinth_place_ball(env, env->path_points[spawn_idx][0], env->path_points[spawn_idx][1]);
    }
}

#endif
