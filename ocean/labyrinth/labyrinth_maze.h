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

    // ---- 4. Pick start + goal corners (diagonally opposite) ----
    // Randomizing adds variety so an RL agent doesn't just memorize a single
    // "always go down-right" policy.
    int corners[4][2] = {{0, 0}, {0, cols - 1}, {rows - 1, 0}, {rows - 1, cols - 1}};
    int start_corner = (int)(lab_rng_next(&rng) % 4);
    int goal_corner = 3 - start_corner; // diagonally opposite in this indexing
    int start_row = corners[start_corner][0], start_col = corners[start_corner][1];
    int goal_row = corners[goal_corner][0], goal_col = corners[goal_corner][1];
    int start_idx = start_row * cols + start_col;
    int goal_idx = goal_row * cols + goal_col;
    float goal_radius_m = 0.3f * (cw < ch ? cw : ch);
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

#endif
