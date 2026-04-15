/* Standalone playable Labyrinth for physics iteration (3D view).
 * Compile: ./build.sh labyrinth --fast
 * Run:     ./labyrinth
 *
 * Controls:
 *   Arrow keys: tilt the board
 *   Shift:      fine-tilt
 *   R:          reset
 *   H:          toggle HUD
 *   ESC:        quit
 */

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "labyrinth.h"

// ---- World → render scaling ----
//
// Everything in the physics is in meters. The board is 0.30m x 0.24m.
// In 3D, we place the board centered at the origin on the XZ plane:
//   board surface normal = +Y when untilted
//   ball_x  → world X  (left-right as player sees it)
//   ball_y  → world Z  (forward-back as player sees it)
// Tilt rotates the board around the board-local X (pitch) and Z (roll).

#define BOARD_THICKNESS 0.008f // visible board slab thickness, m
#define BOARD_CENTER_Y 0.06f   // lift board above the ground grid so tilts don't clip
// Visible wall thickness is locked to the physics constant from the header
// (physics uses ball center + wall half-thickness for collision)
#define WALL_THICKNESS WALL_THICKNESS_M
#define WALL_HEIGHT 0.012f // visible wall height above board surface
#define PI_F 3.14159265358979323846f

static Vector3 ball_world_position(const Labyrinth* env) {
    // Board local coordinates of the ball (centered on board midpoint).
    // labyrinth.h uses (ball_x, ball_y) ∈ [0, BOARD_W] x [0, BOARD_H],
    // where ball_y maps to world Z. Centering gives us an origin at the board center.
    Vector3 local = {
        env->ball_x - 0.5f * BOARD_W,
        0.5f * BOARD_THICKNESS + BALL_RADIUS, // sit on top of slab
        env->ball_y - 0.5f * BOARD_H,
    };
    Matrix tilt = MatrixRotateXYZ((Vector3){env->tilt_y, 0.0f, -env->tilt_x});
    Vector3 rotated = Vector3Transform(local, tilt);
    rotated.y += BOARD_CENTER_Y;
    return rotated;
}

static Matrix board_tilt_matrix(const Labyrinth* env) {
    // Pitch (tilt_y) rotates around world X.
    // Roll  (tilt_x) rotates around world Z (negated: positive tilt_x → right edge down).
    return MatrixRotateXYZ((Vector3){env->tilt_y, 0.0f, -env->tilt_x});
}

// Draw a wall (board-local coordinates) as a capsule on top of the slab:
// a rectangular body along the segment + half-cylinders at each endpoint.
// This matches the physics collision shape (line segment expanded by COLLISION_RADIUS).
// Must be called inside the board's rlPushMatrix block so the wall inherits the tilt.
static void draw_wall_local(const Wall* w) {
    float cx = 0.5f * (w->x1 + w->x2) - 0.5f * BOARD_W;
    float cz = 0.5f * (w->y1 + w->y2) - 0.5f * BOARD_H;
    float dx = w->x2 - w->x1;
    float dz = w->y2 - w->y1;
    float length = sqrtf(dx * dx + dz * dz);
    float angle = atan2f(dz, dx); // rotation around Y axis

    float cy = 0.5f * BOARD_THICKNESS + 0.5f * WALL_HEIGHT; // sit on top of slab
    float cap_radius = 0.5f * WALL_THICKNESS;
    Color wood = (Color){180, 140, 80, 255};

    rlPushMatrix();
    rlTranslatef(cx, cy, cz);
    rlRotatef(-angle * 180.0f / PI_F, 0.0f, 1.0f, 0.0f);

    // Rectangular body
    DrawCubeV((Vector3){0, 0, 0}, (Vector3){length, WALL_HEIGHT, WALL_THICKNESS}, wood);

    // End caps (full cylinders standing vertically at each endpoint).
    // DrawCylinder takes base position, top/bottom radius, height, slices.
    // We shift them by +/- length/2 along local X so they sit at the rectangle's short ends.
    float half_len = 0.5f * length;
    DrawCylinder((Vector3){+half_len, -0.5f * WALL_HEIGHT, 0.0f}, cap_radius, cap_radius,
                 WALL_HEIGHT, 12, wood);
    DrawCylinder((Vector3){-half_len, -0.5f * WALL_HEIGHT, 0.0f}, cap_radius, cap_radius,
                 WALL_HEIGHT, 12, wood);

    rlPopMatrix();
}

// Rotate a body-frame point by the ball's orientation quaternion, giving a world-frame
// offset we then add to the ball's world position.
static Vector3 rotate_by_quat(float qw, float qx, float qy, float qz, Vector3 v) {
    // v' = q * (0, v) * q^-1, using the direct formula
    // (sandwich product for pure-vector quaternion)
    float tx = 2.0f * (qy * v.z - qz * v.y);
    float ty = 2.0f * (qz * v.x - qx * v.z);
    float tz = 2.0f * (qx * v.y - qy * v.x);
    Vector3 r;
    r.x = v.x + qw * tx + (qy * tz - qz * ty);
    r.y = v.y + qw * ty + (qz * tx - qx * tz);
    r.z = v.z + qw * tz + (qx * ty - qy * tx);
    return r;
}

// Draw the ball as a white sphere with flat colored markers on its surface so
// rolling is visible. Each marker is a filled triangle fan lying tangent to
// the sphere (normal pointing outward).
static void draw_ball(const Labyrinth* env) {
    Vector3 ball_pos = ball_world_position(env);
    DrawSphere(ball_pos, BALL_RADIUS, (Color){230, 230, 230, 255});
    DrawSphereWires(ball_pos, BALL_RADIUS, 8, 8, (Color){180, 180, 180, 128});

    float marker_radius = BALL_RADIUS * 0.35f;
    float surface_offset = BALL_RADIUS * 1.001f; // a sliver above to avoid z-fighting
    Vector3 body_axes[6] = {
        {+1, 0, 0}, {-1, 0, 0}, {0, +1, 0}, {0, -1, 0}, {0, 0, +1}, {0, 0, -1},
    };
    Color marker_colors[6] = {
        {230, 70, 70, 255},  {130, 30, 30, 255}, // +X, -X (red)
        {70, 230, 70, 255},  {30, 130, 30, 255}, // +Y, -Y (green)
        {70, 120, 230, 255}, {30, 60, 130, 255}, // +Z, -Z (blue)
    };
    float qw = env->orient_w, qx = env->orient_x, qy = env->orient_y, qz = env->orient_z;

    for (int i = 0; i < 6; i++) {
        // Marker center and outward normal in world frame.
        Vector3 body_unit = body_axes[i];
        Vector3 n = rotate_by_quat(qw, qx, qy, qz, body_unit);
        Vector3 center = {ball_pos.x + n.x * surface_offset, ball_pos.y + n.y * surface_offset,
                          ball_pos.z + n.z * surface_offset};

        // Build an orthonormal basis (u, v) spanning the tangent plane at the marker.
        // Pick any vector not parallel to n, then Gram–Schmidt.
        Vector3 seed = (fabsf(n.y) < 0.9f) ? (Vector3){0, 1, 0} : (Vector3){1, 0, 0};
        Vector3 u = {seed.x - n.x * (n.x * seed.x + n.y * seed.y + n.z * seed.z),
                     seed.y - n.y * (n.x * seed.x + n.y * seed.y + n.z * seed.z),
                     seed.z - n.z * (n.x * seed.x + n.y * seed.y + n.z * seed.z)};
        float u_len = sqrtf(u.x * u.x + u.y * u.y + u.z * u.z);
        u.x /= u_len;
        u.y /= u_len;
        u.z /= u_len;
        Vector3 v = {n.y * u.z - n.z * u.y, n.z * u.x - n.x * u.z, n.x * u.y - n.y * u.x};

        // Fan of triangles around the center.
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

static void draw_board(const Labyrinth* env) {
    Matrix tilt = board_tilt_matrix(env);

    rlPushMatrix();
    // Translate up first (world space), then rotate
    rlTranslatef(0.0f, BOARD_CENTER_Y, 0.0f);
    // Raylib rlMultMatrixf wants column-major 16 floats. Matrix is row-major.
    float m[16] = {tilt.m0, tilt.m1, tilt.m2,  tilt.m3,  tilt.m4,  tilt.m5,  tilt.m6,  tilt.m7,
                   tilt.m8, tilt.m9, tilt.m10, tilt.m11, tilt.m12, tilt.m13, tilt.m14, tilt.m15};
    rlMultMatrixf(m);

    // Slab
    DrawCubeV((Vector3){0, 0, 0}, (Vector3){BOARD_W, BOARD_THICKNESS, BOARD_H},
              (Color){60, 70, 80, 255});
    DrawCubeWiresV((Vector3){0, 0, 0}, (Vector3){BOARD_W, BOARD_THICKNESS, BOARD_H},
                   (Color){0, 187, 187, 255});

    // Border frame: the first 4 walls are the board-edge boundary walls.
    // Render them as thin boxes pushed OUTWARD so the visible inner face
    // aligns with the physics line (otherwise the ball sinks into half the box).
    float cy_wall = 0.5f * BOARD_THICKNESS + 0.5f * WALL_HEIGHT;
    float half_t = 0.5f * WALL_THICKNESS;
    // Bottom edge (y = 0)
    DrawCubeV((Vector3){0, cy_wall, -0.5f * BOARD_H - half_t},
              (Vector3){BOARD_W + WALL_THICKNESS * 2, WALL_HEIGHT, WALL_THICKNESS},
              (Color){180, 140, 80, 255});
    // Top edge (y = BOARD_H)
    DrawCubeV((Vector3){0, cy_wall, 0.5f * BOARD_H + half_t},
              (Vector3){BOARD_W + WALL_THICKNESS * 2, WALL_HEIGHT, WALL_THICKNESS},
              (Color){180, 140, 80, 255});
    // Left edge (x = 0)
    DrawCubeV((Vector3){-0.5f * BOARD_W - half_t, cy_wall, 0},
              (Vector3){WALL_THICKNESS, WALL_HEIGHT, BOARD_H}, (Color){180, 140, 80, 255});
    // Right edge (x = BOARD_W)
    DrawCubeV((Vector3){0.5f * BOARD_W + half_t, cy_wall, 0},
              (Vector3){WALL_THICKNESS, WALL_HEIGHT, BOARD_H}, (Color){180, 140, 80, 255});

    // Interior walls (double-sided — ball can hit either face) stay centered on physics lines.
    for (int i = 4; i < env->num_walls; i++) {
        draw_wall_local(&env->walls[i]);
    }

    rlPopMatrix();
}

int main(void) {
    const int window_w = 1024;
    const int window_h = 720;
    InitWindow(window_w, window_h, "PufferLib Labyrinth (3D)");
    SetTargetFPS(60);

    // Camera: mostly top-down with a small forward lean so depth is still readable
    Camera3D camera = {
        .position = (Vector3){0.0f, 0.50f, 0.10f},
        .target = (Vector3){0.0f, BOARD_CENTER_Y, 0.0f},
        .up = (Vector3){0.0f, 1.0f, 0.0f},
        .fovy = 40.0f,
        .projection = CAMERA_PERSPECTIVE,
    };

    Labyrinth env = {0};
    labyrinth_reset(&env);
    labyrinth_add_demo_walls(&env);

    const float tilt_rate = 0.5f;
    bool show_hud = true;

    // Physics rate measurement
    int physics_steps_in_window = 0;
    float window_elapsed = 0.0f;
    float physics_sps = 0.0f;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        float rate = tilt_rate;
        if (IsKeyDown(KEY_LEFT_SHIFT))
            rate *= 0.3f;

        if (IsKeyDown(KEY_LEFT))
            env.tilt_x -= rate * dt;
        if (IsKeyDown(KEY_RIGHT))
            env.tilt_x += rate * dt;
        if (IsKeyDown(KEY_UP))
            env.tilt_y -= rate * dt;
        if (IsKeyDown(KEY_DOWN))
            env.tilt_y += rate * dt;
        if (IsKeyPressed(KEY_R)) {
            labyrinth_reset(&env);
            labyrinth_add_demo_walls(&env);
        }
        if (IsKeyPressed(KEY_H))
            show_hud = !show_hud;

        if (env.tilt_x > MAX_TILT_RAD)
            env.tilt_x = MAX_TILT_RAD;
        if (env.tilt_x < -MAX_TILT_RAD)
            env.tilt_x = -MAX_TILT_RAD;
        if (env.tilt_y > MAX_TILT_RAD)
            env.tilt_y = MAX_TILT_RAD;
        if (env.tilt_y < -MAX_TILT_RAD)
            env.tilt_y = -MAX_TILT_RAD;

        // Physics at 200 Hz; render at 60 Hz.
        int substeps = (int)(dt / PHYSICS_DT + 0.5f);
        if (substeps < 1)
            substeps = 1;
        if (substeps > 20)
            substeps = 20; // safety cap
        for (int i = 0; i < substeps; i++)
            labyrinth_step(&env);
        physics_steps_in_window += substeps;
        window_elapsed += dt;
        if (window_elapsed >= 1.0f) {
            physics_sps = physics_steps_in_window / window_elapsed;
            physics_steps_in_window = 0;
            window_elapsed = 0.0f;
        }

        BeginDrawing();
        ClearBackground((Color){6, 24, 24, 255});

        BeginMode3D(camera);
        // Ground grid for spatial reference (10 slices, 0.1m spacing = 1m grid)
        DrawGrid(10, 0.1f);

        draw_board(&env);

        draw_ball(&env);
        EndMode3D();

        if (show_hud) {
            DrawText("Labyrinth 3D  —  arrows tilt, shift=fine, R=reset, H=hud", 10, 10, 14,
                     (Color){0, 187, 187, 255});
            DrawText(TextFormat("tilt  x=%+.3f rad (%+.1fdeg)  y=%+.3f rad (%+.1fdeg)", env.tilt_x,
                                env.tilt_x * 180.0f / PI_F, env.tilt_y, env.tilt_y * 180.0f / PI_F),
                     10, 30, 14, WHITE);
            DrawText(TextFormat("ball  pos=(%+.3f, %+.3f)  vel=(%+.3f, %+.3f)",
                                env.ball_x - 0.5f * BOARD_W, env.ball_y - 0.5f * BOARD_H,
                                env.ball_vx, env.ball_vy),
                     10, 46, 14, WHITE);
            DrawText(
                TextFormat("tick=%d  dt=%.4f  substeps/frame=%d  physics=%.0f Hz  (target=%.0f Hz)",
                           env.tick, PHYSICS_DT, substeps, physics_sps, 1.0f / PHYSICS_DT),
                10, 62, 14, WHITE);
            DrawFPS(window_w - 90, 10);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
