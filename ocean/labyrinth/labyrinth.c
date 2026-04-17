/* Standalone playable Labyrinth (3D view).
 * Compile: ./build.sh labyrinth --fast
 * Run:     ./labyrinth
 *
 * Controls:
 *   Arrow keys: tilt the board
 *   Shift:      fine-tilt
 *   R:          reset current maze
 *   N:          advance to next seed (new maze)
 *   , / .:      decrease / increase curriculum difficulty (reloads maze)
 *   H:          toggle HUD
 *   ESC:        quit
 *
 * This file uses the Labyrinth physics + render helpers from labyrinth.h
 * directly (bypassing the RL env's c_render) so we can own the full frame —
 * BeginDrawing → 3D scene + HUD → EndDrawing — in a single pass.
 */

#include "labyrinth.h"

int main(void) {
    const int window_w = 1024;
    const int window_h = 720;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(window_w, window_h, "PufferLib Labyrinth (3D)");
    SetTargetFPS(60);

    Camera3D camera = {
        .position = (Vector3){0.0f, 0.50f, 0.10f},
        .target = (Vector3){0.0f, BOARD_CENTER_Y, 0.0f},
        .up = (Vector3){0.0f, 1.0f, 0.0f},
        .fovy = 40.0f,
        .projection = CAMERA_PERSPECTIVE,
    };

    Labyrinth phys = {0};
    uint32_t seed = 67890;
    float difficulty = 1.0f;
    labyrinth_reset(&phys);
    labyrinth_load_curriculum_maze(&phys, seed, difficulty);

    const float tilt_rate = 0.5f;
    bool show_hud = true;
    int physics_steps_in_window = 0;
    float window_elapsed = 0.0f;
    float physics_sps = 0.0f;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        float rate = tilt_rate;
        if (IsKeyDown(KEY_LEFT_SHIFT))
            rate *= 0.3f;

        if (IsKeyDown(KEY_LEFT))  phys.tilt_x -= rate * dt;
        if (IsKeyDown(KEY_RIGHT)) phys.tilt_x += rate * dt;
        if (IsKeyDown(KEY_UP))    phys.tilt_y -= rate * dt;
        if (IsKeyDown(KEY_DOWN))  phys.tilt_y += rate * dt;
        if (IsKeyPressed(KEY_R)) {
            labyrinth_reset(&phys);
            labyrinth_load_curriculum_maze(&phys, seed, difficulty);
        }
        if (IsKeyPressed(KEY_N)) {
            seed++;
            labyrinth_reset(&phys);
            labyrinth_load_curriculum_maze(&phys, seed, difficulty);
        }
        if (IsKeyPressed(KEY_COMMA)) {
            difficulty -= 0.25f;
            if (difficulty < 0.0f) difficulty = 0.0f;
            labyrinth_reset(&phys);
            labyrinth_load_curriculum_maze(&phys, seed, difficulty);
        }
        if (IsKeyPressed(KEY_PERIOD)) {
            difficulty += 0.25f;
            if (difficulty > 1.0f) difficulty = 1.0f;
            labyrinth_reset(&phys);
            labyrinth_load_curriculum_maze(&phys, seed, difficulty);
        }
        if (IsKeyPressed(KEY_H))
            show_hud = !show_hud;

        if (phys.tilt_x > MAX_TILT_RAD)  phys.tilt_x = MAX_TILT_RAD;
        if (phys.tilt_x < -MAX_TILT_RAD) phys.tilt_x = -MAX_TILT_RAD;
        if (phys.tilt_y > MAX_TILT_RAD)  phys.tilt_y = MAX_TILT_RAD;
        if (phys.tilt_y < -MAX_TILT_RAD) phys.tilt_y = -MAX_TILT_RAD;

        int substeps = (int)(dt / PHYSICS_DT + 0.5f);
        if (substeps < 1)  substeps = 1;
        if (substeps > 20) substeps = 20;
        for (int i = 0; i < substeps; i++)
            labyrinth_step(&phys);
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
        DrawGrid(10, 0.1f);
        draw_board(&phys);
        draw_ball(&phys);
        EndMode3D();

        if (show_hud) {
            DrawText("Labyrinth 3D  —  arrows tilt, shift=fine, R=reset, N=new maze, , .=difficulty, H=hud", 10,
                     10, 14, (Color){0, 187, 187, 255});
            DrawText(TextFormat("difficulty=%.2f  (seed=%u)", difficulty, seed),
                     window_w - 280, 30, 14, (Color){255, 220, 90, 255});
            DrawText(TextFormat("tilt  x=%+.3f rad (%+.1fdeg)  y=%+.3f rad (%+.1fdeg)",
                                phys.tilt_x, phys.tilt_x * 180.0f / PI_F,
                                phys.tilt_y, phys.tilt_y * 180.0f / PI_F),
                     10, 30, 14, WHITE);
            DrawText(TextFormat("ball  pos=(%+.3f, %+.3f)  vel=(%+.3f, %+.3f)",
                                phys.ball_x - 0.5f * BOARD_W,
                                phys.ball_y - 0.5f * BOARD_H,
                                phys.ball_vx, phys.ball_vy),
                     10, 46, 14, WHITE);
            DrawText(
                TextFormat("tick=%d  dt=%.4f  substeps/frame=%d  physics=%.0f Hz  (target=%.0f Hz)",
                           phys.tick, PHYSICS_DT, substeps, physics_sps, 1.0f / PHYSICS_DT),
                10, 62, 14, WHITE);
            if (phys.fell_in_hole) {
                DrawText("FELL IN HOLE — press R to reset", 10, 82, 18,
                         (Color){230, 80, 80, 255});
            } else if (phys.reached_goal) {
                DrawText("GOAL! — press R to reset", 10, 82, 18, (Color){80, 220, 110, 255});
            }
            DrawFPS(window_w - 90, 10);
        }
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
