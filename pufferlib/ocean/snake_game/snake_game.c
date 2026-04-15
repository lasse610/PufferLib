#include "snake_game.h"

int main() {
    SnakeGame env = {
        .width = 25,
        .height = 25,
    };
    env.max_length = env.width * env.height;
    env.observations = (unsigned char*)calloc(env.width * env.height * 4, sizeof(unsigned char));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.grid = (unsigned char*)calloc(env.width * env.height, sizeof(unsigned char));
    env.body = (int*)calloc(env.max_length * 2, sizeof(int));

    c_reset(&env);
    c_render(&env);

    int action = RIGHT; // start moving right
    while (!WindowShouldClose()) {
        // Keyboard input — hold shift to play manually
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) action = UP;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) action = DOWN;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) action = LEFT;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) action = RIGHT;

        env.actions[0] = action;
        c_step(&env);
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    CloseWindow();
    return 0;
}
