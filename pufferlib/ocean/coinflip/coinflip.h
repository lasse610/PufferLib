#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char COIN = 2;

const int TIMEOUT = 100;
const int GRID_SIZE = 11;
const int CENTER = 5;

// Only floats! n must be last.
typedef struct {
    float score;
    float episode_length;
    float n;
} Log;

typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int agent_pos;
    int coin_pos;
    int tick;
} CoinFlip;


void c_reset(CoinFlip* env) {
    env->agent_pos = 5;
    env->tick = 0;
    memset(env->observations, 0, GRID_SIZE);
    env->observations[5] = AGENT;
    if (rand() % 2 == 0){
        env->coin_pos = 2;
    } else {
        env->coin_pos = 8;
    }
    env->observations[env->coin_pos] = COIN;
}

// TODO: Write c_step
void c_step(CoinFlip* env) {
    int action = env->actions[0];
    int new_agent_pos = env->agent_pos + ((action == 0) ? -1 : 1);
    if(new_agent_pos < 0 || new_agent_pos >= GRID_SIZE) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        env->log.score += env->rewards[0];
        env->log.episode_length += env->tick;
        env->log.n++;
        c_reset(env);
        return;
    }
    bool done = (env->coin_pos == new_agent_pos);

    if(done) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0;
        env->log.score += env->rewards[0];
        env->log.episode_length += env->tick;
        env->log.n++;
        c_reset(env);
        return;
    }

    if(env->tick >= TIMEOUT) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        env->log.score += env->rewards[0];
        env->log.episode_length += env->tick;
        env->log.n++;
        c_reset(env);
        return;
    }

    env->rewards[0] = 0.0;
    env->terminals[0] = 0;
    env->observations[env->agent_pos] = EMPTY;
    env->agent_pos = new_agent_pos;
    env->observations[new_agent_pos] = AGENT;
    env->tick++;
    
}

// TODO: Write c_render (optional, can leave empty for now)
void c_render(CoinFlip* env) {
}

// Required
void c_close(CoinFlip* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
