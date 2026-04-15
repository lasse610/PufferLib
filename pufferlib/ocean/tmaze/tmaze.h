#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include <stdbool.h> 

const unsigned char LEFT = 1;
const unsigned char FORWARD = 0;
const unsigned char RIGHT = 2;

const float LEFT_SIGNAL = -1;
const float RIGHT_SIGNAL = 1;
const int TIMEOUT = 100;


const int OBSERVATION_SIZE = 3;

// Only floats! n must be last.
typedef struct {
    float score;
    float episode_length;
    float n;
} Log;

typedef struct {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int agent_pos;
    float signal;
    int junction_pos;
    int tick;
} TMaze;


void c_reset(TMaze* env) {
    env->agent_pos = 0;
    env->tick = 0;
    memset(env->observations, 0, OBSERVATION_SIZE * sizeof(float));
    if (rand() % 2 == 0){
        env->signal = LEFT_SIGNAL;
    } else {
        env->signal = RIGHT_SIGNAL;
    }
    env->junction_pos = (5 + rand() % 11);
    env->observations[0] = env->signal;
}

// TODO: Write c_step
void c_step(TMaze* env) {
    int action = env->actions[0];

    if(env->tick > TIMEOUT){
        env->rewards[0] = 0;
        env->terminals[0] = 1;
        env->log.episode_length += env->tick;
        env->log.score += 0;
        env->log.n++;
        c_reset(env);
        return;
    }

    // Not at junction: always move forward regardless of action
    if(env->agent_pos != env->junction_pos){
        env->agent_pos += 1;
        env->observations[0] = 0;
        env->observations[1] = env->agent_pos;
        env->rewards[0] = (action != FORWARD) ? -0.1 : 0;
        env->tick++;
        env->terminals[0] = 0;
        // Show at_junction if we just arrived
        env->observations[2] = (env->agent_pos == env->junction_pos) ? 1 : 0;
        return;
    }

    // At junction: must choose left or right
    if(action == FORWARD){
        // Wasting time at junction
        env->observations[0] = 0;
        env->observations[1] = env->agent_pos;
        env->observations[2] = 1;
        env->rewards[0] = -0.1;
        env->tick++;
        env->terminals[0] = 0;
        return;
    }

    // At junction, chose left or right
    bool is_correct = (action == RIGHT && env->signal == RIGHT_SIGNAL) ||
                      (action == LEFT && env->signal == LEFT_SIGNAL);

    env->terminals[0] = 1;
    if(is_correct){
        env->rewards[0] = 1;
        env->log.score += 1;
    } else {
        env->rewards[0] = -1;
        env->log.score += 0;
    }
    env->log.episode_length += env->tick;
    env->log.n++;
    c_reset(env);
    
}

// TODO: Write c_render (optional, can leave empty for now)
void c_render(TMaze* env) {
}

// Required
void c_close(TMaze* env) {
    
}
