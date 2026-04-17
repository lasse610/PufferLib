#include "labyrinth.h"

#define OBS_SIZE LABYRINTH_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {LABYRINTH_NUM_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define Env LabyrinthEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->seed = (uint32_t)dict_get(kwargs, "seed")->value;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    env->physics_substeps = (int)dict_get(kwargs, "physics_substeps")->value;
    env->difficulty_start = (float)dict_get(kwargs, "difficulty_start")->value;
    env->curriculum_episodes = (int)dict_get(kwargs, "curriculum_episodes")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "reached_goal", log->reached_goal);
    dict_set(out, "fell_in_hole", log->fell_in_hole);
    dict_set(out, "n", log->n);
}
