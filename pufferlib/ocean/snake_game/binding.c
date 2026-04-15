#include "snake_game.h"

#define Env SnakeGame
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->max_length = env->width * env->height;
    env->start_length = unpack(kwargs, "start_length");
    env->vision = unpack(kwargs, "vision");
    env->grid = (unsigned char*)calloc(env->width * env->height, sizeof(unsigned char));
    env->body = (int*)calloc(env->max_length * 2, sizeof(int));
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "snake_length", log->snake_length);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "food_eaten", log->food_eaten);
    assign_to_dict(dict, "episode_return", log->episode_return);
    return 0;
}
