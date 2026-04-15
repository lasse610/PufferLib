#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const int EMPTY = 0;
const int SNAKE_HEAD = 1;
const int SNAKE_BODY = 2;
const int FOOD = 3;
const int WALL = 4;

const int UP = 0;
const int DOWN = 1;
const int LEFT = 2;
const int RIGHT = 3;

const int GROW = 1;
const int DIE = 2;
const int MOVE = 3;

typedef struct {
    float snake_length;
    float episode_length;
    float food_eaten;
    float episode_return;
    float n;
} Log;

typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int width;
    int height;
    int max_length;
    int head_ptr;
    int snake_length;
    int* body;
    unsigned char* grid;
    int food_eaten;
    float episode_return;
    int tick;
    int steps_since_food;
    int start_length;  // 0 = random, >0 = fixed
    int vision;        // observation radius around head
} SnakeGame;

void handle_grow(SnakeGame* env, int new_head_r, int new_head_c);
void handle_move(SnakeGame* env, int new_head_r, int new_head_c);
void handle_death(SnakeGame* env);


static void grid_set(SnakeGame* env, int row, int col, unsigned char value){
    env->grid[row * env->width + col] = value;
}

static unsigned char grid_get(SnakeGame* env, int row, int col){
    return env->grid[row * env->width + col];
}


void compute_observations(SnakeGame* env){
    int vision = env->vision;
    int window = 2 * vision + 1;
    int obs_size = window * window * 4;
    memset(env->observations, 0, obs_size * sizeof(unsigned char));

    // Find head position
    int head_r = env->body[2 * env->head_ptr];
    int head_c = env->body[2 * env->head_ptr + 1];

    for (int dr = -vision; dr <= vision; dr++) {
        for (int dc = -vision; dc <= vision; dc++) {
            int grid_r = head_r + dr;
            int grid_c = head_c + dc;
            int obs_r = dr + vision;  // local coords: 0 to window-1
            int obs_c = dc + vision;
            int obs_idx = (obs_r * window + obs_c) * 4;

            unsigned char tile;
            if (grid_r < 0 || grid_r >= env->height || grid_c < 0 || grid_c >= env->width) {
                tile = WALL;  // out of bounds = wall
            } else {
                tile = grid_get(env, grid_r, grid_c);
            }

            if (tile != EMPTY) {
                int channel = tile - 1;  // SNAKE_HEAD=1->ch0, BODY=2->ch1, FOOD=3->ch2, WALL=4->ch3
                env->observations[obs_idx + channel] = 1;
            }
        }
    }
}

void c_reset(SnakeGame* env) {
    int width = env->width;
    int height = env->height;
    memset(env->grid, 0, width * height * sizeof(unsigned char));
    memset(env->body, 0, env->max_length * 2 * sizeof(int));

    // Place walls
    for(int row = 0; row < height; row++){
        grid_set(env, row, 0, WALL);
        grid_set(env, row, width - 1, WALL);
    }
    for(int col = 0; col < width; col++){
        grid_set(env, 0, col, WALL);
        grid_set(env, height - 1, col, WALL);
    }

    // Random starting position (away from walls)
    int head_r = 2 + rand() % (height - 4);
    int head_c = 2 + rand() % (width - 4);

    // Random starting length 1-8 (playable area is width-2, so cap at 8)
    int start_length;
    if (env->start_length > 0) {
        start_length = env->start_length;
    } else {
        start_length = 1 + rand() % 12;
    }

    // Build snake body by walking backward from head
    // Pick a random direction for the body to trail
    int dirs_r[] = {-1, 1, 0, 0};
    int dirs_c[] = {0, 0, -1, 1};
    int dir = rand() % 4;
    int body_dr = dirs_r[dir];  // direction body extends FROM head
    int body_dc = dirs_c[dir];

    // Place head
    env->body[0] = head_r;
    env->body[1] = head_c;
    grid_set(env, head_r, head_c, SNAKE_HEAD);
    env->head_ptr = 0;
    env->snake_length = 1;

    // Grow body segments trailing behind head
    int cur_r = head_r;
    int cur_c = head_c;
    for (int i = 1; i < start_length; i++) {
        int next_r = cur_r + body_dr;
        int next_c = cur_c + body_dc;

        // If next position is blocked, try turning
        if (grid_get(env, next_r, next_c) != EMPTY) {
            // Try all 4 directions to find an empty neighbor
            int found = 0;
            for (int d = 0; d < 4; d++) {
                next_r = cur_r + dirs_r[d];
                next_c = cur_c + dirs_c[d];
                if (grid_get(env, next_r, next_c) == EMPTY) {
                    body_dr = dirs_r[d];
                    body_dc = dirs_c[d];
                    found = 1;
                    break;
                }
            }
            if (!found) break;  // boxed in, stop growing
        }

        // Place body segment — store at negative indices in ring buffer
        // We build backward: head is at ptr 0, body at ptr max_length-1, max_length-2, etc.
        int buf_idx = (env->max_length - i) % env->max_length;
        env->body[2 * buf_idx] = next_r;
        env->body[2 * buf_idx + 1] = next_c;
        grid_set(env, next_r, next_c, SNAKE_BODY);
        env->snake_length++;
        cur_r = next_r;
        cur_c = next_c;
    }

    env->tick = 0;
    env->food_eaten = 0;
    env->episode_return = 0;
    env->steps_since_food = 0;

    // Place food
    int food_r, food_c;
    do {
        food_r = rand() % height;
        food_c = rand() % width;
    } while(grid_get(env, food_r, food_c) != EMPTY);
    grid_set(env, food_r, food_c, FOOD);
    compute_observations(env);
}



void c_step(SnakeGame* env){
    if (env->steps_since_food >= 300) {
        handle_death(env);
        return;
    }

    int action = env->actions[0];
    int head_r = env->body[2 * env->head_ptr ];
    int head_c = env->body[2 * env->head_ptr + 1];

    int new_head_r, new_head_c;

    if(action == UP){
        new_head_r = head_r - 1;
        new_head_c = head_c;
    } else if (action == DOWN){
        new_head_r = head_r + 1;
        new_head_c = head_c;
    } else if(action == RIGHT){
        new_head_r = head_r;
        new_head_c = head_c + 1;
    } else if(action == LEFT){
        new_head_r = head_r;
        new_head_c = head_c - 1;
    }

    if(env->snake_length > 1){
        int neck_ptr = (env->head_ptr - 1 + env->max_length) % env->max_length;                                                                                                                                                                                                                                           
        int neck_r = env->body[2 * neck_ptr];                                                                                                                       
        int neck_c = env->body[2 * neck_ptr + 1];
        
        if(new_head_r == neck_r && new_head_c == neck_c){
            new_head_r = head_r - (new_head_r - head_r);
            new_head_c = head_c - (new_head_c - head_c);
        }
    }

    int element_at_new_pos = grid_get(env, new_head_r, new_head_c);

    
    if(element_at_new_pos == WALL || element_at_new_pos == SNAKE_BODY){
        handle_death(env);
        return;
    } else if (element_at_new_pos == FOOD){
        handle_grow(env, new_head_r, new_head_c);
        return;
    } else {
        handle_move(env, new_head_r, new_head_c);
        return;
    }
}

const int CELL_SIZE = 24;
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color COLOR_EMPTY = (Color){6, 24, 24, 255};
const Color COLOR_HEAD = (Color){0, 255, 200, 255};
const Color COLOR_BODY = (Color){0, 187, 187, 255};
const Color COLOR_FOOD = (Color){187, 0, 0, 255};
const Color COLOR_WALL = (Color){80, 80, 80, 255};

static Color tile_color(unsigned char tile) {
    switch (tile) {
        case SNAKE_HEAD: return COLOR_HEAD;
        case SNAKE_BODY: return COLOR_BODY;
        case FOOD:       return COLOR_FOOD;
        case WALL:       return COLOR_WALL;
        default:         return COLOR_EMPTY;
    }
}

void c_render(SnakeGame* env) {
    if (!IsWindowReady()) {
        InitWindow(env->width * CELL_SIZE, env->height * CELL_SIZE, "PufferLib Snake");
        SetTargetFPS(10);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    for (int row = 0; row < env->height; row++) {
        for (int col = 0; col < env->width; col++) {
            unsigned char tile = grid_get(env, row, col);
            if (tile != EMPTY) {
                DrawRectangle(col * CELL_SIZE, row * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE, tile_color(tile));
            }
        }
    }

    DrawText(TextFormat("Length: %d  Food: %d  Tick: %d",
        env->snake_length, env->food_eaten, env->tick),
        10, 10, 16, COLOR_HEAD);

    EndDrawing();
}

void c_close(SnakeGame* env){
    free(env->body);
    free(env->grid);
}

void handle_death(SnakeGame* env) {
    env->terminals[0] = 1;
    env->rewards[0] = -10;

    env->log.snake_length += env->snake_length;
    env->log.episode_length += env->tick;
    env->log.food_eaten += env->food_eaten;
    env->episode_return += env->rewards[0];
    env->log.episode_return += env->episode_return;
    env->log.n++;

    c_reset(env);
}

void handle_move(SnakeGame* env, int new_head_r, int new_head_c){
    int* body = env->body;
    int max_length = env->max_length;
    int snake_length = env -> snake_length;
    int head_ptr = env->head_ptr;
    
    // create new head
    int new_head_ptr = (head_ptr + 1) % max_length;
    body[2 * new_head_ptr] = new_head_r;
    body[2 * new_head_ptr + 1] = new_head_c;
    grid_set(env, new_head_r, new_head_c, SNAKE_HEAD);

    // mark old head body
    int old_head_ptr = head_ptr;
    int old_head_r = body[2 * old_head_ptr];
    int old_head_c = body[2 * old_head_ptr + 1];
    grid_set(env, old_head_r, old_head_c, SNAKE_BODY);
    
    // clear tail (or old head if len 1)
    int tail_ptr = (head_ptr - (snake_length - 1) + max_length) % max_length;
    int tail_r = env->body[2 * tail_ptr];
    int tail_c = env->body[2 * tail_ptr  + 1];
    grid_set(env,tail_r, tail_c, EMPTY);
    
    env->head_ptr = new_head_ptr;
    env->rewards[0] = -0.01;
    env->episode_return += env->rewards[0];
    env->terminals[0] = 0;
    env->tick++;
    env->steps_since_food++;
    compute_observations(env);

}

void handle_grow(SnakeGame* env, int new_head_r, int new_head_c){
    int* body = env->body;
    int height = env->height;
    int width = env->width;
    int max_length = env->max_length;
    int snake_length = env -> snake_length;
    int head_ptr = env->head_ptr;
    
    // create new head
    int new_head_ptr = (head_ptr + 1) % max_length;
    body[2 * new_head_ptr] = new_head_r;
    body[2 * new_head_ptr + 1] = new_head_c;
    grid_set(env, new_head_r, new_head_c, SNAKE_HEAD);

    // mark old head body
    int old_head_ptr = head_ptr;
    int old_head_r = body[2 * old_head_ptr];
    int old_head_c = body[2 * old_head_ptr + 1];
    grid_set(env, old_head_r, old_head_c, SNAKE_BODY);

    int food_r, food_c;

    do{
        food_r = rand() % height;
        food_c = rand() % width;
        
    } while(grid_get(env, food_r, food_c) != EMPTY);
    grid_set(env,food_r, food_c, FOOD);

    env->food_eaten++;
    env->snake_length++;
    env->head_ptr = new_head_ptr;
    env->steps_since_food = 0;
    env->rewards[0] = 3;
    env->terminals[0] = 0;
    env->episode_return += env->rewards[0];
    env->tick++;
    compute_observations(env);
}


