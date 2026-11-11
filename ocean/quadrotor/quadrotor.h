/* Quadrotor RL environment. */

#ifndef QUADROTOR_ENV_H
#define QUADROTOR_ENV_H

#include <stdlib.h>
#include <stdint.h>
#include "quadrotor_physics.h"
#include "quadrotor_control.h"
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define QR_OBS_SIZE 26
#define QR_ROUTE_WALL_OBS_SIZE 6
#define QR_ROUTE_EXTRA_OBS_SIZE 8
#define QR_ROUTE_PERCEPTION_OBS_SIZE 12
#define QR_ROUTE_COMPACT_PERCEPTION_OBS_SIZE 8
#define QR_ROUTE_AFFORDANCE_OBS_SIZE 12
#define QR_ROUTE_PRIVILEGED_OBS_SIZE 12
#define QR_ACTION_HISTORY_OBS_SIZE 4
#define QR_EXPLORE_LEGACY_PATCH_SIDE 4
#define QR_EXPLORE_LEGACY_PATCH_SIZE 16
#define QR_EXPLORE_WIDE_MAP_SIDE 7
#define QR_EXPLORE_WIDE_MAP_CELLS (QR_EXPLORE_WIDE_MAP_SIDE * QR_EXPLORE_WIDE_MAP_SIDE)
#define QR_EXPLORE_FAR_MAP_SIDE 11
#define QR_EXPLORE_FAR_MAP_CELLS (QR_EXPLORE_FAR_MAP_SIDE * QR_EXPLORE_FAR_MAP_SIDE)
#define QR_EXPLORE_PATCH_OBS_SIZE (QR_EXPLORE_LEGACY_PATCH_SIZE + 2 * QR_EXPLORE_WIDE_MAP_CELLS + 2 * QR_EXPLORE_FAR_MAP_CELLS)
#define QR_EXPLORE_GRAPH_CANDIDATE_COUNT 4
#define QR_EXPLORE_GRAPH_CANDIDATE_FEATURES 10
#define QR_EXPLORE_GRAPH_CANDIDATE_OBS_SIZE (QR_EXPLORE_GRAPH_CANDIDATE_COUNT * QR_EXPLORE_GRAPH_CANDIDATE_FEATURES)
#define QR_DEPTH_CAMERA_DEFAULT_WIDTH 16
#define QR_DEPTH_CAMERA_DEFAULT_HEIGHT 12
#define QR_DEPTH_CAMERA_MAX_WIDTH 20
#define QR_DEPTH_CAMERA_MAX_HEIGHT 15
#define QR_DEPTH_CAMERA_STRIDE QR_DEPTH_CAMERA_MAX_WIDTH
#define QR_DEPTH_CAMERA_OBS_SIZE (QR_DEPTH_CAMERA_STRIDE * QR_DEPTH_CAMERA_MAX_HEIGHT)
#define QR_NUM_ACTIONS 4
#define QR_MAX_RAYS 64
#define QR_MAX_OBSTACLES 32
#define QR_PI 3.14159265358979323846f
#define QR_MAX_STEPS 1000
#define QR_ACTION_REPEAT 10
#define QR_MAX_RATE 10.0f
#define QR_SPAWN_RADIUS 2.0f
#define QR_SPAWN_HEIGHT 1.5f
#define QR_INIT_VEL_RADIUS 0.0f
#define QR_INIT_OMEGA_RADIUS 0.0f
#define QR_ROOM_SIZE 0.0f
#define QR_CEILING_HEIGHT 3.0f
#define QR_RAY_LENGTH 3.0f
#define QR_MAX_VEL 1.0f
#define QR_MAX_Z_VEL 0.75f
#define QR_POS_OBS_SCALE 5.0f
#define QR_VEL_OBS_SCALE 1.0f
#define QR_MIN_OFFSET_OBS_SCALE 0.1f
#define QR_CRASH_REWARD -1.0f
#define QR_DIST_SCALE 1.0f
#define QR_CENTER_REWARD 0.5f
#define QR_CENTER_SCALE 10.0f
#define QR_VEL_SCALE 0.06f
#define QR_TILT_SCALE 0.05f
#define QR_RATE_SCALE 0.01f
#define QR_ACTION_SCALE 0.005f
#define QR_HOVER_BONUS 0.1f
#define QR_HOVER_RADIUS 0.3f
#define QR_SETTLE_RADIUS 0.05f
#define QR_SETTLE_VEL 0.05f
#define QR_SETTLE_REWARD 0.15f
#define QR_SETTLE_DIST_SCALE 100.0f
#define QR_SETTLE_VEL_SCALE 100.0f
#define QR_TERMINAL_SETTLE_BONUS 3.0f
#define QR_TERMINAL_UNSETTLED_SCALE 0.5f
#define QR_NAV_STEP_COST 0.01f
#define QR_NAV_PROXIMITY_REWARD 0.03f
#define QR_NAV_DIST_SCALE 0.03f
#define QR_NAV_CENTER_SCALE 2.0f
#define QR_NAV_PROGRESS_SCALE 80.0f
#define QR_NAV_SPEED_SCALE 0.08f
#define QR_NAV_GOAL_RADIUS 0.30f
#define QR_NAV_GOAL_BONUS 25.0f
#define QR_NAV_TIMEOUT_PENALTY 12.0f
#define QR_NAV_TIMEOUT_DIST_SCALE 3.0f
#define QR_NAV_CRASH_REWARD -10.0f
#define QR_NAV_CLEARANCE_RADIUS 0.30f
#define QR_NAV_CLEARANCE_SCALE 0.12f
#define QR_NAV_ROOM_MARGIN 0.25f
#define QR_NAV_MIN_TARGET_DIST 0.5f
#define QR_ROUTE_GOAL_COUNT 3
#define QR_ROUTE_REACH_BONUS 5.0f
#define QR_ROUTE_GOAL_BONUS 25.0f
#define QR_ROUTE_TIMEOUT_PENALTY 12.0f
#define QR_ROUTE_TIMEOUT_GOAL_SCALE 3.0f
#define QR_ROUTE_STEP_COST 0.004f
#define QR_ROUTE_PROXIMITY_REWARD 0.02f
#define QR_ROUTE_DIST_SCALE 0.01f
#define QR_ROUTE_PROGRESS_SCALE 25.0f
#define QR_ROUTE_CRASH_REWARD -24.0f
#define QR_ROUTE_CRASH_GOAL_SCALE 4.0f
#define QR_ROUTE_CRASH_DIST_SCALE 1.5f
#define QR_ROUTE_CLEARANCE_RADIUS 0.45f
#define QR_ROUTE_CLEARANCE_SCALE 0.35f
#define QR_ROUTE_CLEARANCE_SPEED_SCALE 0.0f
#define QR_ROUTE_CLEARANCE_SPEED_MIN 0.0f
#define QR_ROUTE_CLEARANCE_SPEED_FULL 0.0f
#define QR_ROUTE_REFERENCE_CLEARANCE_SCALE 0.0f
#define QR_ROUTE_VEL_SCALE 0.04f
#define QR_ROUTE_ARRIVAL_RADIUS 0.0f
#define QR_ROUTE_ARRIVAL_BONUS 0.0f
#define QR_ROUTE_ARRIVAL_VEL_SCALE 0.0f
#define QR_ROUTE_ARRIVAL_TOWARD_SCALE 0.0f
#define QR_ROUTE_TARGET_WALL_MARGIN 0.0f
#define QR_ROUTE_GOAL_SPEED 2.0f
#define QR_ROUTE_GOAL_BONUS_REF 1.0f
#define QR_DRONE_RADIUS 0.12f
#define QR_NAV_PATH_MAX_NODES (2 + 4 * QR_MAX_OBSTACLES)
#define QR_NAV_PATH_MARGIN (QR_DRONE_RADIUS + 0.08f)
#define QR_NAV_PATH_INF 1.0e9f
#define QR_EXPLORE_MAX_GRID 64
#define QR_EXPLORE_CELL_SIZE 0.25f
#define QR_EXPLORE_GOAL_FRACTION 0.85f
#define QR_EXPLORE_NEW_CELL_REWARD 4.0f
#define QR_EXPLORE_VISIBLE_CELL_REWARD 0.08f
#define QR_EXPLORE_STEP_COST 0.01f
#define QR_EXPLORE_GOAL_BONUS 10.0f
#define QR_EXPLORE_CRASH_REWARD -20.0f
#define QR_EXPLORE_TIMEOUT_PENALTY 0.0f
#define QR_EXPLORE_TIMEOUT_MISSING_SCALE 0.0f
#define QR_EXPLORE_FRONTIER_PROGRESS_SCALE 0.0f
#define QR_EXPLORE_CLEARANCE_RADIUS 0.45f
#define QR_EXPLORE_CLEARANCE_SCALE 0.35f
#define QR_EXPLORE_SOFT_CLEARANCE_RADIUS 1.25f
#define QR_EXPLORE_SOFT_CLEARANCE_SCALE 0.12f
#define QR_EXPLORE_WALL_OUTWARD_SCALE 0.40f
#define QR_EXPLORE_OBSTACLE_INWARD_RADIUS 0.75f
#define QR_EXPLORE_OBSTACLE_INWARD_SCALE 0.20f
#define QR_EXPLORE_TERMINAL_CRASH_SCALE 1.00f
#define QR_EXPLORE_CELL_MARGIN 0.30f
#define QR_EXPLORE_HEIGHT_SCALE 0.03f
#define QR_EXPLORE_FRONTIER_TARGET_NEAREST 0
#define QR_EXPLORE_FRONTIER_TARGET_GAIN 1
#define QR_EXPLORE_FRONTIER_TARGET_UNEXPLORED 2
#define QR_EXPLORE_FRONTIER_TARGET_ONBOARD 3
#define QR_EXPLORE_FRONTIER_TARGET_REGION 4
#define QR_EXPLORE_ONBOARD_STALE_TICKS 96
#define QR_EXPLORE_ONBOARD_STALE_MIN_CELLS 2
#define QR_EXPLORE_ONBOARD_CURRENT_CELL_PENALTY 0.35f
#define QR_EXPLORE_ONBOARD_VIEWPOINT_RADIUS 4
#define QR_EXPLORE_ONBOARD_VIEWPOINT_PREFERRED_DIST 2
#define QR_EXPLORE_ONBOARD_REGION_COMMIT_SCORE_GAP 0.25f
#define QR_EXPLORE_ONBOARD_MAX_REGIONS 32
#define QR_EXPLORE_ONBOARD_REGION_MATCH_RADIUS 4
#define QR_TASK_FIXED_TARGET 0
#define QR_TASK_HOLD_HOVER 1
#define QR_TASK_INDOOR_NAV 2
#define QR_TASK_INDOOR_EXPLORE 3
#define QR_TASK_INDOOR_ROUTE 4
#define QR_TASK_DRONE_HOVER 5
#define QR_DRONE_ALPHA_DIST 0.782192f
#define QR_DRONE_ALPHA_HOVER 0.071445f
#define QR_DRONE_ALPHA_OMEGA 0.00135588f
#define QR_DRONE_ALPHA_SHAPING 3.9754f
#define QR_DRONE_HOVER_TARGET_DIST 5.0f
#define QR_DRONE_HOVER_DIST 0.1f
#define QR_DRONE_HOVER_OMEGA 0.1f
#define QR_DRONE_HOVER_VEL 0.1f
#define QR_ROUTE_TARGET_OBS_LEGACY 0
#define QR_ROUTE_TARGET_OBS_BODY 1
#define QR_ROUTE_TARGET_OBS_MINIMAL 2
#define QR_ROUTE_TARGET_OBS_DRONE 3
#define QR_ACTION_RATE 0
#define QR_ACTION_VELOCITY 1
#define QR_ACTION_MOTOR 2
#define QR_ACTION_MOTOR_HOVER 3
#define QR_ACTION_MOTOR_REF_HOVER 4
#define QR_DYNAMICS_PROFILE_CUSTOM 0
#define QR_DYNAMICS_PROFILE_DRONE_REFERENCE 1
#define QR_OBSTACLES_NONE 0
#define QR_OBSTACLES_CENTER_BLOCK 1
#define QR_OBSTACLES_DOORWAY 2
#define QR_OBSTACLES_CORRIDOR 3
#define QR_OBSTACLES_MAZE 4
#define QR_OBSTACLES_RANDOM_BLOCKS 5
#define QR_OBSTACLES_MIXED_OPEN_BLOCKS 6
#define QR_OBSTACLES_LAYOUT_MIXED 7
#define QR_OBSTACLES_PROCEDURAL_INDOOR 8
#define QR_OBSTACLES_PROCEDURAL_INDOOR_P0 9
#define QR_OBSTACLES_PROCEDURAL_INDOOR_P1 10
#define QR_OBSTACLES_PROCEDURAL_INDOOR_P2 11
#define QR_OBSTACLES_PROCEDURAL_INDOOR_P3 12
#define QR_OBSTACLES_PROCEDURAL_INDOOR_HARD 13
#define QR_OBSTACLES_PROCEDURAL_INDOOR_HARD_P0 14
#define QR_OBSTACLES_PROCEDURAL_INDOOR_HARD_P1 15
#define QR_OBSTACLES_PROCEDURAL_INDOOR_HARD_P2 16
#define QR_OBSTACLES_PROCEDURAL_INDOOR_HARD_P3 17
#define QR_OBSTACLES_PROCEDURAL_INDOOR_TIGHT 18
#define QR_OBSTACLES_PROCEDURAL_INDOOR_TIGHT_P0 19
#define QR_OBSTACLES_PROCEDURAL_INDOOR_TIGHT_P1 20
#define QR_OBSTACLES_PROCEDURAL_INDOOR_TIGHT_P2 21
#define QR_OBSTACLES_PROCEDURAL_INDOOR_TIGHT_P3 22
#define QR_OBSTACLES_PROCEDURAL_HOUSE 23
#define QR_OBSTACLES_PROCEDURAL_HOUSE_P0 24
#define QR_OBSTACLES_PROCEDURAL_HOUSE_P1 25
#define QR_OBSTACLES_PROCEDURAL_HOUSE_P2 26
#define QR_OBSTACLES_PROCEDURAL_HOUSE_P3 27
#define QR_OBSTACLES_PROCEDURAL_HOUSE_MULTIZONE 28
#define QR_OBSTACLES_PROCEDURAL_HOUSE_MULTIZONE_P0 29
#define QR_OBSTACLES_PROCEDURAL_HOUSE_MULTIZONE_P1 30
#define QR_OBSTACLES_PROCEDURAL_HOUSE_MULTIZONE_P2 31
#define QR_OBSTACLES_PROCEDURAL_HOUSE_MULTIZONE_P3 32
#define QR_OBSTACLES_GATE_SLALOM 33
#define QR_OBSTACLES_GATE_SLALOM_P0 34
#define QR_OBSTACLES_GATE_SLALOM_P1 35
#define QR_OBSTACLES_GATE_SLALOM_P2 36
#define QR_OBSTACLES_GATE_SLALOM_P3 37
#define QR_OBSTACLES_GATE_SLALOM_EASY_MIX 38
#define QR_OBSTACLES_GATE_SLALOM_P1_P0_MIX 39
#define QR_OBSTACLES_AGILE_GEOMETRY 40
#define QR_OBSTACLES_AGILE_GEOMETRY_P0 41
#define QR_OBSTACLES_AGILE_GEOMETRY_P1 42
#define QR_OBSTACLES_AGILE_GEOMETRY_P2 43
#define QR_OBSTACLES_AGILE_GEOMETRY_P3 44
#define QR_OBSTACLES_AGILE_GEOMETRY_INTRO 45
#define QR_OBSTACLES_AGILE_PILLAR_WEAVE 46
#define QR_OBSTACLES_AGILE_PILLAR_L0 47
#define QR_OBSTACLES_AGILE_PILLAR_L1 48
#define QR_OBSTACLES_AGILE_PILLAR_L2 49
#define QR_OBSTACLES_AGILE_PILLAR_L3 50
#define QR_OBSTACLES_AGILE_PILLAR_EASY_MIX 51
#define QR_OBSTACLES_AGILE_PILLAR_CURRICULUM 52
#define QR_OBSTACLES_AGILE_PILLAR_L2_OPEN 53
#define QR_OBSTACLES_AGILE_PILLAR_L2_MARKER_TINY 54
#define QR_OBSTACLES_AGILE_PILLAR_L2_MARKER_SMALL 55
#define QR_OBSTACLES_AGILE_PILLAR_L2_MARKER_LARGE 56
#define QR_OBSTACLES_AGILE_PILLAR_BOTTLENECK_CURRICULUM 57
#define QR_ROUTE_REWARD_DENSE 0
#define QR_ROUTE_REWARD_REFERENCE 1
#define QR_ROUTE_REWARD_DRONE_HOVER 2
#define QR_ROUTE_REWARD_DRONE_HOVER_CONTINUOUS 3
#define QR_ROUTE_REWARD_DRONE_HOVER_ARRIVAL 4
#define QR_ROUTE_TARGET_RANDOM 0
#define QR_ROUTE_TARGET_BLOCKED 1
#define QR_CRASH_NONE 0
#define QR_CRASH_GROUND 1
#define QR_CRASH_WALL 2
#define QR_CRASH_OBSTACLE 3
#define QR_CRASH_DIVERGED 4
#define QR_STATUS_DRONE_OOB 12

// Flat float struct: pufferl sums every field across envs and divides by n.
struct Log {
    float perf;
    float success;
    float score;
    float episode_return;
    float episode_length;
    float crashed;
    float avg_dist;
    float avg_vel;
    float avg_tilt;
    float max_dist;
    float final_dist;
    float final_z_error;
    float final_vel;
    float settle_time;
    float settled;
    float hover_rate;
    float timeout;
    float drone_oob;
    float obstacle_hit;
    float wall_hit;
    float ground_hit;
    float min_clearance;
    float path_length;
    float path_efficiency;
    float action_saturation;
    float action_delta;
    float mass_scale;
    float inertia_scale;
    float thrust_scale;
    float torque_scale;
    float drag_scale;
    float motor_tau_scale;
    float coverage;
    float explored_cells;
    float new_cell_rate;
    float frontier_count;
    float stale_frontier_count;
    float stale_marks;
    float frontier_cluster_count;
    float onboard_target_exists;
    float onboard_target_age;
    float onboard_target_cluster_cells;
    float blocked_cells;
    float viewpoint_candidate_count;
    float viewpoint_viable_count;
    float viewpoint_used_rate;
    float viewpoint_miss_rate;
    float viewpoint_best_score;
    float viewpoint_best_clearance;
    float viewpoint_best_depth;
    float waypoint_depth;
    float onboard_target_switch_rate;
    float onboard_objective_switch_rate;
    float unproductive_objective_switch_rate;
    float committed_waypoint_used_rate;
    float committed_waypoint_drop_rate;
    float committed_waypoint_age;
    float region_tracker_regions;
    float region_tracker_active_age;
    float region_tracker_since_progress;
    float region_tracker_switch_rate;
    float region_tracker_stale_rate;
    float frontier_graph_nodes;
    float frontier_graph_unknown_cells;
    float frontier_graph_best_unknown_cells;
    float frontier_graph_best_depth;
    float frontier_graph_best_score;
    float frontier_graph_active_rate;
    float adaptive_lookahead;
    float adaptive_lookahead_long_rate;
    float adaptive_lookahead_clearance_reject_rate;
    float adaptive_lookahead_turn_reject_rate;
    float route_goals;
    float procedural_n[4];
    float procedural_success[4];
    float procedural_timeout[4];
    float procedural_obstacle_hit[4];
    float procedural_coverage[4];
    float n;
};

typedef struct {
    int frontier_count;
    int stale_frontier_count;
    int frontier_cluster_count;
    int blocked_cells;
    int target_exists;
    int target_age;
    int target_idx;
    int target_frontier_idx;
    int target_cluster_cells;
    int viewpoint_candidate_count;
    int viewpoint_viable_count;
    int viewpoint_used;
    int viewpoint_miss;
    int viewpoint_best_idx;
    int waypoint_idx;
    int waypoint_depth;
    float viewpoint_best_score;
    float viewpoint_best_clearance;
    int viewpoint_best_depth;
    int committed_viewpoint_idx;
    int committed_viewpoint_used;
    int committed_viewpoint_age;
    int tracker_region_count;
    int tracker_active_id;
    int tracker_active_age;
    int tracker_since_progress;
    int tracker_stale;
} QrExplorePlannerStats;

typedef struct Client Client;
struct Client {
    Camera3D camera;
    int window_w, window_h;
    float camera_yaw;
    float camera_pitch;
    float camera_distance;
    float render_fps;
};

struct Env {
    // 5.0 trainer binds these per agent after puf_init; qr_bind_agent mirrors
    // them into the legacy flat pointers below that the env code uses.
    Agent agents[1];
    int tag;
    int boundary_reached;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* truncations;
    Log log;
    Client* client;
    int num_agents;
    unsigned int rng;

    Quadrotor phys;
    Quadrotor ctrl_model;
    QuadrotorRateCtrl rate_ctrl;
    QuadrotorPosCtrl vel_ctrl;
    float target[3];
    float max_rate;
    float max_vel;
    float max_z_vel;
    int dynamics_profile;
    float physics_dt;
    int physics_substeps;
    float vel_ctrl_kp_vel;
    float vel_ctrl_ki_z;
    float vel_ctrl_int_limit_z;
    float vel_ctrl_max_z_acc;
    float spawn_radius;
    float spawn_radius_min;
    float episode_spawn_radius;
    float spawn_height;
    float init_vel_radius;
    float init_omega_radius;
    float room_size;
    float ceiling_height;
    float ray_length;
    float hover_radius;
    float drone_alpha_dist;
    float drone_alpha_hover;
    float drone_alpha_omega;
    float drone_alpha_shaping;
    float drone_hover_target_dist;
    float drone_hover_dist;
    float drone_hover_omega;
    float drone_hover_vel;
    float drone_prev_potential;
    int task_mode;
    int num_rays;
    int camera_depth_obs;
    int camera_depth_width;
    int camera_depth_height;
    float camera_depth_fov_x_deg;
    float camera_depth_fov_y_deg;
    int tick;
    int max_steps;
    int action_repeat;
    float episode_return;
    float dist_accum;
    float vel_accum;
    float tilt_accum;
    float max_dist;
    float final_dist;
    float final_z_error;
    float final_vel;
    float prev_dist;
    int last_unsettled_tick;
    float hover_steps;
    int action_mode;
    int obstacle_mode;
    int active_obstacle_mode;
    int procedural_pattern;
    float obstacle_block_prob;
    int num_obstacles;
    float obstacles[QR_MAX_OBSTACLES][6];
    float mass_scale;
    float inertia_scale;
    float thrust_scale;
    float torque_scale;
    float drag_scale;
    float motor_tau_scale;
    float mass_scale_min;
    float mass_scale_max;
    float inertia_scale_min;
    float inertia_scale_max;
    float thrust_scale_min;
    float thrust_scale_max;
    float torque_scale_min;
    float torque_scale_max;
    float drag_scale_min;
    float drag_scale_max;
    float motor_tau_scale_min;
    float motor_tau_scale_max;
    float current_mass_scale;
    float current_inertia_scale;
    float current_thrust_scale;
    float current_torque_scale;
    float current_drag_scale;
    float current_motor_tau_scale;
    float explore_cell_size;
    float explore_goal_fraction;
    float explore_frontier_progress_scale;
    int explore_memory_obs;
    int explore_frontier_target_mode;
    int explore_reachable_coverage;
    int explore_onboard_commit_steps;
    int explore_onboard_path_lookahead;
    int explore_onboard_adaptive_lookahead;
    int explore_onboard_adaptive_lookahead_min;
    int explore_onboard_adaptive_lookahead_max;
    float explore_onboard_adaptive_clearance;
    float explore_onboard_adaptive_turn_dot;
    float explore_onboard_commit_min_clearance;
    int explore_onboard_stall_viewpoint_ticks;
    int explore_onboard_stall_viewpoint_max_depth;
    int explore_onboard_approach_viewpoints;
    int explore_onboard_approach_max_depth;
    int explore_onboard_emit_objective_target;
    int explore_onboard_objective_obs;
    int explore_onboard_region_commit;
    int explore_onboard_region_commit_stall_ticks;
    int explore_onboard_persistent_regions;
    int explore_onboard_region_tracker;
    int explore_onboard_clearance_passable;
    float explore_onboard_passable_radius;
    int explore_onboard_passage_quality;
    float explore_onboard_passage_weight;
    float explore_onboard_clearance_cone_deg;
    int explore_onboard_budget_frontier;
    float explore_onboard_budget_gain_weight;
    float explore_onboard_budget_cluster_weight;
    float explore_onboard_budget_cell_weight;
    float explore_onboard_budget_depth_bias;
    float explore_onboard_budget_depth_weight;
    int explore_onboard_optimistic_unknown;
    int explore_onboard_frontier_graph;
    int explore_onboard_graph_candidate_obs;
    float explore_onboard_graph_gain_weight;
    float explore_onboard_graph_cluster_weight;
    float explore_onboard_graph_depth_weight;
    float explore_onboard_graph_active_bonus;
    float explore_onboard_graph_stale_penalty;
    float explore_onboard_graph_failure_penalty;
    float explore_onboard_graph_attempt_penalty;
    int explore_onboard_graph_gain_cap;
    float route_goal_radius;
    int route_goal_count;
    int route_goal_count_min;
    float route_step_cost;
    float route_proximity_reward;
    float route_dist_scale;
    float route_progress_scale;
    float route_crash_reward;
    float route_crash_goal_scale;
    float route_crash_dist_scale;
    float route_clearance_radius;
    float route_clearance_scale;
    float route_clearance_speed_scale;
    float route_clearance_speed_min;
    float route_clearance_speed_full;
    float route_reference_clearance_radius;
    float route_reference_clearance_scale;
    float route_path_margin;
    float route_vel_scale;
    float route_arrival_radius;
    float route_arrival_bonus;
    float route_arrival_vel_scale;
    float route_arrival_toward_scale;
    float route_target_wall_margin;
    float route_goal_speed;
    float route_goal_bonus;
    float route_reach_bonus;
    float route_target_z_min;
    float route_target_z_max;
    int route_reward_mode;
    int route_walls;
    int route_direct_obs;
    int route_privileged_obs;
    int route_extra_obs;
    int route_wall_obs;
    int route_perception_obs;
    int route_target_obs_mode;
    int action_history_obs;
    int route_target_mode;
    float route_blocked_target_prob;
    int route_hardstate_probe;
    float route_hardstate_start_x;
    float route_hardstate_start_y;
    float route_hardstate_start_z;
    float route_hardstate_target_x;
    float route_hardstate_target_y;
    float route_hardstate_target_z;
    float route_hardstate_yaw_deg;
    float route_hardstate_jitter_xy;
    float route_hardstate_jitter_z;
    int route_goals_reached;
    int episode_route_goal_count;
    int explore_grid_dim;
    int explore_total_cells;
    int explored_cells;
    float explore_new_cells;
    float prev_frontier_dist;
    unsigned char explored[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    unsigned char explore_blocked[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    unsigned char explore_frontier_stale[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    unsigned char explore_reachable[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int explore_onboard_target_idx;
    int explore_onboard_target_frontier_idx;
    int explore_onboard_objective_idx;
    int explore_onboard_objective_depth;
    int explore_onboard_target_cluster_cells;
    int explore_onboard_region_anchor_idx;
    int explore_onboard_region_goal_frontier_idx;
    int explore_onboard_region_goal_unknown_idx;
    int explore_onboard_committed_viewpoint_idx;
    int explore_onboard_committed_unknown_idx;
    int explore_onboard_committed_until_tick;
    int explore_onboard_committed_start_tick;
    int explore_onboard_last_progress_tick;
    int explore_onboard_last_progress_cells;
    int explore_onboard_stale_marks;
    int explore_onboard_viewpoint_calls;
    int explore_onboard_viewpoint_candidate_sum;
    int explore_onboard_viewpoint_viable_sum;
    int explore_onboard_viewpoint_used_count;
    int explore_onboard_viewpoint_miss_count;
    float explore_onboard_viewpoint_score_sum;
    float explore_onboard_viewpoint_clearance_sum;
    int explore_onboard_viewpoint_depth_sum;
    int explore_onboard_target_calls;
    int explore_onboard_waypoint_depth_sum;
    int explore_onboard_selection_calls;
    int explore_onboard_prev_target_idx;
    int explore_onboard_prev_objective_idx;
    int explore_onboard_prev_selection_cells;
    int explore_onboard_target_switch_count;
    int explore_onboard_objective_switch_count;
    int explore_onboard_unproductive_objective_switch_count;
    int explore_onboard_committed_used_count;
    int explore_onboard_committed_drop_count;
    int explore_onboard_committed_age_sum;
    int explore_onboard_committed_last_used;
    int explore_onboard_adaptive_lookahead_calls;
    int explore_onboard_adaptive_lookahead_long_count;
    int explore_onboard_adaptive_clearance_reject_count;
    int explore_onboard_adaptive_turn_reject_count;
    float explore_onboard_adaptive_lookahead_sum;
    int explore_onboard_viewpoint_last_candidates;
    int explore_onboard_viewpoint_last_viable;
    int explore_onboard_viewpoint_last_used;
    int explore_onboard_viewpoint_last_miss;
    int explore_onboard_viewpoint_best_idx;
    int explore_onboard_waypoint_idx;
    int explore_onboard_waypoint_depth;
    float explore_onboard_viewpoint_best_score;
    float explore_onboard_viewpoint_best_clearance;
    int explore_onboard_viewpoint_best_depth;
    int explore_onboard_region_next_id;
    int explore_onboard_active_region_id;
    int explore_onboard_region_id[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_cx[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_cy[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_frontier_idx[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_unknown_idx[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_cells[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_attempts[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_failures[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_first_seen_tick[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_last_seen_tick[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_last_progress_tick[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_region_last_progress_cells[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int explore_onboard_tracker_calls;
    int explore_onboard_tracker_region_sum;
    int explore_onboard_tracker_switch_count;
    int explore_onboard_tracker_stale_count;
    int explore_onboard_tracker_active_age_sum;
    int explore_onboard_tracker_since_progress_sum;
    int explore_onboard_tracker_last_region_count;
    int explore_onboard_tracker_last_active_id;
    int explore_onboard_tracker_last_active_age;
    int explore_onboard_tracker_last_since_progress;
    int explore_onboard_tracker_last_stale;
    int explore_onboard_graph_calls;
    int explore_onboard_graph_node_sum;
    int explore_onboard_graph_unknown_sum;
    int explore_onboard_graph_best_unknown_sum;
    int explore_onboard_graph_best_depth_sum;
    float explore_onboard_graph_score_sum;
    int explore_onboard_graph_active_count;
    int explore_onboard_graph_last_nodes;
    int explore_onboard_graph_last_unknown_cells;
    int explore_onboard_graph_last_best_unknown_cells;
    int explore_onboard_graph_last_best_depth;
    float explore_onboard_graph_last_best_score;
    int explore_onboard_graph_last_active;
    int explore_onboard_graph_candidate_count;
    int explore_onboard_graph_candidate_frontier_idx[QR_EXPLORE_GRAPH_CANDIDATE_COUNT];
    int explore_onboard_graph_candidate_unknown_idx[QR_EXPLORE_GRAPH_CANDIDATE_COUNT];
    int explore_onboard_graph_candidate_region_id[QR_EXPLORE_GRAPH_CANDIDATE_COUNT];
    int explore_onboard_graph_candidate_depth[QR_EXPLORE_GRAPH_CANDIDATE_COUNT];
    int explore_onboard_graph_candidate_cells[QR_EXPLORE_GRAPH_CANDIDATE_COUNT];
    int explore_onboard_graph_candidate_gain[QR_EXPLORE_GRAPH_CANDIDATE_COUNT];
    int explore_onboard_graph_candidate_selected[QR_EXPLORE_GRAPH_CANDIDATE_COUNT];
    float explore_onboard_graph_candidate_passage[QR_EXPLORE_GRAPH_CANDIDATE_COUNT];
    float explore_onboard_graph_candidate_score[QR_EXPLORE_GRAPH_CANDIDATE_COUNT];
    float min_clearance;
    float path_length;
    float start_dist;
    float action_saturation_sum;
    float action_delta_sum;
    float action_steps;
    float prev_action[QR_NUM_ACTIONS];
    int has_prev_action;
    int timed_out;
    int drone_oob;
    int crash_cause;
    int route_goal_event;
    int route_goal_event_index;
    float route_goal_event_dist;
    float route_goal_event_speed;
    float route_goal_event_clearance;
    int route_goal_event_visible;
    int last_episode_status;
    int last_episode_flash;
    int last_episode_goals;
    int last_episode_goal_count;
    float last_episode_dist;
    float last_episode_return;
    float last_episode_clearance;
    float last_episode_pos[3];
    float last_episode_vel[3];
    float last_episode_target[3];
    float last_episode_obs_target[3];
    int seed;
    uint32_t rng_state;
};
typedef Env QuadrotorEnv;

static inline void qr_explore_planner_stats(const QuadrotorEnv* env,
        QrExplorePlannerStats* out);
static inline float qr_norm3(const float v[3]);

static inline uint32_t qr_seed_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x ? x : 0x6d2b79f5u;
}

static inline void qr_seed_rng(QuadrotorEnv* env, int seed) {
    env->rng_state = qr_seed_mix((uint32_t)seed + 0x9e3779b9u);
}

static inline uint32_t qr_rand_u32(QuadrotorEnv* env) {
    env->rng_state = env->rng_state * 1664525u + 1013904223u;
    return env->rng_state;
}

static inline float qr_rand01(QuadrotorEnv* env) {
    return (float)(qr_rand_u32(env) >> 8) * (1.0f / 16777216.0f);
}

static inline float qr_rand_range(QuadrotorEnv* env, float lo, float hi) {
    return lo + qr_rand01(env) * (hi - lo);
}

static inline int qr_quadrotor_state_finite(const Quadrotor* q) {
    for (int i = 0; i < 3; i++) {
        if (!isfinite(q->pos[i]) || !isfinite(q->vel[i]) || !isfinite(q->omega[i]))
            return 0;
    }
    for (int i = 0; i < 4; i++) {
        if (!isfinite(q->quat[i]) || !isfinite(q->motor_speed[i]))
            return 0;
    }
    return 1;
}

static inline int qr_episode_route_goal_count(const QuadrotorEnv* env) {
    return env->episode_route_goal_count > 0
        ? env->episode_route_goal_count
        : env->route_goal_count;
}

static inline int qr_route_reward_uses_direct_dist(const QuadrotorEnv* env) {
    return env->route_reward_mode == QR_ROUTE_REWARD_REFERENCE ||
        env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER ||
        env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER_CONTINUOUS ||
        env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER_ARRIVAL;
}

static inline int qr_route_reward_is_drone_hover_like(const QuadrotorEnv* env) {
    return env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER ||
        env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER_CONTINUOUS ||
        env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER_ARRIVAL;
}

static inline void qr_sample_episode_route_goal_count(QuadrotorEnv* env) {
    int max_goals = env->route_goal_count;
    int min_goals = env->route_goal_count_min;
    if (min_goals <= 0 || min_goals >= max_goals) {
        env->episode_route_goal_count = max_goals;
        return;
    }

    int span = max_goals - min_goals + 1;
    env->episode_route_goal_count = min_goals + (int)(qr_rand_u32(env) % (uint32_t)span);
}

static inline int qr_task_uses_waypoint(int task_mode) {
    return task_mode == QR_TASK_INDOOR_NAV || task_mode == QR_TASK_INDOOR_ROUTE;
}

static inline int qr_task_uses_room_spawn(int task_mode) {
    return task_mode == QR_TASK_INDOOR_NAV ||
        task_mode == QR_TASK_INDOOR_ROUTE ||
        task_mode == QR_TASK_INDOOR_EXPLORE;
}

static inline void qr_clear_explore_graph_candidate_cache(QuadrotorEnv* env) {
    env->explore_onboard_graph_candidate_count = 0;
    for (int i = 0; i < QR_EXPLORE_GRAPH_CANDIDATE_COUNT; i++) {
        env->explore_onboard_graph_candidate_frontier_idx[i] = -1;
        env->explore_onboard_graph_candidate_unknown_idx[i] = -1;
        env->explore_onboard_graph_candidate_region_id[i] = -1;
        env->explore_onboard_graph_candidate_depth[i] = 0;
        env->explore_onboard_graph_candidate_cells[i] = 0;
        env->explore_onboard_graph_candidate_gain[i] = 0;
        env->explore_onboard_graph_candidate_selected[i] = 0;
        env->explore_onboard_graph_candidate_passage[i] = 0.0f;
        env->explore_onboard_graph_candidate_score[i] = -1.0e9f;
    }
}

static inline void qr_update_explore_graph_candidate_selection(QuadrotorEnv* env) {
    for (int i = 0; i < QR_EXPLORE_GRAPH_CANDIDATE_COUNT; i++) {
        int frontier_idx = env->explore_onboard_graph_candidate_frontier_idx[i];
        int region_id = env->explore_onboard_graph_candidate_region_id[i];
        int selected = frontier_idx >= 0 &&
            ((region_id > 0 && region_id == env->explore_onboard_active_region_id) ||
             frontier_idx == env->explore_onboard_target_frontier_idx);
        env->explore_onboard_graph_candidate_selected[i] = selected;
    }
}

static inline void qr_clear_episode_stats(QuadrotorEnv* env) {
    env->tick = 0;
    env->episode_return = 0.0f;
    env->dist_accum = 0.0f;
    env->vel_accum = 0.0f;
    env->tilt_accum = 0.0f;
    env->max_dist = 0.0f;
    env->final_dist = 0.0f;
    env->final_z_error = 0.0f;
    env->final_vel = 0.0f;
    env->prev_dist = 0.0f;
    env->last_unsettled_tick = -1;
    env->hover_steps = 0.0f;
    env->explore_grid_dim = 0;
    env->explore_total_cells = 0;
    env->explored_cells = 0;
    env->explore_new_cells = 0.0f;
    env->prev_frontier_dist = 0.0f;
    env->route_goals_reached = 0;
    memset(env->explored, 0, sizeof(env->explored));
    memset(env->explore_blocked, 0, sizeof(env->explore_blocked));
    memset(env->explore_frontier_stale, 0, sizeof(env->explore_frontier_stale));
    memset(env->explore_reachable, 0, sizeof(env->explore_reachable));
    env->explore_onboard_target_idx = -1;
    env->explore_onboard_target_frontier_idx = -1;
    env->explore_onboard_objective_idx = -1;
    env->explore_onboard_objective_depth = 0;
    env->explore_onboard_target_cluster_cells = 0;
    env->explore_onboard_region_anchor_idx = -1;
    env->explore_onboard_region_goal_frontier_idx = -1;
    env->explore_onboard_region_goal_unknown_idx = -1;
    env->explore_onboard_committed_viewpoint_idx = -1;
    env->explore_onboard_committed_unknown_idx = -1;
    env->explore_onboard_committed_until_tick = 0;
    env->explore_onboard_committed_start_tick = 0;
    env->explore_onboard_last_progress_tick = 0;
    env->explore_onboard_last_progress_cells = 0;
    env->explore_onboard_stale_marks = 0;
    env->explore_onboard_viewpoint_calls = 0;
    env->explore_onboard_viewpoint_candidate_sum = 0;
    env->explore_onboard_viewpoint_viable_sum = 0;
    env->explore_onboard_viewpoint_used_count = 0;
    env->explore_onboard_viewpoint_miss_count = 0;
    env->explore_onboard_viewpoint_score_sum = 0.0f;
    env->explore_onboard_viewpoint_clearance_sum = 0.0f;
    env->explore_onboard_viewpoint_depth_sum = 0;
    env->explore_onboard_target_calls = 0;
    env->explore_onboard_waypoint_depth_sum = 0;
    env->explore_onboard_selection_calls = 0;
    env->explore_onboard_prev_target_idx = -1;
    env->explore_onboard_prev_objective_idx = -1;
    env->explore_onboard_prev_selection_cells = 0;
    env->explore_onboard_target_switch_count = 0;
    env->explore_onboard_objective_switch_count = 0;
    env->explore_onboard_unproductive_objective_switch_count = 0;
    env->explore_onboard_committed_used_count = 0;
    env->explore_onboard_committed_drop_count = 0;
    env->explore_onboard_committed_age_sum = 0;
    env->explore_onboard_committed_last_used = 0;
    env->explore_onboard_adaptive_lookahead_calls = 0;
    env->explore_onboard_adaptive_lookahead_long_count = 0;
    env->explore_onboard_adaptive_clearance_reject_count = 0;
    env->explore_onboard_adaptive_turn_reject_count = 0;
    env->explore_onboard_adaptive_lookahead_sum = 0.0f;
    env->explore_onboard_viewpoint_last_candidates = 0;
    env->explore_onboard_viewpoint_last_viable = 0;
    env->explore_onboard_viewpoint_last_used = 0;
    env->explore_onboard_viewpoint_last_miss = 0;
    env->explore_onboard_viewpoint_best_idx = -1;
    env->explore_onboard_objective_idx = -1;
    env->explore_onboard_objective_depth = 0;
    env->explore_onboard_waypoint_idx = -1;
    env->explore_onboard_waypoint_depth = 0;
    env->explore_onboard_viewpoint_best_score = 0.0f;
    env->explore_onboard_viewpoint_best_clearance = 0.0f;
    env->explore_onboard_viewpoint_best_depth = 0;
    env->explore_onboard_region_next_id = 1;
    env->explore_onboard_active_region_id = -1;
    memset(env->explore_onboard_region_id, 0, sizeof(env->explore_onboard_region_id));
    memset(env->explore_onboard_region_cx, 0, sizeof(env->explore_onboard_region_cx));
    memset(env->explore_onboard_region_cy, 0, sizeof(env->explore_onboard_region_cy));
    memset(env->explore_onboard_region_frontier_idx, -1, sizeof(env->explore_onboard_region_frontier_idx));
    memset(env->explore_onboard_region_unknown_idx, -1, sizeof(env->explore_onboard_region_unknown_idx));
    memset(env->explore_onboard_region_cells, 0, sizeof(env->explore_onboard_region_cells));
    memset(env->explore_onboard_region_attempts, 0, sizeof(env->explore_onboard_region_attempts));
    memset(env->explore_onboard_region_failures, 0, sizeof(env->explore_onboard_region_failures));
    memset(env->explore_onboard_region_first_seen_tick, 0, sizeof(env->explore_onboard_region_first_seen_tick));
    memset(env->explore_onboard_region_last_seen_tick, 0, sizeof(env->explore_onboard_region_last_seen_tick));
    memset(env->explore_onboard_region_last_progress_tick, 0, sizeof(env->explore_onboard_region_last_progress_tick));
    memset(env->explore_onboard_region_last_progress_cells, 0, sizeof(env->explore_onboard_region_last_progress_cells));
    env->explore_onboard_tracker_calls = 0;
    env->explore_onboard_tracker_region_sum = 0;
    env->explore_onboard_tracker_switch_count = 0;
    env->explore_onboard_tracker_stale_count = 0;
    env->explore_onboard_tracker_active_age_sum = 0;
    env->explore_onboard_tracker_since_progress_sum = 0;
    env->explore_onboard_tracker_last_region_count = 0;
    env->explore_onboard_tracker_last_active_id = -1;
    env->explore_onboard_tracker_last_active_age = 0;
    env->explore_onboard_tracker_last_since_progress = 0;
    env->explore_onboard_tracker_last_stale = 0;
    env->explore_onboard_graph_calls = 0;
    env->explore_onboard_graph_node_sum = 0;
    env->explore_onboard_graph_unknown_sum = 0;
    env->explore_onboard_graph_best_unknown_sum = 0;
    env->explore_onboard_graph_best_depth_sum = 0;
    env->explore_onboard_graph_score_sum = 0.0f;
    env->explore_onboard_graph_active_count = 0;
    env->explore_onboard_graph_last_nodes = 0;
    env->explore_onboard_graph_last_unknown_cells = 0;
    env->explore_onboard_graph_last_best_unknown_cells = 0;
    env->explore_onboard_graph_last_best_depth = 0;
    env->explore_onboard_graph_last_best_score = 0.0f;
    env->explore_onboard_graph_last_active = 0;
    qr_clear_explore_graph_candidate_cache(env);
    env->min_clearance = env->ray_length > 0.0f ? env->ray_length : 0.0f;
    env->path_length = 0.0f;
    env->start_dist = 0.0f;
    env->action_saturation_sum = 0.0f;
    env->action_delta_sum = 0.0f;
    env->action_steps = 0.0f;
    memset(env->prev_action, 0, sizeof(env->prev_action));
    env->has_prev_action = 0;
    env->timed_out = 0;
    env->drone_oob = 0;
    env->drone_prev_potential = 0.0f;
    env->crash_cause = QR_CRASH_NONE;
}

static inline void add_log(QuadrotorEnv* env) {
    int nav_success = !env->phys.crashed && env->final_dist < QR_NAV_GOAL_RADIUS;
    int route_success = !env->phys.crashed &&
        (env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER_CONTINUOUS
            ? (!env->drone_oob && env->tick >= env->max_steps)
            : (env->route_goals_reached >= qr_episode_route_goal_count(env)));
    int drone_hover_success = !env->phys.crashed &&
        !env->drone_oob &&
        env->tick >= env->max_steps;
    float coverage = env->explore_total_cells > 0
        ? (float)env->explored_cells / (float)env->explore_total_cells
        : 0.0f;
    int explore_success = !env->phys.crashed &&
        coverage >= env->explore_goal_fraction;
    int success = env->task_mode == QR_TASK_INDOOR_NAV
        ? nav_success
        : (env->task_mode == QR_TASK_INDOOR_ROUTE
            ? route_success
            : (env->task_mode == QR_TASK_INDOOR_EXPLORE
                ? explore_success
                : (env->task_mode == QR_TASK_DRONE_HOVER
                    ? drone_hover_success
                    : (!env->phys.crashed && env->tick >= env->max_steps))));
    int won = success;
    env->log.perf += won ? 1.0f : 0.0f;
    env->log.success += success ? 1.0f : 0.0f;
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.crashed += env->phys.crashed ? 1.0f : 0.0f;
    env->log.avg_dist += env->dist_accum / (float)(env->tick > 0 ? env->tick : 1);
    env->log.avg_vel += env->vel_accum / (float)(env->tick > 0 ? env->tick : 1);
    env->log.avg_tilt += env->tilt_accum / (float)(env->tick > 0 ? env->tick : 1);
    env->log.max_dist += env->max_dist;
    env->log.final_dist += env->final_dist;
    env->log.final_z_error += env->final_z_error;
    env->log.final_vel += env->final_vel;
    float settle_steps = env->last_unsettled_tick < 0 ? 0.0f : (float)(env->last_unsettled_tick + 1);
    env->log.settle_time += settle_steps *
        (float)env->action_repeat * (float)env->physics_substeps * env->physics_dt;
    env->log.settled += (!env->phys.crashed && env->final_dist < QR_SETTLE_RADIUS &&
                         env->final_vel < QR_SETTLE_VEL) ? 1.0f : 0.0f;
    env->log.hover_rate += env->hover_steps / (float)(env->tick > 0 ? env->tick : 1);
    env->log.timeout += env->timed_out ? 1.0f : 0.0f;
    env->log.drone_oob += env->drone_oob ? 1.0f : 0.0f;
    env->log.obstacle_hit += env->crash_cause == QR_CRASH_OBSTACLE ? 1.0f : 0.0f;
    env->log.wall_hit += env->crash_cause == QR_CRASH_WALL ? 1.0f : 0.0f;
    env->log.ground_hit += env->crash_cause == QR_CRASH_GROUND ? 1.0f : 0.0f;
    env->log.min_clearance += env->min_clearance;
    env->log.path_length += env->path_length;
    float path_efficiency = 0.0f;
    if (qr_task_uses_waypoint(env->task_mode) && success && env->path_length > 1e-6f) {
        path_efficiency = env->start_dist / env->path_length;
        if (path_efficiency > 1.0f)
            path_efficiency = 1.0f;
    }
    env->log.path_efficiency += path_efficiency;
    env->log.action_saturation += env->action_steps > 0.0f
        ? env->action_saturation_sum / env->action_steps
        : 0.0f;
    env->log.action_delta += env->action_steps > 0.0f
        ? env->action_delta_sum / env->action_steps
        : 0.0f;
    env->log.mass_scale += env->current_mass_scale;
    env->log.inertia_scale += env->current_inertia_scale;
    env->log.thrust_scale += env->current_thrust_scale;
    env->log.torque_scale += env->current_torque_scale;
    env->log.drag_scale += env->current_drag_scale;
    env->log.motor_tau_scale += env->current_motor_tau_scale;
    env->log.coverage += coverage;
    env->log.explored_cells += (float)env->explored_cells;
    env->log.new_cell_rate += env->tick > 0
        ? env->explore_new_cells / (float)env->tick
        : 0.0f;
    QrExplorePlannerStats planner_stats;
    qr_explore_planner_stats(env, &planner_stats);
    env->log.frontier_count += (float)planner_stats.frontier_count;
    env->log.stale_frontier_count += (float)planner_stats.stale_frontier_count;
    env->log.stale_marks += (float)env->explore_onboard_stale_marks;
    env->log.frontier_cluster_count += (float)planner_stats.frontier_cluster_count;
    env->log.onboard_target_exists += (float)planner_stats.target_exists;
    env->log.onboard_target_age += (float)planner_stats.target_age;
    env->log.onboard_target_cluster_cells +=
        (float)planner_stats.target_cluster_cells;
    env->log.blocked_cells += (float)planner_stats.blocked_cells;
    float viewpoint_calls = (float)env->explore_onboard_viewpoint_calls;
    if (viewpoint_calls > 0.0f) {
        env->log.viewpoint_candidate_count +=
            (float)env->explore_onboard_viewpoint_candidate_sum / viewpoint_calls;
        env->log.viewpoint_viable_count +=
            (float)env->explore_onboard_viewpoint_viable_sum / viewpoint_calls;
        env->log.viewpoint_used_rate +=
            (float)env->explore_onboard_viewpoint_used_count / viewpoint_calls;
        env->log.viewpoint_miss_rate +=
            (float)env->explore_onboard_viewpoint_miss_count / viewpoint_calls;
        env->log.viewpoint_best_score +=
            env->explore_onboard_viewpoint_score_sum / viewpoint_calls;
        env->log.viewpoint_best_clearance +=
            env->explore_onboard_viewpoint_clearance_sum / viewpoint_calls;
        env->log.viewpoint_best_depth +=
            (float)env->explore_onboard_viewpoint_depth_sum / viewpoint_calls;
    }
    float target_calls = (float)env->explore_onboard_target_calls;
    if (target_calls > 0.0f)
        env->log.waypoint_depth +=
            (float)env->explore_onboard_waypoint_depth_sum / target_calls;
    float selection_den = (float)(env->explore_onboard_selection_calls > 1
        ? env->explore_onboard_selection_calls - 1
        : 0);
    if (selection_den > 0.0f) {
        env->log.onboard_target_switch_rate +=
            (float)env->explore_onboard_target_switch_count / selection_den;
        env->log.onboard_objective_switch_rate +=
            (float)env->explore_onboard_objective_switch_count / selection_den;
        env->log.unproductive_objective_switch_rate +=
            (float)env->explore_onboard_unproductive_objective_switch_count /
            selection_den;
    }
    if (target_calls > 0.0f) {
        env->log.committed_waypoint_used_rate +=
            (float)env->explore_onboard_committed_used_count / target_calls;
        env->log.committed_waypoint_drop_rate +=
            (float)env->explore_onboard_committed_drop_count / target_calls;
    }
    if (env->explore_onboard_committed_used_count > 0)
        env->log.committed_waypoint_age +=
            (float)env->explore_onboard_committed_age_sum /
            (float)env->explore_onboard_committed_used_count;
    float tracker_calls = (float)env->explore_onboard_tracker_calls;
    if (tracker_calls > 0.0f) {
        env->log.region_tracker_regions +=
            (float)env->explore_onboard_tracker_region_sum / tracker_calls;
        env->log.region_tracker_active_age +=
            (float)env->explore_onboard_tracker_active_age_sum / tracker_calls;
        env->log.region_tracker_since_progress +=
            (float)env->explore_onboard_tracker_since_progress_sum / tracker_calls;
        env->log.region_tracker_switch_rate +=
            (float)env->explore_onboard_tracker_switch_count / tracker_calls;
        env->log.region_tracker_stale_rate +=
            (float)env->explore_onboard_tracker_stale_count / tracker_calls;
    }
    float graph_calls = (float)env->explore_onboard_graph_calls;
    if (graph_calls > 0.0f) {
        env->log.frontier_graph_nodes +=
            (float)env->explore_onboard_graph_node_sum / graph_calls;
        env->log.frontier_graph_unknown_cells +=
            (float)env->explore_onboard_graph_unknown_sum / graph_calls;
        env->log.frontier_graph_best_unknown_cells +=
            (float)env->explore_onboard_graph_best_unknown_sum / graph_calls;
        env->log.frontier_graph_best_depth +=
            (float)env->explore_onboard_graph_best_depth_sum / graph_calls;
        env->log.frontier_graph_best_score +=
            env->explore_onboard_graph_score_sum / graph_calls;
        env->log.frontier_graph_active_rate +=
            (float)env->explore_onboard_graph_active_count / graph_calls;
    }
    float adaptive_calls = (float)env->explore_onboard_adaptive_lookahead_calls;
    if (adaptive_calls > 0.0f) {
        env->log.adaptive_lookahead +=
            env->explore_onboard_adaptive_lookahead_sum / adaptive_calls;
        env->log.adaptive_lookahead_long_rate +=
            (float)env->explore_onboard_adaptive_lookahead_long_count /
            adaptive_calls;
        env->log.adaptive_lookahead_clearance_reject_rate +=
            (float)env->explore_onboard_adaptive_clearance_reject_count /
            adaptive_calls;
        env->log.adaptive_lookahead_turn_reject_rate +=
            (float)env->explore_onboard_adaptive_turn_reject_count /
            adaptive_calls;
    }
    env->log.route_goals += (float)env->route_goals_reached;
    if (env->procedural_pattern >= 0 && env->procedural_pattern < 4) {
        int p = env->procedural_pattern;
        env->log.procedural_n[p] += 1.0f;
        env->log.procedural_success[p] += success ? 1.0f : 0.0f;
        env->log.procedural_timeout[p] += env->timed_out ? 1.0f : 0.0f;
        env->log.procedural_obstacle_hit[p] +=
            env->crash_cause == QR_CRASH_OBSTACLE ? 1.0f : 0.0f;
        env->log.procedural_coverage[p] += coverage;
    }
    env->log.n += 1.0f;
    env->last_episode_status = success
        ? 10
        : (env->timed_out
            ? 11
            : (env->drone_oob ? QR_STATUS_DRONE_OOB : env->crash_cause));
    env->last_episode_flash = 180;
    env->last_episode_goals = env->route_goals_reached;
    env->last_episode_goal_count = qr_episode_route_goal_count(env);
    env->last_episode_dist = env->final_dist;
    env->last_episode_return = env->episode_return;
}

static inline void qr_add_obstacle(QuadrotorEnv* env, float xmin, float xmax,
        float ymin, float ymax, float zmin, float zmax) {
    if (env->num_obstacles >= QR_MAX_OBSTACLES)
        return;
    float* box = env->obstacles[env->num_obstacles++];
    box[0] = xmin; box[1] = xmax;
    box[2] = ymin; box[3] = ymax;
    box[4] = zmin; box[5] = zmax;
}

static inline int qr_obstacle_overlaps_xy(const QuadrotorEnv* env,
        float xmin, float xmax, float ymin, float ymax, float margin) {
    for (int i = 0; i < env->num_obstacles; i++) {
        const float* box = env->obstacles[i];
        if (xmax + margin < box[0] || xmin - margin > box[1] ||
                ymax + margin < box[2] || ymin - margin > box[3])
            continue;
        return 1;
    }
    return 0;
}

static inline void qr_add_vertical_wall_gap(QuadrotorEnv* env, float x,
        float thickness, float gap_y, float gap_half, float h) {
    float r = env->room_size;
    float xmin = qr_clampf(x - 0.5f * thickness, -r, r);
    float xmax = qr_clampf(x + 0.5f * thickness, -r, r);
    float y0 = -r;
    float y1 = qr_clampf(gap_y - gap_half, -r, r);
    float y2 = qr_clampf(gap_y + gap_half, -r, r);
    float y3 = r;
    if (y1 - y0 > thickness)
        qr_add_obstacle(env, xmin, xmax, y0, y1, 0.0f, h);
    if (y3 - y2 > thickness)
        qr_add_obstacle(env, xmin, xmax, y2, y3, 0.0f, h);
}

static inline void qr_add_horizontal_wall_gap(QuadrotorEnv* env, float y,
        float thickness, float gap_x, float gap_half, float h) {
    float r = env->room_size;
    float ymin = qr_clampf(y - 0.5f * thickness, -r, r);
    float ymax = qr_clampf(y + 0.5f * thickness, -r, r);
    float x0 = -r;
    float x1 = qr_clampf(gap_x - gap_half, -r, r);
    float x2 = qr_clampf(gap_x + gap_half, -r, r);
    float x3 = r;
    if (x1 - x0 > thickness)
        qr_add_obstacle(env, x0, x1, ymin, ymax, 0.0f, h);
    if (x3 - x2 > thickness)
        qr_add_obstacle(env, x2, x3, ymin, ymax, 0.0f, h);
}

static inline void qr_add_horizontal_wall_gap_list(QuadrotorEnv* env,
        float y, float thickness, float xmin, float xmax,
        const float* gap_x, const float* gap_half, int gap_count, float h) {
    float r = env->room_size;
    float ymin = qr_clampf(y - 0.5f * thickness, -r, r);
    float ymax = qr_clampf(y + 0.5f * thickness, -r, r);
    float lo[4];
    float hi[4];
    int n = gap_count;
    if (n > 4) n = 4;
    xmin = qr_clampf(xmin, -r, r);
    xmax = qr_clampf(xmax, -r, r);
    if (xmax < xmin) {
        float tmp = xmin;
        xmin = xmax;
        xmax = tmp;
    }
    for (int i = 0; i < n; i++) {
        lo[i] = qr_clampf(gap_x[i] - gap_half[i], xmin, xmax);
        hi[i] = qr_clampf(gap_x[i] + gap_half[i], xmin, xmax);
        if (hi[i] < lo[i]) {
            float tmp = lo[i];
            lo[i] = hi[i];
            hi[i] = tmp;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (lo[j] < lo[i]) {
                float t = lo[i]; lo[i] = lo[j]; lo[j] = t;
                t = hi[i]; hi[i] = hi[j]; hi[j] = t;
            }
        }
    }
    float cursor = xmin;
    for (int i = 0; i < n; i++) {
        if (lo[i] - cursor > thickness)
            qr_add_obstacle(env, cursor, lo[i], ymin, ymax, 0.0f, h);
        if (hi[i] > cursor)
            cursor = hi[i];
    }
    if (xmax - cursor > thickness)
        qr_add_obstacle(env, cursor, xmax, ymin, ymax, 0.0f, h);
}

static inline void qr_add_vertical_wall_gap_list(QuadrotorEnv* env,
        float x, float thickness, float ymin, float ymax,
        const float* gap_y, const float* gap_half, int gap_count, float h) {
    float r = env->room_size;
    float xmin = qr_clampf(x - 0.5f * thickness, -r, r);
    float xmax = qr_clampf(x + 0.5f * thickness, -r, r);
    float lo[4];
    float hi[4];
    int n = gap_count;
    if (n > 4) n = 4;
    ymin = qr_clampf(ymin, -r, r);
    ymax = qr_clampf(ymax, -r, r);
    if (ymax < ymin) {
        float tmp = ymin;
        ymin = ymax;
        ymax = tmp;
    }
    for (int i = 0; i < n; i++) {
        lo[i] = qr_clampf(gap_y[i] - gap_half[i], ymin, ymax);
        hi[i] = qr_clampf(gap_y[i] + gap_half[i], ymin, ymax);
        if (hi[i] < lo[i]) {
            float tmp = lo[i];
            lo[i] = hi[i];
            hi[i] = tmp;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (lo[j] < lo[i]) {
                float t = lo[i]; lo[i] = lo[j]; lo[j] = t;
                t = hi[i]; hi[i] = hi[j]; hi[j] = t;
            }
        }
    }
    float cursor = ymin;
    for (int i = 0; i < n; i++) {
        if (lo[i] - cursor > thickness)
            qr_add_obstacle(env, xmin, xmax, cursor, lo[i], 0.0f, h);
        if (hi[i] > cursor)
            cursor = hi[i];
    }
    if (ymax - cursor > thickness)
        qr_add_obstacle(env, xmin, xmax, cursor, ymax, 0.0f, h);
}

static inline void qr_configure_procedural_indoor(QuadrotorEnv* env, int forced_pattern) {
    float r = env->room_size;
    float h = env->ceiling_height;
    float wall = qr_rand_range(env, 0.10f, 0.16f);
    float gap = qr_rand_range(env, 0.58f, 0.78f);
    uint32_t choice = forced_pattern >= 0 && forced_pattern < 4
        ? (uint32_t)forced_pattern
        : qr_rand_u32(env) % 4u;
    env->procedural_pattern = (int)choice;

    if (choice == 0u) {
        qr_add_vertical_wall_gap(env,
            qr_rand_range(env, -0.85f, 0.85f), wall,
            qr_rand_range(env, -1.15f, 1.15f), gap, h);
        qr_add_horizontal_wall_gap(env,
            qr_rand_range(env, -0.85f, 0.85f), wall,
            qr_rand_range(env, -1.15f, 1.15f), gap, h);
    } else if (choice == 1u) {
        qr_add_horizontal_wall_gap(env, -1.15f, wall,
            qr_rand_range(env, 0.75f, 1.45f), gap, h);
        qr_add_horizontal_wall_gap(env, 0.05f, wall,
            qr_rand_range(env, -1.45f, -0.75f), gap, h);
        qr_add_horizontal_wall_gap(env, 1.20f, wall,
            qr_rand_range(env, 0.75f, 1.45f), gap, h);
    } else if (choice == 2u) {
        qr_add_vertical_wall_gap(env, -0.95f, wall,
            qr_rand_range(env, -1.25f, -0.25f), gap, h);
        qr_add_horizontal_wall_gap(env, 0.05f, wall,
            qr_rand_range(env, -0.45f, 0.65f), gap, h);
        qr_add_vertical_wall_gap(env, 0.95f, wall,
            qr_rand_range(env, 0.25f, 1.25f), gap, h);
    } else {
        qr_add_obstacle(env, -r, -1.15f, -1.25f, -1.05f, 0.0f, h);
        qr_add_obstacle(env, 0.55f, r, -0.15f, 0.05f, 0.0f, h);
        qr_add_obstacle(env, -r, -0.55f, 1.00f, 1.20f, 0.0f, h);
        float sx = qr_rand_range(env, 0.28f, 0.52f);
        float sy = qr_rand_range(env, 0.28f, 0.52f);
        float cx = qr_rand_range(env, -0.65f, 0.65f);
        float cy = qr_rand_range(env, -0.65f, 0.65f);
        qr_add_obstacle(env,
            cx - 0.5f * sx, cx + 0.5f * sx,
            cy - 0.5f * sy, cy + 0.5f * sy, 0.0f, h);
    }
}

static inline void qr_configure_procedural_indoor_hard_variant(
        QuadrotorEnv* env, int forced_pattern, float gap_min, float gap_max,
        int extra_clutter) {
    float r = env->room_size;
    float h = env->ceiling_height;
    float wall = qr_rand_range(env, 0.11f, 0.18f);
    float gap = qr_rand_range(env, gap_min, gap_max);
    uint32_t choice = forced_pattern >= 0 && forced_pattern < 4
        ? (uint32_t)forced_pattern
        : qr_rand_u32(env) % 4u;
    env->procedural_pattern = (int)choice;

    if (choice == 0u) {
        qr_add_vertical_wall_gap(env, -0.34f * r, wall,
            qr_rand_range(env, -0.58f * r, -0.18f * r), gap, h);
        qr_add_vertical_wall_gap(env, 0.34f * r, wall,
            qr_rand_range(env, 0.18f * r, 0.58f * r), gap, h);
        qr_add_horizontal_wall_gap(env, -0.34f * r, wall,
            qr_rand_range(env, 0.18f * r, 0.58f * r), gap, h);
        qr_add_horizontal_wall_gap(env, 0.34f * r, wall,
            qr_rand_range(env, -0.58f * r, -0.18f * r), gap, h);
    } else if (choice == 1u) {
        int wall_count = extra_clutter ? 7 : 5;
        for (int i = 0; i < wall_count; i++) {
            float y = extra_clutter
                ? (-0.72f + 1.44f * (float)i / (float)(wall_count - 1)) * r
                : (-0.66f + 0.33f * (float)i) * r;
            float sign = (i % 2) == 0 ? 1.0f : -1.0f;
            float gap_x = sign * qr_rand_range(env,
                (extra_clutter ? 0.28f : 0.34f) * r,
                (extra_clutter ? 0.66f : 0.62f) * r);
            qr_add_horizontal_wall_gap(env, y, wall, gap_x, gap, h);
        }
    } else if (choice == 2u) {
        qr_add_vertical_wall_gap(env, -0.55f * r, wall,
            qr_rand_range(env, -0.50f * r, -0.15f * r), gap, h);
        qr_add_vertical_wall_gap(env, 0.00f, wall,
            qr_rand_range(env, 0.15f * r, 0.50f * r), gap, h);
        qr_add_vertical_wall_gap(env, 0.55f * r, wall,
            qr_rand_range(env, -0.50f * r, -0.15f * r), gap, h);
        qr_add_horizontal_wall_gap(env, -0.36f * r, wall,
            qr_rand_range(env, -0.10f * r, 0.35f * r), gap, h);
        qr_add_horizontal_wall_gap(env, 0.36f * r, wall,
            qr_rand_range(env, -0.35f * r, 0.10f * r), gap, h);
    } else {
        qr_add_vertical_wall_gap(env, -0.30f * r, wall,
            qr_rand_range(env, 0.12f * r, 0.50f * r), gap, h);
        qr_add_horizontal_wall_gap(env, 0.25f * r, wall,
            qr_rand_range(env, -0.55f * r, -0.18f * r), gap, h);
        int blocks = extra_clutter ? 8 : 5;
        for (int i = 0; i < blocks; i++) {
            for (int attempt = 0; attempt < 24; attempt++) {
                float sx = qr_rand_range(env,
                    (extra_clutter ? 0.16f : 0.20f) * r,
                    (extra_clutter ? 0.30f : 0.34f) * r);
                float sy = qr_rand_range(env,
                    (extra_clutter ? 0.10f : 0.12f) * r,
                    (extra_clutter ? 0.23f : 0.25f) * r);
                float x_margin = 0.35f + 0.5f * sx;
                float y_margin = 0.35f + 0.5f * sy;
                float cx = qr_rand_range(env, -r + x_margin, r - x_margin);
                float cy = qr_rand_range(env, -r + y_margin, r - y_margin);
                if (cx*cx + cy*cy < 0.45f * 0.45f)
                    continue;
                float xmin = cx - 0.5f * sx;
                float xmax = cx + 0.5f * sx;
                float ymin = cy - 0.5f * sy;
                float ymax = cy + 0.5f * sy;
                if (qr_obstacle_overlaps_xy(env, xmin, xmax, ymin, ymax, 0.30f))
                    continue;
                qr_add_obstacle(env, xmin, xmax, ymin, ymax, 0.0f, h);
                break;
            }
        }
    }
}

static inline void qr_configure_procedural_indoor_hard(
        QuadrotorEnv* env, int forced_pattern) {
    qr_configure_procedural_indoor_hard_variant(env, forced_pattern, 0.46f, 0.64f, 0);
}

static inline void qr_configure_procedural_indoor_tight(
        QuadrotorEnv* env, int forced_pattern) {
    qr_configure_procedural_indoor_hard_variant(env, forced_pattern, 0.34f, 0.50f, 1);
}

static inline void qr_add_house_clutter(QuadrotorEnv* env, int count,
        float sx_min, float sx_max, float sy_min, float sy_max,
        int keep_center_hall_clear) {
    float r = env->room_size;
    float h = env->ceiling_height;
    for (int i = 0; i < count; i++) {
        for (int attempt = 0; attempt < 48; attempt++) {
            float sx = qr_rand_range(env, sx_min * r, sx_max * r);
            float sy = qr_rand_range(env, sy_min * r, sy_max * r);
            float x_margin = 0.35f + 0.5f * sx;
            float y_margin = 0.35f + 0.5f * sy;
            float cx = qr_rand_range(env, -r + x_margin, r - x_margin);
            float cy = qr_rand_range(env, -r + y_margin, r - y_margin);
            if (cx*cx + cy*cy < 0.45f * 0.45f)
                continue;
            if (keep_center_hall_clear && fabsf(cy) < 0.18f * r)
                continue;

            float xmin = cx - 0.5f * sx;
            float xmax = cx + 0.5f * sx;
            float ymin = cy - 0.5f * sy;
            float ymax = cy + 0.5f * sy;
            if (qr_obstacle_overlaps_xy(env, xmin, xmax, ymin, ymax, 0.20f))
                continue;
            qr_add_obstacle(env, xmin, xmax, ymin, ymax, 0.0f, h);
            break;
        }
    }
}

static inline void qr_configure_procedural_house(
        QuadrotorEnv* env, int forced_pattern) {
    float r = env->room_size;
    float h = env->ceiling_height;
    float wall = qr_rand_range(env, 0.10f, 0.16f);
    float door = qr_rand_range(env, 0.38f, 0.54f);
    uint32_t choice = forced_pattern >= 0 && forced_pattern < 4
        ? (uint32_t)forced_pattern
        : qr_rand_u32(env) % 4u;
    env->procedural_pattern = (int)choice;

    if (choice == 0u) {
        float top_gaps[3] = {-0.62f * r, 0.05f * r, 0.62f * r};
        float bot_gaps[3] = {-0.55f * r, 0.18f * r, 0.70f * r};
        float halves[3] = {door, door, door};
        qr_add_horizontal_wall_gap_list(
            env, 0.22f * r, wall, -r, r, top_gaps, halves, 3, h);
        qr_add_horizontal_wall_gap_list(
            env, -0.22f * r, wall, -r, r, bot_gaps, halves, 3, h);
        qr_add_obstacle(env, -0.34f * r - 0.5f * wall,
            -0.34f * r + 0.5f * wall, 0.22f * r, r, 0.0f, h);
        qr_add_obstacle(env, 0.34f * r - 0.5f * wall,
            0.34f * r + 0.5f * wall, 0.22f * r, r, 0.0f, h);
        qr_add_obstacle(env, -0.38f * r - 0.5f * wall,
            -0.38f * r + 0.5f * wall, -r, -0.22f * r, 0.0f, h);
        qr_add_obstacle(env, 0.28f * r - 0.5f * wall,
            0.28f * r + 0.5f * wall, -r, -0.22f * r, 0.0f, h);
        qr_add_house_clutter(env, 5, 0.05f, 0.12f, 0.05f, 0.12f, 1);
    } else if (choice == 1u) {
        float hall_gaps[3] = {-0.58f * r, 0.00f, 0.58f * r};
        float halves[3] = {door, door, door};
        qr_add_vertical_wall_gap_list(
            env, -0.45f * r, wall, -r, r, hall_gaps, halves, 3, h);
        qr_add_obstacle(env, -0.45f * r, r,
            -0.34f * r - 0.5f * wall, -0.34f * r + 0.5f * wall, 0.0f, h);
        qr_add_obstacle(env, -0.45f * r, r,
            0.34f * r - 0.5f * wall, 0.34f * r + 0.5f * wall, 0.0f, h);
        qr_add_obstacle(env, 0.22f * r - 0.5f * wall,
            0.22f * r + 0.5f * wall, -r, -0.34f * r, 0.0f, h);
        qr_add_obstacle(env, 0.22f * r - 0.5f * wall,
            0.22f * r + 0.5f * wall, 0.34f * r, r, 0.0f, h);
        qr_add_obstacle(env, -r, -0.72f * r,
            -0.12f * r, 0.12f * r, 0.0f, h);
        qr_add_house_clutter(env, 6, 0.04f, 0.11f, 0.04f, 0.12f, 0);
    } else if (choice == 2u) {
        float side_gaps_a[2] = {-0.62f * r, 0.45f * r};
        float side_gaps_b[2] = {-0.45f * r, 0.62f * r};
        float cross_gaps_a[2] = {-0.55f * r, 0.35f * r};
        float cross_gaps_b[2] = {-0.35f * r, 0.55f * r};
        float halves2[2] = {door, door};
        qr_add_vertical_wall_gap_list(
            env, -0.55f * r, wall, -r, r, side_gaps_a, halves2, 2, h);
        qr_add_vertical_wall_gap_list(
            env, 0.55f * r, wall, -r, r, side_gaps_b, halves2, 2, h);
        qr_add_horizontal_wall_gap_list(
            env, -0.55f * r, wall, -r, r, cross_gaps_a, halves2, 2, h);
        qr_add_horizontal_wall_gap_list(
            env, 0.55f * r, wall, -r, r, cross_gaps_b, halves2, 2, h);
        qr_add_obstacle(env, -0.16f * r, 0.16f * r,
            -0.16f * r, 0.16f * r, 0.0f, h);
        qr_add_house_clutter(env, 4, 0.05f, 0.12f, 0.05f, 0.12f, 0);
    } else {
        float h_gaps[2] = {-0.52f * r, 0.45f * r};
        float v_gaps[2] = {-0.42f * r, 0.52f * r};
        float halves2[2] = {door, door};
        qr_add_horizontal_wall_gap_list(
            env, 0.08f * r, wall, -r, r, h_gaps, halves2, 2, h);
        qr_add_vertical_wall_gap_list(
            env, -0.12f * r, wall, -r, r, v_gaps, halves2, 2, h);
        qr_add_obstacle(env, 0.28f * r, 0.52f * r,
            -0.18f * r, 0.05f * r, 0.0f, h);
        qr_add_obstacle(env, -0.72f * r, -0.50f * r,
            0.48f * r, 0.72f * r, 0.0f, h);
        qr_add_house_clutter(env, 9, 0.04f, 0.12f, 0.04f, 0.13f, 0);
    }
}

static inline void qr_configure_procedural_house_multizone(
        QuadrotorEnv* env, int forced_pattern) {
    float r = env->room_size;
    float h = env->ceiling_height;
    float wall = qr_rand_range(env, 0.10f, 0.15f);
    float door = qr_rand_range(env, 0.42f, 0.56f);
    uint32_t choice = forced_pattern >= 0 && forced_pattern < 4
        ? (uint32_t)forced_pattern
        : qr_rand_u32(env) % 4u;
    env->procedural_pattern = (int)choice;

    if (choice == 0u) {
        float halves3[3] = {door, door, door};
        float v0[3] = {-0.70f * r, -0.05f * r, 0.62f * r};
        float v2[3] = {-0.62f * r, 0.06f * r, 0.70f * r};
        float h0[3] = {-0.68f * r, -0.05f * r, 0.54f * r};
        float h1[3] = {-0.54f * r, 0.12f * r, 0.68f * r};
        qr_add_vertical_wall_gap_list(env, -0.50f * r, wall, -r, r, v0, halves3, 3, h);
        qr_add_vertical_wall_gap_list(env, 0.50f * r, wall, -r, r, v2, halves3, 3, h);
        qr_add_horizontal_wall_gap_list(env, -0.38f * r, wall, -r, r, h0, halves3, 3, h);
        qr_add_horizontal_wall_gap_list(env, 0.38f * r, wall, -r, r, h1, halves3, 3, h);
        qr_add_house_clutter(env, 4, 0.030f, 0.070f, 0.030f, 0.070f, 0);
    } else if (choice == 1u) {
        for (int i = 0; i < 5; i++) {
            float y = (-0.60f + 0.30f * (float)i) * r;
            float sign = (i % 2) == 0 ? 1.0f : -1.0f;
            float gap = sign * qr_rand_range(env, 0.48f * r, 0.76f * r);
            qr_add_horizontal_wall_gap(env, y, wall, gap, door, h);
        }
        qr_add_obstacle(env, -0.18f * r - 0.5f * wall,
            -0.18f * r + 0.5f * wall, -r, -0.72f * r, 0.0f, h);
        qr_add_obstacle(env, 0.18f * r - 0.5f * wall,
            0.18f * r + 0.5f * wall, 0.72f * r, r, 0.0f, h);
        qr_add_obstacle(env, -r, -0.76f * r,
            -0.08f * r, 0.08f * r, 0.0f, h);
        qr_add_obstacle(env, 0.76f * r, r,
            -0.08f * r, 0.08f * r, 0.0f, h);
        qr_add_house_clutter(env, 5, 0.030f, 0.070f, 0.030f, 0.070f, 0);
    } else if (choice == 2u) {
        float halves2[2] = {door, door};
        float left_v[2] = {-0.62f * r, 0.25f * r};
        float mid_v[2] = {-0.25f * r, 0.62f * r};
        float right_v[2] = {-0.62f * r, 0.25f * r};
        float low_h[2] = {-0.58f * r, 0.22f * r};
        float mid_h[2] = {-0.22f * r, 0.58f * r};
        float high_h[2] = {-0.58f * r, 0.22f * r};
        qr_add_vertical_wall_gap_list(env, -0.58f * r, wall, -r, r, left_v, halves2, 2, h);
        qr_add_vertical_wall_gap_list(env, 0.00f, wall, -r, r, mid_v, halves2, 2, h);
        qr_add_vertical_wall_gap_list(env, 0.58f * r, wall, -r, r, right_v, halves2, 2, h);
        qr_add_horizontal_wall_gap_list(env, -0.58f * r, wall, -r, r, low_h, halves2, 2, h);
        qr_add_horizontal_wall_gap_list(env, 0.00f, wall, -r, r, mid_h, halves2, 2, h);
        qr_add_horizontal_wall_gap_list(env, 0.58f * r, wall, -r, r, high_h, halves2, 2, h);
        qr_add_obstacle(env, -0.18f * r, 0.18f * r,
            -0.18f * r, 0.18f * r, 0.0f, h);
        qr_add_obstacle(env, -0.88f * r, -0.70f * r,
            0.18f * r, 0.46f * r, 0.0f, h);
        qr_add_obstacle(env, 0.70f * r, 0.88f * r,
            -0.46f * r, -0.18f * r, 0.0f, h);
        qr_add_house_clutter(env, 4, 0.030f, 0.070f, 0.030f, 0.070f, 0);
    } else {
        float halves2[2] = {door, door};
        float west_gaps[2] = {-0.55f * r, 0.42f * r};
        float east_gaps[2] = {-0.42f * r, 0.55f * r};
        qr_add_vertical_wall_gap_list(env, -0.36f * r, wall, -r, r, west_gaps, halves2, 2, h);
        qr_add_vertical_wall_gap_list(env, 0.36f * r, wall, -r, r, east_gaps, halves2, 2, h);
        qr_add_obstacle(env, -r, -0.36f * r,
            -0.18f * r - 0.5f * wall, -0.18f * r + 0.5f * wall, 0.0f, h);
        qr_add_obstacle(env, -r, -0.36f * r,
            0.50f * r - 0.5f * wall, 0.50f * r + 0.5f * wall, 0.0f, h);
        qr_add_obstacle(env, 0.36f * r, r,
            -0.50f * r - 0.5f * wall, -0.50f * r + 0.5f * wall, 0.0f, h);
        qr_add_obstacle(env, 0.36f * r, r,
            0.18f * r - 0.5f * wall, 0.18f * r + 0.5f * wall, 0.0f, h);
        qr_add_obstacle(env, -0.12f * r, 0.12f * r,
            -0.72f * r, -0.28f * r, 0.0f, h);
        qr_add_obstacle(env, -0.12f * r, 0.12f * r,
            0.28f * r, 0.72f * r, 0.0f, h);
        qr_add_obstacle(env, -0.70f * r, -0.52f * r,
            -0.76f * r, -0.52f * r, 0.0f, h);
        qr_add_obstacle(env, 0.52f * r, 0.70f * r,
            0.52f * r, 0.76f * r, 0.0f, h);
        qr_add_house_clutter(env, 7, 0.030f, 0.065f, 0.030f, 0.065f, 0);
    }
}

static inline void qr_configure_gate_slalom(
        QuadrotorEnv* env, int forced_pattern) {
    float r = env->room_size;
    float h = env->ceiling_height;
    float wall = qr_rand_range(env, 0.10f, 0.16f);
    uint32_t choice = forced_pattern >= 0 && forced_pattern < 4
        ? (uint32_t)forced_pattern
        : qr_rand_u32(env) % 4u;
    env->procedural_pattern = (int)choice;

    if (choice == 0u) {
        float gap = qr_rand_range(env, 0.18f * r, 0.22f * r);
        qr_add_vertical_wall_gap(env, -0.52f * r, wall,
            -0.34f * r, gap, h);
        qr_add_vertical_wall_gap(env, 0.00f, wall,
            0.34f * r, gap, h);
        qr_add_vertical_wall_gap(env, 0.52f * r, wall,
            -0.34f * r, gap, h);
    } else if (choice == 1u) {
        for (int i = 0; i < 8; i++) {
            float fx = -0.72f + 0.205f * (float)i;
            float side = (i % 2) == 0 ? -1.0f : 1.0f;
            float cx = fx * r + qr_rand_range(env, -0.025f * r, 0.025f * r);
            float cy = side * qr_rand_range(env, 0.18f * r, 0.32f * r);
            float sx = qr_rand_range(env, 0.055f * r, 0.085f * r);
            float sy = qr_rand_range(env, 0.055f * r, 0.085f * r);
            qr_add_obstacle(env, cx - sx, cx + sx, cy - sy, cy + sy, 0.0f, h);
        }
    } else if (choice == 2u) {
        float half = qr_rand_range(env, 0.16f * r, 0.20f * r);
        float gap_x[2] = {-0.42f * r, 0.42f * r};
        float gap_y[2] = {-0.42f * r, 0.42f * r};
        float halves[2] = {half, half};
        qr_add_vertical_wall_gap_list(env, 0.00f, wall, -r, r,
            gap_y, halves, 2, h);
        qr_add_horizontal_wall_gap_list(env, 0.00f, wall, -r, r,
            gap_x, halves, 2, h);
        qr_add_obstacle(env, -0.78f * r, -0.62f * r,
            -0.78f * r, -0.58f * r, 0.0f, h);
        qr_add_obstacle(env, 0.62f * r, 0.78f * r,
            0.58f * r, 0.78f * r, 0.0f, h);
    } else {
        for (int i = 0; i < 5; i++) {
            float fy = -0.66f + 0.33f * (float)i;
            float y0 = fy * r - 0.5f * wall;
            float y1 = fy * r + 0.5f * wall;
            if ((i % 2) == 0) {
                qr_add_obstacle(env, -r, 0.18f * r, y0, y1, 0.0f, h);
            } else {
                qr_add_obstacle(env, -0.18f * r, r, y0, y1, 0.0f, h);
            }
        }
        qr_add_obstacle(env, -0.09f * r, 0.09f * r,
            -0.09f * r, 0.09f * r, 0.0f, h);
    }
}

static inline void qr_configure_agile_geometry(
        QuadrotorEnv* env, int forced_pattern) {
    float r = env->room_size;
    float h = env->ceiling_height;
    float wall = qr_rand_range(env, 0.08f, 0.14f);
    uint32_t choice = forced_pattern >= 0 && forced_pattern < 4
        ? (uint32_t)forced_pattern
        : qr_rand_u32(env) % 4u;
    env->procedural_pattern = (int)choice;

    if (choice == 0u) {
        float half = qr_rand_range(env, 0.22f * r, 0.27f * r);
        qr_add_vertical_wall_gap(env, -0.54f * r, wall,
            -0.26f * r, half, h);
        qr_add_vertical_wall_gap(env, 0.00f, wall,
            0.30f * r, half, h);
        qr_add_vertical_wall_gap(env, 0.54f * r, wall,
            -0.18f * r, half, h);
        qr_add_obstacle(env, -0.86f * r, -0.74f * r,
            0.34f * r, 0.58f * r, 0.0f, h);
        qr_add_obstacle(env, 0.72f * r, 0.86f * r,
            -0.58f * r, -0.34f * r, 0.0f, h);
    } else if (choice == 1u) {
        for (int i = 0; i < 10; i++) {
            float fx = -0.78f + 0.173f * (float)i;
            float side = (i % 2) == 0 ? -1.0f : 1.0f;
            float cx = fx * r + qr_rand_range(env, -0.018f * r, 0.018f * r);
            float cy = side * qr_rand_range(env, 0.18f * r, 0.31f * r);
            float sx = qr_rand_range(env, 0.040f * r, 0.065f * r);
            float sy = qr_rand_range(env, 0.070f * r, 0.105f * r);
            qr_add_obstacle(env, cx - sx, cx + sx, cy - sy, cy + sy, 0.0f, h);
        }
        qr_add_obstacle(env, -0.18f * r, 0.18f * r,
            -0.055f * r, 0.055f * r, 0.0f, h);
    } else if (choice == 2u) {
        qr_add_obstacle(env, -0.20f * r, 0.20f * r,
            -0.16f * r, 0.16f * r, 0.0f, h);
        qr_add_obstacle(env, -0.76f * r, -0.42f * r,
            -0.58f * r, -0.42f * r, 0.0f, h);
        qr_add_obstacle(env, 0.42f * r, 0.76f * r,
            0.42f * r, 0.58f * r, 0.0f, h);
        qr_add_obstacle(env, -0.86f * r, -0.70f * r,
            0.12f * r, 0.42f * r, 0.0f, h);
        qr_add_obstacle(env, 0.70f * r, 0.86f * r,
            -0.42f * r, -0.12f * r, 0.0f, h);
        qr_add_horizontal_wall_gap(env, -0.55f * r, wall,
            0.20f * r, qr_rand_range(env, 0.22f * r, 0.28f * r), h);
        qr_add_horizontal_wall_gap(env, 0.55f * r, wall,
            -0.20f * r, qr_rand_range(env, 0.22f * r, 0.28f * r), h);
    } else {
        qr_add_obstacle(env, -r, -0.24f * r,
            -0.64f * r, -0.52f * r, 0.0f, h);
        qr_add_obstacle(env, 0.24f * r, r,
            -0.18f * r, -0.06f * r, 0.0f, h);
        qr_add_obstacle(env, -r, -0.24f * r,
            0.24f * r, 0.36f * r, 0.0f, h);
        qr_add_obstacle(env, 0.24f * r, r,
            0.62f * r, 0.74f * r, 0.0f, h);
        qr_add_obstacle(env, -0.065f * r, 0.065f * r,
            -0.74f * r, -0.34f * r, 0.0f, h);
        qr_add_obstacle(env, -0.065f * r, 0.065f * r,
            0.34f * r, 0.74f * r, 0.0f, h);
        qr_add_obstacle(env, -0.43f * r, -0.28f * r,
            -0.02f * r, 0.18f * r, 0.0f, h);
        qr_add_obstacle(env, 0.28f * r, 0.43f * r,
            -0.18f * r, 0.02f * r, 0.0f, h);
    }
}

static inline void qr_configure_agile_pillar_weave(
        QuadrotorEnv* env, int level, int center_marker) {
    float r = env->room_size;
    float h = env->ceiling_height;
    if (level < 0) level = 0;
    if (level > 3) level = 3;
    env->procedural_pattern = level;

    int count = level == 0 ? 8 : level == 1 ? 9 : 10;
    float y_min = level <= 1 ? 0.20f * r : 0.18f * r;
    float y_max = level == 0 ? 0.30f * r : level == 1 ? 0.32f * r : 0.31f * r;
    float sx_min = (0.055f - 0.005f * (float)level) * r;
    float sx_max = (0.082f - 0.005f * (float)level) * r;
    float sy_min = (0.055f + 0.005f * (float)level) * r;
    float sy_max = (0.082f + 0.008f * (float)level) * r;
    if (level == 3) {
        sx_min = 0.040f * r;
        sx_max = 0.065f * r;
        sy_min = 0.070f * r;
        sy_max = 0.105f * r;
    }

    for (int i = 0; i < count; i++) {
        float t = count == 1 ? 0.0f : (float)i / (float)(count - 1);
        float fx = -0.72f + 1.44f * t;
        float side = (i % 2) == 0 ? -1.0f : 1.0f;
        float jitter = (0.026f - 0.003f * (float)level) * r;
        if (jitter < 0.012f * r)
            jitter = 0.012f * r;
        float cx = fx * r + qr_rand_range(env, -jitter, jitter);
        float cy = side * qr_rand_range(env, y_min, y_max);
        float sx = qr_rand_range(env, sx_min, sx_max);
        float sy = qr_rand_range(env, sy_min, sy_max);
        qr_add_obstacle(env, cx - sx, cx + sx, cy - sy, cy + sy, 0.0f, h);
    }

    if (center_marker < 0)
        center_marker = level >= 2 ? level : 0;

    if (center_marker >= 3) {
        qr_add_obstacle(env, -0.18f * r, 0.18f * r,
            -0.055f * r, 0.055f * r, 0.0f, h);
    } else if (center_marker == 2) {
        qr_add_obstacle(env, -0.10f * r, 0.10f * r,
            -0.040f * r, 0.040f * r, 0.0f, h);
    } else if (center_marker == 1) {
        qr_add_obstacle(env, -0.055f * r, 0.055f * r,
            -0.026f * r, 0.026f * r, 0.0f, h);
    }
}

static inline void qr_configure_obstacles(QuadrotorEnv* env) {
    env->num_obstacles = 0;
    env->procedural_pattern = -1;
    int obstacle_mode = env->obstacle_mode;
    int forced_procedural_pattern = -1;
    int forced_pillar_center_marker = -1;
    if (obstacle_mode == QR_OBSTACLES_MIXED_OPEN_BLOCKS)
        obstacle_mode = qr_rand01(env) < env->obstacle_block_prob
            ? QR_OBSTACLES_RANDOM_BLOCKS
            : QR_OBSTACLES_NONE;
    else if (obstacle_mode == QR_OBSTACLES_LAYOUT_MIXED) {
        uint32_t choice = qr_rand_u32(env) % 4u;
        obstacle_mode = choice == 0u ? QR_OBSTACLES_CENTER_BLOCK
            : choice == 1u ? QR_OBSTACLES_DOORWAY
            : choice == 2u ? QR_OBSTACLES_CORRIDOR
            : QR_OBSTACLES_MAZE;
    } else if (obstacle_mode >= QR_OBSTACLES_PROCEDURAL_INDOOR_P0 &&
            obstacle_mode <= QR_OBSTACLES_PROCEDURAL_INDOOR_P3) {
        forced_procedural_pattern = obstacle_mode - QR_OBSTACLES_PROCEDURAL_INDOOR_P0;
        obstacle_mode = QR_OBSTACLES_PROCEDURAL_INDOOR;
    } else if (obstacle_mode >= QR_OBSTACLES_PROCEDURAL_INDOOR_HARD_P0 &&
            obstacle_mode <= QR_OBSTACLES_PROCEDURAL_INDOOR_HARD_P3) {
        forced_procedural_pattern = obstacle_mode - QR_OBSTACLES_PROCEDURAL_INDOOR_HARD_P0;
        obstacle_mode = QR_OBSTACLES_PROCEDURAL_INDOOR_HARD;
    } else if (obstacle_mode >= QR_OBSTACLES_PROCEDURAL_INDOOR_TIGHT_P0 &&
            obstacle_mode <= QR_OBSTACLES_PROCEDURAL_INDOOR_TIGHT_P3) {
        forced_procedural_pattern = obstacle_mode - QR_OBSTACLES_PROCEDURAL_INDOOR_TIGHT_P0;
        obstacle_mode = QR_OBSTACLES_PROCEDURAL_INDOOR_TIGHT;
    } else if (obstacle_mode >= QR_OBSTACLES_PROCEDURAL_HOUSE_P0 &&
            obstacle_mode <= QR_OBSTACLES_PROCEDURAL_HOUSE_P3) {
        forced_procedural_pattern = obstacle_mode - QR_OBSTACLES_PROCEDURAL_HOUSE_P0;
        obstacle_mode = QR_OBSTACLES_PROCEDURAL_HOUSE;
    } else if (obstacle_mode >= QR_OBSTACLES_PROCEDURAL_HOUSE_MULTIZONE_P0 &&
            obstacle_mode <= QR_OBSTACLES_PROCEDURAL_HOUSE_MULTIZONE_P3) {
        forced_procedural_pattern =
            obstacle_mode - QR_OBSTACLES_PROCEDURAL_HOUSE_MULTIZONE_P0;
        obstacle_mode = QR_OBSTACLES_PROCEDURAL_HOUSE_MULTIZONE;
    } else if (obstacle_mode >= QR_OBSTACLES_GATE_SLALOM_P0 &&
            obstacle_mode <= QR_OBSTACLES_GATE_SLALOM_P3) {
        forced_procedural_pattern = obstacle_mode - QR_OBSTACLES_GATE_SLALOM_P0;
        obstacle_mode = QR_OBSTACLES_GATE_SLALOM;
    } else if (obstacle_mode == QR_OBSTACLES_GATE_SLALOM_EASY_MIX) {
        uint32_t choice = qr_rand_u32(env) % 10u;
        forced_procedural_pattern = choice < 7u ? 1 : choice < 9u ? 0 : 3;
        obstacle_mode = QR_OBSTACLES_GATE_SLALOM;
    } else if (obstacle_mode == QR_OBSTACLES_GATE_SLALOM_P1_P0_MIX) {
        uint32_t choice = qr_rand_u32(env) % 10u;
        forced_procedural_pattern = choice < 9u ? 1 : 0;
        obstacle_mode = QR_OBSTACLES_GATE_SLALOM;
    } else if (obstacle_mode >= QR_OBSTACLES_AGILE_GEOMETRY_P0 &&
            obstacle_mode <= QR_OBSTACLES_AGILE_GEOMETRY_P3) {
        forced_procedural_pattern = obstacle_mode - QR_OBSTACLES_AGILE_GEOMETRY_P0;
        obstacle_mode = QR_OBSTACLES_AGILE_GEOMETRY;
    } else if (obstacle_mode == QR_OBSTACLES_AGILE_GEOMETRY_INTRO) {
        uint32_t choice = qr_rand_u32(env) % 10u;
        forced_procedural_pattern = choice < 6u ? 1 : choice < 9u ? 0 : 2;
        obstacle_mode = QR_OBSTACLES_AGILE_GEOMETRY;
    } else if (obstacle_mode >= QR_OBSTACLES_AGILE_PILLAR_L0 &&
            obstacle_mode <= QR_OBSTACLES_AGILE_PILLAR_L3) {
        forced_procedural_pattern = obstacle_mode - QR_OBSTACLES_AGILE_PILLAR_L0;
        obstacle_mode = QR_OBSTACLES_AGILE_PILLAR_WEAVE;
    } else if (obstacle_mode == QR_OBSTACLES_AGILE_PILLAR_EASY_MIX) {
        uint32_t choice = qr_rand_u32(env) % 10u;
        forced_procedural_pattern = choice < 7u ? 0 : choice < 9u ? 1 : 2;
        obstacle_mode = QR_OBSTACLES_AGILE_PILLAR_WEAVE;
    } else if (obstacle_mode == QR_OBSTACLES_AGILE_PILLAR_CURRICULUM) {
        uint32_t choice = qr_rand_u32(env) % 20u;
        forced_procedural_pattern = choice < 9u ? 0
            : choice < 15u ? 1
            : choice < 19u ? 2
            : 3;
        obstacle_mode = QR_OBSTACLES_AGILE_PILLAR_WEAVE;
    } else if (obstacle_mode == QR_OBSTACLES_AGILE_PILLAR_L2_OPEN) {
        forced_procedural_pattern = 2;
        forced_pillar_center_marker = 0;
        obstacle_mode = QR_OBSTACLES_AGILE_PILLAR_WEAVE;
    } else if (obstacle_mode >= QR_OBSTACLES_AGILE_PILLAR_L2_MARKER_TINY &&
            obstacle_mode <= QR_OBSTACLES_AGILE_PILLAR_L2_MARKER_LARGE) {
        forced_procedural_pattern = 2;
        forced_pillar_center_marker =
            obstacle_mode - QR_OBSTACLES_AGILE_PILLAR_L2_MARKER_TINY + 1;
        obstacle_mode = QR_OBSTACLES_AGILE_PILLAR_WEAVE;
    } else if (obstacle_mode == QR_OBSTACLES_AGILE_PILLAR_BOTTLENECK_CURRICULUM) {
        uint32_t choice = qr_rand_u32(env) % 16u;
        forced_procedural_pattern = 2;
        forced_pillar_center_marker = choice < 4u ? 0
            : choice < 10u ? 1
            : choice < 15u ? 2
            : 3;
        obstacle_mode = QR_OBSTACLES_AGILE_PILLAR_WEAVE;
    }
    env->active_obstacle_mode = obstacle_mode;

    if (env->room_size <= 0.0f || obstacle_mode == QR_OBSTACLES_NONE)
        return;

    float r = env->room_size;
    float h = env->ceiling_height;
    if (obstacle_mode == QR_OBSTACLES_CENTER_BLOCK) {
        qr_add_obstacle(env, -0.35f, 0.35f, -0.35f, 0.35f, 0.0f, h);
    } else if (obstacle_mode == QR_OBSTACLES_DOORWAY) {
        qr_add_obstacle(env, -r, -0.45f, -0.08f, 0.08f, 0.0f, h);
        qr_add_obstacle(env, 0.45f, r, -0.08f, 0.08f, 0.0f, h);
    } else if (obstacle_mode == QR_OBSTACLES_CORRIDOR) {
        qr_add_obstacle(env, -r, r, 0.45f, r, 0.0f, h);
        qr_add_obstacle(env, -r, r, -r, -0.45f, 0.0f, h);
    } else if (obstacle_mode == QR_OBSTACLES_MAZE) {
        qr_add_obstacle(env, -0.08f, 0.08f, -r, 0.0f, 0.0f, h);
        qr_add_obstacle(env, -0.55f, r, 0.80f, 0.95f, 0.0f, h);
        qr_add_obstacle(env, -r, -1.35f, -0.35f, r, 0.0f, h);
    } else if (obstacle_mode == QR_OBSTACLES_RANDOM_BLOCKS) {
        int target_count = 2 + (int)(qr_rand_u32(env) % 4u);
        for (int i = 0; i < target_count; i++) {
            for (int attempt = 0; attempt < 32; attempt++) {
                float sx = qr_rand_range(env, 0.25f, 0.65f);
                float sy = qr_rand_range(env, 0.25f, 0.65f);
                float x_margin = 0.25f + 0.5f * sx;
                float y_margin = 0.25f + 0.5f * sy;
                if (x_margin >= r || y_margin >= r)
                    break;

                float cx = qr_rand_range(env, -r + x_margin, r - x_margin);
                float cy = qr_rand_range(env, -r + y_margin, r - y_margin);
                if (cx*cx + cy*cy < 0.35f * 0.35f)
                    continue;

                float xmin = cx - 0.5f * sx;
                float xmax = cx + 0.5f * sx;
                float ymin = cy - 0.5f * sy;
                float ymax = cy + 0.5f * sy;
                if (qr_obstacle_overlaps_xy(env, xmin, xmax, ymin, ymax, 0.25f))
                    continue;

                qr_add_obstacle(env, xmin, xmax, ymin, ymax, 0.0f, h);
                break;
            }
        }
    } else if (obstacle_mode == QR_OBSTACLES_PROCEDURAL_INDOOR) {
        qr_configure_procedural_indoor(env, forced_procedural_pattern);
    } else if (obstacle_mode == QR_OBSTACLES_PROCEDURAL_INDOOR_HARD) {
        qr_configure_procedural_indoor_hard(env, forced_procedural_pattern);
    } else if (obstacle_mode == QR_OBSTACLES_PROCEDURAL_INDOOR_TIGHT) {
        qr_configure_procedural_indoor_tight(env, forced_procedural_pattern);
    } else if (obstacle_mode == QR_OBSTACLES_PROCEDURAL_HOUSE) {
        qr_configure_procedural_house(env, forced_procedural_pattern);
    } else if (obstacle_mode == QR_OBSTACLES_PROCEDURAL_HOUSE_MULTIZONE) {
        qr_configure_procedural_house_multizone(env, forced_procedural_pattern);
    } else if (obstacle_mode == QR_OBSTACLES_GATE_SLALOM) {
        qr_configure_gate_slalom(env, forced_procedural_pattern);
    } else if (obstacle_mode == QR_OBSTACLES_AGILE_GEOMETRY) {
        qr_configure_agile_geometry(env, forced_procedural_pattern);
    } else if (obstacle_mode == QR_OBSTACLES_AGILE_PILLAR_WEAVE) {
        qr_configure_agile_pillar_weave(
            env, forced_procedural_pattern, forced_pillar_center_marker);
    }
}

static inline void qr_apply_dynamics_scales(QuadrotorEnv* env) {
    Quadrotor* q = &env->phys;
    q->mass *= env->current_mass_scale;
    for (int i = 0; i < 3; i++) {
        q->inertia[i] *= env->current_inertia_scale;
        q->c_drag[i] *= env->current_drag_scale;
    }
    q->c_thrust *= env->current_thrust_scale;
    q->c_torque *= env->current_torque_scale;
    q->motor_tau *= env->current_motor_tau_scale;
}

static inline float qr_sample_dynamics_scale(
        QuadrotorEnv* env, float fixed, float lo, float hi) {
    if (lo <= 0.0f && hi <= 0.0f)
        return fixed;
    if (lo <= 0.0f) lo = fixed;
    if (hi <= 0.0f) hi = fixed;
    if (hi < lo) {
        float tmp = lo;
        lo = hi;
        hi = tmp;
    }
    if (hi == lo)
        return lo;
    return qr_rand_range(env, lo, hi);
}

static inline void qr_sample_dynamics_scales(QuadrotorEnv* env) {
    env->current_mass_scale = qr_sample_dynamics_scale(
        env, env->mass_scale, env->mass_scale_min, env->mass_scale_max);

    if (env->inertia_scale <= 0.0f &&
            env->inertia_scale_min <= 0.0f &&
            env->inertia_scale_max <= 0.0f) {
        env->current_inertia_scale = env->current_mass_scale;
    } else {
        float fixed = env->inertia_scale > 0.0f
            ? env->inertia_scale
            : env->current_mass_scale;
        env->current_inertia_scale = qr_sample_dynamics_scale(
            env, fixed, env->inertia_scale_min, env->inertia_scale_max);
    }

    env->current_thrust_scale = qr_sample_dynamics_scale(
        env, env->thrust_scale, env->thrust_scale_min, env->thrust_scale_max);

    if (env->torque_scale <= 0.0f &&
            env->torque_scale_min <= 0.0f &&
            env->torque_scale_max <= 0.0f) {
        env->current_torque_scale = env->current_thrust_scale;
    } else {
        float fixed = env->torque_scale > 0.0f
            ? env->torque_scale
            : env->current_thrust_scale;
        env->current_torque_scale = qr_sample_dynamics_scale(
            env, fixed, env->torque_scale_min, env->torque_scale_max);
    }

    env->current_drag_scale = qr_sample_dynamics_scale(
        env, env->drag_scale, env->drag_scale_min, env->drag_scale_max);
    env->current_motor_tau_scale = qr_sample_dynamics_scale(
        env, env->motor_tau_scale,
        env->motor_tau_scale_min, env->motor_tau_scale_max);
}

static inline void qr_configure_dynamics(QuadrotorEnv* env) {
    qr_defaults(&env->phys);
    if (env->dynamics_profile == QR_DYNAMICS_PROFILE_DRONE_REFERENCE)
        qr_drone_reference_defaults(&env->phys);
    qr_sample_dynamics_scales(env);
    qr_apply_dynamics_scales(env);
}

static inline void init(QuadrotorEnv* env) {
    memset(&env->log, 0, sizeof(Log));
    qr_seed_rng(env, env->seed);
    qr_defaults(&env->phys);
    qr_defaults(&env->ctrl_model);
    qr_rate_defaults(&env->rate_ctrl);
    qr_pos_defaults(&env->vel_ctrl);
    if (env->max_steps <= 0) env->max_steps = QR_MAX_STEPS;
    if (env->action_repeat <= 0) env->action_repeat = QR_ACTION_REPEAT;
    if (env->max_rate <= 0.0f) env->max_rate = QR_MAX_RATE;
    if (env->max_vel <= 0.0f) env->max_vel = QR_MAX_VEL;
    if (env->max_z_vel <= 0.0f) env->max_z_vel = QR_MAX_Z_VEL;
    if (env->dynamics_profile != QR_DYNAMICS_PROFILE_CUSTOM &&
            env->dynamics_profile != QR_DYNAMICS_PROFILE_DRONE_REFERENCE)
        env->dynamics_profile = QR_DYNAMICS_PROFILE_CUSTOM;
    if (env->physics_dt <= 0.0f) env->physics_dt = QR_DT;
    if (env->physics_dt > QR_DT) env->physics_dt = QR_DT;
    if (env->physics_substeps <= 0) env->physics_substeps = 1;
    if (env->physics_substeps > 64) env->physics_substeps = 64;
    env->vel_ctrl.max_vel = env->max_vel > env->max_z_vel ? env->max_vel : env->max_z_vel;
    if (env->vel_ctrl_kp_vel > 0.0f)
        env->vel_ctrl.kp_vel = env->vel_ctrl_kp_vel;
    if (env->vel_ctrl_ki_z > 0.0f)
        env->vel_ctrl.ki_vel_z = env->vel_ctrl_ki_z;
    if (env->vel_ctrl_int_limit_z > 0.0f)
        env->vel_ctrl.int_limit_z = env->vel_ctrl_int_limit_z;
    if (env->vel_ctrl_max_z_acc > 0.0f)
        env->vel_ctrl.max_z_acc = env->vel_ctrl_max_z_acc;
    if (env->spawn_radius < 0.0f) env->spawn_radius = QR_SPAWN_RADIUS;
    if (env->spawn_radius_min < 0.0f || env->spawn_radius_min > env->spawn_radius)
        env->spawn_radius_min = env->spawn_radius;
    env->episode_spawn_radius = env->spawn_radius;
    if (env->spawn_height <= 0.0f) env->spawn_height = QR_SPAWN_HEIGHT;
    if (env->init_vel_radius < 0.0f) env->init_vel_radius = QR_INIT_VEL_RADIUS;
    if (env->init_omega_radius < 0.0f) env->init_omega_radius = QR_INIT_OMEGA_RADIUS;
    if (env->room_size < 0.0f) env->room_size = QR_ROOM_SIZE;
    if (env->ceiling_height <= 0.0f) env->ceiling_height = QR_CEILING_HEIGHT;
    if (env->ray_length <= 0.0f) env->ray_length = QR_RAY_LENGTH;
    if (env->num_rays < 0) env->num_rays = 0;
    if (env->num_rays > QR_MAX_RAYS) env->num_rays = QR_MAX_RAYS;
    env->camera_depth_obs = env->camera_depth_obs ? 1 : 0;
    if (env->camera_depth_width <= 0)
        env->camera_depth_width = QR_DEPTH_CAMERA_DEFAULT_WIDTH;
    if (env->camera_depth_height <= 0)
        env->camera_depth_height = QR_DEPTH_CAMERA_DEFAULT_HEIGHT;
    if (env->camera_depth_width > QR_DEPTH_CAMERA_MAX_WIDTH)
        env->camera_depth_width = QR_DEPTH_CAMERA_MAX_WIDTH;
    if (env->camera_depth_height > QR_DEPTH_CAMERA_MAX_HEIGHT)
        env->camera_depth_height = QR_DEPTH_CAMERA_MAX_HEIGHT;
    if (env->camera_depth_fov_x_deg <= 0.0f)
        env->camera_depth_fov_x_deg = 90.0f;
    if (env->camera_depth_fov_y_deg <= 0.0f)
        env->camera_depth_fov_y_deg = 60.0f;
    if (env->camera_depth_fov_x_deg > 170.0f)
        env->camera_depth_fov_x_deg = 170.0f;
    if (env->camera_depth_fov_y_deg > 150.0f)
        env->camera_depth_fov_y_deg = 150.0f;
    if (env->hover_radius <= 0.0f) env->hover_radius = QR_HOVER_RADIUS;
    if (env->drone_alpha_dist < 0.0f) env->drone_alpha_dist = QR_DRONE_ALPHA_DIST;
    if (env->drone_alpha_hover < 0.0f) env->drone_alpha_hover = QR_DRONE_ALPHA_HOVER;
    if (env->drone_alpha_omega < 0.0f) env->drone_alpha_omega = QR_DRONE_ALPHA_OMEGA;
    if (env->drone_alpha_shaping < 0.0f) env->drone_alpha_shaping = QR_DRONE_ALPHA_SHAPING;
    if (env->drone_hover_target_dist <= 0.0f)
        env->drone_hover_target_dist = QR_DRONE_HOVER_TARGET_DIST;
    if (env->drone_hover_dist <= 0.0f) env->drone_hover_dist = QR_DRONE_HOVER_DIST;
    if (env->drone_hover_omega <= 0.0f) env->drone_hover_omega = QR_DRONE_HOVER_OMEGA;
    if (env->drone_hover_vel <= 0.0f) env->drone_hover_vel = QR_DRONE_HOVER_VEL;
    if (env->task_mode != QR_TASK_FIXED_TARGET &&
            env->task_mode != QR_TASK_HOLD_HOVER &&
            env->task_mode != QR_TASK_INDOOR_NAV &&
            env->task_mode != QR_TASK_INDOOR_EXPLORE &&
            env->task_mode != QR_TASK_INDOOR_ROUTE &&
            env->task_mode != QR_TASK_DRONE_HOVER)
        env->task_mode = QR_TASK_FIXED_TARGET;
    if (env->action_mode != QR_ACTION_RATE &&
            env->action_mode != QR_ACTION_VELOCITY &&
            env->action_mode != QR_ACTION_MOTOR &&
            env->action_mode != QR_ACTION_MOTOR_HOVER &&
            env->action_mode != QR_ACTION_MOTOR_REF_HOVER)
        env->action_mode = QR_ACTION_RATE;
    if (env->obstacle_mode < QR_OBSTACLES_NONE ||
            env->obstacle_mode > QR_OBSTACLES_AGILE_PILLAR_BOTTLENECK_CURRICULUM)
        env->obstacle_mode = QR_OBSTACLES_NONE;
    if (env->obstacle_block_prob < 0.0f)
        env->obstacle_block_prob = 0.0f;
    if (env->obstacle_block_prob > 1.0f)
        env->obstacle_block_prob = 1.0f;
    if (env->mass_scale <= 0.0f) env->mass_scale = 1.0f;
    if (env->thrust_scale <= 0.0f) env->thrust_scale = 1.0f;
    if (env->drag_scale <= 0.0f) env->drag_scale = 1.0f;
    if (env->motor_tau_scale <= 0.0f) env->motor_tau_scale = 1.0f;
    if (env->explore_cell_size <= 0.0f) env->explore_cell_size = QR_EXPLORE_CELL_SIZE;
    if (env->explore_goal_fraction <= 0.0f) env->explore_goal_fraction = QR_EXPLORE_GOAL_FRACTION;
    if (env->explore_goal_fraction > 1.0f) env->explore_goal_fraction = 1.0f;
    if (env->explore_frontier_target_mode < QR_EXPLORE_FRONTIER_TARGET_NEAREST ||
            env->explore_frontier_target_mode > QR_EXPLORE_FRONTIER_TARGET_REGION)
        env->explore_frontier_target_mode = QR_EXPLORE_FRONTIER_TARGET_NEAREST;
    env->explore_reachable_coverage = env->explore_reachable_coverage ? 1 : 0;
    if (env->explore_onboard_commit_steps < 0)
        env->explore_onboard_commit_steps = 0;
    if (env->explore_onboard_path_lookahead < 1)
        env->explore_onboard_path_lookahead = 1;
    env->explore_onboard_adaptive_lookahead =
        env->explore_onboard_adaptive_lookahead ? 1 : 0;
    if (env->explore_onboard_adaptive_lookahead_min < 1)
        env->explore_onboard_adaptive_lookahead_min = 1;
    if (env->explore_onboard_adaptive_lookahead_max < 1)
        env->explore_onboard_adaptive_lookahead_max =
            env->explore_onboard_path_lookahead;
    if (env->explore_onboard_adaptive_lookahead_max <
            env->explore_onboard_adaptive_lookahead_min)
        env->explore_onboard_adaptive_lookahead_max =
            env->explore_onboard_adaptive_lookahead_min;
    if (env->explore_onboard_adaptive_clearance < 0.0f)
        env->explore_onboard_adaptive_clearance = 0.0f;
    if (env->explore_onboard_adaptive_turn_dot < -1.0f)
        env->explore_onboard_adaptive_turn_dot = -1.0f;
    if (env->explore_onboard_adaptive_turn_dot > 1.0f)
        env->explore_onboard_adaptive_turn_dot = 1.0f;
    if (env->explore_onboard_commit_min_clearance < 0.0f)
        env->explore_onboard_commit_min_clearance = 0.0f;
    if (env->explore_onboard_stall_viewpoint_ticks < 0)
        env->explore_onboard_stall_viewpoint_ticks = 0;
    if (env->explore_onboard_stall_viewpoint_max_depth < 0)
        env->explore_onboard_stall_viewpoint_max_depth = 0;
    if (env->explore_onboard_approach_viewpoints < 0)
        env->explore_onboard_approach_viewpoints = 0;
    if (env->explore_onboard_approach_max_depth < 1)
        env->explore_onboard_approach_max_depth = 1;
    env->explore_onboard_region_commit =
        env->explore_onboard_region_commit ? 1 : 0;
    if (env->explore_onboard_region_commit_stall_ticks < 0)
        env->explore_onboard_region_commit_stall_ticks = 0;
    env->explore_onboard_persistent_regions =
        env->explore_onboard_persistent_regions ? 1 : 0;
    env->explore_onboard_region_tracker =
        env->explore_onboard_region_tracker ? 1 : 0;
    env->explore_onboard_frontier_graph =
        env->explore_onboard_frontier_graph ? 1 : 0;
    env->explore_onboard_graph_candidate_obs =
        env->explore_onboard_graph_candidate_obs ? 1 : 0;
    if (env->explore_onboard_graph_gain_weight < 0.0f)
        env->explore_onboard_graph_gain_weight = 0.32f;
    if (env->explore_onboard_graph_cluster_weight < 0.0f)
        env->explore_onboard_graph_cluster_weight = 0.12f;
    if (env->explore_onboard_graph_depth_weight < 0.0f)
        env->explore_onboard_graph_depth_weight = 0.045f;
    if (env->explore_onboard_graph_active_bonus < 0.0f)
        env->explore_onboard_graph_active_bonus = 0.75f;
    if (env->explore_onboard_graph_stale_penalty < 0.0f)
        env->explore_onboard_graph_stale_penalty = 0.020f;
    if (env->explore_onboard_graph_failure_penalty < 0.0f)
        env->explore_onboard_graph_failure_penalty = 0.70f;
    if (env->explore_onboard_graph_attempt_penalty < 0.0f)
        env->explore_onboard_graph_attempt_penalty = 0.12f;
    if (env->explore_onboard_graph_gain_cap <= 0)
        env->explore_onboard_graph_gain_cap = 192;
    env->explore_onboard_clearance_passable =
        env->explore_onboard_clearance_passable ? 1 : 0;
    if (env->explore_onboard_passable_radius < 0.0f)
        env->explore_onboard_passable_radius = 0.0f;
    env->explore_onboard_passage_quality =
        env->explore_onboard_passage_quality ? 1 : 0;
    if (env->explore_onboard_passage_weight < 0.0f)
        env->explore_onboard_passage_weight = 0.0f;
    if (env->explore_onboard_clearance_cone_deg <= 0.0f)
        env->explore_onboard_clearance_cone_deg = 30.0f;
    if (env->explore_onboard_clearance_cone_deg > 120.0f)
        env->explore_onboard_clearance_cone_deg = 120.0f;
    if (env->route_goal_radius <= 0.0f) env->route_goal_radius = QR_NAV_GOAL_RADIUS;
    if (env->route_goal_count <= 0) env->route_goal_count = QR_ROUTE_GOAL_COUNT;
    if (env->route_goal_count > 16) env->route_goal_count = 16;
    if (env->route_goal_count_min < 0) env->route_goal_count_min = 0;
    if (env->route_goal_count_min > env->route_goal_count)
        env->route_goal_count_min = env->route_goal_count;
    if (env->route_goal_count_min > 16) env->route_goal_count_min = 16;
    env->episode_route_goal_count = env->route_goal_count;
    if (env->route_reference_clearance_radius <= 0.0f)
        env->route_reference_clearance_radius = QR_ROUTE_CLEARANCE_RADIUS;
    if (env->route_reference_clearance_scale < 0.0f)
        env->route_reference_clearance_scale = QR_ROUTE_REFERENCE_CLEARANCE_SCALE;
    if (env->route_clearance_speed_scale < 0.0f)
        env->route_clearance_speed_scale = QR_ROUTE_CLEARANCE_SPEED_SCALE;
    if (env->route_clearance_speed_min < 0.0f)
        env->route_clearance_speed_min = QR_ROUTE_CLEARANCE_SPEED_MIN;
    if (env->route_clearance_speed_full < 0.0f)
        env->route_clearance_speed_full = QR_ROUTE_CLEARANCE_SPEED_FULL;
    if (env->route_path_margin <= 0.0f)
        env->route_path_margin = QR_NAV_PATH_MARGIN;
    if (env->route_arrival_radius < 0.0f)
        env->route_arrival_radius = QR_ROUTE_ARRIVAL_RADIUS;
    if (env->route_arrival_bonus < 0.0f)
        env->route_arrival_bonus = QR_ROUTE_ARRIVAL_BONUS;
    if (env->route_arrival_vel_scale < 0.0f)
        env->route_arrival_vel_scale = QR_ROUTE_ARRIVAL_VEL_SCALE;
    if (env->route_arrival_toward_scale < 0.0f)
        env->route_arrival_toward_scale = QR_ROUTE_ARRIVAL_TOWARD_SCALE;
    if (env->route_target_wall_margin < 0.0f)
        env->route_target_wall_margin = QR_ROUTE_TARGET_WALL_MARGIN;
    if (env->route_goal_speed <= 0.0f)
        env->route_goal_speed = QR_ROUTE_GOAL_SPEED;
    if (env->route_goal_bonus < 0.0f)
        env->route_goal_bonus = QR_ROUTE_GOAL_BONUS_REF;
    if (env->route_reach_bonus <= 0.0f)
        env->route_reach_bonus = QR_ROUTE_REACH_BONUS;
    if (env->route_target_z_min < 0.0f)
        env->route_target_z_min = 0.0f;
    if (env->route_target_z_max < 0.0f)
        env->route_target_z_max = 0.0f;
    if (env->route_reward_mode < QR_ROUTE_REWARD_DENSE ||
            env->route_reward_mode > QR_ROUTE_REWARD_DRONE_HOVER_ARRIVAL)
        env->route_reward_mode = QR_ROUTE_REWARD_DENSE;
    if (env->route_target_mode < QR_ROUTE_TARGET_RANDOM ||
            env->route_target_mode > QR_ROUTE_TARGET_BLOCKED)
        env->route_target_mode = QR_ROUTE_TARGET_RANDOM;
    if (env->route_target_obs_mode < QR_ROUTE_TARGET_OBS_LEGACY ||
            env->route_target_obs_mode > QR_ROUTE_TARGET_OBS_DRONE)
        env->route_target_obs_mode = QR_ROUTE_TARGET_OBS_LEGACY;
    if (env->route_blocked_target_prob < 0.0f)
        env->route_blocked_target_prob = 0.0f;
    if (env->route_blocked_target_prob > 1.0f)
        env->route_blocked_target_prob = 1.0f;
    env->route_hardstate_probe = env->route_hardstate_probe ? 1 : 0;
    if (env->route_hardstate_jitter_xy < 0.0f)
        env->route_hardstate_jitter_xy = 0.0f;
    if (env->route_hardstate_jitter_z < 0.0f)
        env->route_hardstate_jitter_z = 0.0f;
    if (env->route_walls < 0)
        env->route_walls = 1;
    env->route_privileged_obs = env->route_privileged_obs ? 1 : 0;
    qr_configure_dynamics(env);
    qr_configure_obstacles(env);
    qr_clear_episode_stats(env);
}

static inline void allocate(QuadrotorEnv* env) {
    init(env);
    int obs_size = QR_OBS_SIZE + env->num_rays;
    if (env->route_wall_obs)
        obs_size += QR_ROUTE_WALL_OBS_SIZE;
    if (env->route_extra_obs)
        obs_size += QR_ROUTE_EXTRA_OBS_SIZE;
    if (env->route_perception_obs == 1)
        obs_size += QR_ROUTE_PERCEPTION_OBS_SIZE;
    else if (env->route_perception_obs == 2)
        obs_size += QR_ROUTE_COMPACT_PERCEPTION_OBS_SIZE;
    else if (env->route_perception_obs == 3)
        obs_size += QR_ROUTE_AFFORDANCE_OBS_SIZE;
    if (env->action_history_obs)
        obs_size += QR_ACTION_HISTORY_OBS_SIZE;
    obs_size += QR_EXPLORE_PATCH_OBS_SIZE;
    env->observations = (float*)calloc(obs_size, sizeof(float));
    env->actions = (float*)calloc(QR_NUM_ACTIONS, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
    env->truncations = (unsigned char*)calloc(1, sizeof(unsigned char));
}

static inline float qr_target_offset_scale(const QuadrotorEnv* env) {
    if (qr_task_uses_waypoint(env->task_mode) && env->room_size > 0.0f)
        return 2.0f * env->room_size;

    float scale = env->spawn_radius > env->hover_radius
        ? env->spawn_radius
        : env->hover_radius;
    return scale > QR_MIN_OFFSET_OBS_SCALE ? scale : QR_MIN_OFFSET_OBS_SCALE;
}

static inline float qr_ray_box_xy(float ox, float oy, float dx, float dy,
        const float* box) {
    float tmin = 0.0f;
    float tmax = 1.0e9f;

    if (fabsf(dx) < 1e-6f) {
        if (ox < box[0] || ox > box[1])
            return -1.0f;
    } else {
        float t1 = (box[0] - ox) / dx;
        float t2 = (box[1] - ox) / dx;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
    }

    if (fabsf(dy) < 1e-6f) {
        if (oy < box[2] || oy > box[3])
            return -1.0f;
    } else {
        float t1 = (box[2] - oy) / dy;
        float t2 = (box[3] - oy) / dy;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
    }

    if (tmax < tmin || tmax < 0.0f)
        return -1.0f;
    return tmin > 0.0f ? tmin : 0.0f;
}

static inline float qr_ray_box_3d(float ox, float oy, float oz,
        float dx, float dy, float dz, const float* box) {
    float tmin = 0.0f;
    float tmax = 1.0e9f;

    if (fabsf(dx) < 1e-6f) {
        if (ox < box[0] || ox > box[1])
            return -1.0f;
    } else {
        float t1 = (box[0] - ox) / dx;
        float t2 = (box[1] - ox) / dx;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
    }

    if (fabsf(dy) < 1e-6f) {
        if (oy < box[2] || oy > box[3])
            return -1.0f;
    } else {
        float t1 = (box[2] - oy) / dy;
        float t2 = (box[3] - oy) / dy;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
    }

    if (fabsf(dz) < 1e-6f) {
        if (oz < box[4] || oz > box[5])
            return -1.0f;
    } else {
        float t1 = (box[4] - oz) / dz;
        float t2 = (box[5] - oz) / dz;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
    }

    if (tmax < tmin || tmax < 0.0f)
        return -1.0f;
    return tmin > 0.0f ? tmin : 0.0f;
}

static inline void qr_try_room_plane(float ox, float oy, float oz,
        float dx, float dy, float dz, float t, const QuadrotorEnv* env,
        float* best) {
    if (t < 0.0f || t >= *best)
        return;
    float x = ox + dx * t;
    float y = oy + dy * t;
    float z = oz + dz * t;
    if (x < -env->room_size - 1e-4f || x > env->room_size + 1e-4f ||
            y < -env->room_size - 1e-4f || y > env->room_size + 1e-4f ||
            z < -1e-4f || z > env->ceiling_height + 1e-4f)
        return;
    *best = t;
}

static inline float qr_room_depth_ray_obs(const QuadrotorEnv* env,
        const Quadrotor* q, const float dir_world[3]) {
    if (env->ray_length <= 0.0f)
        return 1.0f;

    float best = env->ray_length;
    int use_room_bounds = env->room_size > 0.0f &&
        (env->task_mode != QR_TASK_INDOOR_ROUTE || env->route_walls);
    if (use_room_bounds) {
        if (fabsf(dir_world[0]) > 1e-6f) {
            qr_try_room_plane(q->pos[0], q->pos[1], q->pos[2],
                dir_world[0], dir_world[1], dir_world[2],
                ( env->room_size - q->pos[0]) / dir_world[0], env, &best);
            qr_try_room_plane(q->pos[0], q->pos[1], q->pos[2],
                dir_world[0], dir_world[1], dir_world[2],
                (-env->room_size - q->pos[0]) / dir_world[0], env, &best);
        }
        if (fabsf(dir_world[1]) > 1e-6f) {
            qr_try_room_plane(q->pos[0], q->pos[1], q->pos[2],
                dir_world[0], dir_world[1], dir_world[2],
                ( env->room_size - q->pos[1]) / dir_world[1], env, &best);
            qr_try_room_plane(q->pos[0], q->pos[1], q->pos[2],
                dir_world[0], dir_world[1], dir_world[2],
                (-env->room_size - q->pos[1]) / dir_world[1], env, &best);
        }
        if (fabsf(dir_world[2]) > 1e-6f) {
            qr_try_room_plane(q->pos[0], q->pos[1], q->pos[2],
                dir_world[0], dir_world[1], dir_world[2],
                (env->ceiling_height - q->pos[2]) / dir_world[2], env, &best);
            qr_try_room_plane(q->pos[0], q->pos[1], q->pos[2],
                dir_world[0], dir_world[1], dir_world[2],
                -q->pos[2] / dir_world[2], env, &best);
        }
    }

    for (int i = 0; i < env->num_obstacles; i++) {
        float t = qr_ray_box_3d(q->pos[0], q->pos[1], q->pos[2],
            dir_world[0], dir_world[1], dir_world[2], env->obstacles[i]);
        if (t >= 0.0f && t < best)
            best = t;
    }

    return qr_clampf(best / env->ray_length, 0.0f, 1.0f);
}

static inline void qr_write_depth_camera_obs(const QuadrotorEnv* env,
        const Quadrotor* q, float* o, int start) {
    if (!env->camera_depth_obs)
        return;

    float tan_x = tanf(0.5f * env->camera_depth_fov_x_deg * QR_PI / 180.0f);
    float tan_y = tanf(0.5f * env->camera_depth_fov_y_deg * QR_PI / 180.0f);
    for (int py = 0; py < env->camera_depth_height; py++) {
        float v = (0.5f - ((float)py + 0.5f) / (float)env->camera_depth_height) *
            2.0f * tan_y;
        for (int px = 0; px < env->camera_depth_width; px++) {
            int out_idx = start + py * QR_DEPTH_CAMERA_STRIDE + px;
            float u = (((float)px + 0.5f) / (float)env->camera_depth_width - 0.5f) *
                2.0f * tan_x;
            float inv = 1.0f / sqrtf(1.0f + u*u + v*v);
            float dir_body[3] = {inv, u * inv, v * inv};
            float dir_world[3];
            qr_body_to_world(q->quat, dir_body, dir_world);
            float dn = sqrtf(dir_world[0]*dir_world[0] +
                             dir_world[1]*dir_world[1] +
                             dir_world[2]*dir_world[2]);
            if (dn <= 1e-6f) {
                o[out_idx] = 1.0f;
                continue;
            }
            dir_world[0] /= dn;
            dir_world[1] /= dn;
            dir_world[2] /= dn;
            o[out_idx] = qr_room_depth_ray_obs(env, q, dir_world);
        }
    }
}

static inline float qr_room_ray_obs_angle(const QuadrotorEnv* env,
        const Quadrotor* q, float angle) {
    if (env->room_size <= 0.0f || env->ray_length <= 0.0f)
        return 1.0f;
    int use_room_bounds = env->task_mode != QR_TASK_INDOOR_ROUTE ||
        env->route_walls;

    float dir_body[3] = {cosf(angle), sinf(angle), 0.0f};
    float dir_world[3];
    qr_body_to_world(q->quat, dir_body, dir_world);

    dir_world[2] = 0.0f;
    float dn = sqrtf(dir_world[0]*dir_world[0] + dir_world[1]*dir_world[1]);
    if (dn <= 1e-6f)
        return 1.0f;

    dir_world[0] /= dn;
    dir_world[1] /= dn;

    float best = env->ray_length;
    if (use_room_bounds) {
        if (dir_world[0] > 1e-6f) {
            float t = (env->room_size - q->pos[0]) / dir_world[0];
            if (t >= 0.0f && t < best) best = t;
        } else if (dir_world[0] < -1e-6f) {
            float t = (-env->room_size - q->pos[0]) / dir_world[0];
            if (t >= 0.0f && t < best) best = t;
        }

        if (dir_world[1] > 1e-6f) {
            float t = (env->room_size - q->pos[1]) / dir_world[1];
            if (t >= 0.0f && t < best) best = t;
        } else if (dir_world[1] < -1e-6f) {
            float t = (-env->room_size - q->pos[1]) / dir_world[1];
            if (t >= 0.0f && t < best) best = t;
        }
    }

    for (int i = 0; i < env->num_obstacles; i++) {
        const float* box = env->obstacles[i];
        if (q->pos[2] < box[4] || q->pos[2] > box[5])
            continue;
        float t = qr_ray_box_xy(q->pos[0], q->pos[1],
            dir_world[0], dir_world[1], box);
        if (t >= 0.0f && t < best)
            best = t;
    }

    return qr_clampf(best / env->ray_length, 0.0f, 1.0f);
}

static inline float qr_room_ray_obs(const QuadrotorEnv* env, const Quadrotor* q,
        int ray_idx) {
    if (env->num_rays <= 0)
        return 1.0f;
    float angle = 2.0f * QR_PI * (float)ray_idx / (float)env->num_rays;
    return qr_room_ray_obs_angle(env, q, angle);
}

static inline void qr_nav_observation_target(const QuadrotorEnv* env,
        const Quadrotor* q, float out[3]);
static inline float qr_target_delta(const QuadrotorEnv* env, const Quadrotor* q,
        float delta[3]);
static inline float qr_nav_path_distance(const QuadrotorEnv* env,
        const Quadrotor* q);
static inline int qr_nav_segment_blocked(const QuadrotorEnv* env,
        float x1, float y1, float x2, float y2);
static inline float qr_clearance(const QuadrotorEnv* env, const Quadrotor* q);
static inline int qr_valid_xy(const QuadrotorEnv* env, float x, float y,
        float margin);
static inline float qr_explore_cell_center(const QuadrotorEnv* env, int idx);
static inline int qr_explore_cell_index(const QuadrotorEnv* env, float x,
        float y);
static inline int qr_explore_valid_neighbor(const QuadrotorEnv* env,
        int dim, int ix, int iy);
static inline int qr_explore_reachable_cell_count(const QuadrotorEnv* env,
        int start_idx, unsigned char* out_reachable);
static inline float qr_explore_coverage(const QuadrotorEnv* env);
static inline float qr_explore_frontier_distance(const QuadrotorEnv* env,
        const Quadrotor* q, float out_dir_body[2]);
static inline float qr_explore_frontier_path_target(QuadrotorEnv* env,
        const Quadrotor* q, float out_dir_body[2]);
static inline void qr_explore_planner_stats(const QuadrotorEnv* env,
        QrExplorePlannerStats* out);

static inline void qr_write_route_target_obs(QuadrotorEnv* env,
        const Quadrotor* q, float* o, float dx, float dy, float dz) {
    float delta_world[3] = {dx, dy, dz};
    float delta_body[3];
    qr_world_to_body(q->quat, delta_world, delta_body);

    float scale = qr_target_offset_scale(env);
    if (scale < QR_MIN_OFFSET_OBS_SCALE)
        scale = QR_MIN_OFFSET_OBS_SCALE;
    float z_scale = env->ceiling_height > QR_MIN_OFFSET_OBS_SCALE
        ? env->ceiling_height
        : QR_MIN_OFFSET_OBS_SCALE;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    float inv_dist = dist > 1e-6f ? 1.0f / dist : 0.0f;
    float xy_dist = sqrtf(delta_body[0]*delta_body[0] +
                          delta_body[1]*delta_body[1]);
    float bearing = xy_dist > 1e-6f
        ? atan2f(delta_body[1], delta_body[0])
        : 0.0f;
    float closing_speed = 0.0f;
    if (dist > 1e-6f)
        closing_speed = (dx*q->vel[0] + dy*q->vel[1] + dz*q->vel[2]) / dist;
    float speed_scale = q->max_vel > 1e-6f ? q->max_vel : QR_MAX_VEL;

    for (int i = 14; i < QR_OBS_SIZE; i++)
        o[i] = 0.0f;

    if (env->route_target_obs_mode == QR_ROUTE_TARGET_OBS_DRONE) {
        o[14] = tanhf(delta_body[0] * 0.1f);
        o[15] = tanhf(delta_body[1] * 0.1f);
        o[16] = tanhf(delta_body[2] * 0.1f);
        o[17] = tanhf(delta_body[0] * 10.0f);
        o[18] = tanhf(delta_body[1] * 10.0f);
        o[19] = tanhf(delta_body[2] * 10.0f);
        return;
    }

    if (env->route_target_obs_mode == QR_ROUTE_TARGET_OBS_BODY) {
        o[14] = qr_clampf(delta_body[0] / scale, -1.0f, 1.0f);
        o[15] = qr_clampf(delta_body[1] / scale, -1.0f, 1.0f);
        o[16] = qr_clampf(delta_body[2] / z_scale, -1.0f, 1.0f);
        o[17] = qr_clampf(delta_body[0] * inv_dist, -1.0f, 1.0f);
        o[18] = qr_clampf(delta_body[1] * inv_dist, -1.0f, 1.0f);
        o[19] = qr_clampf(delta_body[2] * inv_dist, -1.0f, 1.0f);
        o[20] = qr_clampf(dist / scale, 0.0f, 1.0f);
        o[21] = qr_clampf(closing_speed / speed_scale, -1.0f, 1.0f);
        o[22] = xy_dist > 1e-6f ? cosf(bearing) : 1.0f;
        o[23] = xy_dist > 1e-6f ? sinf(bearing) : 0.0f;
        o[24] = env->ceiling_height > 1e-6f
            ? qr_clampf(2.0f * env->target[2] / env->ceiling_height - 1.0f,
                -1.0f, 1.0f)
            : 0.0f;
        o[25] = env->ceiling_height > 1e-6f
            ? qr_clampf(2.0f * q->pos[2] / env->ceiling_height - 1.0f,
                -1.0f, 1.0f)
            : 0.0f;
        return;
    }

    if (env->route_target_obs_mode == QR_ROUTE_TARGET_OBS_MINIMAL) {
        o[14] = qr_clampf(delta_body[0] * inv_dist, -1.0f, 1.0f);
        o[15] = qr_clampf(delta_body[1] * inv_dist, -1.0f, 1.0f);
        o[16] = qr_clampf(delta_body[2] * inv_dist, -1.0f, 1.0f);
        o[17] = qr_clampf(dist / scale, 0.0f, 1.0f);
        o[18] = qr_clampf(delta_body[2] / z_scale, -1.0f, 1.0f);
        o[19] = qr_clampf(closing_speed / speed_scale, -1.0f, 1.0f);
        o[20] = xy_dist > 1e-6f ? cosf(bearing) : 1.0f;
        o[21] = xy_dist > 1e-6f ? sinf(bearing) : 0.0f;
        return;
    }

    o[14] = dx / 30.0f;
    o[15] = dy / 30.0f;
    o[16] = dz / 10.0f;

    o[17] = qr_clampf(dx, -1.0f, 1.0f);
    o[18] = qr_clampf(dy, -1.0f, 1.0f);
    o[19] = qr_clampf(dz, -1.0f, 1.0f);

    o[20] = 0.0f;
    o[21] = 0.0f;
    o[22] = 1.0f;

    o[23] = 0.0f;
    o[24] = 0.0f;
    o[25] = 0.0f;

    if (env->route_extra_obs && qr_task_uses_waypoint(env->task_mode)) {
        float direct_delta[3];
        float direct_dist = qr_target_delta(env, q, direct_delta);
        float extra_scale = env->room_size > 0.0f
            ? 2.0f * env->room_size
            : qr_target_offset_scale(env);
        if (extra_scale < QR_MIN_OFFSET_OBS_SCALE)
            extra_scale = QR_MIN_OFFSET_OBS_SCALE;
        float direct_closing_speed = 0.0f;
        if (direct_dist > 1e-6f) {
            direct_closing_speed = (direct_delta[0]*q->vel[0] +
                                   direct_delta[1]*q->vel[1] +
                                   direct_delta[2]*q->vel[2]) / direct_dist;
        }
        o[20] = qr_clampf(direct_delta[0], -1.0f, 1.0f);
        o[21] = qr_clampf(direct_delta[1], -1.0f, 1.0f);
        o[23] = qr_clampf(direct_delta[2], -1.0f, 1.0f);
        o[24] = qr_clampf(direct_dist / extra_scale, 0.0f, 1.0f);
        o[25] = qr_clampf(direct_closing_speed / speed_scale, -1.0f, 1.0f);
    }
}

static inline void qr_write_explore_memory_obs(QuadrotorEnv* env,
        const Quadrotor* q, float* o) {
    for (int i = 14; i < QR_OBS_SIZE; i++)
        o[i] = 0.0f;

    int dim = env->explore_grid_dim;
    if (dim <= 0 || env->room_size <= 0.0f)
        return;

    float scores[8] = {0};
    float best_score = 0.0f;
    float best_body_x = 1.0f;
    float best_body_y = 0.0f;
    int frontier_count = 0;
    int current_idx = qr_explore_cell_index(env, q->pos[0], q->pos[1]);
    float max_dist = 2.0f * env->room_size;
    if (max_dist < env->explore_cell_size)
        max_dist = env->explore_cell_size;

    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (idx != current_idx && !env->explored[idx])
                continue;

            int frontier = 0;
            const int nx[4] = {ix + 1, ix - 1, ix, ix};
            const int ny[4] = {iy, iy, iy + 1, iy - 1};
            for (int k = 0; k < 4; k++) {
                if (nx[k] < 0 || nx[k] >= dim || ny[k] < 0 || ny[k] >= dim)
                    continue;
                float x = qr_explore_cell_center(env, nx[k]);
                float y = qr_explore_cell_center(env, ny[k]);
                if (!qr_valid_xy(env, x, y, QR_EXPLORE_CELL_MARGIN))
                    continue;
                int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                if (!env->explored[nidx]) {
                    frontier = 1;
                    break;
                }
            }
            if (!frontier)
                continue;

            float x = qr_explore_cell_center(env, ix);
            float y = qr_explore_cell_center(env, iy);
            float delta_world[3] = {x - q->pos[0], y - q->pos[1], 0.0f};
            float delta_body[3];
            qr_world_to_body(q->quat, delta_world, delta_body);
            float dist = sqrtf(delta_world[0]*delta_world[0] +
                               delta_world[1]*delta_world[1]);
            float score = qr_clampf(1.0f - dist / max_dist, 0.0f, 1.0f);
            if (dist < 1e-4f)
                score = 0.25f;

            float angle = atan2f(delta_body[1], delta_body[0]);
            int sector = (int)floorf((angle + QR_PI) * (8.0f / (2.0f * QR_PI)));
            if (sector < 0)
                sector = 0;
            if (sector > 7)
                sector = 7;
            if (score > scores[sector])
                scores[sector] = score;
            if (score > best_score && dist >= 1e-4f) {
                float inv_dist = 1.0f / dist;
                best_score = score;
                best_body_x = qr_clampf(delta_body[0] * inv_dist, -1.0f, 1.0f);
                best_body_y = qr_clampf(delta_body[1] * inv_dist, -1.0f, 1.0f);
            }
            frontier_count++;
        }
    }

    for (int i = 0; i < 8; i++)
        o[14 + i] = scores[i];
    o[22] = best_body_x;
    o[23] = best_body_y;
    o[24] = qr_explore_coverage(env);
    o[25] = qr_clampf((float)frontier_count / 32.0f, 0.0f, 1.0f);
}

static inline void qr_write_explore_frontier_target_obs(QuadrotorEnv* env,
        const Quadrotor* q, float* o) {
    for (int i = 14; i < QR_OBS_SIZE; i++)
        o[i] = 0.0f;

    float frontier_dir[2];
    float frontier_dist = qr_explore_frontier_distance(env, q, frontier_dir);
    float scale = env->room_size > 0.0f
        ? 2.0f * env->room_size
        : QR_MIN_OFFSET_OBS_SCALE;
    if (scale < QR_MIN_OFFSET_OBS_SCALE)
        scale = QR_MIN_OFFSET_OBS_SCALE;

    o[14] = frontier_dir[0];
    o[15] = frontier_dir[1];
    o[16] = qr_clampf(frontier_dist / scale, 0.0f, 1.0f);
    o[17] = qr_explore_coverage(env);
    o[18] = frontier_dist > 0.0f ? 1.0f : 0.0f;
    o[19] = qr_room_ray_obs_angle(env, q, 0.0f);
    o[20] = qr_room_ray_obs_angle(env, q, QR_PI / 2.0f);
    o[21] = qr_room_ray_obs_angle(env, q, -QR_PI / 2.0f);
}

static inline void qr_explore_frontier_sector4(const QuadrotorEnv* env,
        const Quadrotor* q, float* scores) {
    for (int i = 0; i < 4; i++)
        scores[i] = 0.0f;

    int dim = env->explore_grid_dim;
    if (dim <= 0 || env->room_size <= 0.0f)
        return;

    int current_idx = qr_explore_cell_index(env, q->pos[0], q->pos[1]);
    float max_dist = 2.0f * env->room_size;
    if (max_dist < env->explore_cell_size)
        max_dist = env->explore_cell_size;

    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (idx != current_idx && !env->explored[idx])
                continue;

            int frontier = 0;
            const int nx[4] = {ix + 1, ix - 1, ix, ix};
            const int ny[4] = {iy, iy, iy + 1, iy - 1};
            for (int k = 0; k < 4; k++) {
                if (nx[k] < 0 || nx[k] >= dim || ny[k] < 0 || ny[k] >= dim)
                    continue;
                float x = qr_explore_cell_center(env, nx[k]);
                float y = qr_explore_cell_center(env, ny[k]);
                if (!qr_valid_xy(env, x, y, QR_EXPLORE_CELL_MARGIN))
                    continue;
                int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                if (!env->explored[nidx]) {
                    frontier = 1;
                    break;
                }
            }
            if (!frontier)
                continue;

            float x = qr_explore_cell_center(env, ix);
            float y = qr_explore_cell_center(env, iy);
            float delta_world[3] = {x - q->pos[0], y - q->pos[1], 0.0f};
            float delta_body[3];
            qr_world_to_body(q->quat, delta_world, delta_body);
            float dist = sqrtf(delta_world[0]*delta_world[0] +
                               delta_world[1]*delta_world[1]);
            float score = qr_clampf(1.0f - dist / max_dist, 0.0f, 1.0f);
            if (dist < 1e-4f)
                score = 0.25f;

            float angle = atan2f(delta_body[1], delta_body[0]);
            int sector = 2;
            if (fabsf(angle) <= QR_PI / 4.0f)
                sector = 0;
            else if (angle > QR_PI / 4.0f && angle <= 3.0f * QR_PI / 4.0f)
                sector = 1;
            else if (angle < -QR_PI / 4.0f && angle >= -3.0f * QR_PI / 4.0f)
                sector = 3;
            if (score > scores[sector])
                scores[sector] = score;
        }
    }
}

static inline float qr_explore_nearest_frontier_vector(const QuadrotorEnv* env,
        const Quadrotor* q, float* dir) {
    dir[0] = 0.0f;
    dir[1] = 0.0f;

    int dim = env->explore_grid_dim;
    if (dim <= 0 || env->room_size <= 0.0f)
        return 0.0f;

    int current_idx = qr_explore_cell_index(env, q->pos[0], q->pos[1]);
    float best_dist = QR_NAV_PATH_INF;
    float best_body[2] = {0.0f, 0.0f};

    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (idx != current_idx && !env->explored[idx])
                continue;

            int frontier = 0;
            const int nx[4] = {ix + 1, ix - 1, ix, ix};
            const int ny[4] = {iy, iy, iy + 1, iy - 1};
            for (int k = 0; k < 4; k++) {
                if (nx[k] < 0 || nx[k] >= dim || ny[k] < 0 || ny[k] >= dim)
                    continue;
                float x = qr_explore_cell_center(env, nx[k]);
                float y = qr_explore_cell_center(env, ny[k]);
                if (!qr_valid_xy(env, x, y, QR_EXPLORE_CELL_MARGIN))
                    continue;
                int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                if (!env->explored[nidx]) {
                    frontier = 1;
                    break;
                }
            }
            if (!frontier)
                continue;

            float x = qr_explore_cell_center(env, ix);
            float y = qr_explore_cell_center(env, iy);
            float delta_world[3] = {x - q->pos[0], y - q->pos[1], 0.0f};
            float dist = sqrtf(delta_world[0]*delta_world[0] +
                               delta_world[1]*delta_world[1]);
            if (dist >= best_dist)
                continue;

            float delta_body[3];
            qr_world_to_body(q->quat, delta_world, delta_body);
            best_dist = dist;
            best_body[0] = delta_body[0];
            best_body[1] = delta_body[1];
        }
    }

    if (best_dist >= QR_NAV_PATH_INF || best_dist < 1e-6f)
        return 0.0f;

    dir[0] = qr_clampf(best_body[0] / best_dist, -1.0f, 1.0f);
    dir[1] = qr_clampf(best_body[1] / best_dist, -1.0f, 1.0f);
    return best_dist;
}

static inline float qr_explore_target_cell_direction(const QuadrotorEnv* env,
        const Quadrotor* q, int target_idx, float out_dir_body[2]);

static inline void qr_write_explore_frontier_path_obs(QuadrotorEnv* env,
        const Quadrotor* q, float* o) {
    for (int i = 14; i < QR_OBS_SIZE; i++)
        o[i] = 0.0f;

    float target_dir[2];
    float target_dist = qr_explore_frontier_path_target(env, q, target_dir);
    float scale = env->room_size > 0.0f
        ? 2.0f * env->room_size
        : QR_MIN_OFFSET_OBS_SCALE;
    if (scale < QR_MIN_OFFSET_OBS_SCALE)
        scale = QR_MIN_OFFSET_OBS_SCALE;

    o[14] = target_dir[0];
    o[15] = target_dir[1];
    o[16] = qr_clampf(target_dist / scale, 0.0f, 1.0f);
    o[17] = qr_explore_coverage(env);
    o[18] = target_dist > 0.0f ? 1.0f : 0.0f;
    float nearest_frontier_dir[2];
    float nearest_frontier_dist =
        qr_explore_nearest_frontier_vector(env, q, nearest_frontier_dir);
    o[19] = nearest_frontier_dir[0];
    o[20] = nearest_frontier_dir[1];
    o[21] = qr_clampf(nearest_frontier_dist / scale, 0.0f, 1.0f);
    if (env->explore_onboard_objective_obs &&
            env->explore_onboard_objective_idx >= 0) {
        float objective_dir[2];
        float objective_dist = qr_explore_target_cell_direction(
            env, q, env->explore_onboard_objective_idx, objective_dir);
        if (objective_dist > 0.0f) {
            o[19] = objective_dir[0];
            o[20] = objective_dir[1];
            o[21] = qr_clampf(objective_dist / scale, 0.0f, 1.0f);
        }
    }

    float sector_scores[4];
    qr_explore_frontier_sector4(env, q, sector_scores);
    for (int i = 0; i < 4; i++)
        o[22 + i] = sector_scores[i];
}

static inline float qr_explore_patch_value(const QuadrotorEnv* env,
        int idx, int current_idx) {
    if (idx < 0)
        return 0.0f;
    int ix = idx % QR_EXPLORE_MAX_GRID;
    int iy = idx / QR_EXPLORE_MAX_GRID;
    if (ix < 0 || ix >= env->explore_grid_dim ||
            iy < 0 || iy >= env->explore_grid_dim)
        return 0.0f;

    if (idx == current_idx || env->explored[idx])
        return 0.5f;

    const int nx[4] = {ix + 1, ix - 1, ix, ix};
    const int ny[4] = {iy, iy, iy + 1, iy - 1};
    for (int k = 0; k < 4; k++) {
        if (nx[k] < 0 || nx[k] >= env->explore_grid_dim ||
                ny[k] < 0 || ny[k] >= env->explore_grid_dim)
            continue;
        float x = qr_explore_cell_center(env, nx[k]);
        float y = qr_explore_cell_center(env, ny[k]);
        if (!qr_valid_xy(env, x, y, QR_EXPLORE_CELL_MARGIN))
            continue;
        int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
        if (nidx == current_idx || env->explored[nidx])
            return 1.0f;
    }
    return 0.0f;
}

static inline int qr_explore_raw_cell_index(const QuadrotorEnv* env,
        float x, float y) {
    if (env->explore_grid_dim <= 0 || env->room_size <= 0.0f)
        return -1;

    int ix = (int)floorf((x + env->room_size) / env->explore_cell_size);
    int iy = (int)floorf((y + env->room_size) / env->explore_cell_size);
    if (ix < 0 || ix >= env->explore_grid_dim ||
            iy < 0 || iy >= env->explore_grid_dim)
        return -1;
    return iy * QR_EXPLORE_MAX_GRID + ix;
}

static inline int qr_explore_raw_known_free(const QuadrotorEnv* env,
        int idx, int current_idx) {
    if (idx < 0)
        return 0;
    return idx == current_idx || (env->explored[idx] && !env->explore_blocked[idx]);
}

static inline int qr_explore_raw_frontier(const QuadrotorEnv* env,
        int idx, int current_idx) {
    if (idx < 0 || env->explore_blocked[idx] ||
            qr_explore_raw_known_free(env, idx, current_idx))
        return 0;

    int ix = idx % QR_EXPLORE_MAX_GRID;
    int iy = idx / QR_EXPLORE_MAX_GRID;
    if (ix < 0 || ix >= env->explore_grid_dim ||
            iy < 0 || iy >= env->explore_grid_dim)
        return 0;

    const int nx[4] = {ix + 1, ix - 1, ix, ix};
    const int ny[4] = {iy, iy, iy + 1, iy - 1};
    for (int k = 0; k < 4; k++) {
        if (nx[k] < 0 || nx[k] >= env->explore_grid_dim ||
                ny[k] < 0 || ny[k] >= env->explore_grid_dim)
            continue;
        int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
        if (qr_explore_raw_known_free(env, nidx, current_idx))
            return 1;
    }
    return 0;
}

static inline void qr_write_explore_graph_candidate_tail_obs(
    QuadrotorEnv* env, const Quadrotor* q, float* o, int tail_start);

static inline void qr_write_explore_patch_obs(QuadrotorEnv* env,
        const Quadrotor* q, float* o, int* idx) {
    for (int i = 0; i < QR_EXPLORE_PATCH_OBS_SIZE; i++)
        o[(*idx)++] = 0.0f;

    int start = *idx - QR_EXPLORE_PATCH_OBS_SIZE;
    qr_write_depth_camera_obs(env, q, o, start);

    if (env->task_mode != QR_TASK_INDOOR_EXPLORE ||
            env->explore_grid_dim <= 0 || env->room_size <= 0.0f)
        return;

    int current_idx = qr_explore_cell_index(env, q->pos[0], q->pos[1]);
    float spacing = env->explore_cell_size * 3.0f;
    if (spacing < 0.50f)
        spacing = 0.50f;
    if (spacing > 0.75f)
        spacing = 0.75f;

    for (int py = 0; py < QR_EXPLORE_LEGACY_PATCH_SIDE; py++) {
        for (int px = 0; px < QR_EXPLORE_LEGACY_PATCH_SIDE; px++) {
            float body[3] = {
                ((float)px - 1.5f) * spacing,
                ((float)py - 1.5f) * spacing,
                0.0f
            };
            float world[3];
            qr_body_to_world(q->quat, body, world);
            float x = q->pos[0] + world[0];
            float y = q->pos[1] + world[1];
            int cell_idx = qr_explore_cell_index(env, x, y);
            o[start + py * QR_EXPLORE_LEGACY_PATCH_SIDE + px] =
                qr_explore_patch_value(env, cell_idx, current_idx);
        }
    }

    int raw_current_idx = qr_explore_raw_cell_index(env, q->pos[0], q->pos[1]);
    int wide_start = start + QR_EXPLORE_LEGACY_PATCH_SIZE;
    float wide_spacing = env->explore_cell_size * 2.0f;
    if (wide_spacing < 0.35f)
        wide_spacing = 0.35f;
    if (wide_spacing > 0.60f)
        wide_spacing = 0.60f;

    for (int py = 0; py < QR_EXPLORE_WIDE_MAP_SIDE; py++) {
        for (int px = 0; px < QR_EXPLORE_WIDE_MAP_SIDE; px++) {
            float body[3] = {
                ((float)px - 0.5f * (QR_EXPLORE_WIDE_MAP_SIDE - 1)) * wide_spacing,
                ((float)py - 0.5f * (QR_EXPLORE_WIDE_MAP_SIDE - 1)) * wide_spacing,
                0.0f
            };
            float world[3];
            qr_body_to_world(q->quat, body, world);
            float x = q->pos[0] + world[0];
            float y = q->pos[1] + world[1];
            int cell_idx = qr_explore_raw_cell_index(env, x, y);
            int cell = py * QR_EXPLORE_WIDE_MAP_SIDE + px;
            o[wide_start + cell] =
                qr_explore_raw_known_free(env, cell_idx, raw_current_idx) ? 1.0f : 0.0f;
            o[wide_start + QR_EXPLORE_WIDE_MAP_CELLS + cell] =
                qr_explore_raw_frontier(env, cell_idx, raw_current_idx) ? 1.0f : 0.0f;
        }
    }

    int far_start = wide_start + 2 * QR_EXPLORE_WIDE_MAP_CELLS;
    float far_spacing = env->explore_cell_size * 2.5f;
    if (far_spacing < 0.50f)
        far_spacing = 0.50f;
    if (far_spacing > 0.70f)
        far_spacing = 0.70f;

    for (int py = 0; py < QR_EXPLORE_FAR_MAP_SIDE; py++) {
        for (int px = 0; px < QR_EXPLORE_FAR_MAP_SIDE; px++) {
            float body[3] = {
                ((float)px - 0.5f * (QR_EXPLORE_FAR_MAP_SIDE - 1)) * far_spacing,
                ((float)py - 0.5f * (QR_EXPLORE_FAR_MAP_SIDE - 1)) * far_spacing,
                0.0f
            };
            float world[3];
            qr_body_to_world(q->quat, body, world);
            float x = q->pos[0] + world[0];
            float y = q->pos[1] + world[1];
            int cell_idx = qr_explore_raw_cell_index(env, x, y);
            int cell = py * QR_EXPLORE_FAR_MAP_SIDE + px;
            o[far_start + cell] =
                qr_explore_raw_known_free(env, cell_idx, raw_current_idx) ? 1.0f : 0.0f;
            o[far_start + QR_EXPLORE_FAR_MAP_CELLS + cell] =
                qr_explore_raw_frontier(env, cell_idx, raw_current_idx) ? 1.0f : 0.0f;
        }
    }

    if (env->explore_onboard_graph_candidate_obs) {
        int tail_start = start + QR_EXPLORE_PATCH_OBS_SIZE -
            QR_EXPLORE_GRAPH_CANDIDATE_OBS_SIZE;
        qr_write_explore_graph_candidate_tail_obs(env, q, o, tail_start);
    }

    qr_write_depth_camera_obs(env, q, o, start);
}

static inline void qr_write_route_perception_obs(const QuadrotorEnv* env,
        const Quadrotor* q, float* o, int* idx, float dx, float dy, float dz) {
    float delta_world[3] = {dx, dy, dz};
    float delta_body[3];
    qr_world_to_body(q->quat, delta_world, delta_body);

    float scale = qr_target_offset_scale(env);
    float z_scale = env->ceiling_height > QR_MIN_OFFSET_OBS_SCALE
        ? env->ceiling_height
        : QR_MIN_OFFSET_OBS_SCALE;
    float xy_dist = sqrtf(delta_body[0]*delta_body[0] +
                          delta_body[1]*delta_body[1]);
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    float bearing = xy_dist > 1e-6f
        ? atan2f(delta_body[1], delta_body[0])
        : 0.0f;

    o[(*idx)++] = qr_clampf(delta_body[0] / scale, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(delta_body[1] / scale, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(delta_body[2] / z_scale, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(dist / scale, 0.0f, 1.0f);
    o[(*idx)++] = xy_dist > 1e-6f ? cosf(bearing) : 1.0f;
    o[(*idx)++] = xy_dist > 1e-6f ? sinf(bearing) : 0.0f;

    float side = QR_PI / 6.0f;
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, bearing);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, bearing + side);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, bearing - side);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, 0.0f);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, QR_PI / 2.0f);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, -QR_PI / 2.0f);
}

static inline void qr_write_route_compact_perception_obs(const QuadrotorEnv* env,
        const Quadrotor* q, float* o, int* idx, float dx, float dy, float dz) {
    float delta_world[3] = {dx, dy, dz};
    float delta_body[3];
    qr_world_to_body(q->quat, delta_world, delta_body);

    float xy_dist = sqrtf(delta_body[0]*delta_body[0] +
                          delta_body[1]*delta_body[1]);
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    float inv_dist = dist > 1e-6f ? 1.0f / dist : 0.0f;
    float bearing = xy_dist > 1e-6f
        ? atan2f(delta_body[1], delta_body[0])
        : 0.0f;
    float scale = qr_target_offset_scale(env);
    if (scale < QR_MIN_OFFSET_OBS_SCALE)
        scale = QR_MIN_OFFSET_OBS_SCALE;

    o[(*idx)++] = qr_clampf(delta_body[0] * inv_dist, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(delta_body[1] * inv_dist, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(delta_body[2] * inv_dist, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(dist / scale, 0.0f, 1.0f);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, bearing);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, bearing + QR_PI / 4.0f);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, bearing - QR_PI / 4.0f);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, 0.0f);
}

static inline void qr_write_route_affordance_obs(const QuadrotorEnv* env,
        const Quadrotor* q, float* o, int* idx, float dx, float dy, float dz) {
    float delta_world[3] = {dx, dy, dz};
    float delta_body[3];
    qr_world_to_body(q->quat, delta_world, delta_body);

    float xy_dist = sqrtf(delta_body[0]*delta_body[0] +
                          delta_body[1]*delta_body[1]);
    float bearing = xy_dist > 1e-6f
        ? atan2f(delta_body[1], delta_body[0])
        : 0.0f;

    const float max_offset = QR_PI / 2.0f;
    const float step = QR_PI / 12.0f;
    float target_clearance = qr_room_ray_obs_angle(env, q, bearing);
    float left_clearance = 0.0f;
    float right_clearance = 0.0f;
    float best_clearance = target_clearance;
    float best_score = target_clearance;
    float best_offset = 0.0f;

    for (int i = -6; i <= 6; i++) {
        float offset = step * (float)i;
        float clearance = qr_room_ray_obs_angle(env, q, bearing + offset);
        float angle_penalty = 0.25f * fabsf(offset) / max_offset;
        float score = clearance - angle_penalty;

        if (i > 0 && clearance > left_clearance)
            left_clearance = clearance;
        if (i < 0 && clearance > right_clearance)
            right_clearance = clearance;
        if (score > best_score) {
            best_score = score;
            best_clearance = clearance;
            best_offset = offset;
        }
    }

    float best_heading = bearing + best_offset;
    o[(*idx)++] = target_clearance;
    o[(*idx)++] = left_clearance;
    o[(*idx)++] = right_clearance;
    o[(*idx)++] = qr_clampf(left_clearance - right_clearance, -1.0f, 1.0f);
    o[(*idx)++] = cosf(best_heading);
    o[(*idx)++] = sinf(best_heading);
    o[(*idx)++] = best_clearance;
    o[(*idx)++] = qr_clampf(best_offset / max_offset, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(best_score, -1.0f, 1.0f);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, 0.0f);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, QR_PI / 2.0f);
    o[(*idx)++] = qr_room_ray_obs_angle(env, q, -QR_PI / 2.0f);
}

static inline void qr_write_route_extra_obs(const QuadrotorEnv* env,
        const Quadrotor* q, float* o, int* idx) {
    if (!qr_task_uses_waypoint(env->task_mode)) {
        for (int i = 0; i < QR_ROUTE_EXTRA_OBS_SIZE; i++)
            o[(*idx)++] = 0.0f;
        return;
    }

    float delta_world[3];
    float dist = qr_target_delta(env, q, delta_world);
    float delta_body[3];
    qr_world_to_body(q->quat, delta_world, delta_body);

    float inv_dist = dist > 1e-6f ? 1.0f / dist : 0.0f;
    float scale = env->room_size > 0.0f
        ? 2.0f * env->room_size
        : qr_target_offset_scale(env);
    if (scale < QR_MIN_OFFSET_OBS_SCALE)
        scale = QR_MIN_OFFSET_OBS_SCALE;
    float z_scale = env->ceiling_height > QR_MIN_OFFSET_OBS_SCALE
        ? env->ceiling_height
        : QR_MIN_OFFSET_OBS_SCALE;
    float closing_speed = 0.0f;
    if (dist > 1e-6f) {
        closing_speed = (
            delta_world[0] * q->vel[0] +
            delta_world[1] * q->vel[1] +
            delta_world[2] * q->vel[2]
        ) / dist;
    }
    float speed_scale = q->max_vel > 1e-6f ? q->max_vel : QR_MAX_VEL;

    o[(*idx)++] = qr_clampf(delta_body[0] / scale, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(delta_body[1] / scale, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(delta_body[2] / z_scale, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(delta_body[0] * inv_dist, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(delta_body[1] * inv_dist, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(delta_body[2] * inv_dist, -1.0f, 1.0f);
    o[(*idx)++] = qr_clampf(dist / scale, 0.0f, 1.0f);
    o[(*idx)++] = qr_clampf(closing_speed / speed_scale, -1.0f, 1.0f);
}

static inline void qr_write_route_privileged_obs(const QuadrotorEnv* env,
        const Quadrotor* q, float* o, int start) {
    for (int i = 0; i < QR_ROUTE_PRIVILEGED_OBS_SIZE; i++)
        o[start + i] = 0.0f;

    if (!env->route_privileged_obs || !qr_task_uses_waypoint(env->task_mode))
        return;

    float direct_delta[3];
    float direct_dist = qr_target_delta(env, q, direct_delta);
    float path_dist = qr_nav_path_distance(env, q);
    float scale = env->room_size > 0.0f
        ? 2.0f * env->room_size
        : qr_target_offset_scale(env);
    if (scale < QR_MIN_OFFSET_OBS_SCALE)
        scale = QR_MIN_OFFSET_OBS_SCALE;

    float path_target[3] = {env->target[0], env->target[1], env->target[2]};
    qr_nav_observation_target(env, q, path_target);
    float path_delta_world[3] = {
        path_target[0] - q->pos[0],
        path_target[1] - q->pos[1],
        path_target[2] - q->pos[2],
    };
    float path_delta_body[3];
    qr_world_to_body(q->quat, path_delta_world, path_delta_body);
    float path_target_dist = qr_norm3(path_delta_world);
    float inv_path_target = path_target_dist > 1e-6f
        ? 1.0f / path_target_dist
        : 0.0f;

    float direct_delta_body[3];
    qr_world_to_body(q->quat, direct_delta, direct_delta_body);
    float inv_direct = direct_dist > 1e-6f ? 1.0f / direct_dist : 0.0f;
    int blocked = env->num_obstacles > 0 &&
        qr_nav_segment_blocked(env, q->pos[0], q->pos[1],
            env->target[0], env->target[1]);
    float clearance_scale = env->ray_length > 1e-6f ? env->ray_length : scale;
    int goal_count = qr_episode_route_goal_count(env);
    float reached_frac = goal_count > 0
        ? (float)env->route_goals_reached / (float)goal_count
        : 0.0f;
    float remaining_frac = goal_count > 0
        ? (float)(goal_count - env->route_goals_reached) / (float)goal_count
        : 0.0f;

    o[start + 0] = qr_clampf(path_dist / scale, 0.0f, 1.0f);
    o[start + 1] = qr_clampf(direct_dist / scale, 0.0f, 1.0f);
    o[start + 2] = qr_clampf((path_dist - direct_dist) / scale, 0.0f, 1.0f);
    o[start + 3] = blocked ? 1.0f : 0.0f;
    o[start + 4] = qr_clampf(path_delta_body[0] * inv_path_target, -1.0f, 1.0f);
    o[start + 5] = qr_clampf(path_delta_body[1] * inv_path_target, -1.0f, 1.0f);
    o[start + 6] = qr_clampf(path_target_dist / scale, 0.0f, 1.0f);
    o[start + 7] = qr_clampf(direct_delta_body[0] * inv_direct, -1.0f, 1.0f);
    o[start + 8] = qr_clampf(direct_delta_body[1] * inv_direct, -1.0f, 1.0f);
    o[start + 9] = qr_clampf(qr_clearance(env, q) / clearance_scale, 0.0f, 1.0f);
    o[start + 10] = qr_clampf(reached_frac, 0.0f, 1.0f);
    o[start + 11] = qr_clampf(remaining_frac, 0.0f, 1.0f);
}

static inline void compute_observations(QuadrotorEnv* env) {
    const Quadrotor* q = &env->phys;
    float* o = env->observations;
    float vel_body[3];
    qr_world_to_body(q->quat, q->vel, vel_body);

    o[0] = qr_clampf(vel_body[0] / q->max_vel, -1.0f, 1.0f);
    o[1] = qr_clampf(vel_body[1] / q->max_vel, -1.0f, 1.0f);
    o[2] = qr_clampf(vel_body[2] / q->max_vel, -1.0f, 1.0f);

    o[3] = qr_clampf(q->omega[0] / q->max_omega, -1.0f, 1.0f);
    o[4] = qr_clampf(q->omega[1] / q->max_omega, -1.0f, 1.0f);
    o[5] = qr_clampf(q->omega[2] / q->max_omega, -1.0f, 1.0f);

    o[6] = q->quat[0];
    o[7] = q->quat[1];
    o[8] = q->quat[2];
    o[9] = q->quat[3];

    o[10] = qr_clampf(q->motor_speed[0] / q->motor_max, 0.0f, 1.0f);
    o[11] = qr_clampf(q->motor_speed[1] / q->motor_max, 0.0f, 1.0f);
    o[12] = qr_clampf(q->motor_speed[2] / q->motor_max, 0.0f, 1.0f);
    o[13] = qr_clampf(q->motor_speed[3] / q->motor_max, 0.0f, 1.0f);

    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    if (env->task_mode != QR_TASK_INDOOR_EXPLORE) {
        float obs_target[3];
        if (env->route_direct_obs && qr_task_uses_waypoint(env->task_mode)) {
            obs_target[0] = env->target[0];
            obs_target[1] = env->target[1];
            obs_target[2] = env->target[2];
        } else {
            qr_nav_observation_target(env, q, obs_target);
        }
        dx = obs_target[0] - q->pos[0];
        dy = obs_target[1] - q->pos[1];
        dz = obs_target[2] - q->pos[2];
    }

    if (env->task_mode == QR_TASK_INDOOR_EXPLORE && env->explore_memory_obs == 3)
        qr_write_explore_frontier_path_obs(env, q, o);
    else if (env->task_mode == QR_TASK_INDOOR_EXPLORE && env->explore_memory_obs == 2)
        qr_write_explore_frontier_target_obs(env, q, o);
    else if (env->task_mode == QR_TASK_INDOOR_EXPLORE && env->explore_memory_obs)
        qr_write_explore_memory_obs(env, q, o);
    else
        qr_write_route_target_obs(env, q, o, dx, dy, dz);

    int idx = QR_OBS_SIZE;
    for (int i = 0; i < env->num_rays; i++)
        o[idx++] = qr_room_ray_obs(env, q, i);

    if (env->route_wall_obs) {
        float xy = env->room_size > 0.0f ? env->room_size : 0.0f;
        float z = env->ceiling_height > 0.0f ? env->ceiling_height : 0.0f;
        if (xy > 1e-6f) {
            float denom = 2.0f * xy;
            o[idx++] = qr_clampf((xy - q->pos[0]) / denom, 0.0f, 1.0f);
            o[idx++] = qr_clampf((q->pos[0] + xy) / denom, 0.0f, 1.0f);
            o[idx++] = qr_clampf((xy - q->pos[1]) / denom, 0.0f, 1.0f);
            o[idx++] = qr_clampf((q->pos[1] + xy) / denom, 0.0f, 1.0f);
        } else {
            o[idx++] = 1.0f;
            o[idx++] = 1.0f;
            o[idx++] = 1.0f;
            o[idx++] = 1.0f;
        }
        if (z > 1e-6f) {
            o[idx++] = qr_clampf((z - q->pos[2]) / z, 0.0f, 1.0f);
            o[idx++] = qr_clampf(q->pos[2] / z, 0.0f, 1.0f);
        } else {
            o[idx++] = 1.0f;
            o[idx++] = 1.0f;
        }
    }

    if (env->route_extra_obs)
        qr_write_route_extra_obs(env, q, o, &idx);

    if (env->route_perception_obs == 1)
        qr_write_route_perception_obs(env, q, o, &idx, dx, dy, dz);
    else if (env->route_perception_obs == 2)
        qr_write_route_compact_perception_obs(env, q, o, &idx, dx, dy, dz);
    else if (env->route_perception_obs == 3)
        qr_write_route_affordance_obs(env, q, o, &idx, dx, dy, dz);

    if (env->action_history_obs) {
        for (int i = 0; i < QR_NUM_ACTIONS; i++)
            o[idx++] = env->has_prev_action ? env->prev_action[i] : 0.0f;
    }

    int patch_start = idx;
    qr_write_explore_patch_obs(env, q, o, &idx);
    qr_write_route_privileged_obs(env, q, o, patch_start);
}

static inline float qr_room_max_spawn_bound(const QuadrotorEnv* env) {
    if (env->room_size <= 0.0f)
        return env->episode_spawn_radius;

    float room_bound = env->room_size - QR_NAV_ROOM_MARGIN;
    if (room_bound < 0.05f)
        room_bound = env->room_size;
    return room_bound;
}

static inline float qr_room_spawn_bound(const QuadrotorEnv* env) {
    float room_bound = qr_room_max_spawn_bound(env);
    if (env->room_size <= 0.0f)
        return room_bound;

    if (env->episode_spawn_radius > 0.0f && env->episode_spawn_radius < room_bound)
        return env->episode_spawn_radius;

    return room_bound;
}

static inline void qr_sample_episode_spawn_radius(QuadrotorEnv* env) {
    float hi = env->spawn_radius;
    float lo = env->spawn_radius_min;

    if (hi < 0.0f)
        hi = 0.0f;
    if (lo < 0.0f || lo > hi)
        lo = hi;

    env->episode_spawn_radius = hi;
    if (lo < hi)
        env->episode_spawn_radius = qr_rand_range(env, lo, hi);
}

static inline float qr_clamp_room_xy(const QuadrotorEnv* env, float x) {
    if (env->room_size <= 0.0f)
        return x;

    float bound = env->room_size - 0.05f;
    if (bound <= 0.0f)
        bound = env->room_size;

    return qr_clampf(x, -bound, bound);
}

static inline float qr_clamp_room_xy_with_margin(const QuadrotorEnv* env,
        float x, float margin) {
    if (env->room_size <= 0.0f || margin <= 0.0f)
        return qr_clamp_room_xy(env, x);

    float bound = env->room_size - margin;
    if (bound <= 0.0f)
        bound = env->room_size;

    return qr_clampf(x, -bound, bound);
}

static inline float qr_safe_height(const QuadrotorEnv* env, float z) {
    float min_z = 0.3f;
    float max_z = env->ceiling_height - 0.3f;
    if (max_z < min_z)
        max_z = min_z;

    return qr_clampf(z, min_z, max_z);
}

static inline float qr_sample_route_target_z(QuadrotorEnv* env) {
    if (env->route_target_z_max > env->route_target_z_min) {
        return qr_safe_height(env, qr_rand_range(
            env, env->route_target_z_min, env->route_target_z_max));
    }
    return qr_safe_height(env, env->spawn_height);
}

static inline int qr_point_in_obstacle(const QuadrotorEnv* env, float x,
        float y, float z, float margin) {
    for (int i = 0; i < env->num_obstacles; i++) {
        const float* box = env->obstacles[i];
        if (x >= box[0] - margin && x <= box[1] + margin &&
                y >= box[2] - margin && y <= box[3] + margin &&
                z >= box[4] - margin && z <= box[5] + margin)
            return 1;
    }
    return 0;
}

static inline int qr_valid_xy(const QuadrotorEnv* env, float x, float y,
        float margin) {
    if (env->room_size > 0.0f &&
            (x < -env->room_size + margin || x > env->room_size - margin ||
             y < -env->room_size + margin || y > env->room_size - margin))
        return 0;
    return !qr_point_in_obstacle(env, x, y, env->spawn_height, margin);
}

static inline float qr_spawn_xy_margin(const QuadrotorEnv* env) {
    float margin = QR_DRONE_RADIUS;
    if (env->task_mode == QR_TASK_INDOOR_EXPLORE &&
            QR_EXPLORE_CLEARANCE_RADIUS > margin)
        margin = QR_EXPLORE_CLEARANCE_RADIUS;
    return margin;
}

static inline int qr_try_spawn_quadrotor_xy(QuadrotorEnv* env, float bound,
        float margin) {
    Quadrotor* q = &env->phys;
    for (int i = 0; i < 64; i++) {
        float x = qr_rand_range(env, -bound, bound);
        float y = qr_rand_range(env, -bound, bound);
        if (qr_valid_xy(env, x, y, margin)) {
            q->pos[0] = x;
            q->pos[1] = y;
            return 1;
        }
    }

    if (env->room_size <= 0.0f)
        return 0;

    float fallback_bound = qr_room_max_spawn_bound(env);
    for (int i = 0; i < 256; i++) {
        float x = qr_rand_range(env, -fallback_bound, fallback_bound);
        float y = qr_rand_range(env, -fallback_bound, fallback_bound);
        if (qr_valid_xy(env, x, y, margin)) {
            q->pos[0] = x;
            q->pos[1] = y;
            return 1;
        }
    }

    for (int iy = 0; iy < 17; iy++) {
        for (int ix = 0; ix < 17; ix++) {
            float fx = ((float)ix / 16.0f) * 2.0f - 1.0f;
            float fy = ((float)iy / 16.0f) * 2.0f - 1.0f;
            float x = fallback_bound * fx;
            float y = fallback_bound * fy;
            if (qr_valid_xy(env, x, y, margin)) {
                q->pos[0] = x;
                q->pos[1] = y;
                return 1;
            }
        }
    }

    return 0;
}

static inline int qr_route_target_wall_ok(const QuadrotorEnv* env, float x,
        float y) {
    if (env->room_size <= 0.0f || env->route_target_wall_margin <= 0.0f)
        return 1;
    float margin = env->route_target_wall_margin;
    return x >= -env->room_size + margin &&
           x <= env->room_size - margin &&
           y >= -env->room_size + margin &&
           y <= env->room_size - margin;
}

static inline int qr_explore_grid_dim(const QuadrotorEnv* env) {
    if (env->room_size <= 0.0f || env->explore_cell_size <= 0.0f)
        return 0;

    int dim = (int)ceilf((2.0f * env->room_size) / env->explore_cell_size);
    if (dim < 1) dim = 1;
    if (dim > QR_EXPLORE_MAX_GRID) dim = QR_EXPLORE_MAX_GRID;
    return dim;
}

static inline float qr_explore_cell_center(const QuadrotorEnv* env, int idx) {
    return -env->room_size + ((float)idx + 0.5f) * env->explore_cell_size;
}

static inline void qr_reset_exploration_grid(QuadrotorEnv* env) {
    memset(env->explored, 0, sizeof(env->explored));
    memset(env->explore_blocked, 0, sizeof(env->explore_blocked));
    memset(env->explore_frontier_stale, 0, sizeof(env->explore_frontier_stale));
    memset(env->explore_reachable, 0, sizeof(env->explore_reachable));
    env->explore_grid_dim = 0;
    env->explore_total_cells = 0;
    env->explored_cells = 0;
    env->explore_new_cells = 0.0f;
    env->explore_onboard_target_idx = -1;
    env->explore_onboard_target_frontier_idx = -1;
    env->explore_onboard_objective_idx = -1;
    env->explore_onboard_objective_depth = 0;
    env->explore_onboard_target_cluster_cells = 0;
    env->explore_onboard_region_anchor_idx = -1;
    env->explore_onboard_region_goal_frontier_idx = -1;
    env->explore_onboard_region_goal_unknown_idx = -1;
    env->explore_onboard_committed_viewpoint_idx = -1;
    env->explore_onboard_committed_unknown_idx = -1;
    env->explore_onboard_committed_until_tick = 0;
    env->explore_onboard_committed_start_tick = 0;
    env->explore_onboard_last_progress_tick = 0;
    env->explore_onboard_last_progress_cells = 0;
    env->explore_onboard_stale_marks = 0;
    env->explore_onboard_viewpoint_calls = 0;
    env->explore_onboard_viewpoint_candidate_sum = 0;
    env->explore_onboard_viewpoint_viable_sum = 0;
    env->explore_onboard_viewpoint_used_count = 0;
    env->explore_onboard_viewpoint_miss_count = 0;
    env->explore_onboard_viewpoint_score_sum = 0.0f;
    env->explore_onboard_viewpoint_clearance_sum = 0.0f;
    env->explore_onboard_viewpoint_depth_sum = 0;
    env->explore_onboard_target_calls = 0;
    env->explore_onboard_waypoint_depth_sum = 0;
    env->explore_onboard_selection_calls = 0;
    env->explore_onboard_prev_target_idx = -1;
    env->explore_onboard_prev_objective_idx = -1;
    env->explore_onboard_prev_selection_cells = 0;
    env->explore_onboard_target_switch_count = 0;
    env->explore_onboard_objective_switch_count = 0;
    env->explore_onboard_unproductive_objective_switch_count = 0;
    env->explore_onboard_committed_used_count = 0;
    env->explore_onboard_committed_drop_count = 0;
    env->explore_onboard_committed_age_sum = 0;
    env->explore_onboard_committed_last_used = 0;
    env->explore_onboard_viewpoint_last_candidates = 0;
    env->explore_onboard_viewpoint_last_viable = 0;
    env->explore_onboard_viewpoint_last_used = 0;
    env->explore_onboard_viewpoint_last_miss = 0;
    env->explore_onboard_viewpoint_best_idx = -1;
    env->explore_onboard_objective_idx = -1;
    env->explore_onboard_objective_depth = 0;
    env->explore_onboard_waypoint_idx = -1;
    env->explore_onboard_waypoint_depth = 0;
    env->explore_onboard_viewpoint_best_score = 0.0f;
    env->explore_onboard_viewpoint_best_clearance = 0.0f;
    env->explore_onboard_viewpoint_best_depth = 0;
    env->explore_onboard_region_next_id = 1;
    env->explore_onboard_active_region_id = -1;
    memset(env->explore_onboard_region_id, 0, sizeof(env->explore_onboard_region_id));
    memset(env->explore_onboard_region_cx, 0, sizeof(env->explore_onboard_region_cx));
    memset(env->explore_onboard_region_cy, 0, sizeof(env->explore_onboard_region_cy));
    memset(env->explore_onboard_region_frontier_idx, -1, sizeof(env->explore_onboard_region_frontier_idx));
    memset(env->explore_onboard_region_unknown_idx, -1, sizeof(env->explore_onboard_region_unknown_idx));
    memset(env->explore_onboard_region_cells, 0, sizeof(env->explore_onboard_region_cells));
    memset(env->explore_onboard_region_attempts, 0, sizeof(env->explore_onboard_region_attempts));
    memset(env->explore_onboard_region_failures, 0, sizeof(env->explore_onboard_region_failures));
    memset(env->explore_onboard_region_first_seen_tick, 0, sizeof(env->explore_onboard_region_first_seen_tick));
    memset(env->explore_onboard_region_last_seen_tick, 0, sizeof(env->explore_onboard_region_last_seen_tick));
    memset(env->explore_onboard_region_last_progress_tick, 0, sizeof(env->explore_onboard_region_last_progress_tick));
    memset(env->explore_onboard_region_last_progress_cells, 0, sizeof(env->explore_onboard_region_last_progress_cells));
    env->explore_onboard_tracker_calls = 0;
    env->explore_onboard_tracker_region_sum = 0;
    env->explore_onboard_tracker_switch_count = 0;
    env->explore_onboard_tracker_stale_count = 0;
    env->explore_onboard_tracker_active_age_sum = 0;
    env->explore_onboard_tracker_since_progress_sum = 0;
    env->explore_onboard_tracker_last_region_count = 0;
    env->explore_onboard_tracker_last_active_id = -1;
    env->explore_onboard_tracker_last_active_age = 0;
    env->explore_onboard_tracker_last_since_progress = 0;
    env->explore_onboard_tracker_last_stale = 0;

    if (env->task_mode != QR_TASK_INDOOR_EXPLORE)
        return;

    env->explore_grid_dim = qr_explore_grid_dim(env);
    for (int iy = 0; iy < env->explore_grid_dim; iy++) {
        for (int ix = 0; ix < env->explore_grid_dim; ix++) {
            float x = qr_explore_cell_center(env, ix);
            float y = qr_explore_cell_center(env, iy);
            if (qr_valid_xy(env, x, y, QR_EXPLORE_CELL_MARGIN))
                env->explore_total_cells++;
        }
    }

    if (env->explore_reachable_coverage && env->explore_total_cells > 0) {
        int start_idx = qr_explore_cell_index(env, env->phys.pos[0], env->phys.pos[1]);
        int reachable = qr_explore_reachable_cell_count(
            env,
            start_idx,
            env->explore_reachable);
        if (reachable > 0)
            env->explore_total_cells = reachable;
    }
}

static inline int qr_explore_cell_index(const QuadrotorEnv* env, float x, float y) {
    if (env->explore_grid_dim <= 0 || env->explore_total_cells <= 0)
        return -1;
    if (!qr_valid_xy(env, x, y, QR_EXPLORE_CELL_MARGIN))
        return -1;

    int ix = (int)floorf((x + env->room_size) / env->explore_cell_size);
    int iy = (int)floorf((y + env->room_size) / env->explore_cell_size);
    if (ix < 0 || ix >= env->explore_grid_dim ||
            iy < 0 || iy >= env->explore_grid_dim)
        return -1;
    return iy * QR_EXPLORE_MAX_GRID + ix;
}

static inline int qr_explore_reachable_cell_count(const QuadrotorEnv* env,
        int start_idx, unsigned char* out_reachable) {
    int dim = env->explore_grid_dim;
    if (dim <= 0 || start_idx < 0)
        return 0;

    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    unsigned char seen[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    memset(seen, 0, sizeof(seen));

    seen[start_idx] = 1;
    if (out_reachable)
        out_reachable[start_idx] = 1;
    int head = 0;
    int tail = 0;
    int count = 0;
    queue[tail++] = start_idx;

    while (head < tail) {
        int idx = queue[head++];
        int ix = idx % QR_EXPLORE_MAX_GRID;
        int iy = idx / QR_EXPLORE_MAX_GRID;
        if (ix < 0 || ix >= dim || iy < 0 || iy >= dim)
            continue;
        count++;

        const int nx[4] = {ix + 1, ix - 1, ix, ix};
        const int ny[4] = {iy, iy, iy + 1, iy - 1};
        for (int k = 0; k < 4; k++) {
            if (!qr_explore_valid_neighbor(env, dim, nx[k], ny[k]))
                continue;
            int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
            if (seen[nidx])
                continue;
            seen[nidx] = 1;
            if (out_reachable)
                out_reachable[nidx] = 1;
            if (tail < max_cells)
                queue[tail++] = nidx;
        }
    }

    return count;
}

static inline float qr_mark_explored_xy(QuadrotorEnv* env, float x, float y,
        float reward) {
    int idx = qr_explore_cell_index(env, x, y);
    if (idx < 0)
        return 0.0f;
    if (env->explore_reachable_coverage && !env->explore_reachable[idx])
        return 0.0f;
    if (env->explore_blocked[idx])
        return 0.0f;
    if (env->explored[idx])
        return 0.0f;

    env->explore_frontier_stale[idx] = 0;
    env->explored[idx] = 1;
    env->explored_cells++;
    env->explore_new_cells += 1.0f;
    return reward;
}

static inline void qr_mark_blocked_xy(QuadrotorEnv* env, float x, float y) {
    int idx = qr_explore_raw_cell_index(env, x, y);
    if (idx < 0)
        return;
    if (env->explored[idx]) {
        env->explored[idx] = 0;
        if (env->explored_cells > 0)
            env->explored_cells--;
    }
    env->explore_frontier_stale[idx] = 0;
    env->explore_blocked[idx] = 1;
}

static inline float qr_mark_explored(QuadrotorEnv* env, float reward) {
    return qr_mark_explored_xy(env, env->phys.pos[0], env->phys.pos[1], reward);
}

static inline float qr_mark_visible_explored(QuadrotorEnv* env, float reward) {
    float total = qr_mark_explored(env, reward);
    if (env->num_rays <= 0 || env->ray_length <= 0.0f)
        return total;

    const Quadrotor* q = &env->phys;
    float step = env->explore_cell_size * 0.75f;
    if (step < 0.10f)
        step = 0.10f;

    for (int i = 0; i < env->num_rays; i++) {
        float angle = 2.0f * QR_PI * (float)i / (float)env->num_rays;
        float dir_body[3] = {cosf(angle), sinf(angle), 0.0f};
        float dir_world[3];
        qr_body_to_world(q->quat, dir_body, dir_world);
        dir_world[2] = 0.0f;
        float dn = sqrtf(dir_world[0]*dir_world[0] + dir_world[1]*dir_world[1]);
        if (dn <= 1e-6f)
            continue;
        dir_world[0] /= dn;
        dir_world[1] /= dn;

        float dist = qr_room_ray_obs(env, q, i) * env->ray_length;
        if (dist <= 0.0f)
            continue;

        int hit = dist < env->ray_length - 1.0e-3f;
        float free_dist = dist;
        if (hit) {
            free_dist = dist - 0.5f * env->explore_cell_size;
            if (free_dist < 0.0f)
                free_dist = 0.0f;
        }

        for (float t = step; t < free_dist; t += step) {
            total += qr_mark_explored_xy(env,
                q->pos[0] + dir_world[0] * t,
                q->pos[1] + dir_world[1] * t,
                reward);
        }

        if (hit) {
            qr_mark_blocked_xy(env,
                q->pos[0] + dir_world[0] * dist,
                q->pos[1] + dir_world[1] * dist);
        } else {
            total += qr_mark_explored_xy(env,
                q->pos[0] + dir_world[0] * dist,
                q->pos[1] + dir_world[1] * dist,
                reward);
        }
    }

    return total;
}

static inline float qr_explore_coverage(const QuadrotorEnv* env) {
    if (env->explore_total_cells <= 0)
        return 0.0f;
    return (float)env->explored_cells / (float)env->explore_total_cells;
}

static inline float qr_explore_frontier_distance(const QuadrotorEnv* env,
        const Quadrotor* q, float out_dir_body[2]) {
    out_dir_body[0] = 1.0f;
    out_dir_body[1] = 0.0f;

    int dim = env->explore_grid_dim;
    if (dim <= 0 || env->room_size <= 0.0f)
        return 0.0f;

    int current_idx = qr_explore_cell_index(env, q->pos[0], q->pos[1]);
    float best_dist = 1.0e9f;
    float best_body_x = 1.0f;
    float best_body_y = 0.0f;

    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (idx != current_idx && !env->explored[idx])
                continue;

            int frontier = 0;
            const int nx[4] = {ix + 1, ix - 1, ix, ix};
            const int ny[4] = {iy, iy, iy + 1, iy - 1};
            for (int k = 0; k < 4; k++) {
                if (nx[k] < 0 || nx[k] >= dim || ny[k] < 0 || ny[k] >= dim)
                    continue;
                float x = qr_explore_cell_center(env, nx[k]);
                float y = qr_explore_cell_center(env, ny[k]);
                if (!qr_valid_xy(env, x, y, QR_EXPLORE_CELL_MARGIN))
                    continue;
                int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                if (!env->explored[nidx]) {
                    frontier = 1;
                    break;
                }
            }
            if (!frontier)
                continue;

            float x = qr_explore_cell_center(env, ix);
            float y = qr_explore_cell_center(env, iy);
            float delta_world[3] = {x - q->pos[0], y - q->pos[1], 0.0f};
            float dist = sqrtf(delta_world[0]*delta_world[0] +
                               delta_world[1]*delta_world[1]);
            if (dist < best_dist) {
                float delta_body[3];
                qr_world_to_body(q->quat, delta_world, delta_body);
                best_dist = dist;
                if (dist > 1e-4f) {
                    float inv_dist = 1.0f / dist;
                    best_body_x = qr_clampf(delta_body[0] * inv_dist, -1.0f, 1.0f);
                    best_body_y = qr_clampf(delta_body[1] * inv_dist, -1.0f, 1.0f);
                }
            }
        }
    }

    if (best_dist >= 1.0e8f)
        return 0.0f;
    out_dir_body[0] = best_body_x;
    out_dir_body[1] = best_body_y;
    return best_dist;
}

static inline int qr_explore_valid_neighbor(const QuadrotorEnv* env,
        int dim, int ix, int iy) {
    if (ix < 0 || ix >= dim || iy < 0 || iy >= dim)
        return 0;
    float x = qr_explore_cell_center(env, ix);
    float y = qr_explore_cell_center(env, iy);
    return qr_valid_xy(env, x, y, QR_EXPLORE_CELL_MARGIN);
}

static inline int qr_explore_onboard_in_bounds(const QuadrotorEnv* env,
        int dim, int ix, int iy) {
    if (ix < 0 || ix >= dim || iy < 0 || iy >= dim)
        return 0;
    float x = qr_explore_cell_center(env, ix);
    float y = qr_explore_cell_center(env, iy);
    float margin = QR_EXPLORE_CELL_MARGIN;
    return x >= -env->room_size + margin &&
           x <= env->room_size - margin &&
           y >= -env->room_size + margin &&
           y <= env->room_size - margin;
}

static inline int qr_explore_onboard_clearance_ok(
        const QuadrotorEnv* env, int dim, int ix, int iy) {
    if (!env->explore_onboard_clearance_passable ||
            env->explore_onboard_passable_radius <= 0.0f)
        return 1;
    if (!qr_explore_onboard_in_bounds(env, dim, ix, iy))
        return 0;

    float radius = env->explore_onboard_passable_radius;
    float cell = env->explore_cell_size > 1e-6f
        ? env->explore_cell_size
        : QR_EXPLORE_CELL_SIZE;
    int r = (int)ceilf(radius / cell);
    if (r < 1)
        r = 1;

    for (int oy = -r; oy <= r; oy++) {
        for (int ox = -r; ox <= r; ox++) {
            if (ox == 0 && oy == 0)
                continue;
            float dist = sqrtf((float)(ox * ox + oy * oy)) * cell;
            if (dist + 1e-6f >= radius)
                continue;
            int nx = ix + ox;
            int ny = iy + oy;
            if (!qr_explore_onboard_in_bounds(env, dim, nx, ny))
                return 0;
            int nidx = ny * QR_EXPLORE_MAX_GRID + nx;
            if (env->explore_blocked[nidx])
                return 0;
        }
    }
    return 1;
}

static inline int qr_explore_onboard_known_passable_cell(
        const QuadrotorEnv* env, int dim, int idx, int current_idx) {
    if (idx < 0)
        return 0;
    int ix = idx % QR_EXPLORE_MAX_GRID;
    int iy = idx / QR_EXPLORE_MAX_GRID;
    if (!qr_explore_onboard_in_bounds(env, dim, ix, iy))
        return 0;
    if (env->explore_blocked[idx])
        return 0;
    if (idx == current_idx)
        return 1;
    return env->explored[idx] &&
        qr_explore_onboard_clearance_ok(env, dim, ix, iy);
}

static inline int qr_explore_onboard_optimistic_passable_cell(
        const QuadrotorEnv* env, int dim, int idx, int current_idx) {
    if (idx < 0)
        return 0;
    int ix = idx % QR_EXPLORE_MAX_GRID;
    int iy = idx / QR_EXPLORE_MAX_GRID;
    if (!qr_explore_onboard_in_bounds(env, dim, ix, iy))
        return 0;
    if (env->explore_blocked[idx])
        return 0;
    if (idx == current_idx)
        return 1;
    if (env->explored[idx])
        return qr_explore_onboard_clearance_ok(env, dim, ix, iy);
    return !env->explore_frontier_stale[idx];
}

static inline int qr_explore_onboard_unknown_passable_cell(
        const QuadrotorEnv* env, int dim, int idx) {
    if (idx < 0)
        return 0;
    int ix = idx % QR_EXPLORE_MAX_GRID;
    int iy = idx / QR_EXPLORE_MAX_GRID;
    if (!qr_explore_onboard_in_bounds(env, dim, ix, iy))
        return 0;
    return !env->explored[idx] &&
        !env->explore_blocked[idx] &&
        !env->explore_frontier_stale[idx];
}

static inline int qr_explore_onboard_frontier_unknown_count(
        const QuadrotorEnv* env, int dim, int ix, int iy,
        int* first_unknown_idx) {
    const int nx[4] = {ix + 1, ix - 1, ix, ix};
    const int ny[4] = {iy, iy, iy + 1, iy - 1};
    int count = 0;
    if (first_unknown_idx)
        *first_unknown_idx = -1;

    for (int k = 0; k < 4; k++) {
        if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
            continue;
        int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
        if (qr_explore_onboard_unknown_passable_cell(env, dim, nidx)) {
            if (first_unknown_idx && *first_unknown_idx < 0)
                *first_unknown_idx = nidx;
            count++;
        }
    }
    return count;
}

static inline int qr_explore_onboard_unknown_cell(const QuadrotorEnv* env,
        int dim, int idx) {
    if (idx < 0)
        return 0;
    int ix = idx % QR_EXPLORE_MAX_GRID;
    int iy = idx / QR_EXPLORE_MAX_GRID;
    if (!qr_explore_onboard_in_bounds(env, dim, ix, iy))
        return 0;
    return qr_explore_onboard_unknown_passable_cell(env, dim, idx);
}

static inline int qr_explore_onboard_frontier_cell(const QuadrotorEnv* env,
        int dim, int idx, int current_idx) {
    if (idx < 0)
        return 0;
    int ix = idx % QR_EXPLORE_MAX_GRID;
    int iy = idx / QR_EXPLORE_MAX_GRID;
    if (ix < 0 || ix >= dim || iy < 0 || iy >= dim)
        return 0;
    if (!qr_explore_onboard_known_passable_cell(env, dim, idx, current_idx))
        return 0;
    return qr_explore_onboard_frontier_unknown_count(
        env, dim, ix, iy, NULL) > 0;
}

static inline float qr_explore_onboard_nearest_blocked_cells(
        const QuadrotorEnv* env, int dim, int ix, int iy, int radius) {
    if (!qr_explore_onboard_in_bounds(env, dim, ix, iy))
        return 0.0f;

    float best = (float)radius + 1.0f;
    for (int oy = -radius; oy <= radius; oy++) {
        for (int ox = -radius; ox <= radius; ox++) {
            if (ox == 0 && oy == 0)
                continue;
            float dist = sqrtf((float)(ox * ox + oy * oy));
            if (dist > (float)radius + 1e-6f)
                continue;
            int nx = ix + ox;
            int ny = iy + oy;
            int blocked = 0;
            if (!qr_explore_onboard_in_bounds(env, dim, nx, ny)) {
                blocked = 1;
            } else {
                int nidx = ny * QR_EXPLORE_MAX_GRID + nx;
                blocked = env->explore_blocked[nidx] ? 1 : 0;
            }
            if (blocked && dist < best)
                best = dist;
        }
    }
    return best;
}

static inline float qr_explore_onboard_passage_quality_score(
        const QuadrotorEnv* env, int dim, int ix, int iy) {
    if (!env->explore_onboard_passage_quality ||
            env->explore_onboard_passage_weight <= 0.0f)
        return 0.0f;

    float nearest = qr_explore_onboard_nearest_blocked_cells(
        env, dim, ix, iy, 3);
    float centered = qr_clampf(nearest - 1.0f, -1.0f, 2.0f);
    float score = env->explore_onboard_passage_weight * centered;
    if (nearest <= 1.05f)
        score -= env->explore_onboard_passage_weight;
    return score;
}

static inline int qr_explore_onboard_wall_edge_cell(
        const QuadrotorEnv* env, int dim, int idx) {
    if (!env->explore_onboard_passage_quality || idx < 0)
        return 0;
    int ix = idx % QR_EXPLORE_MAX_GRID;
    int iy = idx / QR_EXPLORE_MAX_GRID;
    return qr_explore_onboard_nearest_blocked_cells(env, dim, ix, iy, 2) <= 1.05f;
}

static inline float qr_explore_onboard_frontier_score(
        const QuadrotorEnv* env, int dim, int idx, int current_idx,
        int depth, int unknown_count) {
    int ix = idx % QR_EXPLORE_MAX_GRID;
    int iy = idx / QR_EXPLORE_MAX_GRID;
    float score = (float)unknown_count - 0.03f * (float)depth;
    if (idx == current_idx)
        score -= QR_EXPLORE_ONBOARD_CURRENT_CELL_PENALTY;
    score += qr_explore_onboard_passage_quality_score(env, dim, ix, iy);
    return score;
}

static inline float qr_explore_onboard_budget_cluster_score(
        const QuadrotorEnv* env, float best_cell_score, int cluster_cells,
        int unknown_sum, int depth) {
    float depth_cost = env->explore_onboard_budget_depth_bias +
        env->explore_onboard_budget_depth_weight * (float)(depth > 0 ? depth : 1);
    if (depth_cost < 1.0f)
        depth_cost = 1.0f;

    float gain = env->explore_onboard_budget_gain_weight * (float)unknown_sum;
    gain += env->explore_onboard_budget_cluster_weight *
        sqrtf((float)(cluster_cells > 0 ? cluster_cells : 1));
    float score = gain / depth_cost;
    score += env->explore_onboard_budget_cell_weight * best_cell_score;
    return score;
}

static inline void qr_explore_planner_stats(const QuadrotorEnv* env,
        QrExplorePlannerStats* out) {
    memset(out, 0, sizeof(*out));
    out->target_idx = env->explore_onboard_target_idx;
    out->target_frontier_idx = env->explore_onboard_target_frontier_idx;
    out->target_cluster_cells = env->explore_onboard_target_cluster_cells;
    out->viewpoint_candidate_count = env->explore_onboard_viewpoint_last_candidates;
    out->viewpoint_viable_count = env->explore_onboard_viewpoint_last_viable;
    out->viewpoint_used = env->explore_onboard_viewpoint_last_used;
    out->viewpoint_miss = env->explore_onboard_viewpoint_last_miss;
    out->viewpoint_best_idx = env->explore_onboard_viewpoint_best_idx;
    out->waypoint_idx = env->explore_onboard_waypoint_idx;
    out->waypoint_depth = env->explore_onboard_waypoint_depth;
    out->viewpoint_best_score = env->explore_onboard_viewpoint_best_score;
    out->viewpoint_best_clearance = env->explore_onboard_viewpoint_best_clearance;
    out->viewpoint_best_depth = env->explore_onboard_viewpoint_best_depth;
    out->committed_viewpoint_idx = env->explore_onboard_committed_viewpoint_idx;
    out->committed_viewpoint_used = env->explore_onboard_committed_last_used;
    out->committed_viewpoint_age =
        env->explore_onboard_committed_viewpoint_idx >= 0
            ? env->tick - env->explore_onboard_committed_start_tick
            : 0;
    out->tracker_region_count = env->explore_onboard_tracker_last_region_count;
    out->tracker_active_id = env->explore_onboard_tracker_last_active_id;
    out->tracker_active_age = env->explore_onboard_tracker_last_active_age;
    out->tracker_since_progress =
        env->explore_onboard_tracker_last_since_progress;
    out->tracker_stale = env->explore_onboard_tracker_last_stale;

    if (env->task_mode != QR_TASK_INDOOR_EXPLORE ||
            env->explore_grid_dim <= 0)
        return;

    int dim = env->explore_grid_dim;
    int current_idx = qr_explore_cell_index(
        env, env->phys.pos[0], env->phys.pos[1]);
    int target_idx = env->explore_onboard_target_idx;
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    if (target_idx >= 0 && target_idx < max_cells &&
            !env->explored[target_idx] &&
            !env->explore_blocked[target_idx]) {
        out->target_exists = 1;
        out->target_age = env->tick - env->explore_onboard_last_progress_tick;
        if (out->target_age < 0)
            out->target_age = 0;
    }

    unsigned char frontier_seen[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    memset(frontier_seen, 0, sizeof(frontier_seen));

    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (env->explore_blocked[idx])
                out->blocked_cells++;
            if (env->explore_frontier_stale[idx])
                out->stale_frontier_count++;
            if (!qr_explore_onboard_frontier_cell(env, dim, idx, current_idx))
                continue;
            out->frontier_count++;
            if (frontier_seen[idx])
                continue;
            out->frontier_cluster_count++;

            int head = 0;
            int tail = 0;
            frontier_seen[idx] = 1;
            queue[tail++] = idx;
            while (head < tail) {
                int fidx = queue[head++];
                int fx = fidx % QR_EXPLORE_MAX_GRID;
                int fy = fidx / QR_EXPLORE_MAX_GRID;
                const int nx[4] = {fx + 1, fx - 1, fx, fx};
                const int ny[4] = {fy, fy, fy + 1, fy - 1};
                for (int k = 0; k < 4; k++) {
                    if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
                        continue;
                    int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                    if (frontier_seen[nidx] ||
                            !qr_explore_onboard_frontier_cell(env, dim, nidx, current_idx))
                        continue;
                    frontier_seen[nidx] = 1;
                    if (tail < max_cells)
                        queue[tail++] = nidx;
                }
            }
        }
    }
}

static inline int qr_explore_first_unknown_neighbor(const QuadrotorEnv* env,
        int dim, int ix, int iy) {
    const int nx[4] = {ix + 1, ix - 1, ix, ix};
    const int ny[4] = {iy, iy, iy + 1, iy - 1};
    for (int k = 0; k < 4; k++) {
        if (!qr_explore_valid_neighbor(env, dim, nx[k], ny[k]))
            continue;
        int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
        if (!env->explored[nidx])
            return nidx;
    }
    return -1;
}

static inline int qr_explore_unknown_neighbor_count(const QuadrotorEnv* env,
        int dim, int ix, int iy) {
    const int nx[4] = {ix + 1, ix - 1, ix, ix};
    const int ny[4] = {iy, iy, iy + 1, iy - 1};
    int count = 0;
    for (int k = 0; k < 4; k++) {
        if (!qr_explore_valid_neighbor(env, dim, nx[k], ny[k]))
            continue;
        int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
        if (!env->explored[nidx])
            count++;
    }
    return count;
}

static inline int qr_explore_nearest_valid_cell_index(const QuadrotorEnv* env,
        float x, float y) {
    int dim = env->explore_grid_dim;
    if (dim <= 0)
        return -1;

    int best_idx = -1;
    float best_dist2 = 1.0e30f;
    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            if (!qr_explore_valid_neighbor(env, dim, ix, iy))
                continue;
            float cx = qr_explore_cell_center(env, ix);
            float cy = qr_explore_cell_center(env, iy);
            float dx = cx - x;
            float dy = cy - y;
            float dist2 = dx*dx + dy*dy;
            if (best_idx < 0 || dist2 < best_dist2) {
                best_idx = iy * QR_EXPLORE_MAX_GRID + ix;
                best_dist2 = dist2;
            }
        }
    }
    return best_idx;
}

static inline float qr_explore_target_cell_direction(const QuadrotorEnv* env,
        const Quadrotor* q, int idx, float out_dir_body[2]) {
    int ix = idx % QR_EXPLORE_MAX_GRID;
    int iy = idx / QR_EXPLORE_MAX_GRID;
    if (ix < 0 || ix >= env->explore_grid_dim ||
            iy < 0 || iy >= env->explore_grid_dim)
        return 0.0f;

    float x = qr_explore_cell_center(env, ix);
    float y = qr_explore_cell_center(env, iy);
    float delta_world[3] = {x - q->pos[0], y - q->pos[1], 0.0f};
    float dist = sqrtf(delta_world[0]*delta_world[0] +
                       delta_world[1]*delta_world[1]);
    if (dist <= 1e-4f)
        return 0.0f;

    float delta_body[3];
    qr_world_to_body(q->quat, delta_world, delta_body);
    float inv_dist = 1.0f / dist;
    out_dir_body[0] = qr_clampf(delta_body[0] * inv_dist, -1.0f, 1.0f);
    out_dir_body[1] = qr_clampf(delta_body[1] * inv_dist, -1.0f, 1.0f);
    return dist;
}

static inline float qr_explore_privileged_unexplored_path_target(
        const QuadrotorEnv* env, const Quadrotor* q, int current_idx,
        float out_dir_body[2]) {
    int dim = env->explore_grid_dim;
    if (dim <= 0 || current_idx < 0)
        return 0.0f;

    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    int parent[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    for (int i = 0; i < max_cells; i++)
        parent[i] = -2;

    parent[current_idx] = -1;
    int head = 0;
    int tail = 0;
    queue[tail++] = current_idx;

    int target_idx = -1;
    while (head < tail) {
        int idx = queue[head++];
        int ix = idx % QR_EXPLORE_MAX_GRID;
        int iy = idx / QR_EXPLORE_MAX_GRID;
        if (ix >= dim || iy >= dim)
            continue;

        if (idx != current_idx && !env->explored[idx]) {
            target_idx = idx;
            break;
        }

        const int nx[4] = {ix + 1, ix - 1, ix, ix};
        const int ny[4] = {iy, iy, iy + 1, iy - 1};
        for (int k = 0; k < 4; k++) {
            if (!qr_explore_valid_neighbor(env, dim, nx[k], ny[k]))
                continue;
            int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
            if (parent[nidx] != -2)
                continue;
            parent[nidx] = idx;
            if (tail < max_cells)
                queue[tail++] = nidx;
        }
    }

    if (target_idx < 0)
        return 0.0f;

    int waypoint_idx = target_idx;
    int path_steps = 0;
    int cursor = target_idx;
    while (cursor != current_idx && cursor >= 0) {
        path_steps++;
        cursor = parent[cursor];
    }
    if (path_steps <= 0)
        path_steps = 1;
    while (parent[waypoint_idx] >= 0 && parent[waypoint_idx] != current_idx)
        waypoint_idx = parent[waypoint_idx];

    if (qr_explore_target_cell_direction(env, q, waypoint_idx, out_dir_body) <= 0.0f)
        return 0.0f;
    return (float)path_steps * env->explore_cell_size;
}

static inline void qr_explore_reset_onboard_viewpoint_step_debug(
        QuadrotorEnv* env);
static inline void qr_explore_record_onboard_selection(
        QuadrotorEnv* env, int target_idx, int objective_idx);

static inline int qr_explore_privileged_line_clear(const QuadrotorEnv* env,
        float x0, float y0, float x1, float y1) {
    float dx = x1 - x0;
    float dy = y1 - y0;
    float dist = sqrtf(dx*dx + dy*dy);
    float step = env->explore_cell_size * 0.50f;
    if (step < 0.05f)
        step = 0.05f;
    int steps = (int)ceilf(dist / step);
    if (steps < 1)
        steps = 1;
    for (int i = 0; i <= steps; i++) {
        float t = (float)i / (float)steps;
        float x = x0 + dx * t;
        float y = y0 + dy * t;
        if (!qr_valid_xy(env, x, y, QR_EXPLORE_CELL_MARGIN))
            return 0;
    }
    return 1;
}

static inline int qr_explore_privileged_visible_unknown_gain(
        const QuadrotorEnv* env, int dim, int center_idx, int radius) {
    int cx = center_idx % QR_EXPLORE_MAX_GRID;
    int cy = center_idx / QR_EXPLORE_MAX_GRID;
    int gain = 0;

    for (int oy = -radius; oy <= radius; oy++) {
        for (int ox = -radius; ox <= radius; ox++) {
            int ax = ox < 0 ? -ox : ox;
            int ay = oy < 0 ? -oy : oy;
            if (ax + ay > radius)
                continue;
            int ix = cx + ox;
            int iy = cy + oy;
            if (ix < 0 || ix >= dim || iy < 0 || iy >= dim)
                continue;
            if (!qr_explore_valid_neighbor(env, dim, ix, iy))
                continue;
            int idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (env->explored[idx])
                continue;
            gain++;
        }
    }
    return gain;
}

static inline float qr_explore_privileged_clearance_score(
        const QuadrotorEnv* env, int dim, int center_idx) {
    int cx = center_idx % QR_EXPLORE_MAX_GRID;
    int cy = center_idx / QR_EXPLORE_MAX_GRID;
    int valid = 0;
    int total = 0;
    for (int oy = -2; oy <= 2; oy++) {
        for (int ox = -2; ox <= 2; ox++) {
            if (ox == 0 && oy == 0)
                continue;
            int ax = ox < 0 ? -ox : ox;
            int ay = oy < 0 ? -oy : oy;
            if (ax + ay > 2)
                continue;
            total++;
            int ix = cx + ox;
            int iy = cy + oy;
            if (qr_explore_valid_neighbor(env, dim, ix, iy))
                valid++;
        }
    }
    return total > 0 ? (float)valid / (float)total : 0.0f;
}

static inline float qr_explore_privileged_region_path_target(
        QuadrotorEnv* env, const Quadrotor* q, int current_idx,
        float out_dir_body[2]) {
    int dim = env->explore_grid_dim;
    if (dim <= 0 || current_idx < 0)
        return 0.0f;
    qr_explore_reset_onboard_viewpoint_step_debug(env);

    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    int parent[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int depth[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    for (int i = 0; i < max_cells; i++) {
        parent[i] = -2;
        depth[i] = -1;
    }

    parent[current_idx] = -1;
    depth[current_idx] = 0;
    int head = 0;
    int tail = 0;
    queue[tail++] = current_idx;

    while (head < tail) {
        int idx = queue[head++];
        int ix = idx % QR_EXPLORE_MAX_GRID;
        int iy = idx / QR_EXPLORE_MAX_GRID;
        if (ix >= dim || iy >= dim)
            continue;

        const int nx[4] = {ix + 1, ix - 1, ix, ix};
        const int ny[4] = {iy, iy, iy + 1, iy - 1};
        for (int k = 0; k < 4; k++) {
            if (!qr_explore_valid_neighbor(env, dim, nx[k], ny[k]))
                continue;
            int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
            if (parent[nidx] != -2)
                continue;
            parent[nidx] = idx;
            depth[nidx] = depth[idx] + 1;
            if (tail < max_cells)
                queue[tail++] = nidx;
        }
    }

    unsigned char seen[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int region_queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    memset(seen, 0, sizeof(seen));

    int best_target_idx = -1;
    int best_region_cells = 0;
    int best_depth = 0;
    int best_gain = 0;
    float best_score = -1.0e9f;

    for (int sy = 0; sy < dim; sy++) {
        for (int sx = 0; sx < dim; sx++) {
            int seed_idx = sy * QR_EXPLORE_MAX_GRID + sx;
            if (seen[seed_idx] || env->explored[seed_idx] ||
                    parent[seed_idx] == -2 ||
                    !qr_explore_valid_neighbor(env, dim, sx, sy))
                continue;

            int rhead = 0;
            int rtail = 0;
            region_queue[rtail++] = seed_idx;
            seen[seed_idx] = 1;
            int region_cells = 0;
            int region_best_idx = -1;
            int region_best_depth = 0;
            int region_best_gain = 0;
            float region_best_clearance = 0.0f;
            float region_best_candidate_score = -1.0e9f;

            while (rhead < rtail) {
                int idx = region_queue[rhead++];
                region_cells++;
                int ix = idx % QR_EXPLORE_MAX_GRID;
                int iy = idx / QR_EXPLORE_MAX_GRID;

                int gain = qr_explore_privileged_visible_unknown_gain(
                    env, dim, idx, 4);
                float clearance = qr_explore_privileged_clearance_score(
                    env, dim, idx);
                int d = depth[idx] >= 0 ? depth[idx] : 9999;
                float candidate_score = 1.20f * (float)gain;
                candidate_score += 1.40f * clearance;
                candidate_score -= 0.055f * (float)d;
                if (d <= 1)
                    candidate_score -= 0.35f;
                if (region_best_idx < 0 || candidate_score > region_best_candidate_score ||
                        (candidate_score == region_best_candidate_score &&
                            d < region_best_depth)) {
                    region_best_idx = idx;
                    region_best_depth = d;
                    region_best_gain = gain;
                    region_best_clearance = clearance;
                    region_best_candidate_score = candidate_score;
                }

                const int nx[4] = {ix + 1, ix - 1, ix, ix};
                const int ny[4] = {iy, iy, iy + 1, iy - 1};
                for (int k = 0; k < 4; k++) {
                    if (!qr_explore_valid_neighbor(env, dim, nx[k], ny[k]))
                        continue;
                    int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                    if (seen[nidx] || env->explored[nidx] || parent[nidx] == -2)
                        continue;
                    seen[nidx] = 1;
                    if (rtail < max_cells)
                        region_queue[rtail++] = nidx;
                }
            }

            if (region_best_idx < 0)
                continue;
            float size_gain = sqrtf((float)region_cells);
            float travel = sqrtf((float)(region_best_depth + 1));
            float score = (2.30f * (float)region_best_gain +
                    0.35f * size_gain +
                    0.90f * region_best_clearance) / travel;
            score -= 0.012f * (float)region_best_depth;
            if (region_best_depth <= 1)
                score += 0.35f;

            if (best_target_idx < 0 || score > best_score ||
                    (score == best_score && region_best_depth < best_depth)) {
                best_target_idx = region_best_idx;
                best_region_cells = region_cells;
                best_depth = region_best_depth;
                best_gain = region_best_gain;
                best_score = score;
            }
        }
    }

    if (best_target_idx < 0)
        return qr_explore_privileged_unexplored_path_target(
            env, q, current_idx, out_dir_body);

    env->explore_onboard_target_idx = best_target_idx;
    env->explore_onboard_target_frontier_idx = best_target_idx;
    env->explore_onboard_target_cluster_cells = best_region_cells;
    env->explore_onboard_waypoint_depth = best_depth;
    env->explore_onboard_waypoint_depth_sum += best_depth > 0 ? best_depth : 1;
    env->explore_onboard_target_calls++;
    env->explore_onboard_viewpoint_best_idx = best_target_idx;
    env->explore_onboard_viewpoint_best_score = best_score;
    env->explore_onboard_viewpoint_best_clearance =
        qr_explore_privileged_clearance_score(env, dim, best_target_idx);
    env->explore_onboard_viewpoint_best_depth = best_depth;
    env->explore_onboard_viewpoint_calls++;
    env->explore_onboard_viewpoint_candidate_sum += best_region_cells;
    env->explore_onboard_viewpoint_viable_sum += best_gain;
    env->explore_onboard_viewpoint_score_sum += best_score;
    env->explore_onboard_viewpoint_clearance_sum +=
        env->explore_onboard_viewpoint_best_clearance;
    env->explore_onboard_viewpoint_depth_sum += best_depth;
    env->explore_onboard_viewpoint_used_count++;
    env->explore_onboard_viewpoint_last_candidates = best_region_cells;
    env->explore_onboard_viewpoint_last_viable = best_gain;
    env->explore_onboard_viewpoint_last_used = 1;

    int waypoint_idx = best_target_idx;
    while (parent[waypoint_idx] >= 0 && parent[waypoint_idx] != current_idx)
        waypoint_idx = parent[waypoint_idx];
    env->explore_onboard_waypoint_idx = waypoint_idx;
    qr_explore_record_onboard_selection(env, best_target_idx, best_target_idx);

    if (qr_explore_target_cell_direction(env, q, waypoint_idx, out_dir_body) <= 0.0f)
        return 0.0f;
    int path_steps = best_depth > 0 ? best_depth : 1;
    return (float)path_steps * env->explore_cell_size;
}

static inline int qr_explore_select_onboard_frontier_region_target(
        const QuadrotorEnv* env, int dim, const int* parent, const int* depth,
        int seed_idx, int current_idx, int* out_frontier_idx,
        int* out_unknown_idx, int* out_depth, int* out_cluster_cells) {
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    if (seed_idx < 0 || seed_idx >= max_cells || parent[seed_idx] == -2 ||
            !qr_explore_onboard_frontier_cell(env, dim, seed_idx, current_idx))
        return 0;

    unsigned char seen[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    memset(seen, 0, sizeof(seen));

    int head = 0;
    int tail = 0;
    seen[seed_idx] = 1;
    queue[tail++] = seed_idx;

    int best_frontier_idx = -1;
    int best_unknown_idx = -1;
    int best_depth = 0;
    float best_score = -1.0e9f;
    int cluster_cells = 0;
    int commit_ready = env->explore_onboard_region_commit &&
        (env->explore_onboard_region_commit_stall_ticks <= 0 ||
         env->tick - env->explore_onboard_last_progress_tick >=
            env->explore_onboard_region_commit_stall_ticks);
    int committed_frontier_idx = commit_ready
        ? env->explore_onboard_region_goal_frontier_idx
        : -1;
    int committed_found = 0;
    int committed_unknown_idx = -1;
    int committed_depth = 0;
    float committed_score = -1.0e9f;

    while (head < tail) {
        int idx = queue[head++];
        cluster_cells++;
        int ix = idx % QR_EXPLORE_MAX_GRID;
        int iy = idx / QR_EXPLORE_MAX_GRID;

        int first_unknown = -1;
        int unknown_count = qr_explore_onboard_frontier_unknown_count(
            env, dim, ix, iy, &first_unknown);
        if (unknown_count > 0 && first_unknown >= 0) {
            float score = qr_explore_onboard_frontier_score(
                env, dim, idx, current_idx, depth[idx], unknown_count);
            if (idx == committed_frontier_idx) {
                committed_found = 1;
                committed_unknown_idx = first_unknown;
                committed_depth = depth[idx];
                committed_score = score;
            }
            if (best_frontier_idx < 0 || score > best_score ||
                    (score == best_score && depth[idx] < best_depth)) {
                best_score = score;
                best_depth = depth[idx];
                best_frontier_idx = idx;
                best_unknown_idx = first_unknown;
            }
        }

        const int nx[4] = {ix + 1, ix - 1, ix, ix};
        const int ny[4] = {iy, iy, iy + 1, iy - 1};
        for (int k = 0; k < 4; k++) {
            if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
                continue;
            int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
            if (seen[nidx] || parent[nidx] == -2 ||
                    !qr_explore_onboard_frontier_cell(env, dim, nidx, current_idx))
                continue;
            seen[nidx] = 1;
            if (tail < max_cells)
                queue[tail++] = nidx;
        }
    }

    if (committed_found &&
            (best_frontier_idx < 0 ||
             committed_score >= best_score - QR_EXPLORE_ONBOARD_REGION_COMMIT_SCORE_GAP)) {
        *out_frontier_idx = committed_frontier_idx;
        *out_unknown_idx = committed_unknown_idx;
        *out_depth = committed_depth;
        *out_cluster_cells = cluster_cells;
        return 1;
    }

    if (best_frontier_idx < 0)
        return 0;
    *out_frontier_idx = best_frontier_idx;
    *out_unknown_idx = best_unknown_idx;
    *out_depth = best_depth;
    *out_cluster_cells = cluster_cells;
    return 1;
}

static inline int qr_explore_select_onboard_budget_frontier_target(
        QuadrotorEnv* env, int dim, const int* parent, const int* depth,
        int current_idx, int* out_frontier_idx, int* out_unknown_idx,
        int* out_depth, int* out_cluster_cells) {
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    unsigned char seen[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    memset(seen, 0, sizeof(seen));

    int best_frontier_idx = -1;
    int best_unknown_idx = -1;
    int best_depth = 0;
    int best_cluster_cells = 0;
    int region_count = 0;
    int region_unknown_sum = 0;
    float best_region_score = -1.0e9f;

    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int seed_idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (seen[seed_idx] || parent[seed_idx] == -2 ||
                    !qr_explore_onboard_frontier_cell(env, dim, seed_idx, current_idx))
                continue;

            int head = 0;
            int tail = 0;
            int cluster_cells = 0;
            int unknown_sum = 0;
            int region_best_frontier_idx = -1;
            int region_best_unknown_idx = -1;
            int region_best_depth = 0;
            float region_best_cell_score = -1.0e9f;

            seen[seed_idx] = 1;
            queue[tail++] = seed_idx;
            while (head < tail) {
                int idx = queue[head++];
                int fx = idx % QR_EXPLORE_MAX_GRID;
                int fy = idx / QR_EXPLORE_MAX_GRID;
                cluster_cells++;

                int first_unknown = -1;
                int unknown_count = qr_explore_onboard_frontier_unknown_count(
                    env, dim, fx, fy, &first_unknown);
                if (unknown_count > 0 && first_unknown >= 0) {
                    unknown_sum += unknown_count;
                    float score = qr_explore_onboard_frontier_score(
                        env, dim, idx, current_idx, depth[idx], unknown_count);
                    if (region_best_frontier_idx < 0 ||
                            score > region_best_cell_score ||
                            (score == region_best_cell_score &&
                                depth[idx] < region_best_depth)) {
                        region_best_cell_score = score;
                        region_best_depth = depth[idx];
                        region_best_frontier_idx = idx;
                        region_best_unknown_idx = first_unknown;
                    }
                }

                const int nx[4] = {fx + 1, fx - 1, fx, fx};
                const int ny[4] = {fy, fy, fy + 1, fy - 1};
                for (int k = 0; k < 4; k++) {
                    if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
                        continue;
                    int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                    if (seen[nidx] || parent[nidx] == -2 ||
                            !qr_explore_onboard_frontier_cell(env, dim, nidx, current_idx))
                        continue;
                    seen[nidx] = 1;
                    if (tail < max_cells)
                        queue[tail++] = nidx;
                }
            }

            if (region_best_frontier_idx < 0 || cluster_cells <= 0)
                continue;

            float region_score = qr_explore_onboard_budget_cluster_score(
                env, region_best_cell_score, cluster_cells, unknown_sum,
                region_best_depth);
            region_count++;
            region_unknown_sum += unknown_sum;
            if (best_frontier_idx < 0 || region_score > best_region_score ||
                    (region_score == best_region_score &&
                        region_best_depth < best_depth)) {
                best_region_score = region_score;
                best_frontier_idx = region_best_frontier_idx;
                best_unknown_idx = region_best_unknown_idx;
                best_depth = region_best_depth;
                best_cluster_cells = cluster_cells;
            }
        }
    }

    if (best_frontier_idx < 0)
        return 0;

    env->explore_onboard_viewpoint_calls++;
    env->explore_onboard_viewpoint_candidate_sum += region_count;
    env->explore_onboard_viewpoint_viable_sum += region_unknown_sum;
    env->explore_onboard_viewpoint_score_sum += best_region_score;
    env->explore_onboard_viewpoint_depth_sum += best_depth;
    env->explore_onboard_viewpoint_last_candidates = region_count;
    env->explore_onboard_viewpoint_last_viable = region_unknown_sum;
    env->explore_onboard_viewpoint_last_used = 1;
    env->explore_onboard_viewpoint_best_idx = best_frontier_idx;
    env->explore_onboard_viewpoint_best_score = best_region_score;
    env->explore_onboard_viewpoint_best_depth = best_depth;

    *out_frontier_idx = best_frontier_idx;
    *out_unknown_idx = best_unknown_idx;
    *out_depth = best_depth;
    *out_cluster_cells = best_cluster_cells;
    return 1;
}

static inline int qr_explore_select_onboard_optimistic_unknown_target(
        QuadrotorEnv* env, int dim, const int* parent, const int* depth,
        int current_idx, int* out_target_idx, int* out_unknown_idx,
        int* out_depth, int* out_cluster_cells) {
    int best_idx = -1;
    int best_depth = 0;
    float best_score = -1.0e9f;
    int candidate_count = 0;

    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (idx == current_idx || parent[idx] == -2 || env->explored[idx] ||
                    env->explore_blocked[idx] || env->explore_frontier_stale[idx])
                continue;
            int d = depth[idx] > 0 ? depth[idx] : 1;
            candidate_count++;
            float score = -1.0f * (float)d;
            score += qr_explore_onboard_passage_quality_score(env, dim, ix, iy);
            if (best_idx < 0 || score > best_score ||
                    (score == best_score && d < best_depth)) {
                best_idx = idx;
                best_depth = d;
                best_score = score;
            }
        }
    }

    if (best_idx < 0)
        return 0;

    env->explore_onboard_viewpoint_calls++;
    env->explore_onboard_viewpoint_candidate_sum += candidate_count;
    env->explore_onboard_viewpoint_viable_sum += candidate_count;
    env->explore_onboard_viewpoint_score_sum += best_score;
    env->explore_onboard_viewpoint_depth_sum += best_depth;
    env->explore_onboard_viewpoint_last_candidates = candidate_count;
    env->explore_onboard_viewpoint_last_viable = candidate_count;
    env->explore_onboard_viewpoint_last_used = 1;
    env->explore_onboard_viewpoint_best_idx = best_idx;
    env->explore_onboard_viewpoint_best_score = best_score;
    env->explore_onboard_viewpoint_best_depth = best_depth;

    *out_target_idx = best_idx;
    *out_unknown_idx = best_idx;
    *out_depth = best_depth;
    *out_cluster_cells = 1;
    return 1;
}

static inline int qr_explore_onboard_known_blocked_nearby(
        const QuadrotorEnv* env, int dim, int ix, int iy, int radius) {
    int blocked = 0;
    for (int oy = -radius; oy <= radius; oy++) {
        for (int ox = -radius; ox <= radius; ox++) {
            if (ox == 0 && oy == 0)
                continue;
            int ax = ox < 0 ? -ox : ox;
            int ay = oy < 0 ? -oy : oy;
            if (ax > radius || ay > radius)
                continue;
            int nx = ix + ox;
            int ny = iy + oy;
            if (!qr_explore_onboard_in_bounds(env, dim, nx, ny)) {
                blocked += 2;
                continue;
            }
            int nidx = ny * QR_EXPLORE_MAX_GRID + nx;
            if (env->explore_blocked[nidx])
                blocked++;
        }
    }
    return blocked;
}

static inline int qr_explore_onboard_known_line_clear(
        const QuadrotorEnv* env, int dim, int from_idx, int to_idx) {
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    if (from_idx < 0 || from_idx >= max_cells ||
            to_idx < 0 || to_idx >= max_cells)
        return 0;

    int fx = from_idx % QR_EXPLORE_MAX_GRID;
    int fy = from_idx / QR_EXPLORE_MAX_GRID;
    int tx = to_idx % QR_EXPLORE_MAX_GRID;
    int ty = to_idx / QR_EXPLORE_MAX_GRID;
    if (!qr_explore_onboard_in_bounds(env, dim, fx, fy) ||
            !qr_explore_onboard_in_bounds(env, dim, tx, ty))
        return 0;

    int dx = tx - fx;
    int dy = ty - fy;
    int ax = dx < 0 ? -dx : dx;
    int ay = dy < 0 ? -dy : dy;
    int steps = ax > ay ? ax : ay;
    if (steps <= 0)
        return 1;

    for (int s = 1; s <= steps; s++) {
        float t = (float)s / (float)steps;
        int ix = (int)floorf((float)fx + (float)dx * t + 0.5f);
        int iy = (int)floorf((float)fy + (float)dy * t + 0.5f);
        if (!qr_explore_onboard_in_bounds(env, dim, ix, iy))
            return 0;
        int idx = iy * QR_EXPLORE_MAX_GRID + ix;
        if (idx == to_idx)
            continue;
        if (!qr_explore_onboard_known_passable_cell(env, dim, idx, from_idx))
            return 0;
    }
    return 1;
}

static inline float qr_explore_onboard_direction_cone_clearance(
        const QuadrotorEnv* env, const Quadrotor* q, const float dir_body[2]) {
    if (env->ray_length <= 0.0f)
        return 1.0e6f;
    float angle = atan2f(dir_body[1], dir_body[0]);
    float cone = env->explore_onboard_clearance_cone_deg * QR_PI / 180.0f;
    float center = qr_room_ray_obs_angle(env, q, angle);
    float left = qr_room_ray_obs_angle(env, q, angle + cone);
    float right = qr_room_ray_obs_angle(env, q, angle - cone);
    return fminf(center, fminf(left, right)) * env->ray_length;
}

static inline void qr_explore_reset_onboard_viewpoint_step_debug(
        QuadrotorEnv* env) {
    env->explore_onboard_viewpoint_last_candidates = 0;
    env->explore_onboard_viewpoint_last_viable = 0;
    env->explore_onboard_viewpoint_last_used = 0;
    env->explore_onboard_viewpoint_last_miss = 0;
    env->explore_onboard_viewpoint_best_idx = -1;
    env->explore_onboard_waypoint_idx = -1;
    env->explore_onboard_waypoint_depth = 0;
    env->explore_onboard_viewpoint_best_score = 0.0f;
    env->explore_onboard_viewpoint_best_clearance = 0.0f;
    env->explore_onboard_viewpoint_best_depth = 0;
    env->explore_onboard_committed_last_used = 0;
}

static inline int qr_explore_onboard_first_path_step(
        const int* parent, int current_idx, int target_idx) {
    int waypoint_idx = target_idx;
    while (parent[waypoint_idx] >= 0 && parent[waypoint_idx] != current_idx)
        waypoint_idx = parent[waypoint_idx];
    return waypoint_idx;
}

static inline int qr_explore_onboard_path_step(
        const int* parent, int current_idx, int target_idx, int lookahead) {
    if (lookahead <= 1)
        return qr_explore_onboard_first_path_step(parent, current_idx, target_idx);

    int path[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int count = 0;
    int idx = target_idx;
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    while (idx >= 0 && idx < max_cells && idx != current_idx && count < max_cells) {
        path[count++] = idx;
        idx = parent[idx];
    }
    if (count <= 0 || idx != current_idx)
        return qr_explore_onboard_first_path_step(parent, current_idx, target_idx);

    int step_from_current = lookahead - 1;
    if (step_from_current >= count)
        step_from_current = count - 1;
    return path[count - 1 - step_from_current];
}

static inline int qr_explore_onboard_select_path_step(
        QuadrotorEnv* env, const Quadrotor* q, int dim,
        const int* parent, int current_idx, int target_idx) {
    int fixed_lookahead = env->explore_onboard_path_lookahead;
    if (fixed_lookahead < 1)
        fixed_lookahead = 1;
    if (!env->explore_onboard_adaptive_lookahead)
        return qr_explore_onboard_path_step(
            parent, current_idx, target_idx, fixed_lookahead);

    int short_lookahead = env->explore_onboard_adaptive_lookahead_min;
    int long_lookahead = env->explore_onboard_adaptive_lookahead_max;
    if (short_lookahead < 1)
        short_lookahead = 1;
    if (long_lookahead < short_lookahead)
        long_lookahead = short_lookahead;

    int short_idx = qr_explore_onboard_path_step(
        parent, current_idx, target_idx, short_lookahead);
    int selected_idx = short_idx;
    int selected_lookahead = short_lookahead;

    env->explore_onboard_adaptive_lookahead_calls++;

    int long_idx = qr_explore_onboard_path_step(
        parent, current_idx, target_idx, long_lookahead);
    if (long_idx != short_idx) {
        float short_dir[2];
        float long_dir[2];
        int reject_clearance = 0;
        int reject_turn = 0;

        if (qr_explore_target_cell_direction(env, q, short_idx, short_dir) <= 0.0f ||
                qr_explore_target_cell_direction(env, q, long_idx, long_dir) <= 0.0f) {
            reject_turn = 1;
        } else {
            float long_clearance =
                qr_explore_onboard_direction_cone_clearance(env, q, long_dir);
            if (long_clearance < env->explore_onboard_adaptive_clearance) {
                reject_clearance = 1;
            } else if (!qr_explore_onboard_known_line_clear(
                    env, dim, current_idx, long_idx)) {
                reject_turn = 1;
            } else {
                float turn_dot = short_dir[0] * long_dir[0] +
                    short_dir[1] * long_dir[1];
                if (turn_dot < env->explore_onboard_adaptive_turn_dot) {
                    reject_turn = 1;
                } else {
                    selected_idx = long_idx;
                    selected_lookahead = long_lookahead;
                    env->explore_onboard_adaptive_lookahead_long_count++;
                }
            }
        }

        if (reject_clearance)
            env->explore_onboard_adaptive_clearance_reject_count++;
        if (reject_turn)
            env->explore_onboard_adaptive_turn_reject_count++;
    }

    env->explore_onboard_adaptive_lookahead_sum += (float)selected_lookahead;
    return selected_idx;
}

static inline void qr_explore_clear_onboard_committed_viewpoint(
        QuadrotorEnv* env, int count_drop) {
    if (count_drop && env->explore_onboard_committed_viewpoint_idx >= 0)
        env->explore_onboard_committed_drop_count++;
    env->explore_onboard_committed_viewpoint_idx = -1;
    env->explore_onboard_committed_unknown_idx = -1;
    env->explore_onboard_committed_until_tick = 0;
    env->explore_onboard_committed_start_tick = 0;
}

static inline void qr_explore_set_onboard_committed_viewpoint(
        QuadrotorEnv* env, int viewpoint_idx, int target_unknown_idx) {
    if (env->explore_onboard_commit_steps <= 0 || viewpoint_idx < 0)
        return;
    env->explore_onboard_committed_viewpoint_idx = viewpoint_idx;
    env->explore_onboard_committed_unknown_idx = target_unknown_idx;
    env->explore_onboard_committed_until_tick =
        env->tick + env->explore_onboard_commit_steps;
    env->explore_onboard_committed_start_tick = env->tick;
}

static inline int qr_explore_use_onboard_committed_viewpoint(
        QuadrotorEnv* env, const Quadrotor* q, int dim,
        const int* parent, int current_idx, int target_unknown_idx,
        int* waypoint_idx, int* path_steps) {
    int committed_idx = env->explore_onboard_committed_viewpoint_idx;
    if (env->explore_onboard_commit_steps <= 0 || committed_idx < 0)
        return 0;
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    if (committed_idx >= max_cells || env->tick > env->explore_onboard_committed_until_tick ||
            target_unknown_idx != env->explore_onboard_committed_unknown_idx ||
            parent[committed_idx] == -2 || env->explore_blocked[committed_idx] ||
            !env->explored[committed_idx]) {
        qr_explore_clear_onboard_committed_viewpoint(env, 1);
        return 0;
    }
    if (target_unknown_idx >= 0 &&
            !qr_explore_onboard_known_line_clear(env, dim, committed_idx, target_unknown_idx)) {
        qr_explore_clear_onboard_committed_viewpoint(env, 1);
        return 0;
    }

    int first_step_idx = qr_explore_onboard_select_path_step(
        env, q, dim, parent, current_idx, committed_idx);
    float dir[2];
    if (qr_explore_target_cell_direction(env, q, first_step_idx, dir) <= 0.0f) {
        qr_explore_clear_onboard_committed_viewpoint(env, 1);
        return 0;
    }
    float clearance = qr_explore_onboard_direction_cone_clearance(env, q, dir);
    if (clearance < env->explore_onboard_commit_min_clearance) {
        qr_explore_clear_onboard_committed_viewpoint(env, 1);
        return 0;
    }

    *waypoint_idx = first_step_idx;
    *path_steps = committed_idx == current_idx ? 1 : 0;
    int path_idx = committed_idx;
    while (path_idx >= 0 && path_idx != current_idx) {
        (*path_steps)++;
        path_idx = parent[path_idx];
    }
    if (*path_steps <= 0)
        *path_steps = 1;
    env->explore_onboard_committed_used_count++;
    env->explore_onboard_committed_last_used = 1;
    env->explore_onboard_committed_age_sum +=
        env->tick - env->explore_onboard_committed_start_tick;
    return 1;
}

static inline void qr_explore_record_onboard_selection(
        QuadrotorEnv* env, int target_idx, int objective_idx) {
    if (env->explore_onboard_selection_calls > 0) {
        if (target_idx != env->explore_onboard_prev_target_idx)
            env->explore_onboard_target_switch_count++;
        if (objective_idx != env->explore_onboard_prev_objective_idx) {
            env->explore_onboard_objective_switch_count++;
            if (env->explored_cells <= env->explore_onboard_prev_selection_cells)
                env->explore_onboard_unproductive_objective_switch_count++;
        }
    }
    env->explore_onboard_selection_calls++;
    env->explore_onboard_prev_target_idx = target_idx;
    env->explore_onboard_prev_objective_idx = objective_idx;
    env->explore_onboard_prev_selection_cells = env->explored_cells;
}

static inline int qr_explore_select_onboard_local_viewpoint(
        QuadrotorEnv* env, const Quadrotor* q, int dim,
        const int* parent, const int* depth, int frontier_idx,
        int unknown_idx, int current_idx) {
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    if (frontier_idx < 0 || frontier_idx >= max_cells ||
            current_idx < 0 || current_idx >= max_cells)
        return -1;

    int fx = frontier_idx % QR_EXPLORE_MAX_GRID;
    int fy = frontier_idx / QR_EXPLORE_MAX_GRID;
    int ux = unknown_idx >= 0 ? unknown_idx % QR_EXPLORE_MAX_GRID : fx;
    int uy = unknown_idx >= 0 ? unknown_idx / QR_EXPLORE_MAX_GRID : fy;
    int sight_idx = unknown_idx >= 0 ? unknown_idx : frontier_idx;

    int best_idx = -1;
    int best_depth = 0;
    int best_blocked = 0;
    float best_score = -1.0e9f;
    float best_clearance = 0.0f;
    int candidate_count = 0;
    int viable_count = 0;

    for (int iy = fy - QR_EXPLORE_ONBOARD_VIEWPOINT_RADIUS;
            iy <= fy + QR_EXPLORE_ONBOARD_VIEWPOINT_RADIUS; iy++) {
        for (int ix = fx - QR_EXPLORE_ONBOARD_VIEWPOINT_RADIUS;
                ix <= fx + QR_EXPLORE_ONBOARD_VIEWPOINT_RADIUS; ix++) {
            if (!qr_explore_onboard_in_bounds(env, dim, ix, iy))
                continue;
            int idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (parent[idx] == -2 || !env->explored[idx] ||
                    env->explore_blocked[idx])
                continue;
            candidate_count++;
            int dx_frontier = ix - fx;
            int dy_frontier = iy - fy;
            if (dx_frontier < 0)
                dx_frontier = -dx_frontier;
            if (dy_frontier < 0)
                dy_frontier = -dy_frontier;
            int frontier_dist = dx_frontier + dy_frontier;
            int dx_current = ix - (current_idx % QR_EXPLORE_MAX_GRID);
            int dy_current = iy - (current_idx / QR_EXPLORE_MAX_GRID);
            if (dx_current < 0)
                dx_current = -dx_current;
            if (dy_current < 0)
                dy_current = -dy_current;
            int current_dist = dx_current + dy_current;
            if (frontier_dist > QR_EXPLORE_ONBOARD_VIEWPOINT_RADIUS &&
                    current_dist > QR_EXPLORE_ONBOARD_VIEWPOINT_RADIUS)
                continue;
            if (!qr_explore_onboard_known_line_clear(env, dim, idx, sight_idx))
                continue;

            int first_step_idx = idx;
            while (parent[first_step_idx] >= 0 &&
                    parent[first_step_idx] != current_idx)
                first_step_idx = parent[first_step_idx];
            float first_step_dir[2];
            if (qr_explore_target_cell_direction(
                    env, q, first_step_idx, first_step_dir) <= 0.0f)
                continue;
            float cone_clearance = qr_explore_onboard_direction_cone_clearance(
                env, q, first_step_dir);
            viable_count++;

            int dx_unknown = ix - ux;
            int dy_unknown = iy - uy;
            if (dx_unknown < 0)
                dx_unknown = -dx_unknown;
            if (dy_unknown < 0)
                dy_unknown = -dy_unknown;
            int unknown_dist = dx_unknown + dy_unknown;
            int preferred_error = unknown_dist - QR_EXPLORE_ONBOARD_VIEWPOINT_PREFERRED_DIST;
            if (preferred_error < 0)
                preferred_error = -preferred_error;

            int blocked1 = qr_explore_onboard_known_blocked_nearby(
                env, dim, ix, iy, 1);
            int blocked2 = qr_explore_onboard_known_blocked_nearby(
                env, dim, ix, iy, 2);
            int local_unknowns = qr_explore_onboard_frontier_unknown_count(
                env, dim, ix, iy, NULL);
            int candidate_depth = depth[idx] >= 0 ? depth[idx] : 9999;
            int viewpoint_dist = frontier_dist < current_dist ?
                frontier_dist : current_dist;

            float score = 0.45f * (float)local_unknowns;
            score += 0.20f * (float)(QR_EXPLORE_ONBOARD_VIEWPOINT_RADIUS - viewpoint_dist);
            score -= 0.06f * (float)candidate_depth;
            score -= 0.40f * (float)blocked1;
            score -= 0.10f * (float)blocked2;
            score -= 0.12f * (float)preferred_error;
            score += 0.55f * qr_clampf(
                (cone_clearance - 0.25f) / 0.45f, 0.0f, 1.0f);
            score += qr_explore_onboard_passage_quality_score(env, dim, ix, iy);
            if (cone_clearance < 0.35f)
                score -= 0.80f;
            if (idx == current_idx)
                score -= 1.25f;
            if (idx == frontier_idx)
                score -= 0.45f;
            if (candidate_depth > 0)
                score += 0.65f;

            if (best_idx < 0 || score > best_score ||
                    (score == best_score && candidate_depth < best_depth) ||
                    (score == best_score && candidate_depth == best_depth &&
                        blocked1 < best_blocked)) {
                best_idx = idx;
                best_depth = candidate_depth;
                best_blocked = blocked1;
                best_score = score;
                best_clearance = cone_clearance;
            }
        }
    }

    env->explore_onboard_viewpoint_calls++;
    env->explore_onboard_viewpoint_candidate_sum += candidate_count;
    env->explore_onboard_viewpoint_viable_sum += viable_count;
    env->explore_onboard_viewpoint_last_candidates = candidate_count;
    env->explore_onboard_viewpoint_last_viable = viable_count;
    if (best_idx == current_idx)
        best_idx = -1;
    env->explore_onboard_viewpoint_best_idx = best_idx;
    env->explore_onboard_viewpoint_best_score =
        best_idx >= 0 ? best_score : 0.0f;
    env->explore_onboard_viewpoint_best_clearance =
        best_idx >= 0 ? best_clearance : 0.0f;
    env->explore_onboard_viewpoint_best_depth =
        best_idx >= 0 ? best_depth : 0;
    if (best_idx >= 0) {
        env->explore_onboard_viewpoint_score_sum += best_score;
        env->explore_onboard_viewpoint_clearance_sum += best_clearance;
        env->explore_onboard_viewpoint_depth_sum += best_depth;
    }

    if (best_idx < 0) {
        env->explore_onboard_viewpoint_miss_count++;
        env->explore_onboard_viewpoint_last_miss = 1;
        return -1;
    }
    return best_idx;
}

static inline int qr_explore_try_onboard_local_viewpoint(
        QuadrotorEnv* env, const Quadrotor* q, int dim,
        const int* parent, const int* depth, int target_idx,
        int target_unknown_idx, int current_idx,
        int require_distinct_target, int* objective_idx,
        int* waypoint_idx, int* path_steps) {
    int viewpoint_idx = qr_explore_select_onboard_local_viewpoint(
        env, q, dim, parent, depth, target_idx, target_unknown_idx,
        current_idx);
    if (viewpoint_idx < 0)
        return 0;
    if (require_distinct_target && viewpoint_idx == target_idx)
        return 0;

    env->explore_onboard_viewpoint_used_count++;
    env->explore_onboard_viewpoint_last_used = 1;
    qr_explore_set_onboard_committed_viewpoint(
        env, viewpoint_idx, target_unknown_idx);
    *objective_idx = viewpoint_idx;
    *waypoint_idx = viewpoint_idx;
    *path_steps = depth[viewpoint_idx] > 0 ? depth[viewpoint_idx] : 1;
    *waypoint_idx = qr_explore_onboard_select_path_step(
        env, q, dim, parent, current_idx, *waypoint_idx);
    return 1;
}

static inline int qr_explore_mark_onboard_frontier_region_stale(
        QuadrotorEnv* env, int seed_idx, int current_idx) {
    int dim = env->explore_grid_dim;
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    if (dim <= 0 || seed_idx < 0 || seed_idx >= max_cells ||
            !qr_explore_onboard_frontier_cell(env, dim, seed_idx, current_idx))
        return 0;

    unsigned char seen[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    memset(seen, 0, sizeof(seen));

    int head = 0;
    int tail = 0;
    int cluster_cells = 0;
    int marked_unknowns = 0;
    seen[seed_idx] = 1;
    queue[tail++] = seed_idx;

    while (head < tail) {
        int idx = queue[head++];
        cluster_cells++;
        int ix = idx % QR_EXPLORE_MAX_GRID;
        int iy = idx / QR_EXPLORE_MAX_GRID;
        const int nx[4] = {ix + 1, ix - 1, ix, ix};
        const int ny[4] = {iy, iy, iy + 1, iy - 1};

        for (int k = 0; k < 4; k++) {
            if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
                continue;
            int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
            if (!env->explored[nidx] && !env->explore_blocked[nidx] &&
                    !env->explore_frontier_stale[nidx]) {
                env->explore_frontier_stale[nidx] = 1;
                marked_unknowns++;
            }
            if (seen[nidx] ||
                    !qr_explore_onboard_frontier_cell(env, dim, nidx, current_idx))
                continue;
            seen[nidx] = 1;
            if (tail < max_cells)
                queue[tail++] = nidx;
        }
    }

    env->explore_onboard_target_cluster_cells = cluster_cells;
    return marked_unknowns;
}

static inline void qr_explore_update_onboard_stale_target(QuadrotorEnv* env) {
    int idx = env->explore_onboard_target_idx;
    if (idx < 0)
        return;

    int gained = env->explored_cells - env->explore_onboard_last_progress_cells;
    if (gained >= QR_EXPLORE_ONBOARD_STALE_MIN_CELLS) {
        env->explore_onboard_last_progress_cells = env->explored_cells;
        env->explore_onboard_last_progress_tick = env->tick;
        return;
    }

    if (idx >= QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID ||
            env->explored[idx] || env->explore_blocked[idx]) {
        qr_explore_clear_onboard_committed_viewpoint(env, 0);
        env->explore_onboard_target_idx = -1;
        env->explore_onboard_target_frontier_idx = -1;
        env->explore_onboard_target_cluster_cells = 0;
        env->explore_onboard_region_goal_frontier_idx = -1;
        env->explore_onboard_region_goal_unknown_idx = -1;
        return;
    }

    if (env->tick - env->explore_onboard_last_progress_tick >=
            QR_EXPLORE_ONBOARD_STALE_TICKS) {
        int current_idx = qr_explore_cell_index(
            env, env->phys.pos[0], env->phys.pos[1]);
        int seed_idx = env->explore_onboard_region_anchor_idx >= 0
            ? env->explore_onboard_region_anchor_idx
            : env->explore_onboard_target_frontier_idx;
        int marked = qr_explore_mark_onboard_frontier_region_stale(
            env, seed_idx, current_idx);
        if (marked <= 0)
            env->explore_frontier_stale[idx] = 1;
        qr_explore_clear_onboard_committed_viewpoint(env, 1);
        env->explore_onboard_target_idx = -1;
        env->explore_onboard_target_frontier_idx = -1;
        env->explore_onboard_target_cluster_cells = 0;
        env->explore_onboard_region_anchor_idx = -1;
        env->explore_onboard_region_goal_frontier_idx = -1;
        env->explore_onboard_region_goal_unknown_idx = -1;
        env->explore_onboard_stale_marks++;
    }
}

static inline void qr_explore_set_onboard_target(QuadrotorEnv* env,
        int target_unknown_idx, int target_frontier_idx, int cluster_cells) {
    if (target_unknown_idx < 0)
        return;
    int new_region = 0;
    if (env->explore_onboard_region_anchor_idx < 0) {
        env->explore_onboard_region_anchor_idx = target_frontier_idx;
        new_region = 1;
    }
    if (env->explore_onboard_region_commit) {
        if (new_region ||
                env->explore_onboard_region_goal_frontier_idx < 0 ||
                env->explore_onboard_region_goal_frontier_idx != target_frontier_idx) {
            env->explore_onboard_region_goal_frontier_idx = target_frontier_idx;
            env->explore_onboard_region_goal_unknown_idx = target_unknown_idx;
        }
    } else {
        env->explore_onboard_region_goal_frontier_idx = -1;
        env->explore_onboard_region_goal_unknown_idx = -1;
    }
    if (env->explore_onboard_target_idx == target_unknown_idx &&
            env->explore_onboard_target_frontier_idx == target_frontier_idx) {
        qr_update_explore_graph_candidate_selection(env);
        return;
    }
    qr_explore_clear_onboard_committed_viewpoint(env, 1);
    env->explore_onboard_target_idx = target_unknown_idx;
    env->explore_onboard_target_frontier_idx = target_frontier_idx;
    env->explore_onboard_target_cluster_cells = cluster_cells;
    qr_update_explore_graph_candidate_selection(env);
    if (new_region) {
        env->explore_onboard_last_progress_tick = env->tick;
        env->explore_onboard_last_progress_cells = env->explored_cells;
    }
}

static inline int qr_explore_persistent_region_slot_by_id(
        const QuadrotorEnv* env, int id) {
    if (id <= 0)
        return -1;
    for (int i = 0; i < QR_EXPLORE_ONBOARD_MAX_REGIONS; i++) {
        if (env->explore_onboard_region_id[i] == id)
            return i;
    }
    return -1;
}

static inline void qr_explore_update_persistent_region_progress(
        QuadrotorEnv* env) {
    int slot = qr_explore_persistent_region_slot_by_id(
        env, env->explore_onboard_active_region_id);
    if (slot < 0) {
        env->explore_onboard_active_region_id = -1;
        return;
    }

    int gained = env->explored_cells -
        env->explore_onboard_region_last_progress_cells[slot];
    if (gained >= QR_EXPLORE_ONBOARD_STALE_MIN_CELLS) {
        env->explore_onboard_region_last_progress_tick[slot] = env->tick;
        env->explore_onboard_region_last_progress_cells[slot] =
            env->explored_cells;
        env->explore_onboard_last_progress_tick = env->tick;
        env->explore_onboard_last_progress_cells = env->explored_cells;
        return;
    }

    if (env->tick - env->explore_onboard_region_last_progress_tick[slot] >=
            QR_EXPLORE_ONBOARD_STALE_TICKS) {
        env->explore_onboard_region_failures[slot]++;
        env->explore_onboard_active_region_id = -1;
        env->explore_onboard_region_anchor_idx = -1;
        env->explore_onboard_region_goal_frontier_idx = -1;
        env->explore_onboard_region_goal_unknown_idx = -1;
    }
}

static inline int qr_explore_persistent_region_match_slot(
        QuadrotorEnv* env, int cx, int cy, int frontier_idx,
        unsigned char* used_slots) {
    int best_slot = -1;
    int best_dist = 999999;
    for (int i = 0; i < QR_EXPLORE_ONBOARD_MAX_REGIONS; i++) {
        if (used_slots[i] || env->explore_onboard_region_id[i] <= 0)
            continue;
        int dx = env->explore_onboard_region_cx[i] - cx;
        int dy = env->explore_onboard_region_cy[i] - cy;
        if (dx < 0) dx = -dx;
        if (dy < 0) dy = -dy;
        int dist = dx + dy;
        if (env->explore_onboard_region_frontier_idx[i] == frontier_idx)
            dist = 0;
        if (dist <= QR_EXPLORE_ONBOARD_REGION_MATCH_RADIUS &&
                dist < best_dist) {
            best_dist = dist;
            best_slot = i;
        }
    }

    if (best_slot >= 0)
        return best_slot;

    for (int i = 0; i < QR_EXPLORE_ONBOARD_MAX_REGIONS; i++) {
        if (used_slots[i] || env->explore_onboard_region_id[i] <= 0)
            return i;
    }

    int replace_slot = 0;
    int oldest_seen = env->explore_onboard_region_last_seen_tick[0];
    for (int i = 1; i < QR_EXPLORE_ONBOARD_MAX_REGIONS; i++) {
        if (used_slots[i])
            continue;
        if (env->explore_onboard_region_last_seen_tick[i] < oldest_seen) {
            oldest_seen = env->explore_onboard_region_last_seen_tick[i];
            replace_slot = i;
        }
    }
    return replace_slot;
}

static inline void qr_explore_update_passive_region_tracker(
        QuadrotorEnv* env, int dim, const int* parent, const int* depth,
        int current_idx, int selected_frontier_idx) {
    if (!env->explore_onboard_region_tracker)
        return;

    int active_slot = qr_explore_persistent_region_slot_by_id(
        env, env->explore_onboard_active_region_id);
    if (active_slot >= 0) {
        int gained = env->explored_cells -
            env->explore_onboard_region_last_progress_cells[active_slot];
        if (gained >= QR_EXPLORE_ONBOARD_STALE_MIN_CELLS) {
            env->explore_onboard_region_last_progress_tick[active_slot] =
                env->tick;
            env->explore_onboard_region_last_progress_cells[active_slot] =
                env->explored_cells;
        }
    } else {
        env->explore_onboard_active_region_id = -1;
    }

    unsigned char seen[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    unsigned char used_slots[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    memset(seen, 0, sizeof(seen));
    memset(used_slots, 0, sizeof(used_slots));

    int region_count = 0;
    int selected_slot = -1;
    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int seed_idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (seen[seed_idx] || parent[seed_idx] == -2 ||
                    !qr_explore_onboard_frontier_cell(env, dim, seed_idx, current_idx))
                continue;

            int head = 0;
            int tail = 0;
            int cluster_cells = 0;
            int sum_x = 0;
            int sum_y = 0;
            int best_frontier_idx = -1;
            int best_unknown_idx = -1;
            int best_depth = 0;
            float best_score = -1.0e9f;
            int contains_selected = 0;

            seen[seed_idx] = 1;
            queue[tail++] = seed_idx;
            while (head < tail) {
                int idx = queue[head++];
                int fx = idx % QR_EXPLORE_MAX_GRID;
                int fy = idx / QR_EXPLORE_MAX_GRID;
                cluster_cells++;
                sum_x += fx;
                sum_y += fy;
                if (idx == selected_frontier_idx)
                    contains_selected = 1;

                int first_unknown = -1;
                int unknown_count = qr_explore_onboard_frontier_unknown_count(
                    env, dim, fx, fy, &first_unknown);
                if (unknown_count > 0 && first_unknown >= 0) {
                    float score = qr_explore_onboard_frontier_score(
                        env, dim, idx, current_idx, depth[idx], unknown_count);
                    if (best_frontier_idx < 0 || score > best_score ||
                            (score == best_score && depth[idx] < best_depth)) {
                        best_score = score;
                        best_depth = depth[idx];
                        best_frontier_idx = idx;
                        best_unknown_idx = first_unknown;
                    }
                }

                const int nx[4] = {fx + 1, fx - 1, fx, fx};
                const int ny[4] = {fy, fy, fy + 1, fy - 1};
                for (int k = 0; k < 4; k++) {
                    if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
                        continue;
                    int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                    if (seen[nidx] || parent[nidx] == -2 ||
                            !qr_explore_onboard_frontier_cell(env, dim, nidx, current_idx))
                        continue;
                    seen[nidx] = 1;
                    if (tail < QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID)
                        queue[tail++] = nidx;
                }
            }

            if (best_frontier_idx < 0 || cluster_cells <= 0)
                continue;

            int cx = (sum_x + cluster_cells / 2) / cluster_cells;
            int cy = (sum_y + cluster_cells / 2) / cluster_cells;
            int slot = qr_explore_persistent_region_match_slot(
                env, cx, cy, best_frontier_idx, used_slots);
            used_slots[slot] = 1;
            if (env->explore_onboard_region_id[slot] <= 0) {
                env->explore_onboard_region_id[slot] =
                    env->explore_onboard_region_next_id++;
                env->explore_onboard_region_attempts[slot] = 0;
                env->explore_onboard_region_failures[slot] = 0;
                env->explore_onboard_region_first_seen_tick[slot] = env->tick;
                env->explore_onboard_region_last_progress_tick[slot] = env->tick;
                env->explore_onboard_region_last_progress_cells[slot] =
                    env->explored_cells;
            }
            env->explore_onboard_region_cx[slot] = cx;
            env->explore_onboard_region_cy[slot] = cy;
            env->explore_onboard_region_frontier_idx[slot] = best_frontier_idx;
            env->explore_onboard_region_unknown_idx[slot] = best_unknown_idx;
            env->explore_onboard_region_cells[slot] = cluster_cells;
            env->explore_onboard_region_last_seen_tick[slot] = env->tick;
            region_count++;
            if (contains_selected)
                selected_slot = slot;
        }
    }

    if (selected_slot >= 0) {
        int selected_id = env->explore_onboard_region_id[selected_slot];
        if (selected_id != env->explore_onboard_active_region_id) {
            if (env->explore_onboard_active_region_id > 0)
                env->explore_onboard_tracker_switch_count++;
            env->explore_onboard_active_region_id = selected_id;
            env->explore_onboard_region_attempts[selected_slot]++;
            env->explore_onboard_region_last_progress_tick[selected_slot] =
                env->tick;
            env->explore_onboard_region_last_progress_cells[selected_slot] =
                env->explored_cells;
        }
        active_slot = selected_slot;
    } else {
        active_slot = qr_explore_persistent_region_slot_by_id(
            env, env->explore_onboard_active_region_id);
    }

    int active_age = 0;
    int since_progress = 0;
    int stale = 0;
    int active_id = -1;
    if (active_slot >= 0) {
        active_id = env->explore_onboard_region_id[active_slot];
        active_age = env->tick -
            env->explore_onboard_region_first_seen_tick[active_slot];
        since_progress = env->tick -
            env->explore_onboard_region_last_progress_tick[active_slot];
        if (active_age < 0) active_age = 0;
        if (since_progress < 0) since_progress = 0;
        stale = since_progress >= QR_EXPLORE_ONBOARD_STALE_TICKS;
    }

    env->explore_onboard_tracker_calls++;
    env->explore_onboard_tracker_region_sum += region_count;
    env->explore_onboard_tracker_active_age_sum += active_age;
    env->explore_onboard_tracker_since_progress_sum += since_progress;
    env->explore_onboard_tracker_stale_count += stale;
    env->explore_onboard_tracker_last_region_count = region_count;
    env->explore_onboard_tracker_last_active_id = active_id;
    env->explore_onboard_tracker_last_active_age = active_age;
    env->explore_onboard_tracker_last_since_progress = since_progress;
    env->explore_onboard_tracker_last_stale = stale;
}

static inline int qr_explore_select_onboard_persistent_region_target(
        QuadrotorEnv* env, int dim, const int* parent, const int* depth,
        int current_idx, int* out_frontier_idx, int* out_unknown_idx,
        int* out_depth, int* out_cluster_cells) {
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    qr_clear_explore_graph_candidate_cache(env);
    qr_explore_update_persistent_region_progress(env);

    unsigned char seen[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    unsigned char used_slots[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    memset(seen, 0, sizeof(seen));
    memset(used_slots, 0, sizeof(used_slots));

    int best_frontier_idx = -1;
    int best_unknown_idx = -1;
    int best_depth = 0;
    int best_cluster_cells = 0;
    int best_slot = -1;
    int region_count = 0;
    int region_unknown_sum = 0;
    float best_region_score = -1.0e9f;

    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int seed_idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (seen[seed_idx] || parent[seed_idx] == -2 ||
                    !qr_explore_onboard_frontier_cell(env, dim, seed_idx, current_idx))
                continue;

            int head = 0;
            int tail = 0;
            int cluster_cells = 0;
            int sum_x = 0;
            int sum_y = 0;
            int unknown_sum = 0;
            int region_best_frontier_idx = -1;
            int region_best_unknown_idx = -1;
            int region_best_depth = 0;
            float region_best_cell_score = -1.0e9f;

            seen[seed_idx] = 1;
            queue[tail++] = seed_idx;
            while (head < tail) {
                int idx = queue[head++];
                int fx = idx % QR_EXPLORE_MAX_GRID;
                int fy = idx / QR_EXPLORE_MAX_GRID;
                cluster_cells++;
                sum_x += fx;
                sum_y += fy;

                int first_unknown = -1;
                int unknown_count = qr_explore_onboard_frontier_unknown_count(
                    env, dim, fx, fy, &first_unknown);
                if (unknown_count > 0 && first_unknown >= 0) {
                    unknown_sum += unknown_count;
                    float score = qr_explore_onboard_frontier_score(
                        env, dim, idx, current_idx, depth[idx], unknown_count);
                    if (region_best_frontier_idx < 0 ||
                            score > region_best_cell_score ||
                            (score == region_best_cell_score &&
                                depth[idx] < region_best_depth)) {
                        region_best_cell_score = score;
                        region_best_depth = depth[idx];
                        region_best_frontier_idx = idx;
                        region_best_unknown_idx = first_unknown;
                    }
                }

                const int nx[4] = {fx + 1, fx - 1, fx, fx};
                const int ny[4] = {fy, fy, fy + 1, fy - 1};
                for (int k = 0; k < 4; k++) {
                    if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
                        continue;
                    int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                    if (seen[nidx] || parent[nidx] == -2 ||
                            !qr_explore_onboard_frontier_cell(env, dim, nidx, current_idx))
                        continue;
                    seen[nidx] = 1;
                    if (tail < max_cells)
                        queue[tail++] = nidx;
                }
            }

            if (region_best_frontier_idx < 0 || cluster_cells <= 0)
                continue;

            int cx = (sum_x + cluster_cells / 2) / cluster_cells;
            int cy = (sum_y + cluster_cells / 2) / cluster_cells;
            int slot = qr_explore_persistent_region_match_slot(
                env, cx, cy, region_best_frontier_idx, used_slots);
            used_slots[slot] = 1;

            int is_new = env->explore_onboard_region_id[slot] <= 0;
            if (is_new) {
                env->explore_onboard_region_id[slot] =
                    env->explore_onboard_region_next_id++;
                env->explore_onboard_region_attempts[slot] = 0;
                env->explore_onboard_region_failures[slot] = 0;
                env->explore_onboard_region_first_seen_tick[slot] = env->tick;
                env->explore_onboard_region_last_progress_tick[slot] = env->tick;
                env->explore_onboard_region_last_progress_cells[slot] =
                    env->explored_cells;
            }
            env->explore_onboard_region_cx[slot] = cx;
            env->explore_onboard_region_cy[slot] = cy;
            env->explore_onboard_region_frontier_idx[slot] =
                region_best_frontier_idx;
            env->explore_onboard_region_unknown_idx[slot] =
                region_best_unknown_idx;
            env->explore_onboard_region_cells[slot] = cluster_cells;
            env->explore_onboard_region_last_seen_tick[slot] = env->tick;

            int region_id = env->explore_onboard_region_id[slot];
            int active = region_id == env->explore_onboard_active_region_id;
            int since_progress = env->tick -
                env->explore_onboard_region_last_progress_tick[slot];
            float unknown_bonus = (float)(unknown_sum < 32 ? unknown_sum : 32);
            float region_score = region_best_cell_score;
            region_score += 0.06f * sqrtf((float)cluster_cells);
            region_score += 0.02f * unknown_bonus;
            region_score -= 0.18f *
                (float)env->explore_onboard_region_attempts[slot];
            region_score -= 0.45f *
                (float)env->explore_onboard_region_failures[slot];
            if (active) {
                region_score += 0.55f;
                if (since_progress > QR_EXPLORE_ONBOARD_STALE_TICKS / 2)
                    region_score -= 0.015f *
                        (float)(since_progress - QR_EXPLORE_ONBOARD_STALE_TICKS / 2);
            }
            if (region_best_depth <= 1)
                region_score -= 0.15f;

            region_count++;
            region_unknown_sum += unknown_sum;
            if (best_frontier_idx < 0 || region_score > best_region_score ||
                    (region_score == best_region_score &&
                        region_best_depth < best_depth)) {
                best_region_score = region_score;
                best_frontier_idx = region_best_frontier_idx;
                best_unknown_idx = region_best_unknown_idx;
                best_depth = region_best_depth;
                best_cluster_cells = cluster_cells;
                best_slot = slot;
            }
        }
    }

    if (best_frontier_idx < 0 || best_slot < 0)
        return 0;

    int selected_id = env->explore_onboard_region_id[best_slot];
    if (selected_id != env->explore_onboard_active_region_id) {
        env->explore_onboard_active_region_id = selected_id;
        env->explore_onboard_region_attempts[best_slot]++;
    }

    env->explore_onboard_region_anchor_idx = best_frontier_idx;
    env->explore_onboard_region_goal_frontier_idx = best_frontier_idx;
    env->explore_onboard_region_goal_unknown_idx = best_unknown_idx;
    env->explore_onboard_viewpoint_calls++;
    env->explore_onboard_viewpoint_candidate_sum += region_count;
    env->explore_onboard_viewpoint_viable_sum += region_unknown_sum;
    env->explore_onboard_viewpoint_used_count++;
    env->explore_onboard_viewpoint_score_sum += best_region_score;
    env->explore_onboard_viewpoint_depth_sum += best_depth;
    env->explore_onboard_viewpoint_last_candidates = region_count;
    env->explore_onboard_viewpoint_last_viable = region_unknown_sum;
    env->explore_onboard_viewpoint_last_used = 1;
    env->explore_onboard_viewpoint_best_idx = best_frontier_idx;
    env->explore_onboard_viewpoint_best_score = best_region_score;
    env->explore_onboard_viewpoint_best_depth = best_depth;

    *out_frontier_idx = best_frontier_idx;
    *out_unknown_idx = best_unknown_idx;
    *out_depth = best_depth;
    *out_cluster_cells = best_cluster_cells;
    return 1;
}

static inline int qr_explore_onboard_unknown_components(
        const QuadrotorEnv* env, int dim, int* component_id,
        int* component_size) {
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    for (int i = 0; i < max_cells; i++) {
        component_id[i] = -1;
        component_size[i] = 0;
    }

    int component_count = 0;
    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int seed_idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (component_id[seed_idx] >= 0 ||
                    !qr_explore_onboard_unknown_passable_cell(
                        env, dim, seed_idx))
                continue;

            int head = 0;
            int tail = 0;
            int count = 0;
            component_id[seed_idx] = component_count;
            queue[tail++] = seed_idx;
            while (head < tail) {
                int idx = queue[head++];
                count++;
                int cx = idx % QR_EXPLORE_MAX_GRID;
                int cy = idx / QR_EXPLORE_MAX_GRID;
                const int nx[4] = {cx + 1, cx - 1, cx, cx};
                const int ny[4] = {cy, cy, cy + 1, cy - 1};
                for (int k = 0; k < 4; k++) {
                    if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
                        continue;
                    int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                    if (component_id[nidx] >= 0 ||
                            !qr_explore_onboard_unknown_passable_cell(
                                env, dim, nidx))
                        continue;
                    component_id[nidx] = component_count;
                    if (tail < max_cells)
                        queue[tail++] = nidx;
                }
            }

            component_size[component_count] = count;
            component_count++;
            if (component_count >= max_cells)
                return component_count;
        }
    }
    return component_count;
}

static inline void qr_explore_record_frontier_graph_metrics(
        QuadrotorEnv* env, int node_count, int unknown_cells,
        int best_unknown_cells, int best_depth, float best_score,
        int active_selected) {
    env->explore_onboard_graph_calls++;
    env->explore_onboard_graph_node_sum += node_count;
    env->explore_onboard_graph_unknown_sum += unknown_cells;
    env->explore_onboard_graph_best_unknown_sum += best_unknown_cells;
    env->explore_onboard_graph_best_depth_sum += best_depth;
    env->explore_onboard_graph_score_sum += best_score;
    env->explore_onboard_graph_active_count += active_selected ? 1 : 0;
    env->explore_onboard_graph_last_nodes = node_count;
    env->explore_onboard_graph_last_unknown_cells = unknown_cells;
    env->explore_onboard_graph_last_best_unknown_cells = best_unknown_cells;
    env->explore_onboard_graph_last_best_depth = best_depth;
    env->explore_onboard_graph_last_best_score = best_score;
    env->explore_onboard_graph_last_active = active_selected ? 1 : 0;
}

static inline void qr_graph_candidate_insert(
        int* frontier_idx, int* unknown_idx, int* region_id, int* depth,
        int* cluster_cells, int* component_gain, float* passage,
        float* score, int fidx, int uidx, int rid, int d, int cells, int gain,
        float passage_score, float candidate_score);

static inline int qr_explore_select_onboard_frontier_graph_target(
        QuadrotorEnv* env, int dim, const int* parent, const int* depth,
        int current_idx, int* out_frontier_idx, int* out_unknown_idx,
        int* out_depth, int* out_cluster_cells) {
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    qr_explore_update_persistent_region_progress(env);

    int component_id[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int component_size[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int component_seen_stamp[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int component_count = qr_explore_onboard_unknown_components(
        env, dim, component_id, component_size);
    memset(component_seen_stamp, 0, sizeof(component_seen_stamp));

    unsigned char seen[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    unsigned char used_slots[QR_EXPLORE_ONBOARD_MAX_REGIONS];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    memset(seen, 0, sizeof(seen));
    memset(used_slots, 0, sizeof(used_slots));

    int best_frontier_idx = -1;
    int best_unknown_idx = -1;
    int best_depth = 0;
    int best_cluster_cells = 0;
    int best_slot = -1;
    int best_active = 0;
    int best_unknown_gain = 0;
    int graph_nodes = 0;
    int graph_unknown_sum = 0;
    float best_region_score = -1.0e9f;
    int stamp = 1;

    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int seed_idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (seen[seed_idx] || parent[seed_idx] == -2 ||
                    !qr_explore_onboard_frontier_cell(env, dim, seed_idx,
                        current_idx))
                continue;

            int head = 0;
            int tail = 0;
            int cluster_cells = 0;
            int sum_x = 0;
            int sum_y = 0;
            int unknown_sum = 0;
            int component_gain = 0;
            int region_best_frontier_idx = -1;
            int region_best_unknown_idx = -1;
            int region_best_depth = 0;
            float region_best_cell_score = -1.0e9f;
            float region_best_passage = 0.0f;

            if (stamp == 2147483647) {
                memset(component_seen_stamp, 0, sizeof(component_seen_stamp));
                stamp = 1;
            }
            int cluster_stamp = stamp++;
            seen[seed_idx] = 1;
            queue[tail++] = seed_idx;
            while (head < tail) {
                int idx = queue[head++];
                int fx = idx % QR_EXPLORE_MAX_GRID;
                int fy = idx / QR_EXPLORE_MAX_GRID;
                cluster_cells++;
                sum_x += fx;
                sum_y += fy;

                int first_unknown = -1;
                int unknown_count = qr_explore_onboard_frontier_unknown_count(
                    env, dim, fx, fy, &first_unknown);
                if (unknown_count > 0 && first_unknown >= 0) {
                    unknown_sum += unknown_count;
                    int local_component_gain = 0;
                    const int unx[4] = {fx + 1, fx - 1, fx, fx};
                    const int uny[4] = {fy, fy, fy + 1, fy - 1};
                    for (int k = 0; k < 4; k++) {
                        if (!qr_explore_onboard_in_bounds(
                                env, dim, unx[k], uny[k]))
                            continue;
                        int uidx = uny[k] * QR_EXPLORE_MAX_GRID + unx[k];
                        int cid = component_id[uidx];
                        if (cid < 0 || cid >= component_count)
                            continue;
                        int gain = component_size[cid];
                        if (env->explore_onboard_graph_gain_cap > 0 &&
                                gain > env->explore_onboard_graph_gain_cap)
                            gain = env->explore_onboard_graph_gain_cap;
                        local_component_gain += gain;
                        if (component_seen_stamp[cid] != cluster_stamp) {
                            component_seen_stamp[cid] = cluster_stamp;
                            component_gain += gain;
                        }
                    }
                    float score = qr_explore_onboard_frontier_score(
                        env, dim, idx, current_idx, depth[idx], unknown_count);
                    score += 0.045f * sqrtf((float)local_component_gain);
                    if (region_best_frontier_idx < 0 ||
                            score > region_best_cell_score ||
                            (score == region_best_cell_score &&
                                depth[idx] < region_best_depth)) {
                        region_best_cell_score = score;
                        region_best_depth = depth[idx];
                        region_best_frontier_idx = idx;
                        region_best_unknown_idx = first_unknown;
                        region_best_passage =
                            qr_explore_onboard_passage_quality_score(
                                env, dim, fx, fy);
                    }
                }

                const int nx[4] = {fx + 1, fx - 1, fx, fx};
                const int ny[4] = {fy, fy, fy + 1, fy - 1};
                for (int k = 0; k < 4; k++) {
                    if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
                        continue;
                    int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
                    if (seen[nidx] || parent[nidx] == -2 ||
                            !qr_explore_onboard_frontier_cell(
                                env, dim, nidx, current_idx))
                        continue;
                    seen[nidx] = 1;
                    if (tail < max_cells)
                        queue[tail++] = nidx;
                }
            }

            if (region_best_frontier_idx < 0 || cluster_cells <= 0)
                continue;

            int cx = (sum_x + cluster_cells / 2) / cluster_cells;
            int cy = (sum_y + cluster_cells / 2) / cluster_cells;
            int slot = qr_explore_persistent_region_match_slot(
                env, cx, cy, region_best_frontier_idx, used_slots);
            used_slots[slot] = 1;

            int is_new = env->explore_onboard_region_id[slot] <= 0;
            if (is_new) {
                env->explore_onboard_region_id[slot] =
                    env->explore_onboard_region_next_id++;
                env->explore_onboard_region_attempts[slot] = 0;
                env->explore_onboard_region_failures[slot] = 0;
                env->explore_onboard_region_first_seen_tick[slot] = env->tick;
                env->explore_onboard_region_last_progress_tick[slot] = env->tick;
                env->explore_onboard_region_last_progress_cells[slot] =
                    env->explored_cells;
            }
            env->explore_onboard_region_cx[slot] = cx;
            env->explore_onboard_region_cy[slot] = cy;
            env->explore_onboard_region_frontier_idx[slot] =
                region_best_frontier_idx;
            env->explore_onboard_region_unknown_idx[slot] =
                region_best_unknown_idx;
            env->explore_onboard_region_cells[slot] = cluster_cells;
            env->explore_onboard_region_last_seen_tick[slot] = env->tick;

            int region_id = env->explore_onboard_region_id[slot];
            int active = region_id == env->explore_onboard_active_region_id;
            int since_progress = env->tick -
                env->explore_onboard_region_last_progress_tick[slot];
            if (since_progress < 0)
                since_progress = 0;

            float region_score = region_best_cell_score;
            region_score += env->explore_onboard_graph_gain_weight *
                sqrtf((float)(component_gain > 0 ? component_gain : unknown_sum));
            region_score += env->explore_onboard_graph_cluster_weight *
                sqrtf((float)cluster_cells);
            region_score -= env->explore_onboard_graph_depth_weight *
                (float)region_best_depth;
            region_score -= env->explore_onboard_graph_attempt_penalty *
                (float)env->explore_onboard_region_attempts[slot];
            region_score -= env->explore_onboard_graph_failure_penalty *
                (float)env->explore_onboard_region_failures[slot];
            if (active) {
                region_score += env->explore_onboard_graph_active_bonus;
                if (since_progress > QR_EXPLORE_ONBOARD_STALE_TICKS / 2) {
                    region_score -= env->explore_onboard_graph_stale_penalty *
                        (float)(since_progress - QR_EXPLORE_ONBOARD_STALE_TICKS / 2);
                }
            }
            if (region_best_depth <= 1)
                region_score -= 0.12f;

            graph_nodes++;
            graph_unknown_sum += component_gain;
            qr_graph_candidate_insert(
                env->explore_onboard_graph_candidate_frontier_idx,
                env->explore_onboard_graph_candidate_unknown_idx,
                env->explore_onboard_graph_candidate_region_id,
                env->explore_onboard_graph_candidate_depth,
                env->explore_onboard_graph_candidate_cells,
                env->explore_onboard_graph_candidate_gain,
                env->explore_onboard_graph_candidate_passage,
                env->explore_onboard_graph_candidate_score,
                region_best_frontier_idx, region_best_unknown_idx,
                region_id, region_best_depth, cluster_cells, component_gain,
                region_best_passage, region_score);
            if (best_frontier_idx < 0 || region_score > best_region_score ||
                    (region_score == best_region_score &&
                        region_best_depth < best_depth)) {
                best_region_score = region_score;
                best_frontier_idx = region_best_frontier_idx;
                best_unknown_idx = region_best_unknown_idx;
                best_depth = region_best_depth;
                best_cluster_cells = cluster_cells;
                best_slot = slot;
                best_active = active;
                best_unknown_gain = component_gain;
            }
        }
    }

    for (int i = 0; i < QR_EXPLORE_GRAPH_CANDIDATE_COUNT; i++) {
        if (env->explore_onboard_graph_candidate_frontier_idx[i] >= 0)
            env->explore_onboard_graph_candidate_count++;
    }

    if (best_frontier_idx < 0 || best_slot < 0) {
        qr_explore_record_frontier_graph_metrics(
            env, graph_nodes, graph_unknown_sum, 0, 0, 0.0f, 0);
        return 0;
    }

    int selected_id = env->explore_onboard_region_id[best_slot];
    if (selected_id != env->explore_onboard_active_region_id) {
        env->explore_onboard_active_region_id = selected_id;
        env->explore_onboard_region_attempts[best_slot]++;
        env->explore_onboard_last_progress_tick = env->tick;
        env->explore_onboard_last_progress_cells = env->explored_cells;
    }

    env->explore_onboard_region_anchor_idx = best_frontier_idx;
    env->explore_onboard_region_goal_frontier_idx = best_frontier_idx;
    env->explore_onboard_region_goal_unknown_idx = best_unknown_idx;
    qr_update_explore_graph_candidate_selection(env);

    env->explore_onboard_viewpoint_calls++;
    env->explore_onboard_viewpoint_candidate_sum += graph_nodes;
    env->explore_onboard_viewpoint_viable_sum += graph_unknown_sum;
    env->explore_onboard_viewpoint_used_count++;
    env->explore_onboard_viewpoint_score_sum += best_region_score;
    env->explore_onboard_viewpoint_depth_sum += best_depth;
    env->explore_onboard_viewpoint_last_candidates = graph_nodes;
    env->explore_onboard_viewpoint_last_viable = graph_unknown_sum;
    env->explore_onboard_viewpoint_last_used = 1;
    env->explore_onboard_viewpoint_best_idx = best_frontier_idx;
    env->explore_onboard_viewpoint_best_score = best_region_score;
    env->explore_onboard_viewpoint_best_depth = best_depth;

    qr_explore_record_frontier_graph_metrics(
        env, graph_nodes, graph_unknown_sum, best_unknown_gain, best_depth,
        best_region_score, best_active);

    *out_frontier_idx = best_frontier_idx;
    *out_unknown_idx = best_unknown_idx;
    *out_depth = best_depth;
    *out_cluster_cells = best_cluster_cells;
    return 1;
}

static inline void qr_graph_candidate_insert(
        int* frontier_idx, int* unknown_idx, int* region_id, int* depth,
        int* cluster_cells, int* component_gain, float* passage,
        float* score, int fidx, int uidx, int rid, int d, int cells, int gain,
        float passage_score, float candidate_score) {
    for (int i = 0; i < QR_EXPLORE_GRAPH_CANDIDATE_COUNT; i++) {
        if (frontier_idx[i] >= 0 && candidate_score <= score[i])
            continue;
        for (int j = QR_EXPLORE_GRAPH_CANDIDATE_COUNT - 1; j > i; j--) {
            frontier_idx[j] = frontier_idx[j - 1];
            unknown_idx[j] = unknown_idx[j - 1];
            region_id[j] = region_id[j - 1];
            depth[j] = depth[j - 1];
            cluster_cells[j] = cluster_cells[j - 1];
            component_gain[j] = component_gain[j - 1];
            passage[j] = passage[j - 1];
            score[j] = score[j - 1];
        }
        frontier_idx[i] = fidx;
        unknown_idx[i] = uidx;
        region_id[i] = rid;
        depth[i] = d;
        cluster_cells[i] = cells;
        component_gain[i] = gain;
        passage[i] = passage_score;
        score[i] = candidate_score;
        break;
    }
}

static inline void qr_write_explore_graph_candidate_tail_obs(
        QuadrotorEnv* env, const Quadrotor* q, float* o, int tail_start) {
    for (int i = 0; i < QR_EXPLORE_GRAPH_CANDIDATE_OBS_SIZE; i++)
        o[tail_start + i] = 0.0f;

    if (!env->explore_onboard_graph_candidate_obs ||
            env->task_mode != QR_TASK_INDOOR_EXPLORE ||
            env->explore_grid_dim <= 0)
        return;

    int dim = env->explore_grid_dim;
    int current_idx = qr_explore_cell_index(env, q->pos[0], q->pos[1]);
    if (current_idx < 0)
        return;

    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    int parent[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int depth[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    for (int i = 0; i < max_cells; i++) {
        parent[i] = -2;
        depth[i] = -1;
    }

    parent[current_idx] = -1;
    depth[current_idx] = 0;
    int head = 0;
    int tail = 0;
    queue[tail++] = current_idx;
    while (head < tail) {
        int idx = queue[head++];
        int ix = idx % QR_EXPLORE_MAX_GRID;
        int iy = idx / QR_EXPLORE_MAX_GRID;
        const int nx[4] = {ix + 1, ix - 1, ix, ix};
        const int ny[4] = {iy, iy, iy + 1, iy - 1};
        for (int k = 0; k < 4; k++) {
            if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
                continue;
            int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
            int passable = env->explore_onboard_optimistic_unknown
                ? qr_explore_onboard_optimistic_passable_cell(
                    env, dim, nidx, current_idx)
                : qr_explore_onboard_known_passable_cell(
                    env, dim, nidx, current_idx);
            if (!passable || parent[nidx] != -2)
                continue;
            parent[nidx] = idx;
            depth[nidx] = depth[idx] + 1;
            if (tail < max_cells)
                queue[tail++] = nidx;
        }
    }

    int refresh_candidates =
        env->explore_onboard_graph_candidate_count <= 0 ||
        env->explore_frontier_target_mode != QR_EXPLORE_FRONTIER_TARGET_ONBOARD;
    if (refresh_candidates) {
        int frontier_idx = -1;
        int unknown_idx = -1;
        int best_depth = 0;
        int cluster_cells = 0;
        qr_explore_select_onboard_frontier_graph_target(
            env, dim, parent, depth, current_idx, &frontier_idx, &unknown_idx,
            &best_depth, &cluster_cells);
        qr_update_explore_graph_candidate_selection(env);
        if (env->explore_onboard_graph_candidate_count <= 0)
            return;
    }

    float depth_norm = (float)dim > 1.0f ? (float)dim : 1.0f;
    float gain_norm = env->explore_onboard_graph_gain_cap > 0
        ? (float)env->explore_onboard_graph_gain_cap
        : 192.0f;
    for (int i = 0; i < QR_EXPLORE_GRAPH_CANDIDATE_COUNT; i++) {
        int out = tail_start + i * QR_EXPLORE_GRAPH_CANDIDATE_FEATURES;
        int frontier_idx = env->explore_onboard_graph_candidate_frontier_idx[i];
        if (frontier_idx < 0)
            continue;
        if (frontier_idx >= max_cells || parent[frontier_idx] == -2)
            continue;

        int waypoint_idx = frontier_idx;
        while (parent[waypoint_idx] >= 0 && parent[waypoint_idx] != current_idx)
            waypoint_idx = parent[waypoint_idx];
        int candidate_depth = depth[frontier_idx] >= 0
            ? depth[frontier_idx]
            : env->explore_onboard_graph_candidate_depth[i];
        float action_dir[2] = {0.0f, 0.0f};
        float frontier_dir[2] = {0.0f, 0.0f};
        qr_explore_target_cell_direction(env, q, waypoint_idx, action_dir);
        qr_explore_target_cell_direction(env, q, frontier_idx, frontier_dir);
        float passage_norm = 0.5f;
        if (env->explore_onboard_passage_weight > 1e-6f) {
            passage_norm = 0.5f + env->explore_onboard_graph_candidate_passage[i] /
                (4.0f * env->explore_onboard_passage_weight);
        }

        o[out + 0] = action_dir[0];
        o[out + 1] = action_dir[1];
        o[out + 2] = frontier_dir[0];
        o[out + 3] = frontier_dir[1];
        o[out + 4] = qr_clampf(
            (float)candidate_depth / depth_norm,
            0.0f, 1.0f);
        o[out + 5] = qr_clampf(
            (float)env->explore_onboard_graph_candidate_gain[i] / gain_norm,
            0.0f, 1.0f);
        o[out + 6] = qr_clampf(
            (float)env->explore_onboard_graph_candidate_cells[i] / 32.0f,
            0.0f, 1.0f);
        o[out + 7] = qr_clampf(passage_norm, 0.0f, 1.0f);
        o[out + 8] = env->explore_onboard_graph_candidate_selected[i] ? 1.0f : 0.0f;
        o[out + 9] = 1.0f;
    }
}

static inline float qr_explore_onboard_frontier_path_target(
        QuadrotorEnv* env, const Quadrotor* q, int current_idx,
        float out_dir_body[2]) {
    int dim = env->explore_grid_dim;
    if (dim <= 0 || current_idx < 0)
        return 0.0f;

    qr_explore_reset_onboard_viewpoint_step_debug(env);
    qr_explore_update_onboard_stale_target(env);

    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    int parent[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int depth[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    for (int i = 0; i < max_cells; i++) {
        parent[i] = -2;
        depth[i] = -1;
    }

    parent[current_idx] = -1;
    depth[current_idx] = 0;
    int head = 0;
    int tail = 0;
    queue[tail++] = current_idx;

    while (head < tail) {
        int idx = queue[head++];
        int ix = idx % QR_EXPLORE_MAX_GRID;
        int iy = idx / QR_EXPLORE_MAX_GRID;
            if (ix >= dim || iy >= dim)
                continue;

        const int nx[4] = {ix + 1, ix - 1, ix, ix};
        const int ny[4] = {iy, iy, iy + 1, iy - 1};
        for (int k = 0; k < 4; k++) {
            if (!qr_explore_onboard_in_bounds(env, dim, nx[k], ny[k]))
                continue;
            int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
            int passable = env->explore_onboard_optimistic_unknown
                ? qr_explore_onboard_optimistic_passable_cell(
                    env, dim, nidx, current_idx)
                : qr_explore_onboard_known_passable_cell(
                    env, dim, nidx, current_idx);
            if (!passable ||
                    parent[nidx] != -2)
                continue;
            parent[nidx] = idx;
            depth[nidx] = depth[idx] + 1;
            if (tail < max_cells)
                queue[tail++] = nidx;
        }
    }

    int target_idx = -1;
    int target_unknown_idx = -1;
    int target_cluster_cells = 1;
    int best_depth = 0;

    int persistent_ready = env->explore_onboard_persistent_regions &&
        (env->explore_onboard_region_commit_stall_ticks <= 0 ||
         env->tick - env->explore_onboard_last_progress_tick >=
            env->explore_onboard_region_commit_stall_ticks);
    if (persistent_ready) {
        qr_explore_select_onboard_persistent_region_target(
            env, dim, parent, depth, current_idx, &target_idx,
            &target_unknown_idx, &best_depth, &target_cluster_cells);
    } else if (env->explore_onboard_region_anchor_idx >= 0 &&
            !qr_explore_select_onboard_frontier_region_target(
                env, dim, parent, depth, env->explore_onboard_region_anchor_idx,
                current_idx, &target_idx, &target_unknown_idx, &best_depth,
                &target_cluster_cells)) {
        env->explore_onboard_region_anchor_idx = -1;
        env->explore_onboard_region_goal_frontier_idx = -1;
        env->explore_onboard_region_goal_unknown_idx = -1;
    }

    float best_score = -1.0e9f;
    if (target_idx < 0 && env->explore_onboard_frontier_graph) {
        qr_explore_select_onboard_frontier_graph_target(
            env, dim, parent, depth, current_idx, &target_idx,
            &target_unknown_idx, &best_depth, &target_cluster_cells);
    }

    if (target_idx < 0 && env->explore_onboard_optimistic_unknown) {
        qr_explore_select_onboard_optimistic_unknown_target(
            env, dim, parent, depth, current_idx, &target_idx,
            &target_unknown_idx, &best_depth, &target_cluster_cells);
    }

    if (target_idx < 0 && env->explore_onboard_budget_frontier) {
        qr_explore_select_onboard_budget_frontier_target(
            env, dim, parent, depth, current_idx, &target_idx,
            &target_unknown_idx, &best_depth, &target_cluster_cells);
    }

    if (target_idx < 0) {
        for (int iy = 0; iy < dim; iy++) {
            for (int ix = 0; ix < dim; ix++) {
                int idx = iy * QR_EXPLORE_MAX_GRID + ix;
                if (parent[idx] == -2)
                    continue;

                int first_unknown = -1;
                int unknown_count = qr_explore_onboard_frontier_unknown_count(
                    env, dim, ix, iy, &first_unknown);
                if (unknown_count <= 0 || first_unknown < 0)
                    continue;
                float score = qr_explore_onboard_frontier_score(
                    env, dim, idx, current_idx, depth[idx], unknown_count);
                if (target_idx < 0 || score > best_score ||
                        (score == best_score && depth[idx] < best_depth)) {
                    best_score = score;
                    best_depth = depth[idx];
                    target_idx = idx;
                    target_unknown_idx = first_unknown;
                }
            }
        }

        if (target_idx >= 0) {
            int region_target_idx = target_idx;
            int region_unknown_idx = target_unknown_idx;
            int region_depth = best_depth;
            int region_cells = 1;
            if (qr_explore_select_onboard_frontier_region_target(
                    env, dim, parent, depth, target_idx, current_idx,
                    &region_target_idx, &region_unknown_idx, &region_depth,
                    &region_cells)) {
                target_idx = region_target_idx;
                target_unknown_idx = region_unknown_idx;
                best_depth = region_depth;
                target_cluster_cells = region_cells;
            }
        }
    }

    if (target_idx < 0)
        return 0.0f;

    qr_explore_update_passive_region_tracker(
        env, dim, parent, depth, current_idx, target_idx);

    qr_explore_set_onboard_target(
        env, target_unknown_idx, target_idx, target_cluster_cells);

    int waypoint_idx = target_idx;
    int objective_idx = target_idx;
    int path_steps = depth[target_idx] > 0 ? depth[target_idx] : 1;
    int committed_used = target_unknown_idx >= 0 &&
        qr_explore_use_onboard_committed_viewpoint(
            env, q, dim, parent, current_idx, target_unknown_idx,
            &waypoint_idx, &path_steps);
    if (committed_used) {
        objective_idx = env->explore_onboard_committed_viewpoint_idx;
    } else if (target_idx == current_idx && target_unknown_idx >= 0) {
        if (!qr_explore_try_onboard_local_viewpoint(
                env, q, dim, parent, depth, target_idx, target_unknown_idx,
                current_idx, 0, &objective_idx, &waypoint_idx, &path_steps)) {
            if (objective_idx == target_idx) {
                waypoint_idx = target_unknown_idx;
                objective_idx = target_unknown_idx;
            }
        }
    } else {
        waypoint_idx = qr_explore_onboard_select_path_step(
            env, q, dim, parent, current_idx, waypoint_idx);
    }

    if (!committed_used && target_idx != current_idx && target_unknown_idx >= 0) {
        int selected_viewpoint = 0;
        if (env->explore_onboard_approach_viewpoints &&
                path_steps <= env->explore_onboard_approach_max_depth) {
            selected_viewpoint = qr_explore_try_onboard_local_viewpoint(
                env, q, dim, parent, depth, target_idx, target_unknown_idx,
                current_idx, 1, &objective_idx, &waypoint_idx, &path_steps);
        }
        if (env->explore_onboard_stall_viewpoint_ticks > 0 &&
                env->tick - env->explore_onboard_last_progress_tick >=
                    env->explore_onboard_stall_viewpoint_ticks &&
                path_steps <= env->explore_onboard_stall_viewpoint_max_depth) {
            selected_viewpoint = qr_explore_try_onboard_local_viewpoint(
                env, q, dim, parent, depth, target_idx, target_unknown_idx,
                current_idx, 1, &objective_idx, &waypoint_idx, &path_steps);
        }
        float current_dir[2];
        if (!selected_viewpoint &&
                qr_explore_target_cell_direction(env, q, waypoint_idx, current_dir) > 0.0f &&
                qr_explore_onboard_direction_cone_clearance(env, q, current_dir) < 0.50f) {
            selected_viewpoint = qr_explore_try_onboard_local_viewpoint(
                env, q, dim, parent, depth, target_idx, target_unknown_idx,
                current_idx, 1, &objective_idx, &waypoint_idx, &path_steps);
        }
        if (!selected_viewpoint &&
                qr_explore_onboard_wall_edge_cell(env, dim, waypoint_idx)) {
            qr_explore_try_onboard_local_viewpoint(
                env, q, dim, parent, depth, target_idx, target_unknown_idx,
                current_idx, 1, &objective_idx, &waypoint_idx, &path_steps);
        }
    }

    env->explore_onboard_target_calls++;
    env->explore_onboard_waypoint_depth_sum += path_steps;
    env->explore_onboard_waypoint_idx = waypoint_idx;
    env->explore_onboard_waypoint_depth = path_steps;
    env->explore_onboard_objective_idx = objective_idx;
    env->explore_onboard_objective_depth =
        (objective_idx >= 0 && objective_idx < max_cells && depth[objective_idx] >= 0)
            ? depth[objective_idx]
            : path_steps;
    qr_explore_record_onboard_selection(env, target_idx, objective_idx);

    int emit_idx = waypoint_idx;
    if (env->explore_onboard_emit_objective_target && objective_idx >= 0)
        emit_idx = objective_idx;
    if (qr_explore_target_cell_direction(env, q, emit_idx, out_dir_body) <= 0.0f)
        return 0.0f;
    return (float)path_steps * env->explore_cell_size;
}

static inline float qr_explore_frontier_path_target(QuadrotorEnv* env,
        const Quadrotor* q, float out_dir_body[2]) {
    out_dir_body[0] = 1.0f;
    out_dir_body[1] = 0.0f;

    int dim = env->explore_grid_dim;
    if (dim <= 0 || env->room_size <= 0.0f)
        return 0.0f;

    int current_idx = qr_explore_cell_index(env, q->pos[0], q->pos[1]);
    if (current_idx < 0) {
        int nearest_idx = qr_explore_nearest_valid_cell_index(env, q->pos[0], q->pos[1]);
        if (nearest_idx < 0)
            return 0.0f;
        return qr_explore_target_cell_direction(env, q, nearest_idx, out_dir_body);
    }

    if (env->explore_frontier_target_mode == QR_EXPLORE_FRONTIER_TARGET_UNEXPLORED)
        return qr_explore_privileged_unexplored_path_target(env, q, current_idx, out_dir_body);
    if (env->explore_frontier_target_mode == QR_EXPLORE_FRONTIER_TARGET_REGION)
        return qr_explore_privileged_region_path_target(env, q, current_idx, out_dir_body);
    if (env->explore_frontier_target_mode == QR_EXPLORE_FRONTIER_TARGET_ONBOARD)
        return qr_explore_onboard_frontier_path_target(env, q, current_idx, out_dir_body);

    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    int parent[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int depth[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    int queue[QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID];
    for (int i = 0; i < max_cells; i++) {
        parent[i] = -2;
        depth[i] = -1;
    }

    parent[current_idx] = -1;
    depth[current_idx] = 0;
    int head = 0;
    int tail = 0;
    queue[tail++] = current_idx;

    int target_idx = -1;
    int target_unknown_idx = -1;
    float best_score = -1.0e9f;
    int best_depth = 0;
    while (head < tail) {
        int idx = queue[head++];
        int ix = idx % QR_EXPLORE_MAX_GRID;
        int iy = idx / QR_EXPLORE_MAX_GRID;
        if (ix >= dim || iy >= dim)
            continue;

        int unknown_idx = qr_explore_first_unknown_neighbor(env, dim, ix, iy);
        if (unknown_idx >= 0) {
            if (env->explore_frontier_target_mode == QR_EXPLORE_FRONTIER_TARGET_GAIN) {
                int unknown_count = qr_explore_unknown_neighbor_count(env, dim, ix, iy);
                float score = (float)unknown_count - 0.05f * (float)depth[idx];
                if (target_idx < 0 || score > best_score ||
                        (score == best_score && depth[idx] < best_depth)) {
                    best_score = score;
                    best_depth = depth[idx];
                    target_idx = idx;
                    target_unknown_idx = unknown_idx;
                }
            } else {
                target_idx = idx;
                target_unknown_idx = unknown_idx;
                break;
            }
        }

        const int nx[4] = {ix + 1, ix - 1, ix, ix};
        const int ny[4] = {iy, iy, iy + 1, iy - 1};
        for (int k = 0; k < 4; k++) {
            if (!qr_explore_valid_neighbor(env, dim, nx[k], ny[k]))
                continue;
            int nidx = ny[k] * QR_EXPLORE_MAX_GRID + nx[k];
            if (!env->explored[nidx] || parent[nidx] != -2)
                continue;
            parent[nidx] = idx;
            depth[nidx] = depth[idx] + 1;
            if (tail < max_cells)
                queue[tail++] = nidx;
        }
    }

    if (target_idx < 0)
        return qr_explore_privileged_unexplored_path_target(
            env, q, current_idx, out_dir_body);

    int waypoint_idx = target_idx;
    int path_steps = depth[target_idx] > 0 ? depth[target_idx] : 1;
    if (target_idx == current_idx && target_unknown_idx >= 0) {
        waypoint_idx = target_unknown_idx;
    } else {
        while (parent[waypoint_idx] >= 0 && parent[waypoint_idx] != current_idx)
            waypoint_idx = parent[waypoint_idx];
    }

    if (qr_explore_target_cell_direction(env, q, waypoint_idx, out_dir_body) <= 0.0f)
        return 0.0f;
    return (float)path_steps * env->explore_cell_size;
}

static inline float qr_explore_progress_distance(QuadrotorEnv* env,
        const Quadrotor* q, float out_dir_body[2]) {
    if (env->explore_memory_obs == 3)
        return qr_explore_frontier_path_target(env, q, out_dir_body);
    return qr_explore_frontier_distance(env, q, out_dir_body);
}

static inline float qr_box_xy_clearance(float x, float y, const float* box) {
    float dx = 0.0f;
    if (x < box[0]) dx = box[0] - x;
    else if (x > box[1]) dx = x - box[1];

    float dy = 0.0f;
    if (y < box[2]) dy = box[2] - y;
    else if (y > box[3]) dy = y - box[3];

    return sqrtf(dx*dx + dy*dy);
}

static inline float qr_clearance(const QuadrotorEnv* env, const Quadrotor* q) {
    float best = env->ray_length > 0.0f ? env->ray_length : 1.0e6f;
    int use_room_bounds = env->room_size > 0.0f &&
        (env->task_mode != QR_TASK_INDOOR_ROUTE || env->route_walls);
    if (use_room_bounds) {
        float x_wall = env->room_size - fabsf(q->pos[0]);
        float y_wall = env->room_size - fabsf(q->pos[1]);
        float ceiling = env->ceiling_height - q->pos[2];
        if (x_wall < best) best = x_wall;
        if (y_wall < best) best = y_wall;
        if (q->pos[2] < best) best = q->pos[2];
        if (ceiling < best) best = ceiling;
    }

    for (int i = 0; i < env->num_obstacles; i++) {
        const float* box = env->obstacles[i];
        if (q->pos[2] < box[4] || q->pos[2] > box[5])
            continue;
        float clearance = qr_box_xy_clearance(q->pos[0], q->pos[1], box);
        if (clearance < best)
            best = clearance;
    }

    return best > 0.0f ? best : 0.0f;
}

static inline float qr_yaw_from_quat(const float q[4]) {
    return atan2f(2.0f * (q[0]*q[3] + q[1]*q[2]),
                  1.0f - 2.0f * (q[2]*q[2] + q[3]*q[3]));
}

static inline void qr_controller_state(const QuadrotorEnv* env, Quadrotor* out) {
    *out = env->phys;
    out->mass = env->ctrl_model.mass;
    for (int i = 0; i < 3; i++) {
        out->inertia[i] = env->ctrl_model.inertia[i];
        out->c_drag[i] = env->ctrl_model.c_drag[i];
    }
    out->arm_length = env->ctrl_model.arm_length;
    out->c_thrust = env->ctrl_model.c_thrust;
    out->c_torque = env->ctrl_model.c_torque;
    out->motor_tau = env->ctrl_model.motor_tau;
    out->motor_max = env->ctrl_model.motor_max;
}

static inline float qr_norm3(const float v[3]) {
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static inline float qr_target_delta(const QuadrotorEnv* env, const Quadrotor* q,
        float delta[3]) {
    delta[0] = env->target[0] - q->pos[0];
    delta[1] = env->target[1] - q->pos[1];
    delta[2] = env->target[2] - q->pos[2];
    return qr_norm3(delta);
}

static inline float qr_dist2_xy(float ax, float ay, float bx, float by) {
    float dx = ax - bx;
    float dy = ay - by;
    return dx*dx + dy*dy;
}

static inline int qr_segment_intersects_box_xy(float x1, float y1,
        float x2, float y2, const float* box, float margin) {
    float xmin = box[0] - margin;
    float xmax = box[1] + margin;
    float ymin = box[2] - margin;
    float ymax = box[3] + margin;

    if ((x1 >= xmin && x1 <= xmax && y1 >= ymin && y1 <= ymax) ||
            (x2 >= xmin && x2 <= xmax && y2 >= ymin && y2 <= ymax))
        return 1;

    float tmin = 0.0f;
    float tmax = 1.0f;
    float dx = x2 - x1;
    float dy = y2 - y1;

    if (fabsf(dx) < 1e-6f) {
        if (x1 < xmin || x1 > xmax)
            return 0;
    } else {
        float t1 = (xmin - x1) / dx;
        float t2 = (xmax - x1) / dx;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
        if (tmin > tmax)
            return 0;
    }

    if (fabsf(dy) < 1e-6f) {
        if (y1 < ymin || y1 > ymax)
            return 0;
    } else {
        float t1 = (ymin - y1) / dy;
        float t2 = (ymax - y1) / dy;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
        if (tmin > tmax)
            return 0;
    }

    return tmax > 1e-5f && tmin < 1.0f - 1e-5f;
}

static inline float qr_route_path_margin(const QuadrotorEnv* env) {
    return env->route_path_margin > 0.0f
        ? env->route_path_margin
        : QR_NAV_PATH_MARGIN;
}

static inline int qr_nav_segment_blocked(const QuadrotorEnv* env,
        float x1, float y1, float x2, float y2) {
    for (int i = 0; i < env->num_obstacles; i++) {
        if (qr_segment_intersects_box_xy(x1, y1, x2, y2,
                env->obstacles[i], QR_DRONE_RADIUS))
            return 1;
    }
    return 0;
}

static inline int qr_add_path_node(const QuadrotorEnv* env,
        float xs[QR_NAV_PATH_MAX_NODES], float ys[QR_NAV_PATH_MAX_NODES],
        int count, float x, float y, float margin) {
    if (count >= QR_NAV_PATH_MAX_NODES)
        return count;

    if (env->room_size > 0.0f) {
        float bound = env->room_size - QR_DRONE_RADIUS;
        x = qr_clampf(x, -bound, bound);
        y = qr_clampf(y, -bound, bound);
    }
    if (!qr_valid_xy(env, x, y, QR_DRONE_RADIUS))
        return count;

    for (int i = 0; i < count; i++) {
        if (qr_dist2_xy(xs[i], ys[i], x, y) < margin * margin)
            return count;
    }

    xs[count] = x;
    ys[count] = y;
    return count + 1;
}

static inline float qr_nav_path_distance(const QuadrotorEnv* env,
        const Quadrotor* q) {
    float direct_delta[3];
    float direct_dist = qr_target_delta(env, q, direct_delta);
    if (!qr_task_uses_waypoint(env->task_mode) || env->num_obstacles <= 0)
        return direct_dist;

    float xs[QR_NAV_PATH_MAX_NODES];
    float ys[QR_NAV_PATH_MAX_NODES];
    int node_count = 0;

    xs[node_count] = q->pos[0];
    ys[node_count] = q->pos[1];
    node_count++;
    xs[node_count] = env->target[0];
    ys[node_count] = env->target[1];
    node_count++;

    float path_margin = qr_route_path_margin(env);
    for (int i = 0; i < env->num_obstacles; i++) {
        const float* box = env->obstacles[i];
        float xmin = box[0] - path_margin;
        float xmax = box[1] + path_margin;
        float ymin = box[2] - path_margin;
        float ymax = box[3] + path_margin;
        node_count = qr_add_path_node(env, xs, ys, node_count, xmin, ymin, 0.02f);
        node_count = qr_add_path_node(env, xs, ys, node_count, xmin, ymax, 0.02f);
        node_count = qr_add_path_node(env, xs, ys, node_count, xmax, ymin, 0.02f);
        node_count = qr_add_path_node(env, xs, ys, node_count, xmax, ymax, 0.02f);
    }

    float best[QR_NAV_PATH_MAX_NODES];
    int visited[QR_NAV_PATH_MAX_NODES];
    for (int i = 0; i < node_count; i++) {
        best[i] = QR_NAV_PATH_INF;
        visited[i] = 0;
    }
    best[0] = 0.0f;

    for (int iter = 0; iter < node_count; iter++) {
        int u = -1;
        float u_best = QR_NAV_PATH_INF;
        for (int i = 0; i < node_count; i++) {
            if (!visited[i] && best[i] < u_best) {
                u = i;
                u_best = best[i];
            }
        }
        if (u < 0 || u == 1)
            break;

        visited[u] = 1;
        for (int v = 0; v < node_count; v++) {
            if (visited[v] || v == u)
                continue;
            if (qr_nav_segment_blocked(env, xs[u], ys[u], xs[v], ys[v]))
                continue;

            float edge = sqrtf(qr_dist2_xy(xs[u], ys[u], xs[v], ys[v]));
            float candidate = best[u] + edge;
            if (candidate < best[v])
                best[v] = candidate;
        }
    }

    if (best[1] >= QR_NAV_PATH_INF * 0.5f)
        return direct_dist;

    float dz = env->target[2] - q->pos[2];
    return sqrtf(best[1]*best[1] + dz*dz);
}

static inline void qr_nav_observation_target(const QuadrotorEnv* env,
        const Quadrotor* q, float out[3]) {
    out[0] = env->target[0];
    out[1] = env->target[1];
    out[2] = env->target[2];

    if (!qr_task_uses_waypoint(env->task_mode) || env->num_obstacles <= 0)
        return;

    float path_margin = qr_route_path_margin(env);
    if (!qr_nav_segment_blocked(env, q->pos[0], q->pos[1],
            env->target[0], env->target[1]))
        return;

    float xs[QR_NAV_PATH_MAX_NODES];
    float ys[QR_NAV_PATH_MAX_NODES];
    int node_count = 0;

    xs[node_count] = q->pos[0];
    ys[node_count] = q->pos[1];
    node_count++;
    xs[node_count] = env->target[0];
    ys[node_count] = env->target[1];
    node_count++;

    for (int i = 0; i < env->num_obstacles; i++) {
        const float* box = env->obstacles[i];
        float xmin = box[0] - path_margin;
        float xmax = box[1] + path_margin;
        float ymin = box[2] - path_margin;
        float ymax = box[3] + path_margin;
        node_count = qr_add_path_node(env, xs, ys, node_count, xmin, ymin, 0.02f);
        node_count = qr_add_path_node(env, xs, ys, node_count, xmin, ymax, 0.02f);
        node_count = qr_add_path_node(env, xs, ys, node_count, xmax, ymin, 0.02f);
        node_count = qr_add_path_node(env, xs, ys, node_count, xmax, ymax, 0.02f);
    }

    float best[QR_NAV_PATH_MAX_NODES];
    int parent[QR_NAV_PATH_MAX_NODES];
    int visited[QR_NAV_PATH_MAX_NODES];
    for (int i = 0; i < node_count; i++) {
        best[i] = QR_NAV_PATH_INF;
        parent[i] = -1;
        visited[i] = 0;
    }
    best[0] = 0.0f;

    for (int iter = 0; iter < node_count; iter++) {
        int u = -1;
        float u_best = QR_NAV_PATH_INF;
        for (int i = 0; i < node_count; i++) {
            if (!visited[i] && best[i] < u_best) {
                u = i;
                u_best = best[i];
            }
        }
        if (u < 0 || u == 1)
            break;

        visited[u] = 1;
        for (int v = 0; v < node_count; v++) {
            if (visited[v] || v == u)
                continue;
            if (qr_nav_segment_blocked(env, xs[u], ys[u], xs[v], ys[v]))
                continue;

            float edge = sqrtf(qr_dist2_xy(xs[u], ys[u], xs[v], ys[v]));
            float candidate = best[u] + edge;
            if (candidate < best[v]) {
                best[v] = candidate;
                parent[v] = u;
            }
        }
    }

    if (best[1] >= QR_NAV_PATH_INF * 0.5f)
        return;

    int next = 1;
    while (parent[next] > 0)
        next = parent[next];

    if (parent[next] == 0 && next != 1) {
        out[0] = xs[next];
        out[1] = ys[next];
    }
}

static inline float qr_upright_tilt(const Quadrotor* q) {
    float body_z[3] = {0, 0, 1}, world_z[3];
    qr_body_to_world(q->quat, body_z, world_z);
    return 1.0f - world_z[2];
}

static inline int qr_out_of_room(const QuadrotorEnv* env, const Quadrotor* q) {
    if (env->task_mode == QR_TASK_INDOOR_ROUTE && !env->route_walls)
        return env->ceiling_height > 0.0f && q->pos[2] > env->ceiling_height;
    return env->room_size > 0.0f &&
        (q->pos[0] < -env->room_size || q->pos[0] > env->room_size ||
         q->pos[1] < -env->room_size || q->pos[1] > env->room_size ||
         q->pos[2] > env->ceiling_height);
}

static inline int qr_hits_ground(const QuadrotorEnv* env, const Quadrotor* q) {
    return q->pos[2] < 0.0f;
}

static inline int qr_hits_obstacle(const QuadrotorEnv* env, const Quadrotor* q) {
    return qr_point_in_obstacle(env, q->pos[0], q->pos[1], q->pos[2], QR_DRONE_RADIUS);
}

static inline void qr_action_to_control(QuadrotorEnv* env, float* thrust,
        float rate_des[3], float* action_effort) {
    Quadrotor* q = &env->phys;
    Quadrotor ctrl_q;
    qr_controller_state(env, &ctrl_q);
    float* a = env->actions;
    float hover_thrust = ctrl_q.mass * QR_GRAVITY;

    if (env->action_mode == QR_ACTION_MOTOR ||
            env->action_mode == QR_ACTION_MOTOR_HOVER ||
            env->action_mode == QR_ACTION_MOTOR_REF_HOVER) {
        *thrust = 0.0f;
        rate_des[0] = 0.0f;
        rate_des[1] = 0.0f;
        rate_des[2] = 0.0f;
        *action_effort = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3];
        return;
    }

    if (env->action_mode == QR_ACTION_VELOCITY) {
        float vel_body_des[3] = {
            qr_clampf(a[0], -1.0f, 1.0f) * env->max_vel,
            qr_clampf(a[1], -1.0f, 1.0f) * env->max_vel,
            0.0f
        };
        float vel_world_des[3];
        qr_body_to_world(q->quat, vel_body_des, vel_world_des);
        vel_world_des[2] = qr_clampf(a[2], -1.0f, 1.0f) * env->max_z_vel;

        qr_vel_control(&ctrl_q, &env->vel_ctrl, vel_world_des, qr_yaw_from_quat(q->quat),
            thrust, rate_des);
        rate_des[2] = qr_clampf(a[3], -1.0f, 1.0f) * env->max_rate;
        *action_effort = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3];
        return;
    }

    *thrust = qr_clampf(a[0], 0.0f, 1.0f) * 2.0f * hover_thrust;
    rate_des[0] = qr_clampf(a[1], -1.0f, 1.0f) * env->max_rate;
    rate_des[1] = qr_clampf(a[2], -1.0f, 1.0f) * env->max_rate;
    rate_des[2] = qr_clampf(a[3], -1.0f, 1.0f) * env->max_rate;

    float thrust_err = a[0] - 0.5f;
    *action_effort = thrust_err*thrust_err + a[1]*a[1] + a[2]*a[2] + a[3]*a[3];
}

static inline float qr_action_effort(const QuadrotorEnv* env) {
    const float* a = env->actions;
    if (env->action_mode == QR_ACTION_RATE) {
        float thrust_err = a[0] - 0.5f;
        return thrust_err*thrust_err + a[1]*a[1] + a[2]*a[2] + a[3]*a[3];
    }

    return a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3];
}

static inline float qr_hovering(float dist, float tilt, float hover_radius) {
    return (dist < hover_radius && tilt < 0.2f) ? 1.0f : 0.0f;
}

static inline float qr_drone_hover_potential(const QuadrotorEnv* env,
        float dist, float vel_mag, float omega_mag);

static inline float qr_goal_radius(const QuadrotorEnv* env) {
    return env->route_goal_radius > 0.0f ? env->route_goal_radius : QR_NAV_GOAL_RADIUS;
}

static inline float qr_nav_step_reward(const Quadrotor* q, const float delta[3],
        float direct_dist, float path_dist, float vel_mag, float tilt,
        float omega_mag, float action_effort, float clearance,
        float goal_radius, float* prev_path_dist, int* terminal) {
    float progress = *prev_path_dist - path_dist;
    *prev_path_dist = path_dist;

    float toward_goal = 0.0f;
    if (direct_dist > 1e-6f) {
        toward_goal = (delta[0]*q->vel[0] + delta[1]*q->vel[1] +
                       delta[2]*q->vel[2]) / direct_dist;
        if (toward_goal < 0.0f)
            toward_goal = 0.0f;
    }

    if (direct_dist < goal_radius)
        *terminal = 1;

    float clearance_penalty = 0.0f;
    if (clearance < QR_NAV_CLEARANCE_RADIUS) {
        float margin = (QR_NAV_CLEARANCE_RADIUS - clearance) / QR_NAV_CLEARANCE_RADIUS;
        clearance_penalty = QR_NAV_CLEARANCE_SCALE * margin * margin;
    }

    return QR_NAV_PROXIMITY_REWARD * expf(-QR_NAV_CENTER_SCALE * direct_dist * direct_dist)
        + QR_NAV_PROGRESS_SCALE * progress
        + QR_NAV_SPEED_SCALE * toward_goal
        - QR_NAV_STEP_COST
        - QR_NAV_DIST_SCALE * path_dist
        - clearance_penalty
        - QR_VEL_SCALE * 0.5f * vel_mag
        - QR_TILT_SCALE * tilt
        - QR_RATE_SCALE * omega_mag
        - QR_ACTION_SCALE * action_effort;
}

static inline float qr_route_step_reward(QuadrotorEnv* env,
        float direct_dist, float path_dist, float vel_mag, float toward_goal,
        float tilt,
        float omega_mag, float action_effort, float clearance,
        float goal_radius, float* prev_path_dist, int* terminal) {
    if (qr_route_reward_is_drone_hover_like(env)) {
        float prev_direct_dist = *prev_path_dist;
        float curr = qr_drone_hover_potential(env, direct_dist, vel_mag, omega_mag);
        float reward = env->drone_alpha_dist * (prev_direct_dist - direct_dist)
            + env->drone_alpha_hover * curr
            + env->drone_alpha_shaping * (curr - env->drone_prev_potential)
            - env->drone_alpha_omega * omega_mag
            - env->route_step_cost;
        if (env->route_clearance_scale > 0.0f &&
                env->route_clearance_radius > 1e-6f &&
                clearance < env->route_clearance_radius) {
            float margin = (env->route_clearance_radius - clearance) /
                env->route_clearance_radius;
            reward -= env->route_clearance_scale * margin * margin;
        }
        if (env->route_clearance_speed_scale > 0.0f &&
                env->route_clearance_speed_full >
                    env->route_clearance_speed_min &&
                toward_goal > 0.0f) {
            float gate = qr_clampf(
                (clearance - env->route_clearance_speed_min) /
                    (env->route_clearance_speed_full -
                        env->route_clearance_speed_min),
                0.0f, 1.0f);
            float goal_speed = env->route_goal_speed > 0.0f
                ? env->route_goal_speed
                : QR_ROUTE_GOAL_SPEED;
            float bounded_toward = toward_goal < goal_speed
                ? toward_goal
                : goal_speed;
            reward += env->route_clearance_speed_scale *
                gate * bounded_toward;
        }
        if (env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER_ARRIVAL &&
                env->route_arrival_radius > goal_radius &&
                direct_dist < env->route_arrival_radius) {
            float margin = (env->route_arrival_radius - direct_dist) /
                env->route_arrival_radius;
            float goal_speed = env->route_goal_speed > 0.0f
                ? env->route_goal_speed
                : QR_ROUTE_GOAL_SPEED;
            float excess_speed = vel_mag - goal_speed;
            if (excess_speed > 0.0f)
                reward -= env->route_arrival_vel_scale *
                    margin * excess_speed * excess_speed;
        }
        env->drone_prev_potential = curr;
        *prev_path_dist = direct_dist;
        if (env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER &&
                direct_dist < goal_radius)
            *terminal = 1;
        if (env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER_ARRIVAL) {
            float goal_speed = env->route_goal_speed > 0.0f
                ? env->route_goal_speed
                : QR_ROUTE_GOAL_SPEED;
            if (direct_dist < goal_radius && vel_mag <= goal_speed)
                *terminal = 1;
        }
        return reward;
    }

    if (env->route_reward_mode == QR_ROUTE_REWARD_REFERENCE) {
        float progress = *prev_path_dist - direct_dist;
        *prev_path_dist = direct_dist;

        float reward = 0.1f * progress - 0.001f - 0.001f * vel_mag;
        if (env->route_reference_clearance_scale > 0.0f &&
                env->route_reference_clearance_radius > 1e-6f &&
                clearance < env->route_reference_clearance_radius) {
            float margin = (env->route_reference_clearance_radius - clearance) /
                env->route_reference_clearance_radius;
            reward -= env->route_reference_clearance_scale * margin * margin;
        }
        if (direct_dist <= goal_radius) {
            float goal_speed = env->route_goal_speed > 0.0f
                ? env->route_goal_speed
                : QR_ROUTE_GOAL_SPEED;
            if (vel_mag <= goal_speed) {
                *terminal = 1;
                reward += env->route_goal_bonus;
            } else {
                reward -= 0.02f * (vel_mag - goal_speed);
            }
        }
        return reward;
    }

    float progress = *prev_path_dist - path_dist;
    *prev_path_dist = path_dist;

    if (direct_dist < goal_radius)
        *terminal = 1;

    float clearance_penalty = 0.0f;
    if (clearance < env->route_clearance_radius &&
            env->route_clearance_radius > 1e-6f) {
        float margin = (env->route_clearance_radius - clearance) /
            env->route_clearance_radius;
        clearance_penalty = env->route_clearance_scale * margin * margin;
    }

    float arrival_bonus = 0.0f;
    float arrival_vel_penalty = 0.0f;
    float arrival_toward_bonus = 0.0f;
    if (env->route_arrival_radius > goal_radius &&
            direct_dist < env->route_arrival_radius) {
        float margin = (env->route_arrival_radius - direct_dist) /
            env->route_arrival_radius;
        arrival_bonus = env->route_arrival_bonus * margin;
        arrival_vel_penalty = env->route_arrival_vel_scale * margin * vel_mag;
        if (env->route_arrival_toward_scale > 0.0f && toward_goal > 0.0f) {
            float max_toward = env->max_vel > 1e-6f ? env->max_vel : QR_MAX_VEL;
            float bounded_toward = toward_goal < max_toward ? toward_goal : max_toward;
            arrival_toward_bonus = env->route_arrival_toward_scale *
                margin * bounded_toward;
        }
    }

    return env->route_proximity_reward *
            expf(-QR_NAV_CENTER_SCALE * direct_dist * direct_dist)
        + env->route_progress_scale * progress
        + arrival_bonus
        + arrival_toward_bonus
        - env->route_step_cost
        - env->route_dist_scale * path_dist
        - clearance_penalty
        - arrival_vel_penalty
        - env->route_vel_scale * vel_mag
        - QR_TILT_SCALE * tilt
        - QR_RATE_SCALE * omega_mag
        - QR_ACTION_SCALE * action_effort;
}

static inline float qr_episode_crash_reward(const QuadrotorEnv* env,
        float final_dist) {
    if (env->task_mode == QR_TASK_INDOOR_ROUTE) {
        float missing = (float)(qr_episode_route_goal_count(env) - env->route_goals_reached);
        if (missing < 0.0f)
            missing = 0.0f;
        return env->route_crash_reward
            - env->route_crash_goal_scale * missing
            - env->route_crash_dist_scale * final_dist;
    }
    if (env->task_mode == QR_TASK_INDOOR_NAV)
        return QR_NAV_CRASH_REWARD;
    if (env->task_mode == QR_TASK_INDOOR_EXPLORE)
        return QR_EXPLORE_CRASH_REWARD;
    return QR_CRASH_REWARD;
}

static inline float qr_explore_wall_outward_penalty(
        const QuadrotorEnv* env, const Quadrotor* q) {
    if (env->room_size <= 0.0f)
        return 0.0f;

    float radius = QR_EXPLORE_OBSTACLE_INWARD_RADIUS;
    float max_vel = env->max_vel > 1e-6f ? env->max_vel : QR_MAX_VEL;
    float penalty = 0.0f;
    for (int axis = 0; axis < 2; axis++) {
        float pos = q->pos[axis];
        float wall_clearance = env->room_size - fabsf(pos);
        if (wall_clearance >= radius)
            continue;
        if (wall_clearance < 0.0f)
            wall_clearance = 0.0f;

        float outward_vel = pos >= 0.0f ? q->vel[axis] : -q->vel[axis];
        if (outward_vel <= 0.0f)
            continue;

        float margin = (radius - wall_clearance) / radius;
        float speed = qr_clampf(outward_vel / max_vel, 0.0f, 1.0f);
        float axis_penalty = QR_EXPLORE_WALL_OUTWARD_SCALE * margin * speed;
        if (axis_penalty > penalty)
            penalty = axis_penalty;
    }
    return penalty;
}

static inline float qr_explore_obstacle_inward_penalty(
        const QuadrotorEnv* env, const Quadrotor* q) {
    if (env->num_obstacles <= 0)
        return 0.0f;

    float radius = QR_EXPLORE_SOFT_CLEARANCE_RADIUS;
    float max_vel = env->max_vel > 1e-6f ? env->max_vel : QR_MAX_VEL;
    float penalty = 0.0f;
    for (int i = 0; i < env->num_obstacles; i++) {
        const float* box = env->obstacles[i];
        if (q->pos[2] < box[4] || q->pos[2] > box[5])
            continue;

        float nx = qr_clampf(q->pos[0], box[0], box[1]);
        float ny = qr_clampf(q->pos[1], box[2], box[3]);
        float dx = nx - q->pos[0];
        float dy = ny - q->pos[1];
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist >= radius || dist <= 1e-6f)
            continue;

        float inv_dist = 1.0f / dist;
        float ux = dx * inv_dist;
        float uy = dy * inv_dist;
        float vel_toward = q->vel[0] * ux + q->vel[1] * uy;

        float cmd_toward = 0.0f;
        if (env->action_mode == QR_ACTION_VELOCITY) {
            float vel_body_des[3] = {
                qr_clampf(env->actions[0], -1.0f, 1.0f) * max_vel,
                qr_clampf(env->actions[1], -1.0f, 1.0f) * max_vel,
                0.0f
            };
            float vel_world_des[3];
            qr_body_to_world(q->quat, vel_body_des, vel_world_des);
            cmd_toward = vel_world_des[0] * ux + vel_world_des[1] * uy;
        }

        float inward = vel_toward > cmd_toward ? vel_toward : cmd_toward;
        if (inward <= 0.0f)
            continue;

        float margin = (radius - dist) / radius;
        float speed = qr_clampf(inward / max_vel, 0.0f, 1.0f);
        float obstacle_penalty = QR_EXPLORE_OBSTACLE_INWARD_SCALE * margin * speed;
        if (obstacle_penalty > penalty)
            penalty = obstacle_penalty;
    }
    return penalty;
}

static inline float qr_explore_step_reward(QuadrotorEnv* env,
        float new_cell_reward, float vel_mag, float tilt, float omega_mag,
        float action_effort, float clearance, int* terminal) {
    float coverage = qr_explore_coverage(env);
    if (coverage >= env->explore_goal_fraction)
        *terminal = 1;

    float clearance_penalty = 0.0f;
    if (clearance < QR_EXPLORE_SOFT_CLEARANCE_RADIUS) {
        float margin = (QR_EXPLORE_SOFT_CLEARANCE_RADIUS - clearance) /
            QR_EXPLORE_SOFT_CLEARANCE_RADIUS;
        clearance_penalty += QR_EXPLORE_SOFT_CLEARANCE_SCALE * margin * margin;
    }
    if (clearance < QR_EXPLORE_CLEARANCE_RADIUS) {
        float margin = (QR_EXPLORE_CLEARANCE_RADIUS - clearance) /
            QR_EXPLORE_CLEARANCE_RADIUS;
        clearance_penalty += QR_EXPLORE_CLEARANCE_SCALE * margin * margin;
    }

    float frontier_progress_reward = 0.0f;
    if (env->explore_frontier_progress_scale > 0.0f) {
        float frontier_dir[2];
        float frontier_dist = qr_explore_progress_distance(env, &env->phys, frontier_dir);
        if (env->prev_frontier_dist > 0.0f && frontier_dist > 0.0f) {
            float progress = qr_clampf(
                env->prev_frontier_dist - frontier_dist, -0.10f, 0.10f);
            frontier_progress_reward =
                env->explore_frontier_progress_scale * progress;
        }
        env->prev_frontier_dist = frontier_dist;
    }

    float height_err = fabsf(env->phys.pos[2] - env->spawn_height);
    return new_cell_reward
        + frontier_progress_reward
        - QR_EXPLORE_STEP_COST
        - clearance_penalty
        - qr_explore_wall_outward_penalty(env, &env->phys)
        - qr_explore_obstacle_inward_penalty(env, &env->phys)
        - QR_EXPLORE_HEIGHT_SCALE * height_err
        - QR_VEL_SCALE * 0.25f * vel_mag
        - QR_TILT_SCALE * tilt
        - QR_RATE_SCALE * omega_mag
        - QR_ACTION_SCALE * action_effort;
}

static inline float qr_hover_step_reward(float dist, float vel_mag, float tilt,
        float omega_mag, float action_effort, float hovering) {
    float reward = QR_CENTER_REWARD * expf(-QR_CENTER_SCALE * dist * dist)
        - QR_DIST_SCALE * dist
        - QR_VEL_SCALE * vel_mag
        - QR_TILT_SCALE * tilt
        - QR_RATE_SCALE * omega_mag
        - QR_ACTION_SCALE * action_effort;

    if (hovering > 0.0f)
        reward += QR_HOVER_BONUS;

    reward += QR_SETTLE_REWARD * expf(
        -QR_SETTLE_DIST_SCALE * dist * dist
        -QR_SETTLE_VEL_SCALE * vel_mag * vel_mag);
    return reward;
}

static inline float qr_drone_hover_potential(const QuadrotorEnv* env,
        float dist, float vel_mag, float omega_mag) {
    float d = 1.0f / (1.0f + dist / env->drone_hover_dist);
    float v = 1.0f / (1.0f + vel_mag / env->drone_hover_vel);
    float w = 1.0f / (1.0f + omega_mag / env->drone_hover_omega);
    return d * (0.7f + 0.15f * v + 0.15f * w);
}

static inline float qr_drone_hover_score(const QuadrotorEnv* env,
        float dist, float vel_mag, float omega_mag) {
    float d = dist / (env->drone_hover_dist * 10.0f);
    float v = vel_mag / (env->drone_hover_vel * 10.0f);
    float w = omega_mag / (env->drone_hover_omega * 10.0f);
    float score = 1.0f - 0.7f * d - 0.15f * v - 0.15f * w;
    return score > 0.0f ? score : 0.0f;
}

static inline float qr_drone_hover_step_reward(QuadrotorEnv* env,
        float prev_dist, float dist, float vel_mag, float omega_mag) {
    float curr = qr_drone_hover_potential(env, dist, vel_mag, omega_mag);
    float reward = env->drone_alpha_dist * (prev_dist - dist)
        + env->drone_alpha_hover * curr
        + env->drone_alpha_shaping * (curr - env->drone_prev_potential)
        - env->drone_alpha_omega * omega_mag;
    env->drone_prev_potential = curr;
    return reward;
}

static inline float qr_terminal_reward(const QuadrotorEnv* env,
        float final_dist, float final_vel) {
    if (env->task_mode == QR_TASK_DRONE_HOVER)
        return 0.0f;

    if (env->task_mode == QR_TASK_INDOOR_EXPLORE) {
        if (qr_explore_coverage(env) >= env->explore_goal_fraction)
            return QR_EXPLORE_GOAL_BONUS;
        if (env->tick >= env->max_steps) {
            float goal = env->explore_goal_fraction > 1e-6f
                ? env->explore_goal_fraction
                : QR_EXPLORE_GOAL_FRACTION;
            float missing = (goal - qr_explore_coverage(env)) / goal;
            missing = qr_clampf(missing, 0.0f, 1.0f);
            return -QR_EXPLORE_TIMEOUT_PENALTY
                - QR_EXPLORE_TIMEOUT_MISSING_SCALE * missing;
        }
        return 0.0f;
    }

    if (env->task_mode == QR_TASK_INDOOR_NAV) {
        if (final_dist < qr_goal_radius(env))
            return QR_NAV_GOAL_BONUS;
        if (env->tick >= env->max_steps)
            return -QR_NAV_TIMEOUT_PENALTY - QR_NAV_TIMEOUT_DIST_SCALE * final_dist;
        return 0.0f;
    }

    if (env->task_mode == QR_TASK_INDOOR_ROUTE) {
        if (env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER_CONTINUOUS)
            return 0.0f;
        if (qr_route_reward_uses_direct_dist(env) &&
                env->route_reward_mode == QR_ROUTE_REWARD_REFERENCE)
            return 0.0f;
        if (env->route_goals_reached >= qr_episode_route_goal_count(env))
            return QR_ROUTE_GOAL_BONUS;
        if (env->tick >= env->max_steps) {
            float missing = (float)(qr_episode_route_goal_count(env) - env->route_goals_reached);
            if (missing < 0.0f)
                missing = 0.0f;
            return -QR_ROUTE_TIMEOUT_PENALTY
                - QR_ROUTE_TIMEOUT_GOAL_SCALE * missing
                - QR_NAV_TIMEOUT_DIST_SCALE * final_dist;
        }
        return 0.0f;
    }

    if (final_dist < QR_SETTLE_RADIUS && final_vel < QR_SETTLE_VEL)
        return QR_TERMINAL_SETTLE_BONUS;

    float dist_ratio = final_dist / QR_SETTLE_RADIUS;
    float vel_ratio = final_vel / QR_SETTLE_VEL;
    return -QR_TERMINAL_UNSETTLED_SCALE *
        (dist_ratio * dist_ratio + vel_ratio * vel_ratio);
}

static inline void qr_spawn_quadrotor(QuadrotorEnv* env, float bound) {
    Quadrotor* q = &env->phys;
    float spawn_margin = qr_spawn_xy_margin(env);
    if (qr_task_uses_room_spawn(env->task_mode) && bound > 0.0f) {
        int spawned = 0;
        float margins[3] = {
            spawn_margin,
            0.5f * (spawn_margin + QR_DRONE_RADIUS),
            QR_DRONE_RADIUS,
        };
        for (int i = 0; !spawned && i < 3; i++) {
            if (i > 0 && fabsf(margins[i] - margins[i - 1]) < 1e-6f)
                continue;
            if (qr_try_spawn_quadrotor_xy(env, bound, margins[i]))
                spawned = 1;
        }
        if (!spawned) {
            q->pos[0] = qr_clamp_room_xy(env, -0.5f * bound);
            q->pos[1] = qr_clamp_room_xy(env, -0.5f * bound);
        }
        q->pos[2] = qr_safe_height(
            env, env->spawn_height + qr_rand_range(env, -0.1f, 0.1f));
        return;
    }

    float rx = qr_rand_range(env, -env->spawn_radius, env->spawn_radius);
    float ry = qr_rand_range(env, -env->spawn_radius, env->spawn_radius);
    float rz = qr_rand_range(env, -0.5f * env->spawn_radius, 0.5f * env->spawn_radius)
               + env->spawn_height;
    q->pos[0] = qr_clamp_room_xy(env, rx);
    q->pos[1] = qr_clamp_room_xy(env, ry);
    if (!qr_valid_xy(env, q->pos[0], q->pos[1], spawn_margin)) {
        float margins[3] = {
            spawn_margin,
            0.5f * (spawn_margin + QR_DRONE_RADIUS),
            QR_DRONE_RADIUS,
        };
        int spawned = 0;
        for (int i = 0; !spawned && i < 3; i++) {
            if (i > 0 && fabsf(margins[i] - margins[i - 1]) < 1e-6f)
                continue;
            if (qr_valid_xy(env, q->pos[0], q->pos[1], margins[i])) {
                spawned = 1;
                break;
            }
        }
        if (!spawned) {
            q->pos[0] = 0.0f;
            q->pos[1] = 0.0f;
        }
    }
    q->pos[2] = qr_safe_height(env, rz);
}

static inline void qr_select_drone_hover_target(QuadrotorEnv* env, float bound) {
    Quadrotor* q = &env->phys;
    float u = qr_rand01(env);
    float v = qr_rand01(env);
    float z = 2.0f * v - 1.0f;
    float a = 2.0f * QR_PI * u;
    float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float rad = env->drone_hover_target_dist * cbrtf(qr_rand01(env));

    float tx = q->pos[0] + rad * r_xy * cosf(a);
    float ty = q->pos[1] + rad * r_xy * sinf(a);
    float tz = q->pos[2] + rad * z;

    if (bound > 0.0f) {
        tx = qr_clamp_room_xy_with_margin(env, tx, env->route_target_wall_margin);
        ty = qr_clamp_room_xy_with_margin(env, ty, env->route_target_wall_margin);
    }

    env->target[0] = tx;
    env->target[1] = ty;
    env->target[2] = qr_safe_height(env, tz);
}

static inline void qr_select_target(QuadrotorEnv* env, float bound) {
    Quadrotor* q = &env->phys;
    if (env->task_mode == QR_TASK_HOLD_HOVER ||
            env->task_mode == QR_TASK_INDOOR_EXPLORE) {
        env->target[0] = q->pos[0];
        env->target[1] = q->pos[1];
        env->target[2] = env->task_mode == QR_TASK_INDOOR_EXPLORE
            ? qr_safe_height(env, env->spawn_height)
            : q->pos[2];
        return;
    }

    if (env->task_mode == QR_TASK_DRONE_HOVER ||
            (env->task_mode == QR_TASK_INDOOR_ROUTE &&
             env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER_CONTINUOUS)) {
        qr_select_drone_hover_target(env, bound);
        return;
    }

    env->target[0] = 0.0f;
    env->target[1] = 0.0f;
    env->target[2] = qr_sample_route_target_z(env);
    if (!qr_task_uses_waypoint(env->task_mode) || bound <= 0.0f)
        return;

    float min_target_dist = QR_NAV_MIN_TARGET_DIST;
    if (env->task_mode == QR_TASK_INDOOR_ROUTE) {
        float radius_dist = 1.5f * qr_goal_radius(env);
        if (radius_dist > min_target_dist)
            min_target_dist = radius_dist;
    }

    int sample_blocked = env->route_target_mode == QR_ROUTE_TARGET_BLOCKED &&
        env->num_obstacles > 0 &&
        qr_rand01(env) < env->route_blocked_target_prob;
    if (sample_blocked) {
        for (int i = 0; i < 64; i++) {
            float tx = qr_rand_range(env, -bound, bound);
            float ty = qr_rand_range(env, -bound, bound);
            float dx = tx - q->pos[0];
            float dy = ty - q->pos[1];
            int far_enough = sqrtf(dx*dx + dy*dy) >= min_target_dist;
            if (far_enough &&
                    qr_route_target_wall_ok(env, tx, ty) &&
                    qr_valid_xy(env, tx, ty, QR_DRONE_RADIUS) &&
                    qr_nav_segment_blocked(env, q->pos[0], q->pos[1], tx, ty)) {
                env->target[0] = tx;
                env->target[1] = ty;
                env->target[2] = qr_sample_route_target_z(env);
                return;
            }
        }
    }

    for (int i = 0; i < 16; i++) {
        float tx = qr_rand_range(env, -bound, bound);
        float ty = qr_rand_range(env, -bound, bound);
        float dx = tx - q->pos[0];
        float dy = ty - q->pos[1];
        int far_enough = sqrtf(dx*dx + dy*dy) >= min_target_dist || i == 15;
        if (far_enough &&
                qr_route_target_wall_ok(env, tx, ty) &&
                qr_valid_xy(env, tx, ty, QR_DRONE_RADIUS)) {
            env->target[0] = tx;
            env->target[1] = ty;
            env->target[2] = qr_sample_route_target_z(env);
            return;
        }
    }

    float fallback_x = qr_clamp_room_xy_with_margin(
        env, -q->pos[0], env->route_target_wall_margin);
    float fallback_y = qr_clamp_room_xy_with_margin(
        env, -q->pos[1], env->route_target_wall_margin);
    if (qr_valid_xy(env, fallback_x, fallback_y, QR_DRONE_RADIUS)) {
        env->target[0] = fallback_x;
        env->target[1] = fallback_y;
        env->target[2] = qr_sample_route_target_z(env);
    }
}

static inline void qr_randomize_initial_motion(QuadrotorEnv* env) {
    for (int i = 0; i < 3; i++) {
        float rv = qr_rand_range(env, -1.0f, 1.0f);
        float rw = qr_rand_range(env, -1.0f, 1.0f);
        env->phys.vel[i] = rv * env->init_vel_radius;
        env->phys.omega[i] = rw * env->init_omega_radius;
    }
}

static inline void qr_apply_route_hardstate_probe(QuadrotorEnv* env) {
    if (!env->route_hardstate_probe)
        return;

    Quadrotor* q = &env->phys;
    float start_x = env->route_hardstate_start_x;
    float start_y = env->route_hardstate_start_y;
    float start_z = env->route_hardstate_start_z;
    float target_x = env->route_hardstate_target_x;
    float target_y = env->route_hardstate_target_y;
    float target_z = env->route_hardstate_target_z;

    if (env->route_hardstate_jitter_xy > 0.0f) {
        start_x += qr_rand_range(env,
            -env->route_hardstate_jitter_xy, env->route_hardstate_jitter_xy);
        start_y += qr_rand_range(env,
            -env->route_hardstate_jitter_xy, env->route_hardstate_jitter_xy);
        target_x += qr_rand_range(env,
            -env->route_hardstate_jitter_xy, env->route_hardstate_jitter_xy);
        target_y += qr_rand_range(env,
            -env->route_hardstate_jitter_xy, env->route_hardstate_jitter_xy);
    }
    if (env->route_hardstate_jitter_z > 0.0f) {
        start_z += qr_rand_range(env,
            -env->route_hardstate_jitter_z, env->route_hardstate_jitter_z);
        target_z += qr_rand_range(env,
            -env->route_hardstate_jitter_z, env->route_hardstate_jitter_z);
    }

    q->pos[0] = qr_clamp_room_xy(env, start_x);
    q->pos[1] = qr_clamp_room_xy(env, start_y);
    q->pos[2] = qr_safe_height(env, start_z);
    env->target[0] = qr_clamp_room_xy(env, target_x);
    env->target[1] = qr_clamp_room_xy(env, target_y);
    env->target[2] = qr_safe_height(env, target_z);

    float yaw = env->route_hardstate_yaw_deg * QR_PI / 180.0f;
    q->quat[0] = cosf(0.5f * yaw);
    q->quat[1] = 0.0f;
    q->quat[2] = 0.0f;
    q->quat[3] = sinf(0.5f * yaw);
    memset(q->vel, 0, sizeof(q->vel));
    memset(q->omega, 0, sizeof(q->omega));
    q->crashed = 0;
    q->tick = 0;
    env->route_goals_reached = 0;
    if (env->action_mode == QR_ACTION_RATE ||
            env->action_mode == QR_ACTION_VELOCITY ||
            env->action_mode == QR_ACTION_MOTOR_HOVER ||
            env->action_mode == QR_ACTION_MOTOR_REF_HOVER) {
        float hover = qr_hover_motor_speed(q);
        for (int i = 0; i < QR_NUM_ACTIONS; i++)
            q->motor_speed[i] = hover;
    }
}

static inline void qr_refresh_episode_metrics(QuadrotorEnv* env) {
    float delta[3];
    env->final_dist = qr_target_delta(env, &env->phys, delta);
    env->final_z_error = fabsf(delta[2]);
    env->prev_dist = qr_route_reward_uses_direct_dist(env)
        ? env->final_dist
        : qr_nav_path_distance(env, &env->phys);
    env->max_dist = env->final_dist;
    env->start_dist = env->prev_dist;
    env->final_vel = qr_norm3(env->phys.vel);
    env->min_clearance = qr_clearance(env, &env->phys);
    if (env->task_mode == QR_TASK_INDOOR_EXPLORE) {
        env->final_dist = 0.0f;
        env->final_z_error = 0.0f;
        env->prev_dist = 0.0f;
        env->max_dist = 0.0f;
        env->start_dist = 0.0f;
    } else if (env->task_mode == QR_TASK_DRONE_HOVER) {
        env->prev_dist = env->final_dist;
        env->start_dist = env->final_dist;
        env->drone_prev_potential = qr_drone_hover_potential(
            env, env->final_dist, env->final_vel, qr_norm3(env->phys.omega));
    } else if (env->task_mode == QR_TASK_INDOOR_ROUTE &&
            qr_route_reward_is_drone_hover_like(env)) {
        env->drone_prev_potential = qr_drone_hover_potential(
            env, env->final_dist, env->final_vel, qr_norm3(env->phys.omega));
    }
    if (env->final_dist >= QR_SETTLE_RADIUS || env->final_vel >= QR_SETTLE_VEL)
        env->last_unsettled_tick = 0;
}

static inline void c_reset(QuadrotorEnv* env) {
    qr_clear_episode_stats(env);
    qr_configure_dynamics(env);
    qr_reset(&env->phys);
    if (env->action_mode == QR_ACTION_RATE ||
            env->action_mode == QR_ACTION_VELOCITY ||
            env->action_mode == QR_ACTION_MOTOR_HOVER ||
            env->action_mode == QR_ACTION_MOTOR_REF_HOVER) {
        float hover = qr_hover_motor_speed(&env->phys);
        for (int i = 0; i < QR_NUM_ACTIONS; i++)
            env->phys.motor_speed[i] = hover;
    }
    qr_rate_reset(&env->rate_ctrl);
    qr_pos_reset(&env->vel_ctrl);
    qr_configure_obstacles(env);
    qr_sample_episode_route_goal_count(env);

    qr_sample_episode_spawn_radius(env);
    float bound = qr_room_spawn_bound(env);
    qr_spawn_quadrotor(env, bound);
    qr_select_target(env, bound);
    qr_apply_route_hardstate_probe(env);
    qr_reset_exploration_grid(env);
    qr_mark_explored(env, 0.0f);
    if (env->task_mode == QR_TASK_INDOOR_EXPLORE) {
        float frontier_dir[2];
        env->prev_frontier_dist = qr_explore_progress_distance(
            env, &env->phys, frontier_dir);
    }
    qr_randomize_initial_motion(env);
    qr_refresh_episode_metrics(env);
    compute_observations(env);
}

static inline void c_step(QuadrotorEnv* env) {
    Quadrotor* q = &env->phys;
    env->route_goal_event = 0;
    env->route_goal_event_index = -1;
    env->route_goal_event_dist = 0.0f;
    env->route_goal_event_speed = 0.0f;
    env->route_goal_event_clearance = 0.0f;
    env->route_goal_event_visible = 0;

    float action_effort = qr_action_effort(env);

    float saturation = 0.0f;
    float action_delta = 0.0f;
    for (int i = 0; i < QR_NUM_ACTIONS; i++) {
        float a = env->actions[i];
        if (env->action_mode == QR_ACTION_RATE && i == 0) {
            if (a < 0.05f || a > 0.95f) saturation += 1.0f;
        } else if (fabsf(a) > 0.95f) {
            saturation += 1.0f;
        }
        if (env->has_prev_action)
            action_delta += fabsf(a - env->prev_action[i]);
        env->prev_action[i] = a;
    }
    env->has_prev_action = 1;
    env->action_saturation_sum += saturation / (float)QR_NUM_ACTIONS;
    env->action_delta_sum += action_delta / (float)QR_NUM_ACTIONS;
    env->action_steps += 1.0f;

    float reward_sum = 0.0f;
    float dist_sum = 0.0f;
    float vel_sum = 0.0f;
    float tilt_sum = 0.0f;
    float hover_sum = 0.0f;
    float final_dist = env->final_dist;
    float final_z_error = env->final_z_error;
    float final_vel = env->final_vel;
    float prev_path_dist = env->prev_dist;
    int steps = 0;
    int terminal = 0;
    for (int i = 0; i < env->action_repeat; i++) {
        float prev_pos[3] = {q->pos[0], q->pos[1], q->pos[2]};
        if (env->action_mode == QR_ACTION_MOTOR) {
            for (int s = 0; s < env->physics_substeps; s++)
                qr_step_normalized_dt(q, env->actions, env->physics_dt);
        } else if (env->action_mode == QR_ACTION_MOTOR_HOVER) {
            for (int s = 0; s < env->physics_substeps; s++)
                qr_step_hover_normalized_dt(q, env->actions, env->physics_dt);
        } else if (env->action_mode == QR_ACTION_MOTOR_REF_HOVER) {
            for (int s = 0; s < env->physics_substeps; s++)
                qr_step_hover_centered_normalized_dt(
                    q, env->actions, env->physics_dt);
        } else {
            float thrust;
            float rate_des[3];
            float control_effort;
            float motor_cmd[4];
            Quadrotor ctrl_q;
            qr_action_to_control(env, &thrust, rate_des, &control_effort);
            qr_controller_state(env, &ctrl_q);
            qr_rate_control(&ctrl_q, &env->rate_ctrl, thrust, rate_des, motor_cmd);
            for (int s = 0; s < env->physics_substeps; s++)
                qr_step_dt(q, motor_cmd, env->physics_dt);
        }

        if (!qr_quadrotor_state_finite(q)) {
            q->crashed = 1;
            env->crash_cause = QR_CRASH_DIVERGED;
            reward_sum += qr_episode_crash_reward(env, 50.0f);
            dist_sum += 50.0f;
            vel_sum += QR_BASE_MAX_VEL;
            tilt_sum += QR_PI;
            final_dist = 50.0f;
            final_z_error = 50.0f;
            final_vel = QR_BASE_MAX_VEL;
            terminal = 1;
            steps++;
            break;
        }

        float step_delta[3] = {
            q->pos[0] - prev_pos[0],
            q->pos[1] - prev_pos[1],
            q->pos[2] - prev_pos[2]
        };
        env->path_length += qr_norm3(step_delta);
        float clearance = qr_clearance(env, q);
        if (clearance < env->min_clearance)
            env->min_clearance = clearance;

        float delta[3];
        float dist = qr_target_delta(env, q, delta);
        float path_dist = qr_nav_path_distance(env, q);
        dist_sum += dist;
        if (dist > env->max_dist)
            env->max_dist = dist;

        float tilt = qr_upright_tilt(q);
        float vel_mag = qr_norm3(q->vel);
        float toward_goal = 0.0f;
        if (dist > 1e-6f)
            toward_goal = (delta[0]*q->vel[0] + delta[1]*q->vel[1] +
                           delta[2]*q->vel[2]) / dist;
        final_dist = dist;
        final_z_error = fabsf(delta[2]);
        final_vel = vel_mag;

        float omega_mag = qr_norm3(q->omega);

        float hover = qr_hovering(dist, tilt, env->hover_radius);
        float r;
        if (qr_task_uses_waypoint(env->task_mode)) {
            int target_reached = 0;
            if (env->task_mode == QR_TASK_INDOOR_NAV) {
                r = qr_nav_step_reward(q, delta, dist, path_dist, vel_mag, tilt,
                    omega_mag, action_effort, clearance, qr_goal_radius(env),
                    &prev_path_dist, &target_reached);
                terminal = target_reached;
            } else {
                r = qr_route_step_reward(env, dist, path_dist, vel_mag,
                    toward_goal, tilt, omega_mag, action_effort, clearance,
                    qr_goal_radius(env),
                    &prev_path_dist, &target_reached);
                if (target_reached) {
                    env->route_goal_event = 1;
                    env->route_goal_event_index = env->route_goals_reached;
                    env->route_goal_event_dist = dist;
                    env->route_goal_event_speed = vel_mag;
                    env->route_goal_event_clearance = clearance;
                    env->route_goal_event_visible = !qr_nav_segment_blocked(
                        env, q->pos[0], q->pos[1],
                        env->target[0], env->target[1]);
                    env->route_goals_reached++;
                    if (env->route_reward_mode != QR_ROUTE_REWARD_REFERENCE &&
                            env->route_reward_mode != QR_ROUTE_REWARD_DRONE_HOVER_CONTINUOUS)
                        r += env->route_reach_bonus;
                    if (env->route_goals_reached >= qr_episode_route_goal_count(env)) {
                        terminal = 1;
                    } else {
                        qr_select_target(env, qr_room_spawn_bound(env));
                        final_dist = qr_target_delta(env, q, delta);
                        final_z_error = fabsf(delta[2]);
                        prev_path_dist = qr_route_reward_uses_direct_dist(env)
                            ? final_dist
                            : qr_nav_path_distance(env, q);
                        if (qr_route_reward_is_drone_hover_like(env)) {
                            env->drone_prev_potential = qr_drone_hover_potential(
                                env, final_dist, final_vel, omega_mag);
                        }
                    }
                }
            }
        } else if (env->task_mode == QR_TASK_INDOOR_EXPLORE) {
            float new_cell_reward = qr_mark_visible_explored(env,
                QR_EXPLORE_VISIBLE_CELL_REWARD);
            r = qr_explore_step_reward(env, new_cell_reward, vel_mag, tilt,
                omega_mag, action_effort, clearance, &terminal);
        } else if (env->task_mode == QR_TASK_DRONE_HOVER) {
            float pdx = env->target[0] - prev_pos[0];
            float pdy = env->target[1] - prev_pos[1];
            float pdz = env->target[2] - prev_pos[2];
            float prev_direct_dist = sqrtf(pdx*pdx + pdy*pdy + pdz*pdz);
            r = qr_drone_hover_step_reward(
                env, prev_direct_dist, dist, vel_mag, omega_mag);
            hover = qr_drone_hover_score(env, dist, vel_mag, omega_mag);
            prev_path_dist = dist;
            if (dist > env->drone_hover_target_dist + 1.0f) {
                env->drone_oob = 1;
                terminal = 1;
            }
        } else {
            r = qr_hover_step_reward(dist, vel_mag, tilt, omega_mag,
                action_effort, hover);
        }

        if (env->task_mode == QR_TASK_INDOOR_ROUTE &&
                env->route_reward_mode == QR_ROUTE_REWARD_DRONE_HOVER_CONTINUOUS) {
            hover = qr_drone_hover_score(env, dist, vel_mag, omega_mag);
            if (dist > env->drone_hover_target_dist + 1.0f) {
                env->drone_oob = 1;
                terminal = 1;
            }
        }

        if (q->crashed || qr_hits_ground(env, q)) {
            q->crashed = 1;
            if (env->crash_cause == QR_CRASH_NONE)
                env->crash_cause = QR_CRASH_GROUND;
            r = qr_episode_crash_reward(env, dist);
            terminal = 1;
        } else if (qr_hits_obstacle(env, q)) {
            q->crashed = 1;
            env->crash_cause = QR_CRASH_OBSTACLE;
            r = qr_episode_crash_reward(env, dist);
            terminal = 1;
        } else if (qr_out_of_room(env, q)) {
            q->crashed = 1;
            env->crash_cause = QR_CRASH_WALL;
            r = qr_episode_crash_reward(env, dist);
            terminal = 1;
        } else if (q->pos[2] > 50.0f || dist > 50.0f) {
            q->crashed = 1;
            env->crash_cause = QR_CRASH_DIVERGED;
            r = qr_episode_crash_reward(env, dist);
            terminal = 1;
        }

        reward_sum += r;
        vel_sum += vel_mag;
        tilt_sum += tilt;
        hover_sum += hover;
        steps++;
        if (terminal) break;
    }

    env->tick++;
    if (env->tick >= env->max_steps) {
        terminal = 1;
        env->timed_out = !q->crashed && (
            (env->task_mode == QR_TASK_INDOOR_NAV &&
                final_dist >= qr_goal_radius(env)) ||
            (env->task_mode == QR_TASK_INDOOR_ROUTE &&
                env->route_reward_mode != QR_ROUTE_REWARD_DRONE_HOVER_CONTINUOUS &&
                env->route_goals_reached < qr_episode_route_goal_count(env)) ||
            (env->task_mode == QR_TASK_INDOOR_EXPLORE &&
                qr_explore_coverage(env) < env->explore_goal_fraction));
    }

    float r = reward_sum / (float)(steps > 0 ? steps : 1);
    env->dist_accum += dist_sum / (float)(steps > 0 ? steps : 1);
    env->vel_accum += vel_sum / (float)(steps > 0 ? steps : 1);
    env->tilt_accum += tilt_sum / (float)(steps > 0 ? steps : 1);
    env->hover_steps += hover_sum / (float)(steps > 0 ? steps : 1);
    env->final_dist = final_dist;
    env->final_z_error = final_z_error;
    env->final_vel = final_vel;
    env->prev_dist = prev_path_dist;
    if (final_dist >= QR_SETTLE_RADIUS || final_vel >= QR_SETTLE_VEL)
        env->last_unsettled_tick = env->tick - 1;

    if (terminal && q->crashed &&
            env->task_mode == QR_TASK_INDOOR_EXPLORE)
        r += QR_EXPLORE_TERMINAL_CRASH_SCALE * QR_EXPLORE_CRASH_REWARD;
    else if (terminal && !q->crashed)
        r += qr_terminal_reward(env, final_dist, final_vel);

    env->rewards[0] = r;
    env->terminals[0] = terminal ? 1.0f : 0.0f;
    env->episode_return += r;

    if (terminal) {
        env->last_episode_clearance = qr_clearance(env, &env->phys);
        for (int i = 0; i < 3; i++) {
            env->last_episode_pos[i] = env->phys.pos[i];
            env->last_episode_vel[i] = env->phys.vel[i];
            env->last_episode_target[i] = env->target[i];
        }
        qr_nav_observation_target(env, &env->phys, env->last_episode_obs_target);
        add_log(env);
        c_reset(env);
    } else {
        compute_observations(env);
    }
}

static void qr_draw_drone(const Quadrotor* q);
static void qr_draw_rays(const QuadrotorEnv* env);
static void qr_draw_room(const QuadrotorEnv* env);
static void qr_draw_obstacles(const QuadrotorEnv* env);
static void qr_draw_explored(const QuadrotorEnv* env);
static void qr_draw_exploration_planner(const QuadrotorEnv* env);
static void qr_draw_ground(void);
static const char* qr_status_name(int status);
static Color qr_status_color(int status);

static inline void qr_reset_render_camera(QuadrotorEnv* env) {
    Client* client = env->client;
    client->camera_yaw = (env->room_size > 0.0f) ? 135.0f : 135.0f;
    client->camera_pitch = (env->room_size > 0.0f) ? 36.0f : 32.0f;
    client->camera_distance = (env->room_size > 0.0f) ? 8.5f : 5.0f;
}

static inline void qr_update_render_controls(QuadrotorEnv* env) {
    Client* client = env->client;
    float dt = GetFrameTime();
    if (dt <= 0.0f || dt > 0.1f) dt = 1.0f / fmaxf(client->render_fps, 1.0f);

    float yaw_step = 95.0f * dt;
    float pitch_step = 75.0f * dt;
    float zoom_step = 6.0f * dt;
    if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
        yaw_step *= 2.0f;
        pitch_step *= 2.0f;
        zoom_step *= 2.0f;
    }

    if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) client->camera_yaw -= yaw_step;
    if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) client->camera_yaw += yaw_step;
    if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) client->camera_pitch += pitch_step;
    if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) client->camera_pitch -= pitch_step;

    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) client->camera_distance -= wheel * 0.9f;
    if (IsKeyDown(KEY_Q)) client->camera_distance += zoom_step;
    if (IsKeyDown(KEY_E)) client->camera_distance -= zoom_step;

    if (IsKeyPressed(KEY_R)) qr_reset_render_camera(env);

    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
        client->render_fps = fminf(client->render_fps * 1.25f, 240.0f);
        SetTargetFPS((int)client->render_fps);
    }
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
        client->render_fps = fmaxf(client->render_fps / 1.25f, 5.0f);
        SetTargetFPS((int)client->render_fps);
    }

    client->camera_pitch = fminf(fmaxf(client->camera_pitch, 8.0f), 82.0f);
    client->camera_distance = fminf(fmaxf(client->camera_distance, 1.5f), 35.0f);
}

static inline void qr_apply_render_camera(QuadrotorEnv* env, Vector3 drone_pos) {
    Client* client = env->client;
    float yaw = client->camera_yaw * QR_PI / 180.0f;
    float pitch = client->camera_pitch * QR_PI / 180.0f;
    float horizontal = client->camera_distance * cosf(pitch);

    client->camera.target = drone_pos;
    client->camera.position = (Vector3){
        drone_pos.x + horizontal * cosf(yaw),
        drone_pos.y + client->camera_distance * sinf(pitch),
        drone_pos.z + horizontal * sinf(yaw),
    };
}

static inline void c_render(QuadrotorEnv* env) {
    if (env->client == NULL) {
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->window_w = 1200;
        env->client->window_h = 800;
        InitWindow(env->client->window_w, env->client->window_h, "PufferLib Quadrotor");
        env->client->render_fps = 60.0f;
        SetTargetFPS((int)env->client->render_fps);
        env->client->camera = (Camera3D){
            .position = {-3.0f, 3.0f, 3.0f},
            .target = {0, 1, 0},
            .up = {0, 1, 0},
            .fovy = 55.0f,
            .projection = CAMERA_PERSPECTIVE,
        };
        qr_reset_render_camera(env);
    }
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) exit(0);

    Quadrotor* q = &env->phys;
    Vector3 drone_pos = {q->pos[0], q->pos[2], -q->pos[1]};
    qr_update_render_controls(env);
    qr_apply_render_camera(env, drone_pos);

    BeginDrawing();
    ClearBackground((Color){15, 15, 25, 255});
    BeginMode3D(env->client->camera);

    qr_draw_ground();
    qr_draw_room(env);
    qr_draw_explored(env);
    qr_draw_exploration_planner(env);
    qr_draw_obstacles(env);
    qr_draw_rays(env);
    qr_draw_drone(q);

    if (env->task_mode != QR_TASK_INDOOR_EXPLORE) {
        Vector3 tv = {env->target[0], env->target[2], -env->target[1]};
        Vector3 dp = {q->pos[0], q->pos[2], -q->pos[1]};
        DrawSphereWires(tv, qr_goal_radius(env), 16, 16, Fade(RED, 0.35f));
        DrawSphere(tv, 0.08f, Fade(RED, 0.85f));
        DrawLine3D(dp, tv, Fade(RED, 0.45f));
        DrawLine3D((Vector3){tv.x, 0, tv.z}, tv, Fade(RED, 0.3f));

        if (qr_task_uses_waypoint(env->task_mode) && !env->route_direct_obs) {
            float obs_target[3];
            qr_nav_observation_target(env, q, obs_target);
            float ox = obs_target[0] - env->target[0];
            float oy = obs_target[1] - env->target[1];
            if (ox*ox + oy*oy > 0.0025f) {
                Vector3 ov = {obs_target[0], obs_target[2], -obs_target[1]};
                DrawSphereWires(ov, 0.22f, 12, 12, Fade(ORANGE, 0.70f));
                DrawSphere(ov, 0.06f, Fade(ORANGE, 0.90f));
                DrawLine3D(dp, ov, Fade(ORANGE, 0.55f));
                DrawLine3D((Vector3){ov.x, 0, ov.z}, ov, Fade(ORANGE, 0.35f));
            }
        }
    }

    EndMode3D();

    float dx = env->target[0] - q->pos[0];
    float dy = env->target[1] - q->pos[1];
    float dz = env->target[2] - q->pos[2];
    if (env->task_mode == QR_TASK_INDOOR_EXPLORE) {
        QrExplorePlannerStats planner_stats;
        qr_explore_planner_stats(env, &planner_stats);
        DrawText(TextFormat("step %d/%d  cov=%.2f  cells=%d/%d  rew=%.3f",
            env->tick, env->max_steps, qr_explore_coverage(env),
            env->explored_cells, env->explore_total_cells, env->rewards[0]),
            10, 10, 16, (Color){0, 187, 187, 255});
        DrawText(TextFormat("planner frontier=%d/%d stale=%d marks=%d target=%s age=%d cluster=%d blocked=%d",
            planner_stats.frontier_count,
            planner_stats.frontier_cluster_count,
            planner_stats.stale_frontier_count,
            env->explore_onboard_stale_marks,
            planner_stats.target_exists ? "yes" : "no",
            planner_stats.target_age,
            planner_stats.target_cluster_cells,
            planner_stats.blocked_cells),
            10, 32, 16, RAYWHITE);
        DrawText(TextFormat("view candidates=%d viable=%d used=%d miss=%d best=%.2f clear=%.2fm depth=%d wp_depth=%d",
            planner_stats.viewpoint_candidate_count,
            planner_stats.viewpoint_viable_count,
            planner_stats.viewpoint_used,
            planner_stats.viewpoint_miss,
            planner_stats.viewpoint_best_score,
            planner_stats.viewpoint_best_clearance,
            planner_stats.viewpoint_best_depth,
            planner_stats.waypoint_depth),
            10, 54, 16, RAYWHITE);
        DrawText(TextFormat("commit idx=%d used=%d age=%d target_sw=%d obj_sw=%d unprod=%d drops=%d",
            planner_stats.committed_viewpoint_idx,
            planner_stats.committed_viewpoint_used,
            planner_stats.committed_viewpoint_age,
            env->explore_onboard_target_switch_count,
            env->explore_onboard_objective_switch_count,
            env->explore_onboard_unproductive_objective_switch_count,
            env->explore_onboard_committed_drop_count),
            10, 76, 16, RAYWHITE);
        if (env->explore_onboard_region_tracker) {
            DrawText(TextFormat("region_tracker regions=%d active=%d age=%d no_progress=%d stale=%d switches=%d",
                planner_stats.tracker_region_count,
                planner_stats.tracker_active_id,
                planner_stats.tracker_active_age,
                planner_stats.tracker_since_progress,
                planner_stats.tracker_stale,
                env->explore_onboard_tracker_switch_count),
                10, 98, 16, RAYWHITE);
        }
    } else {
        DrawText(TextFormat("step %d/%d  dist=%.2f  cov=%.2f  rew=%.3f",
            env->tick, env->max_steps,
            sqrtf(dx*dx+dy*dy+dz*dz), qr_explore_coverage(env), env->rewards[0]),
            10, 10, 16, (Color){0, 187, 187, 255});
    }
    if (env->last_episode_flash > 0) {
        int last_y = env->task_mode == QR_TASK_INDOOR_EXPLORE
            ? (env->explore_onboard_region_tracker ? 120 : 98)
            : 32;
        DrawText(TextFormat("last: %s  goals=%d/%d  dist=%.2f  return=%.1f",
            qr_status_name(env->last_episode_status),
            env->last_episode_goals, env->last_episode_goal_count,
            env->last_episode_dist, env->last_episode_return),
            10, last_y, 16, qr_status_color(env->last_episode_status));
        env->last_episode_flash--;
    }
    DrawFPS(env->client->window_w - 90, 10);
    float action_xy = env->has_prev_action
        ? sqrtf(env->prev_action[0]*env->prev_action[0] +
                env->prev_action[1]*env->prev_action[1])
        : 0.0f;
    DrawRectangle(8, env->client->window_h - 76, 930, 68, Fade(BLACK, 0.55f));
    DrawText("Camera: A/D rotate  W/S pitch  wheel/Q/E zoom  R reset",
        14, env->client->window_h - 70, 16, RAYWHITE);
    DrawText(TextFormat("Speed: -/+ render fps %.0f  |  Shift = faster camera controls  |  Esc quit",
        env->client->render_fps),
        14, env->client->window_h - 48, 16, RAYWHITE);
    DrawText(TextFormat("Debug: clearance %.2fm  prev action xy %.2f yaw %.2f",
        env->min_clearance, action_xy, env->has_prev_action ? env->prev_action[3] : 0.0f),
        14, env->client->window_h - 26, 16, RAYWHITE);

    EndDrawing();
}

static const char* qr_status_name(int status) {
    switch (status) {
        case 10: return "success";
        case 11: return "timeout";
        case QR_STATUS_DRONE_OOB: return "target oob";
        case QR_CRASH_GROUND: return "ground crash";
        case QR_CRASH_WALL: return "wall crash";
        case QR_CRASH_OBSTACLE: return "obstacle crash";
        case QR_CRASH_DIVERGED: return "diverged";
        default: return "none";
    }
}

static Color qr_status_color(int status) {
    if (status == 10)
        return (Color){90, 220, 120, 255};
    if (status == 11)
        return (Color){240, 210, 80, 255};
    if (status == QR_STATUS_DRONE_OOB)
        return (Color){250, 150, 80, 255};
    if (status == QR_CRASH_OBSTACLE)
        return ORANGE;
    if (status == QR_CRASH_WALL)
        return SKYBLUE;
    if (status == QR_CRASH_GROUND)
        return (Color){230, 90, 90, 255};
    return LIGHTGRAY;
}

static inline void c_close(QuadrotorEnv* env) {
    if (env->client) {
        if (IsWindowReady()) CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

static void qr_draw_drone(const Quadrotor* q) {
    Vector3 pos = {q->pos[0], q->pos[2], -q->pos[1]};
    float L = q->arm_length;
    float arm_body[4][3] = {{+L,+L,0},{+L,-L,0},{-L,-L,0},{-L,+L,0}};
    Color mc[4] = {RED, GREEN, BLUE, YELLOW};

    DrawSphere(pos, 0.02f, DARKGRAY);
    for (int i = 0; i < 4; i++) {
        float w[3];
        qr_body_to_world(q->quat, arm_body[i], w);
        Vector3 tip = {q->pos[0]+w[0], q->pos[2]+w[2], -(q->pos[1]+w[1])};
        DrawLine3D(pos, tip, WHITE);
        DrawSphere(tip, 0.015f, mc[i]);
    }
    float bz[3]={0,0,0.1f}, fw[3]={0.08f,0,0}, wz[3], wf[3];
    qr_body_to_world(q->quat, bz, wz);
    qr_body_to_world(q->quat, fw, wf);
    DrawLine3D(pos, (Vector3){q->pos[0]+wz[0], q->pos[2]+wz[2], -(q->pos[1]+wz[1])}, SKYBLUE);
    DrawLine3D(pos, (Vector3){q->pos[0]+wf[0], q->pos[2]+wf[2], -(q->pos[1]+wf[1])}, RED);
}

static void qr_draw_rays(const QuadrotorEnv* env) {
    if (env->num_rays <= 0 || env->ray_length <= 0.0f)
        return;

    const Quadrotor* q = &env->phys;
    Vector3 start = {q->pos[0], q->pos[2], -q->pos[1]};
    for (int i = 0; i < env->num_rays; i++) {
        float angle = 2.0f * QR_PI * (float)i / (float)env->num_rays;
        float dir_body[3] = {cosf(angle), sinf(angle), 0.0f};
        float dir_world[3];
        qr_body_to_world(q->quat, dir_body, dir_world);
        dir_world[2] = 0.0f;
        float dn = sqrtf(dir_world[0]*dir_world[0] + dir_world[1]*dir_world[1]);
        if (dn <= 1e-6f)
            continue;
        dir_world[0] /= dn;
        dir_world[1] /= dn;

        float dist = qr_room_ray_obs(env, q, i) * env->ray_length;
        Vector3 end = {
            q->pos[0] + dir_world[0] * dist,
            q->pos[2],
            -(q->pos[1] + dir_world[1] * dist)
        };
        Color color = Fade(SKYBLUE, 0.25f);
        if (dist < 0.5f)
            color = Fade(RED, 0.75f);
        else if (dist < 1.0f)
            color = Fade(YELLOW, 0.65f);
        DrawLine3D(start, end, color);
    }
}

static void qr_draw_room(const QuadrotorEnv* env) {
    if (env->room_size <= 0.0f)
        return;

    float r = env->room_size;
    float h = env->ceiling_height > 0.0f ? env->ceiling_height : QR_CEILING_HEIGHT;
    float t = 0.03f;
    Color edge = (Color){110, 220, 255, 190};
    Color ceiling = (Color){110, 220, 255, 110};

    DrawCubeWires((Vector3){ r, 0.5f*h, 0.0f}, t, h, 2.0f*r, edge);
    DrawCubeWires((Vector3){-r, 0.5f*h, 0.0f}, t, h, 2.0f*r, edge);
    DrawCubeWires((Vector3){0.0f, 0.5f*h,  r}, 2.0f*r, h, t, edge);
    DrawCubeWires((Vector3){0.0f, 0.5f*h, -r}, 2.0f*r, h, t, edge);

    DrawLine3D((Vector3){-r, h, -r}, (Vector3){ r, h, -r}, ceiling);
    DrawLine3D((Vector3){ r, h, -r}, (Vector3){ r, h,  r}, ceiling);
    DrawLine3D((Vector3){ r, h,  r}, (Vector3){-r, h,  r}, ceiling);
    DrawLine3D((Vector3){-r, h,  r}, (Vector3){-r, h, -r}, ceiling);
}

static void qr_draw_obstacles(const QuadrotorEnv* env) {
    for (int i = 0; i < env->num_obstacles; i++) {
        const float* box = env->obstacles[i];
        float cx = 0.5f * (box[0] + box[1]);
        float cy = 0.5f * (box[2] + box[3]);
        float cz = 0.5f * (box[4] + box[5]);
        float sx = box[1] - box[0];
        float sy = box[3] - box[2];
        float sz = box[5] - box[4];
        Vector3 center = {cx, cz, -cy};
        DrawCube(center, sx, sz, sy, Fade(ORANGE, 0.45f));
        DrawCubeWires(center, sx, sz, sy, ORANGE);
    }
}

static void qr_draw_explored(const QuadrotorEnv* env) {
    if (env->task_mode != QR_TASK_INDOOR_EXPLORE || env->explore_grid_dim <= 0)
        return;

    float size = env->explore_cell_size * 0.95f;
    for (int iy = 0; iy < env->explore_grid_dim; iy++) {
        for (int ix = 0; ix < env->explore_grid_dim; ix++) {
            int idx = iy * QR_EXPLORE_MAX_GRID + ix;
            if (!env->explored[idx])
                continue;
            float x = qr_explore_cell_center(env, ix);
            float y = qr_explore_cell_center(env, iy);
            DrawCube((Vector3){x, 0.01f, -y}, size, 0.01f, size,
                Fade(SKYBLUE, 0.35f));
        }
    }
}

static void qr_draw_exploration_planner(const QuadrotorEnv* env) {
    if (env->task_mode != QR_TASK_INDOOR_EXPLORE || env->explore_grid_dim <= 0)
        return;

    int dim = env->explore_grid_dim;
    float size = env->explore_cell_size * 0.82f;
    int current_idx = qr_explore_cell_index(
        env, env->phys.pos[0], env->phys.pos[1]);

    for (int iy = 0; iy < dim; iy++) {
        for (int ix = 0; ix < dim; ix++) {
            int idx = iy * QR_EXPLORE_MAX_GRID + ix;
            float x = qr_explore_cell_center(env, ix);
            float y = qr_explore_cell_center(env, iy);
            Vector3 center = {x, 0.025f, -y};

            if (env->explore_blocked[idx]) {
                DrawCube(center, size, 0.02f, size, Fade(RED, 0.40f));
                continue;
            }

            if (env->explore_frontier_stale[idx])
                DrawCube(center, size, 0.018f, size, Fade(PURPLE, 0.50f));

            if ((idx == current_idx || env->explored[idx]) &&
                    qr_explore_onboard_frontier_unknown_count(
                        env, dim, ix, iy, NULL) > 0) {
                DrawCube(center, size, 0.014f, size, Fade(YELLOW, 0.42f));
                DrawCubeWires(center, size, 0.014f, size, Fade(YELLOW, 0.80f));
            }

            if (idx == current_idx) {
                DrawCubeWires((Vector3){x, 0.045f, -y}, size, 0.04f, size, GREEN);
            }
        }
    }

    int target_idx = env->explore_onboard_target_idx;
    const int max_cells = QR_EXPLORE_MAX_GRID * QR_EXPLORE_MAX_GRID;
    if (target_idx >= 0 && target_idx < max_cells) {
        int tx = target_idx % QR_EXPLORE_MAX_GRID;
        int ty = target_idx / QR_EXPLORE_MAX_GRID;
        if (tx >= 0 && tx < dim && ty >= 0 && ty < dim &&
                !env->explore_blocked[target_idx]) {
            float x = qr_explore_cell_center(env, tx);
            float y = qr_explore_cell_center(env, ty);
            Vector3 floor = {x, 0.04f, -y};
            Vector3 top = {x, 0.55f, -y};
            DrawLine3D(floor, top, Fade(RED, 0.90f));
            DrawSphere(top, 0.075f, Fade(RED, 0.90f));
            DrawCubeWires(floor, size * 1.18f, 0.04f, size * 1.18f, RED);
        }
    }

    int frontier_idx = env->explore_onboard_target_frontier_idx;
    if (frontier_idx >= 0 && frontier_idx < max_cells) {
        int fx = frontier_idx % QR_EXPLORE_MAX_GRID;
        int fy = frontier_idx / QR_EXPLORE_MAX_GRID;
        if (fx >= 0 && fx < dim && fy >= 0 && fy < dim) {
            float x = qr_explore_cell_center(env, fx);
            float y = qr_explore_cell_center(env, fy);
            Vector3 center = {x, 0.075f, -y};
            DrawCubeWires(center, size * 1.30f, 0.07f, size * 1.30f, ORANGE);
        }
    }

    int waypoint_idx = env->explore_onboard_waypoint_idx;
    if (waypoint_idx >= 0 && waypoint_idx < max_cells) {
        int wx = waypoint_idx % QR_EXPLORE_MAX_GRID;
        int wy = waypoint_idx / QR_EXPLORE_MAX_GRID;
        if (wx >= 0 && wx < dim && wy >= 0 && wy < dim) {
            float x = qr_explore_cell_center(env, wx);
            float y = qr_explore_cell_center(env, wy);
            Vector3 floor = {x, 0.065f, -y};
            Vector3 top = {x, 0.40f, -y};
            DrawLine3D(floor, top, Fade(BLUE, 0.85f));
            DrawSphere(top, 0.055f, Fade(BLUE, 0.90f));
            DrawCubeWires(floor, size * 1.08f, 0.04f, size * 1.08f, BLUE);
        }
    }

    int viewpoint_idx = env->explore_onboard_viewpoint_best_idx;
    if (viewpoint_idx >= 0 && viewpoint_idx < max_cells) {
        int vx = viewpoint_idx % QR_EXPLORE_MAX_GRID;
        int vy = viewpoint_idx / QR_EXPLORE_MAX_GRID;
        if (vx >= 0 && vx < dim && vy >= 0 && vy < dim) {
            float x = qr_explore_cell_center(env, vx);
            float y = qr_explore_cell_center(env, vy);
            Vector3 center = {x, 0.105f, -y};
            Color color = env->explore_onboard_viewpoint_last_used ? GREEN : LIME;
            DrawCubeWires(center, size * 1.45f, 0.10f, size * 1.45f, color);
            DrawSphere((Vector3){x, 0.62f, -y}, 0.05f, Fade(color, 0.90f));
        }
    }

    int committed_idx = env->explore_onboard_committed_viewpoint_idx;
    if (committed_idx >= 0 && committed_idx < max_cells) {
        int cx = committed_idx % QR_EXPLORE_MAX_GRID;
        int cy = committed_idx / QR_EXPLORE_MAX_GRID;
        if (cx >= 0 && cx < dim && cy >= 0 && cy < dim) {
            float x = qr_explore_cell_center(env, cx);
            float y = qr_explore_cell_center(env, cy);
            Vector3 center = {x, 0.13f, -y};
            DrawCubeWires(center, size * 1.70f, 0.13f, size * 1.70f, SKYBLUE);
            DrawSphere((Vector3){x, 0.72f, -y}, 0.055f, Fade(SKYBLUE, 0.95f));
        }
    }
}

static void qr_draw_ground(void) {
    DrawPlane((Vector3){0,0,0}, (Vector2){40,40}, (Color){40,50,40,255});
    DrawGrid(40, 1.0f);
}

// PufferLib 5.0 API

// Upper bound of the configurable obs layout. puf_init asserts that the
// [env] config produces exactly this size.
#define OBS_SIZE (QR_OBS_SIZE + 24 + QR_ROUTE_WALL_OBS_SIZE + QR_ACTION_HISTORY_OBS_SIZE + QR_EXPLORE_PATCH_OBS_SIZE)
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}

static inline int qget_int(Dict* kwargs, const char* key, int fallback) {
    DictItem* item = dict_find(kwargs, key);
    return item == NULL ? fallback : (int)item->value;
}

static inline float qget_float(Dict* kwargs, const char* key, float fallback) {
    DictItem* item = dict_find(kwargs, key);
    return item == NULL ? fallback : (float)item->value;
}

static inline int configured_obs_size(const QuadrotorEnv* env) {
    int obs_size = QR_OBS_SIZE + env->num_rays;
    if (env->route_wall_obs)
        obs_size += QR_ROUTE_WALL_OBS_SIZE;
    if (env->route_extra_obs)
        obs_size += QR_ROUTE_EXTRA_OBS_SIZE;
    if (env->route_perception_obs == 1)
        obs_size += QR_ROUTE_PERCEPTION_OBS_SIZE;
    else if (env->route_perception_obs == 2)
        obs_size += QR_ROUTE_COMPACT_PERCEPTION_OBS_SIZE;
    else if (env->route_perception_obs == 3)
        obs_size += QR_ROUTE_AFFORDANCE_OBS_SIZE;
    if (env->action_history_obs)
        obs_size += QR_ACTION_HISTORY_OBS_SIZE;
    obs_size += QR_EXPLORE_PATCH_OBS_SIZE;
    return obs_size;
}

static inline void qr_bind_agent(Env* env) {
    Agent* a = &env->agents[0];
    env->observations = a->observations;
    env->actions = a->actions;
    env->rewards = a->rewards;
    env->terminals = a->terminals;
}

void puf_init(Env* env, Dict* kwargs) {
    // pufferl seeds env->rng with the env index before calling puf_init.
    int seed_offset = (int)env->rng;

    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    env->max_steps = qget_int(kwargs, "max_steps", QR_MAX_STEPS);
    env->action_repeat = qget_int(kwargs, "action_repeat", 1);
    env->max_rate = qget_float(kwargs, "max_rate", QR_MAX_RATE);
    env->max_vel = qget_float(kwargs, "max_vel", QR_MAX_VEL);
    env->max_z_vel = qget_float(kwargs, "max_z_vel", QR_MAX_Z_VEL);
    env->dynamics_profile = qget_int(kwargs, "dynamics_profile", QR_DYNAMICS_PROFILE_CUSTOM);
    env->physics_dt = qget_float(kwargs, "physics_dt", QR_DT);
    env->physics_substeps = qget_int(kwargs, "physics_substeps", 1);
    env->vel_ctrl_kp_vel = qget_float(kwargs, "vel_ctrl_kp_vel", 0.0f);
    env->vel_ctrl_ki_z = qget_float(kwargs, "vel_ctrl_ki_z", 0.0f);
    env->vel_ctrl_int_limit_z = qget_float(kwargs, "vel_ctrl_int_limit_z", 0.0f);
    env->vel_ctrl_max_z_acc = qget_float(kwargs, "vel_ctrl_max_z_acc", 0.0f);
    env->spawn_radius = qget_float(kwargs, "spawn_radius", QR_SPAWN_RADIUS);
    env->spawn_radius_min = qget_float(kwargs, "spawn_radius_min", QR_SPAWN_RADIUS);
    env->spawn_height = qget_float(kwargs, "spawn_height", QR_SPAWN_HEIGHT);
    env->init_vel_radius = qget_float(kwargs, "init_vel_radius", QR_INIT_VEL_RADIUS);
    env->init_omega_radius = qget_float(kwargs, "init_omega_radius", QR_INIT_OMEGA_RADIUS);
    env->room_size = qget_float(kwargs, "room_size", QR_ROOM_SIZE);
    env->ceiling_height = qget_float(kwargs, "ceiling_height", QR_CEILING_HEIGHT);
    env->ray_length = qget_float(kwargs, "ray_length", QR_RAY_LENGTH);
    env->num_rays = qget_int(kwargs, "num_rays", 24);
    env->camera_depth_obs = qget_int(kwargs, "camera_depth_obs", 0);
    env->camera_depth_width = qget_int(kwargs, "camera_depth_width", QR_DEPTH_CAMERA_DEFAULT_WIDTH);
    env->camera_depth_height = qget_int(kwargs, "camera_depth_height", QR_DEPTH_CAMERA_DEFAULT_HEIGHT);
    env->camera_depth_fov_x_deg = qget_float(kwargs, "camera_depth_fov_x_deg", 90.0f);
    env->camera_depth_fov_y_deg = qget_float(kwargs, "camera_depth_fov_y_deg", 60.0f);
    env->hover_radius = qget_float(kwargs, "hover_radius", QR_HOVER_RADIUS);
    env->drone_alpha_dist = qget_float(kwargs, "drone_alpha_dist", QR_DRONE_ALPHA_DIST);
    env->drone_alpha_hover = qget_float(kwargs, "drone_alpha_hover", QR_DRONE_ALPHA_HOVER);
    env->drone_alpha_omega = qget_float(kwargs, "drone_alpha_omega", QR_DRONE_ALPHA_OMEGA);
    env->drone_alpha_shaping = qget_float(kwargs, "drone_alpha_shaping", QR_DRONE_ALPHA_SHAPING);
    env->drone_hover_target_dist = qget_float(kwargs, "drone_hover_target_dist", QR_DRONE_HOVER_TARGET_DIST);
    env->drone_hover_dist = qget_float(kwargs, "drone_hover_dist", QR_DRONE_HOVER_DIST);
    env->drone_hover_omega = qget_float(kwargs, "drone_hover_omega", QR_DRONE_HOVER_OMEGA);
    env->drone_hover_vel = qget_float(kwargs, "drone_hover_vel", QR_DRONE_HOVER_VEL);
    env->task_mode = qget_int(kwargs, "task_mode", QR_TASK_INDOOR_ROUTE);
    env->action_mode = qget_int(kwargs, "action_mode", QR_ACTION_MOTOR);
    env->obstacle_mode = qget_int(kwargs, "obstacle_mode", QR_OBSTACLES_LAYOUT_MIXED);
    env->obstacle_block_prob = qget_float(kwargs, "obstacle_block_prob", 0.5f);
    env->mass_scale = qget_float(kwargs, "mass_scale", 1.0f);
    env->inertia_scale = qget_float(kwargs, "inertia_scale", 0.0f);
    env->thrust_scale = qget_float(kwargs, "thrust_scale", 1.0f);
    env->torque_scale = qget_float(kwargs, "torque_scale", 0.0f);
    env->drag_scale = qget_float(kwargs, "drag_scale", 1.0f);
    env->motor_tau_scale = qget_float(kwargs, "motor_tau_scale", 1.0f);
    env->mass_scale_min = qget_float(kwargs, "mass_scale_min", 0.0f);
    env->mass_scale_max = qget_float(kwargs, "mass_scale_max", 0.0f);
    env->inertia_scale_min = qget_float(kwargs, "inertia_scale_min", 0.0f);
    env->inertia_scale_max = qget_float(kwargs, "inertia_scale_max", 0.0f);
    env->thrust_scale_min = qget_float(kwargs, "thrust_scale_min", 0.0f);
    env->thrust_scale_max = qget_float(kwargs, "thrust_scale_max", 0.0f);
    env->torque_scale_min = qget_float(kwargs, "torque_scale_min", 0.0f);
    env->torque_scale_max = qget_float(kwargs, "torque_scale_max", 0.0f);
    env->drag_scale_min = qget_float(kwargs, "drag_scale_min", 0.0f);
    env->drag_scale_max = qget_float(kwargs, "drag_scale_max", 0.0f);
    env->motor_tau_scale_min = qget_float(kwargs, "motor_tau_scale_min", 0.0f);
    env->motor_tau_scale_max = qget_float(kwargs, "motor_tau_scale_max", 0.0f);
    env->explore_cell_size = qget_float(kwargs, "explore_cell_size", QR_EXPLORE_CELL_SIZE);
    env->explore_goal_fraction = qget_float(kwargs, "explore_goal_fraction", QR_EXPLORE_GOAL_FRACTION);
    env->explore_frontier_progress_scale = qget_float(kwargs,
        "explore_frontier_progress_scale", QR_EXPLORE_FRONTIER_PROGRESS_SCALE);
    env->explore_memory_obs = qget_int(kwargs, "explore_memory_obs", 0);
    env->explore_frontier_target_mode = qget_int(
        kwargs,
        "explore_frontier_target_mode",
        QR_EXPLORE_FRONTIER_TARGET_NEAREST);
    env->explore_reachable_coverage = qget_int(kwargs, "explore_reachable_coverage", 0);
    env->explore_onboard_commit_steps = qget_int(kwargs, "explore_onboard_commit_steps", 0);
    env->explore_onboard_path_lookahead = qget_int(kwargs, "explore_onboard_path_lookahead", 1);
    env->explore_onboard_adaptive_lookahead = qget_int(kwargs, "explore_onboard_adaptive_lookahead", 0);
    env->explore_onboard_adaptive_lookahead_min = qget_int(kwargs, "explore_onboard_adaptive_lookahead_min", 2);
    env->explore_onboard_adaptive_lookahead_max = qget_int(kwargs, "explore_onboard_adaptive_lookahead_max", env->explore_onboard_path_lookahead);
    env->explore_onboard_adaptive_clearance = qget_float(kwargs, "explore_onboard_adaptive_clearance", 0.55f);
    env->explore_onboard_adaptive_turn_dot = qget_float(kwargs, "explore_onboard_adaptive_turn_dot", 0.55f);
    env->explore_onboard_commit_min_clearance = qget_float(kwargs, "explore_onboard_commit_min_clearance", 0.35f);
    env->explore_onboard_stall_viewpoint_ticks = qget_int(kwargs, "explore_onboard_stall_viewpoint_ticks", 0);
    env->explore_onboard_stall_viewpoint_max_depth = qget_int(kwargs, "explore_onboard_stall_viewpoint_max_depth", 1);
    env->explore_onboard_approach_viewpoints = qget_int(kwargs, "explore_onboard_approach_viewpoints", 0);
    env->explore_onboard_approach_max_depth = qget_int(kwargs, "explore_onboard_approach_max_depth", 1);
    env->explore_onboard_emit_objective_target = qget_int(kwargs, "explore_onboard_emit_objective_target", 0);
    env->explore_onboard_objective_obs = qget_int(kwargs, "explore_onboard_objective_obs", 0);
    env->explore_onboard_region_commit = qget_int(kwargs, "explore_onboard_region_commit", 0);
    env->explore_onboard_region_commit_stall_ticks = qget_int(kwargs, "explore_onboard_region_commit_stall_ticks", 0);
    env->explore_onboard_persistent_regions = qget_int(kwargs, "explore_onboard_persistent_regions", 0);
    env->explore_onboard_region_tracker = qget_int(kwargs, "explore_onboard_region_tracker", 0);
    env->explore_onboard_clearance_passable = qget_int(kwargs, "explore_onboard_clearance_passable", 0);
    env->explore_onboard_passable_radius = qget_float(kwargs, "explore_onboard_passable_radius", 0.0f);
    env->explore_onboard_passage_quality = qget_int(kwargs, "explore_onboard_passage_quality", 0);
    env->explore_onboard_passage_weight = qget_float(kwargs, "explore_onboard_passage_weight", 0.35f);
    env->explore_onboard_clearance_cone_deg = qget_float(kwargs, "explore_onboard_clearance_cone_deg", 30.0f);
    env->explore_onboard_budget_frontier = qget_int(kwargs, "explore_onboard_budget_frontier", 0);
    env->explore_onboard_budget_gain_weight = qget_float(kwargs, "explore_onboard_budget_gain_weight", 1.0f);
    env->explore_onboard_budget_cluster_weight = qget_float(kwargs, "explore_onboard_budget_cluster_weight", 0.35f);
    env->explore_onboard_budget_cell_weight = qget_float(kwargs, "explore_onboard_budget_cell_weight", 0.25f);
    env->explore_onboard_budget_depth_bias = qget_float(kwargs, "explore_onboard_budget_depth_bias", 6.0f);
    env->explore_onboard_budget_depth_weight = qget_float(kwargs, "explore_onboard_budget_depth_weight", 1.0f);
    env->explore_onboard_optimistic_unknown = qget_int(kwargs, "explore_onboard_optimistic_unknown", 0);
    env->explore_onboard_frontier_graph = qget_int(kwargs, "explore_onboard_frontier_graph", 0);
    env->explore_onboard_graph_candidate_obs = qget_int(kwargs, "explore_onboard_graph_candidate_obs", 0);
    env->explore_onboard_graph_gain_weight = qget_float(kwargs, "explore_onboard_graph_gain_weight", 0.32f);
    env->explore_onboard_graph_cluster_weight = qget_float(kwargs, "explore_onboard_graph_cluster_weight", 0.12f);
    env->explore_onboard_graph_depth_weight = qget_float(kwargs, "explore_onboard_graph_depth_weight", 0.045f);
    env->explore_onboard_graph_active_bonus = qget_float(kwargs, "explore_onboard_graph_active_bonus", 0.75f);
    env->explore_onboard_graph_stale_penalty = qget_float(kwargs, "explore_onboard_graph_stale_penalty", 0.020f);
    env->explore_onboard_graph_failure_penalty = qget_float(kwargs, "explore_onboard_graph_failure_penalty", 0.70f);
    env->explore_onboard_graph_attempt_penalty = qget_float(kwargs, "explore_onboard_graph_attempt_penalty", 0.12f);
    env->explore_onboard_graph_gain_cap = qget_int(kwargs, "explore_onboard_graph_gain_cap", 192);
    env->route_goal_radius = qget_float(kwargs, "route_goal_radius", QR_NAV_GOAL_RADIUS);
    env->route_goal_count = qget_int(kwargs, "route_goal_count", QR_ROUTE_GOAL_COUNT);
    env->route_goal_count_min = qget_int(kwargs, "route_goal_count_min", 0);
    env->route_step_cost = qget_float(kwargs, "route_step_cost", QR_ROUTE_STEP_COST);
    env->route_proximity_reward = qget_float(kwargs, "route_proximity_reward", QR_ROUTE_PROXIMITY_REWARD);
    env->route_dist_scale = qget_float(kwargs, "route_dist_scale", QR_ROUTE_DIST_SCALE);
    env->route_progress_scale = qget_float(kwargs, "route_progress_scale", QR_ROUTE_PROGRESS_SCALE);
    env->route_crash_reward = qget_float(kwargs, "route_crash_reward", QR_ROUTE_CRASH_REWARD);
    env->route_crash_goal_scale = qget_float(kwargs, "route_crash_goal_scale", QR_ROUTE_CRASH_GOAL_SCALE);
    env->route_crash_dist_scale = qget_float(kwargs, "route_crash_dist_scale", QR_ROUTE_CRASH_DIST_SCALE);
    env->route_clearance_radius = qget_float(kwargs, "route_clearance_radius", QR_ROUTE_CLEARANCE_RADIUS);
    env->route_clearance_scale = qget_float(kwargs, "route_clearance_scale", QR_ROUTE_CLEARANCE_SCALE);
    env->route_clearance_speed_scale = qget_float(kwargs, "route_clearance_speed_scale", QR_ROUTE_CLEARANCE_SPEED_SCALE);
    env->route_clearance_speed_min = qget_float(kwargs, "route_clearance_speed_min", QR_ROUTE_CLEARANCE_SPEED_MIN);
    env->route_clearance_speed_full = qget_float(kwargs, "route_clearance_speed_full", QR_ROUTE_CLEARANCE_SPEED_FULL);
    env->route_reference_clearance_radius = qget_float(kwargs, "route_reference_clearance_radius", QR_ROUTE_CLEARANCE_RADIUS);
    env->route_reference_clearance_scale = qget_float(kwargs, "route_reference_clearance_scale", QR_ROUTE_REFERENCE_CLEARANCE_SCALE);
    env->route_path_margin = qget_float(kwargs, "route_path_margin", QR_NAV_PATH_MARGIN);
    env->route_vel_scale = qget_float(kwargs, "route_vel_scale", QR_ROUTE_VEL_SCALE);
    env->route_arrival_radius = qget_float(kwargs, "route_arrival_radius", QR_ROUTE_ARRIVAL_RADIUS);
    env->route_arrival_bonus = qget_float(kwargs, "route_arrival_bonus", QR_ROUTE_ARRIVAL_BONUS);
    env->route_arrival_vel_scale = qget_float(kwargs, "route_arrival_vel_scale", QR_ROUTE_ARRIVAL_VEL_SCALE);
    env->route_arrival_toward_scale = qget_float(kwargs, "route_arrival_toward_scale", QR_ROUTE_ARRIVAL_TOWARD_SCALE);
    env->route_target_wall_margin = qget_float(kwargs, "route_target_wall_margin", QR_ROUTE_TARGET_WALL_MARGIN);
    env->route_goal_speed = qget_float(kwargs, "route_goal_speed", QR_ROUTE_GOAL_SPEED);
    env->route_goal_bonus = qget_float(kwargs, "route_goal_bonus", QR_ROUTE_GOAL_BONUS_REF);
    env->route_reach_bonus = qget_float(kwargs, "route_reach_bonus", QR_ROUTE_REACH_BONUS);
    env->route_target_z_min = qget_float(kwargs, "route_target_z_min", 0.0f);
    env->route_target_z_max = qget_float(kwargs, "route_target_z_max", 0.0f);
    env->route_reward_mode = qget_int(kwargs, "route_reward_mode", QR_ROUTE_REWARD_REFERENCE);
    env->route_target_mode = qget_int(kwargs, "route_target_mode", QR_ROUTE_TARGET_RANDOM);
    env->route_blocked_target_prob = qget_float(kwargs, "route_blocked_target_prob", 1.0f);
    env->route_hardstate_probe = qget_int(kwargs, "route_hardstate_probe", 0);
    env->route_hardstate_start_x = qget_float(kwargs, "route_hardstate_start_x", 0.0f);
    env->route_hardstate_start_y = qget_float(kwargs, "route_hardstate_start_y", 0.0f);
    env->route_hardstate_start_z = qget_float(kwargs, "route_hardstate_start_z", QR_SPAWN_HEIGHT);
    env->route_hardstate_target_x = qget_float(kwargs, "route_hardstate_target_x", 0.0f);
    env->route_hardstate_target_y = qget_float(kwargs, "route_hardstate_target_y", 0.0f);
    env->route_hardstate_target_z = qget_float(kwargs, "route_hardstate_target_z", QR_SPAWN_HEIGHT);
    env->route_hardstate_yaw_deg = qget_float(kwargs, "route_hardstate_yaw_deg", 0.0f);
    env->route_hardstate_jitter_xy = qget_float(kwargs, "route_hardstate_jitter_xy", 0.0f);
    env->route_hardstate_jitter_z = qget_float(kwargs, "route_hardstate_jitter_z", 0.0f);
    env->route_walls = qget_int(kwargs, "route_walls", 1);
    env->route_direct_obs = qget_int(kwargs, "route_direct_obs", 1);
    env->route_privileged_obs = qget_int(kwargs, "route_privileged_obs", 0);
    env->route_extra_obs = qget_int(kwargs, "route_extra_obs", 0);
    env->route_wall_obs = qget_int(kwargs, "route_wall_obs", 0);
    env->route_perception_obs = qget_int(kwargs, "route_perception_obs", 0);
    env->route_target_obs_mode = qget_int(kwargs, "route_target_obs_mode", QR_ROUTE_TARGET_OBS_LEGACY);
    env->action_history_obs = qget_int(kwargs, "action_history_obs", 0);
    env->seed = qget_int(kwargs, "seed", 0) + 9973 * seed_offset;

    init(env);

    int obs_size = configured_obs_size(env);
    if (obs_size != OBS_SIZE) {
        fprintf(stderr, "quadrotor obs size mismatch: config gives %d, build expects %d\n",
            obs_size, OBS_SIZE);
        abort();
    }
}

void puf_reset(Env* env) {
    qr_bind_agent(env);
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    c_reset(env);
}

void puf_step(Env* env) {
    qr_bind_agent(env);
    c_step(env);
}

void puf_render(Env* env) {
    qr_bind_agent(env);
    c_render(env);
}

void puf_close(Env* env) {
    c_close(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "success", log->success);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "crashed", log->crashed);
    dict_set(out, "avg_dist", log->avg_dist);
    dict_set(out, "avg_vel", log->avg_vel);
    dict_set(out, "avg_tilt", log->avg_tilt);
    dict_set(out, "max_dist", log->max_dist);
    dict_set(out, "final_dist", log->final_dist);
    dict_set(out, "final_z_error", log->final_z_error);
    dict_set(out, "final_vel", log->final_vel);
    dict_set(out, "settle_time", log->settle_time);
    dict_set(out, "settled", log->settled);
    dict_set(out, "hover_rate", log->hover_rate);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "drone_oob", log->drone_oob);
    dict_set(out, "obstacle_hit", log->obstacle_hit);
    dict_set(out, "wall_hit", log->wall_hit);
    dict_set(out, "ground_hit", log->ground_hit);
    dict_set(out, "min_clearance", log->min_clearance);
    dict_set(out, "path_length", log->path_length);
    dict_set(out, "path_efficiency", log->path_efficiency);
    dict_set(out, "action_saturation", log->action_saturation);
    dict_set(out, "action_delta", log->action_delta);
    dict_set(out, "coverage", log->coverage);
    dict_set(out, "explored_cells", log->explored_cells);
    dict_set(out, "new_cell_rate", log->new_cell_rate);
    dict_set(out, "frontier_count", log->frontier_count);
    dict_set(out, "stale_frontier_count", log->stale_frontier_count);
    dict_set(out, "stale_marks", log->stale_marks);
    dict_set(out, "frontier_cluster_count", log->frontier_cluster_count);
    dict_set(out, "onboard_target_exists", log->onboard_target_exists);
    dict_set(out, "onboard_target_age", log->onboard_target_age);
    dict_set(out, "onboard_target_cluster_cells", log->onboard_target_cluster_cells);
    dict_set(out, "blocked_cells", log->blocked_cells);
    dict_set(out, "viewpoint_candidate_count", log->viewpoint_candidate_count);
    dict_set(out, "viewpoint_viable_count", log->viewpoint_viable_count);
    dict_set(out, "viewpoint_used_rate", log->viewpoint_used_rate);
    dict_set(out, "viewpoint_miss_rate", log->viewpoint_miss_rate);
    dict_set(out, "viewpoint_best_score", log->viewpoint_best_score);
    dict_set(out, "viewpoint_best_clearance", log->viewpoint_best_clearance);
    dict_set(out, "viewpoint_best_depth", log->viewpoint_best_depth);
    dict_set(out, "waypoint_depth", log->waypoint_depth);
    dict_set(out, "onboard_target_switch_rate", log->onboard_target_switch_rate);
    dict_set(out, "onboard_objective_switch_rate", log->onboard_objective_switch_rate);
    dict_set(out, "unproductive_objective_switch_rate", log->unproductive_objective_switch_rate);
    dict_set(out, "committed_waypoint_used_rate", log->committed_waypoint_used_rate);
    dict_set(out, "committed_waypoint_drop_rate", log->committed_waypoint_drop_rate);
    dict_set(out, "committed_waypoint_age", log->committed_waypoint_age);
    dict_set(out, "region_tracker_regions", log->region_tracker_regions);
    dict_set(out, "region_tracker_active_age", log->region_tracker_active_age);
    dict_set(out, "region_tracker_since_progress", log->region_tracker_since_progress);
    dict_set(out, "region_tracker_switch_rate", log->region_tracker_switch_rate);
    dict_set(out, "region_tracker_stale_rate", log->region_tracker_stale_rate);
    dict_set(out, "frontier_graph_nodes", log->frontier_graph_nodes);
    dict_set(out, "frontier_graph_unknown_cells", log->frontier_graph_unknown_cells);
    dict_set(out, "frontier_graph_best_unknown_cells", log->frontier_graph_best_unknown_cells);
    dict_set(out, "frontier_graph_best_depth", log->frontier_graph_best_depth);
    dict_set(out, "frontier_graph_best_score", log->frontier_graph_best_score);
    dict_set(out, "frontier_graph_active_rate", log->frontier_graph_active_rate);
    dict_set(out, "adaptive_lookahead", log->adaptive_lookahead);
    dict_set(out, "adaptive_lookahead_long_rate", log->adaptive_lookahead_long_rate);
    dict_set(out, "adaptive_lookahead_clearance_reject_rate", log->adaptive_lookahead_clearance_reject_rate);
    dict_set(out, "adaptive_lookahead_turn_reject_rate", log->adaptive_lookahead_turn_reject_rate);
    dict_set(out, "route_goals", log->route_goals);
    dict_set(out, "procedural_0_n", log->procedural_n[0]);
    dict_set(out, "procedural_0_success", log->procedural_success[0]);
    dict_set(out, "procedural_0_timeout", log->procedural_timeout[0]);
    dict_set(out, "procedural_0_obstacle_hit", log->procedural_obstacle_hit[0]);
    dict_set(out, "procedural_0_coverage", log->procedural_coverage[0]);
    dict_set(out, "procedural_1_n", log->procedural_n[1]);
    dict_set(out, "procedural_1_success", log->procedural_success[1]);
    dict_set(out, "procedural_1_timeout", log->procedural_timeout[1]);
    dict_set(out, "procedural_1_obstacle_hit", log->procedural_obstacle_hit[1]);
    dict_set(out, "procedural_1_coverage", log->procedural_coverage[1]);
    dict_set(out, "procedural_2_n", log->procedural_n[2]);
    dict_set(out, "procedural_2_success", log->procedural_success[2]);
    dict_set(out, "procedural_2_timeout", log->procedural_timeout[2]);
    dict_set(out, "procedural_2_obstacle_hit", log->procedural_obstacle_hit[2]);
    dict_set(out, "procedural_2_coverage", log->procedural_coverage[2]);
    dict_set(out, "procedural_3_n", log->procedural_n[3]);
    dict_set(out, "procedural_3_success", log->procedural_success[3]);
    dict_set(out, "procedural_3_timeout", log->procedural_timeout[3]);
    dict_set(out, "procedural_3_obstacle_hit", log->procedural_obstacle_hit[3]);
    dict_set(out, "procedural_3_coverage", log->procedural_coverage[3]);
}

#endif // QUADROTOR_ENV_H
