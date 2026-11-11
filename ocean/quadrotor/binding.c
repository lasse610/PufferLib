#include "quadrotor.h"
#include <stdio.h>

#define OBS_SIZE (QR_OBS_SIZE + 24 + QR_ROUTE_WALL_OBS_SIZE + QR_ACTION_HISTORY_OBS_SIZE + QR_EXPLORE_PATCH_OBS_SIZE)
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env QuadrotorEnv
#include "vecenv.h"

static inline int qget_int(Dict* kwargs, const char* key, int fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : (int)item->value;
}

static inline float qget_float(Dict* kwargs, const char* key, float fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
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

void my_init(Env* env, Dict* kwargs) {
    int seed_offset = (int)env->rng;

    env->num_agents = 1;
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

void my_log(Log* log, Dict* out) {
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
