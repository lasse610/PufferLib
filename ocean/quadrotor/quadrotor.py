import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.quadrotor import binding

BASE_OBSERVATION_SIZE = 26
WALL_OBSERVATION_SIZE = 6
ROUTE_EXTRA_OBSERVATION_SIZE = 8
ROUTE_PERCEPTION_OBSERVATION_SIZE = 12
ROUTE_COMPACT_PERCEPTION_OBSERVATION_SIZE = 8
ROUTE_AFFORDANCE_OBSERVATION_SIZE = 12

TASK_MODES = {
    'fixed_target': 0,
    'fixed_hover': 0,
    'recovery_hover': 0,
    'hold_hover': 1,
    'hold': 1,
    'indoor_nav': 2,
    'nav': 2,
    'indoor_explore': 3,
    'explore': 3,
    'exploration': 3,
    'indoor_route': 4,
    'route': 4,
    'waypoint_route': 4,
    'waypoints': 4,
    'drone_hover': 5,
    'drone_style_hover': 5,
    'puffer_drone_hover': 5,
}

ACTION_MODES = {
    'rate': 0,
    'body_rate': 0,
    'thrust_rate': 0,
    'velocity': 1,
    'vel': 1,
    'motor': 2,
    'motors': 2,
    'direct': 2,
    'direct_motor': 2,
    'motor_hover': 3,
    'hover_motor': 3,
    'hover_centered_motor': 3,
    'motor_ref_hover': 4,
    'reference_motor_hover': 4,
    'drone_motor_hover': 4,
}

OBSTACLE_MODES = {
    'none': 0,
    'open': 0,
    'center_block': 1,
    'block': 1,
    'doorway': 2,
    'corridor': 3,
    'maze': 4,
    'random_blocks': 5,
    'random': 5,
    'mixed': 6,
    'mixed_blocks': 6,
    'open_blocks': 6,
    'layout_mixed': 7,
    'mixed_layouts': 7,
    'indoor_layouts': 7,
    'fixed_layouts': 7,
    'procedural_indoor': 8,
    'procedural': 8,
    'rooms': 8,
    'procedural_indoor_p0': 9,
    'procedural_indoor_p1': 10,
    'procedural_indoor_p2': 11,
    'procedural_indoor_p3': 12,
    'procedural_indoor_hard': 13,
    'procedural_indoor_hard_p0': 14,
    'procedural_indoor_hard_p1': 15,
    'procedural_indoor_hard_p2': 16,
    'procedural_indoor_hard_p3': 17,
    'procedural_indoor_tight': 18,
    'procedural_indoor_tight_p0': 19,
    'procedural_indoor_tight_p1': 20,
    'procedural_indoor_tight_p2': 21,
    'procedural_indoor_tight_p3': 22,
    'procedural_house': 23,
    'house': 23,
    'procedural_house_p0': 24,
    'procedural_house_p1': 25,
    'procedural_house_p2': 26,
    'procedural_house_p3': 27,
    'procedural_house_multizone': 28,
    'house_multizone': 28,
    'procedural_house_multizone_p0': 29,
    'procedural_house_multizone_p1': 30,
    'procedural_house_multizone_p2': 31,
    'procedural_house_multizone_p3': 32,
    'gate_slalom': 33,
    'gates': 33,
    'slalom': 33,
    'creative_obstacles': 33,
    'gate_slalom_p0': 34,
    'gate_slalom_p1': 35,
    'gate_slalom_p2': 36,
    'gate_slalom_p3': 37,
    'gate_slalom_easy_mix': 38,
    'gate_slalom_curriculum': 38,
    'easy_gates': 38,
    'gate_slalom_p1_p0_mix': 39,
    'p1_p0_mix': 39,
    'p1_gate_intro': 39,
    'agile_geometry': 40,
    'agile_course': 40,
    'creative_course': 40,
    'agile_geometry_p0': 41,
    'agile_geometry_p1': 42,
    'agile_geometry_p2': 43,
    'agile_geometry_p3': 44,
    'agile_geometry_intro': 45,
    'agile_course_intro': 45,
    'agile_pillar_weave': 46,
    'agile_pillars': 46,
    'agile_pillar_l0': 47,
    'agile_pillar_l1': 48,
    'agile_pillar_l2': 49,
    'agile_pillar_l3': 50,
    'agile_pillar_easy_mix': 51,
    'agile_pillar_curriculum': 52,
    'agile_pillar_l2_open': 53,
    'agile_pillar_l2_marker_tiny': 54,
    'agile_pillar_l2_marker_small': 55,
    'agile_pillar_l2_marker_large': 56,
    'agile_pillar_bottleneck_curriculum': 57,
}

ROUTE_REWARD_MODES = {
    'dense': 0,
    'shaped': 0,
    'current': 0,
    'reference': 1,
    'drone': 1,
    'drone_reference': 1,
    'drone_hover': 2,
    'hover': 2,
    'drone_hover_route': 2,
    'puffer_drone_hover': 2,
    'drone_hover_continuous': 3,
    'hover_continuous': 3,
    'continuous_hover': 3,
    'puffer_drone_hover_continuous': 3,
    'drone_hover_arrival': 4,
    'hover_arrival': 4,
    'controlled_arrival': 4,
    'drone_controlled_arrival': 4,
}

ROUTE_TARGET_MODES = {
    'random': 0,
    'uniform': 0,
    'blocked': 1,
    'blocked_line': 1,
    'vision': 1,
}


ROUTE_TARGET_OBS_MODES = {
    'legacy': 0,
    'world': 0,
    'current': 0,
    'body': 1,
    'body_frame': 1,
    'local': 1,
    'minimal': 2,
    'body_minimal': 2,
    'unit': 2,
    'drone': 3,
    'native_drone': 3,
    'puffer_drone': 3,
    'drone_hover': 3,
}


def obs_mode_id(value):
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('', '0', 'false', 'off', 'none', 'no'):
            return 0
        if text in ('1', 'true', 'on', 'legacy', 'full', 'yes'):
            return 1
        if text in ('2', 'compact', 'small'):
            return 2
        if text in ('3', 'affordance', 'local_affordance', 'gap'):
            return 3
    return int(value)


def route_target_obs_mode_id(value):
    if isinstance(value, str):
        try:
            return ROUTE_TARGET_OBS_MODES[value.strip().lower()]
        except KeyError as e:
            valid = ', '.join(sorted(ROUTE_TARGET_OBS_MODES))
            raise ValueError(
                f'Unknown route_target_obs_mode: {value!r}. '
                f'Valid modes: {valid}'
            ) from e
    return int(value)


def task_mode_id(task_mode):
    if isinstance(task_mode, str):
        try:
            return TASK_MODES[task_mode]
        except KeyError as e:
            valid = ', '.join(sorted(TASK_MODES))
            raise ValueError(f'Unknown quadrotor task_mode {task_mode!r}. Valid modes: {valid}') from e

    return int(task_mode)


def action_mode_id(action_mode):
    if isinstance(action_mode, str):
        try:
            return ACTION_MODES[action_mode]
        except KeyError as e:
            valid = ', '.join(sorted(ACTION_MODES))
            raise ValueError(f'Unknown quadrotor action_mode {action_mode!r}. Valid modes: {valid}') from e

    return int(action_mode)


def obstacle_mode_id(obstacle_mode):
    if isinstance(obstacle_mode, str):
        try:
            return OBSTACLE_MODES[obstacle_mode]
        except KeyError as e:
            valid = ', '.join(sorted(OBSTACLE_MODES))
            raise ValueError(f'Unknown quadrotor obstacle_mode {obstacle_mode!r}. Valid modes: {valid}') from e

    return int(obstacle_mode)


def route_reward_mode_id(route_reward_mode):
    if isinstance(route_reward_mode, str):
        try:
            return ROUTE_REWARD_MODES[route_reward_mode]
        except KeyError as e:
            valid = ', '.join(sorted(ROUTE_REWARD_MODES))
            raise ValueError(
                f'Unknown quadrotor route_reward_mode {route_reward_mode!r}. '
                f'Valid modes: {valid}'
            ) from e

    return int(route_reward_mode)


def route_target_mode_id(route_target_mode):
    if isinstance(route_target_mode, str):
        try:
            return ROUTE_TARGET_MODES[route_target_mode]
        except KeyError as e:
            valid = ', '.join(sorted(ROUTE_TARGET_MODES))
            raise ValueError(
                f'Unknown quadrotor route_target_mode {route_target_mode!r}. '
                f'Valid modes: {valid}'
            ) from e

    return int(route_target_mode)


class Quadrotor(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, max_steps=1000, action_repeat=10,
                 max_rate=10.0, max_vel=1.0, max_z_vel=0.75,
                 dynamics_profile=0, physics_dt=0.05, physics_substeps=1,
                 vel_ctrl_kp_vel=0.0, vel_ctrl_ki_z=0.0,
                 vel_ctrl_int_limit_z=0.0, vel_ctrl_max_z_acc=0.0,
                 spawn_radius=2.0, spawn_radius_min=-1.0, spawn_height=1.5,
                 init_vel_radius=0.0, init_omega_radius=0.0,
                 room_size=0.0, ceiling_height=3.0, num_rays=0, ray_length=3.0,
                 hover_radius=0.3, obstacle_mode='none', obstacle_block_prob=0.5,
                 obs_noise=0.0, action_noise=0.0, action_delay=0,
                 action_delay_min=-1, action_delay_max=-1,
                 action_history_obs=0,
                 mass_scale=1.0, inertia_scale=0.0, thrust_scale=1.0,
                 torque_scale=0.0, drag_scale=1.0, motor_tau_scale=1.0,
                 mass_scale_min=0.0, mass_scale_max=0.0,
                 inertia_scale_min=0.0, inertia_scale_max=0.0,
                 thrust_scale_min=0.0, thrust_scale_max=0.0,
                 torque_scale_min=0.0, torque_scale_max=0.0,
                 drag_scale_min=0.0, drag_scale_max=0.0,
                 motor_tau_scale_min=0.0, motor_tau_scale_max=0.0,
                 explore_cell_size=0.25, explore_goal_fraction=0.85,
                 route_goal_radius=0.3, route_goal_count=3,
                 route_goal_count_min=0,
                 route_step_cost=0.004, route_proximity_reward=0.02,
                 route_dist_scale=0.01, route_progress_scale=25.0,
                 route_crash_reward=-24.0, route_crash_goal_scale=4.0,
                 route_crash_dist_scale=1.5, route_clearance_radius=0.45,
                 route_clearance_scale=0.35,
                 route_clearance_speed_scale=0.0,
                 route_clearance_speed_min=0.0,
                 route_clearance_speed_full=0.0,
                 route_vel_scale=0.04,
                 route_reference_clearance_radius=0.45,
                 route_reference_clearance_scale=0.0,
                 route_path_margin=0.0,
                 route_arrival_radius=0.0, route_arrival_bonus=0.0,
                 route_arrival_vel_scale=0.0,
                 route_arrival_toward_scale=0.0,
                 route_target_wall_margin=0.0,
                 route_goal_speed=2.0, route_goal_bonus=1.0,
                 route_reach_bonus=5.0,
                 route_reward_mode='dense',
                 route_target_z_min=0.0, route_target_z_max=0.0,
                 route_walls=True, route_direct_obs=False,
                 route_privileged_obs=False,
                 route_extra_obs=False, route_wall_obs=False,
                 route_perception_obs=False,
                 route_target_obs_mode='legacy',
                 route_target_mode='random',
                 route_blocked_target_prob=1.0,
                 drone_alpha_dist=0.782192, drone_alpha_hover=0.071445,
                 drone_alpha_omega=0.00135588,
                 drone_alpha_shaping=3.9754,
                 drone_hover_target_dist=5.0,
                 drone_hover_dist=0.1, drone_hover_omega=0.1,
                 drone_hover_vel=0.1,
                 task_mode='fixed_target', action_mode='rate',
                 render_mode='human', report_interval=1024,
                 buf=None, seed=0):
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0
        self.action_mode = action_mode_id(action_mode)
        self.route_target_obs_mode = route_target_obs_mode_id(route_target_obs_mode)
        self.obs_noise = float(obs_noise)
        self.action_noise = float(action_noise)
        fixed_delay = max(int(action_delay), 0)
        min_delay = fixed_delay if int(action_delay_min) < 0 else max(int(action_delay_min), 0)
        max_delay = fixed_delay if int(action_delay_max) < 0 else max(int(action_delay_max), 0)
        if max_delay < min_delay:
            min_delay, max_delay = max_delay, min_delay
        self.action_delay_min = min_delay
        self.action_delay_max = max_delay
        self.action_delay = max_delay
        self.rng = np.random.default_rng(seed)

        self.num_rays = int(np.clip(num_rays, 0, 64))
        self.num_act = 4
        self.action_history_obs = int(np.clip(action_history_obs, 0, 4))
        self.action_history_size = self.action_history_obs * self.num_act
        self.route_wall_obs = bool(route_wall_obs)
        self.route_extra_obs = bool(route_extra_obs)
        self.route_perception_obs = obs_mode_id(route_perception_obs)
        self.num_obs = BASE_OBSERVATION_SIZE + self.num_rays
        if self.route_wall_obs:
            self.num_obs += WALL_OBSERVATION_SIZE
        if self.route_extra_obs:
            self.num_obs += ROUTE_EXTRA_OBSERVATION_SIZE
        if self.route_perception_obs == 1:
            self.num_obs += ROUTE_PERCEPTION_OBSERVATION_SIZE
        elif self.route_perception_obs == 2:
            self.num_obs += ROUTE_COMPACT_PERCEPTION_OBSERVATION_SIZE
        elif self.route_perception_obs == 3:
            self.num_obs += ROUTE_AFFORDANCE_OBSERVATION_SIZE
        self.num_obs += self.action_history_size
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_obs,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_act,), dtype=np.float32
        )

        super().__init__(buf)
        self.actions = np.zeros((num_envs, self.num_act), dtype=np.float32)
        self.action_queue = np.zeros(
            (self.action_delay_max + 1, num_envs, self.num_act),
            dtype=np.float32,
        )
        self.command_history = np.zeros(
            (self.action_history_obs, num_envs, self.num_act),
            dtype=np.float32,
        )
        self.current_action_delay = np.zeros(num_envs, dtype=np.int64)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            max_steps=float(max_steps),
            action_repeat=float(action_repeat),
            max_rate=float(max_rate),
            max_vel=float(max_vel),
            max_z_vel=float(max_z_vel),
            dynamics_profile=int(dynamics_profile),
            physics_dt=float(physics_dt),
            physics_substeps=int(physics_substeps),
            vel_ctrl_kp_vel=float(vel_ctrl_kp_vel),
            vel_ctrl_ki_z=float(vel_ctrl_ki_z),
            vel_ctrl_int_limit_z=float(vel_ctrl_int_limit_z),
            vel_ctrl_max_z_acc=float(vel_ctrl_max_z_acc),
            spawn_radius=float(spawn_radius),
            spawn_radius_min=float(spawn_radius_min),
            spawn_height=float(spawn_height),
            init_vel_radius=float(init_vel_radius),
            init_omega_radius=float(init_omega_radius),
            room_size=float(room_size),
            ceiling_height=float(ceiling_height),
            ray_length=float(ray_length),
            num_rays=int(self.num_rays),
            hover_radius=float(hover_radius),
            drone_alpha_dist=float(drone_alpha_dist),
            drone_alpha_hover=float(drone_alpha_hover),
            drone_alpha_omega=float(drone_alpha_omega),
            drone_alpha_shaping=float(drone_alpha_shaping),
            drone_hover_target_dist=float(drone_hover_target_dist),
            drone_hover_dist=float(drone_hover_dist),
            drone_hover_omega=float(drone_hover_omega),
            drone_hover_vel=float(drone_hover_vel),
            task_mode=task_mode_id(task_mode),
            action_mode=int(self.action_mode),
            obstacle_mode=obstacle_mode_id(obstacle_mode),
            obstacle_block_prob=float(obstacle_block_prob),
            mass_scale=float(mass_scale),
            inertia_scale=float(inertia_scale),
            thrust_scale=float(thrust_scale),
            torque_scale=float(torque_scale),
            drag_scale=float(drag_scale),
            motor_tau_scale=float(motor_tau_scale),
            mass_scale_min=float(mass_scale_min),
            mass_scale_max=float(mass_scale_max),
            inertia_scale_min=float(inertia_scale_min),
            inertia_scale_max=float(inertia_scale_max),
            thrust_scale_min=float(thrust_scale_min),
            thrust_scale_max=float(thrust_scale_max),
            torque_scale_min=float(torque_scale_min),
            torque_scale_max=float(torque_scale_max),
            drag_scale_min=float(drag_scale_min),
            drag_scale_max=float(drag_scale_max),
            motor_tau_scale_min=float(motor_tau_scale_min),
            motor_tau_scale_max=float(motor_tau_scale_max),
            explore_cell_size=float(explore_cell_size),
            explore_goal_fraction=float(explore_goal_fraction),
            route_goal_radius=float(route_goal_radius),
            route_goal_count=int(route_goal_count),
            route_goal_count_min=int(route_goal_count_min),
            route_step_cost=float(route_step_cost),
            route_proximity_reward=float(route_proximity_reward),
            route_dist_scale=float(route_dist_scale),
            route_progress_scale=float(route_progress_scale),
            route_crash_reward=float(route_crash_reward),
            route_crash_goal_scale=float(route_crash_goal_scale),
            route_crash_dist_scale=float(route_crash_dist_scale),
            route_clearance_radius=float(route_clearance_radius),
            route_clearance_scale=float(route_clearance_scale),
            route_clearance_speed_scale=float(route_clearance_speed_scale),
            route_clearance_speed_min=float(route_clearance_speed_min),
            route_clearance_speed_full=float(route_clearance_speed_full),
            route_reference_clearance_radius=float(route_reference_clearance_radius),
            route_reference_clearance_scale=float(route_reference_clearance_scale),
            route_path_margin=float(route_path_margin),
            route_vel_scale=float(route_vel_scale),
            route_arrival_radius=float(route_arrival_radius),
            route_arrival_bonus=float(route_arrival_bonus),
            route_arrival_vel_scale=float(route_arrival_vel_scale),
            route_arrival_toward_scale=float(route_arrival_toward_scale),
            route_target_wall_margin=float(route_target_wall_margin),
            route_goal_speed=float(route_goal_speed),
            route_goal_bonus=float(route_goal_bonus),
            route_reach_bonus=float(route_reach_bonus),
            route_target_z_min=float(route_target_z_min),
            route_target_z_max=float(route_target_z_max),
            route_reward_mode=route_reward_mode_id(route_reward_mode),
            route_target_mode=route_target_mode_id(route_target_mode),
            route_blocked_target_prob=float(route_blocked_target_prob),
            route_walls=int(bool(route_walls)),
            route_direct_obs=int(bool(route_direct_obs)),
            route_privileged_obs=int(bool(route_privileged_obs)),
            route_extra_obs=int(self.route_extra_obs),
            route_wall_obs=int(self.route_wall_obs),
            route_perception_obs=int(self.route_perception_obs),
            route_target_obs_mode=int(self.route_target_obs_mode),
            action_history_obs=int(self.action_history_obs),
        )

    def _sample_action_delay(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
            size = self.num_agents
        else:
            env_ids = np.asarray(env_ids, dtype=np.int64)
            size = len(env_ids)

        if self.action_delay_min == self.action_delay_max:
            self.current_action_delay[env_ids] = self.action_delay_max
        else:
            self.current_action_delay[env_ids] = self.rng.integers(
                self.action_delay_min,
                self.action_delay_max + 1,
                size=size,
                dtype=np.int64,
            )

    def _apply_obs_noise(self):
        if self.obs_noise > 0.0:
            obs_end = self.num_obs - self.action_history_size
            if obs_end <= 0:
                return self.observations
            noise = self.rng.normal(
                0.0, self.obs_noise, (self.num_agents, obs_end)
            ).astype(np.float32)
            self.observations[:, :obs_end] = np.clip(
                self.observations[:, :obs_end] + noise,
                -1.0,
                1.0,
            )
        return self.observations

    def _write_action_history_obs(self):
        if self.action_history_size <= 0:
            return
        start = self.num_obs - self.action_history_size
        self.observations[:, start:] = self.command_history.transpose(
            1, 0, 2
        ).reshape(self.num_agents, self.action_history_size)

    def _push_command_history(self, commands):
        if self.action_history_obs <= 0:
            return
        if self.action_history_obs > 1:
            self.command_history[1:] = self.command_history[:-1]
        self.command_history[0] = commands

    def reset(self, seed=None):
        self.tick = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.action_queue[:] = 0.0
        self.command_history[:] = 0.0
        self._sample_action_delay()
        binding.vec_reset(self.c_envs, seed or 0)
        self._write_action_history_obs()
        self.rewards[:] = 0
        self.terminals[:] = 0
        self.truncations[:] = 0
        return self._apply_obs_noise(), []

    @property
    def agents_per_batch(self):
        return self.num_agents

    def step(self, actions):
        commanded = np.clip(
            actions.reshape(self.num_agents, self.num_act),
            -1.0,
            1.0,
        ).astype(np.float32)
        requested = commanded.copy()
        if self.action_noise > 0.0:
            requested = requested + self.rng.normal(
                0.0, self.action_noise, requested.shape
            ).astype(np.float32)
            requested = np.clip(requested, -1.0, 1.0)

        if self.action_delay_max > 0:
            self.action_queue[:-1] = self.action_queue[1:].copy()
            self.action_queue[-1] = requested
            delay_indices = self.action_delay_max - self.current_action_delay
            requested = self.action_queue[
                delay_indices,
                np.arange(self.num_agents),
            ]

        self.actions[:] = requested
        if self.action_mode == ACTION_MODES['rate']:
            # The C env expects thrust in [0, 1], while the public action space is [-1, 1].
            self.actions[:, 0] = 0.5 * (self.actions[:, 0] + 1.0)
        self.tick += 1
        binding.vec_step(self.c_envs)
        self._push_command_history(commanded)
        done = np.logical_or(self.terminals, self.truncations)
        if np.any(done):
            done_ids = np.flatnonzero(done)
            self.action_queue[:, done_ids, :] = 0.0
            self.command_history[:, done_ids, :] = 0.0
            self._sample_action_delay(done_ids)
        self._write_action_history_obs()

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        return self._apply_obs_noise(), self.rewards, self.terminals, self.truncations, info

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def debug(self):
        return binding.vec_debug(self.c_envs)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024):
    num_envs = 1024
    env = Quadrotor(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.uniform(-1, 1, (atn_cache, num_envs, 4)).astype(np.float32)

    import time
    start = time.time()
    while time.time() - start < timeout:
        env.step(actions[tick % atn_cache])
        tick += 1
    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,.0f}')
    env.close()

if __name__ == '__main__':
    test_performance()
