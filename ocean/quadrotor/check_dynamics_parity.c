#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "quadrotor_physics.h"
#include "../drone/dronelib.h"

static uint32_t next_u32(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float next_action(uint32_t *state) {
    uint32_t value = next_u32(state) >> 8;
    return 2.0f * ((float)value / 16777215.0f) - 1.0f;
}

static float maxf(float a, float b) {
    return a > b ? a : b;
}

static float vec_diff(Vec3 a, const float b[3]) {
    float dx = fabsf(a.x - b[0]);
    float dy = fabsf(a.y - b[1]);
    float dz = fabsf(a.z - b[2]);
    return maxf(dx, maxf(dy, dz));
}

static float quat_diff(Quat a, const float b[4]) {
    float dw = fabsf(a.w - b[0]);
    float dx = fabsf(a.x - b[1]);
    float dy = fabsf(a.y - b[2]);
    float dz = fabsf(a.z - b[3]);
    return maxf(maxf(dw, dx), maxf(dy, dz));
}

int main(int argc, char **argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 200;
    if (steps <= 0) {
        fprintf(stderr, "steps must be positive\n");
        return 2;
    }

    Drone drone = {0};
    init_drone(&drone, 0.2f, 0.0f);
    drone.params.gravity = BASE_GRAVITY;

    drone.state.pos = (Vec3){0.35f, -0.25f, 1.4f};
    drone.state.vel = (Vec3){0.15f, -0.08f, 0.05f};
    drone.state.quat = (Quat){0.994702f, 0.049912f, -0.074869f, 0.049912f};
    quat_normalize(&drone.state.quat);
    drone.state.omega = (Vec3){0.08f, -0.04f, 0.02f};
    for (int i = 0; i < 4; i++) {
        drone.state.rpms[i] = 100.0f + 20.0f * (float)i;
    }

    Quadrotor quad = {0};
    qr_defaults(&quad);
    qr_reset(&quad);
    quad.mass = drone.params.mass;
    quad.inertia[0] = drone.params.ixx;
    quad.inertia[1] = drone.params.iyy;
    quad.inertia[2] = drone.params.izz;
    quad.arm_length = drone.params.arm_len;
    quad.c_thrust = drone.params.k_thrust;
    quad.c_torque = drone.params.k_drag;
    quad.c_drag[0] = drone.params.b_drag;
    quad.c_drag[1] = drone.params.b_drag;
    quad.c_drag[2] = drone.params.b_drag;
    quad.c_ang_damp = drone.params.k_ang_damp;
    quad.motor_tau = drone.params.k_mot;
    quad.motor_inertia = drone.params.j_mot;
    quad.motor_max = drone.params.max_rpm;
    quad.max_vel = drone.params.max_vel;
    quad.max_omega = drone.params.max_omega;

    quad.pos[0] = drone.state.pos.x;
    quad.pos[1] = drone.state.pos.y;
    quad.pos[2] = drone.state.pos.z;
    quad.vel[0] = drone.state.vel.x;
    quad.vel[1] = drone.state.vel.y;
    quad.vel[2] = drone.state.vel.z;
    quad.quat[0] = drone.state.quat.w;
    quad.quat[1] = drone.state.quat.x;
    quad.quat[2] = drone.state.quat.y;
    quad.quat[3] = drone.state.quat.z;
    quad.omega[0] = drone.state.omega.x;
    quad.omega[1] = drone.state.omega.y;
    quad.omega[2] = drone.state.omega.z;
    for (int i = 0; i < 4; i++) {
        quad.motor_speed[i] = drone.state.rpms[i];
    }

    uint32_t rng = 0xdecafbadU;
    float max_pos = 0.0f;
    float max_vel = 0.0f;
    float max_quat = 0.0f;
    float max_omega = 0.0f;
    float max_motor = 0.0f;

    for (int step = 0; step < steps; step++) {
        float action[4];
        float action_copy[4];
        for (int i = 0; i < 4; i++) {
            action[i] = next_action(&rng);
            action_copy[i] = action[i];
        }

        move_drone(&drone, action_copy);
        qr_step_normalized(&quad, action);

        max_pos = maxf(max_pos, vec_diff(drone.state.pos, quad.pos));
        max_vel = maxf(max_vel, vec_diff(drone.state.vel, quad.vel));
        max_quat = maxf(max_quat, quat_diff(drone.state.quat, quad.quat));
        max_omega = maxf(max_omega, vec_diff(drone.state.omega, quad.omega));
        for (int i = 0; i < 4; i++) {
            max_motor = maxf(max_motor, fabsf(drone.state.rpms[i] - quad.motor_speed[i]));
        }
    }

    float worst = maxf(maxf(max_pos, max_vel), maxf(max_quat, maxf(max_omega, max_motor)));
    printf("steps=%d\n", steps);
    printf("max_pos_diff=%.9g\n", max_pos);
    printf("max_vel_diff=%.9g\n", max_vel);
    printf("max_quat_diff=%.9g\n", max_quat);
    printf("max_omega_diff=%.9g\n", max_omega);
    printf("max_motor_diff=%.9g\n", max_motor);
    printf("worst_diff=%.9g\n", worst);

    return worst <= 1e-4f ? 0 : 1;
}
