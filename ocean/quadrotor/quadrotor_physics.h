/* Puffer drone-style rigid-body dynamics for the custom indoor env. */

#ifndef QUADROTOR_PHYSICS_H
#define QUADROTOR_PHYSICS_H

#include <math.h>
#include <string.h>

#define QR_GRAVITY       9.81f
#define QR_DT            0.05f
#define QR_SQRT2_INV     0.70710678f
#define QR_BASE_MASS     1.0f
#define QR_BASE_IXX      0.01f
#define QR_BASE_IYY      0.01f
#define QR_BASE_IZZ      0.02f
#define QR_BASE_ARM_LEN  0.1f
#define QR_BASE_K_THRUST 3e-5f
#define QR_BASE_K_DRAG   1e-6f
#define QR_BASE_B_DRAG   0.1f
#define QR_BASE_MAX_RPM  750.0f
#define QR_BASE_MAX_VEL  50.0f
#define QR_BASE_MAX_OMEGA 50.0f
#define QR_BASE_K_MOT    0.1f
#define QR_BASE_J_MOT    1e-5f
#define QR_BASE_K_ANG_DAMP 0.2f

#define QR_REF_MASS      0.027f
#define QR_REF_IXX       3.85e-6f
#define QR_REF_IYY       3.85e-6f
#define QR_REF_IZZ       5.9675e-6f
#define QR_REF_ARM_LEN   0.0396f
#define QR_REF_K_THRUST  3.16e-10f
#define QR_REF_K_DRAG    0.005964552f
#define QR_REF_B_DRAG    0.0f
#define QR_REF_MAX_RPM   21702.0f
#define QR_REF_MAX_VEL   20.0f
#define QR_REF_MAX_OMEGA 20.0f
#define QR_REF_K_MOT     0.15f
#define QR_REF_J_MOT     0.0f
#define QR_REF_K_ANG_DAMP 0.0f

typedef struct Quadrotor {
    float pos[3];
    float vel[3];
    float quat[4];
    float omega[3];
    float motor_speed[4];

    float mass;
    float inertia[3];
    float arm_length;
    float c_thrust;
    float c_torque;
    float c_drag[3];
    float c_ang_damp;
    float motor_tau;
    float motor_inertia;
    float motor_max;
    float max_vel;
    float max_omega;

    int crashed;
    int tick;
} Quadrotor;

typedef struct QuadrotorDerivative {
    float vel[3];
    float v_dot[3];
    float q_dot[4];
    float omega_dot[3];
    float motor_dot[4];
} QuadrotorDerivative;

#ifndef QR_CLAMPF_DEFINED
#define QR_CLAMPF_DEFINED
static inline float qr_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
#endif

static inline void qr_defaults(Quadrotor* q) {
    q->mass       = QR_BASE_MASS;
    q->inertia[0] = QR_BASE_IXX;
    q->inertia[1] = QR_BASE_IYY;
    q->inertia[2] = QR_BASE_IZZ;
    q->arm_length = QR_BASE_ARM_LEN;
    q->c_thrust   = QR_BASE_K_THRUST;
    q->c_torque   = QR_BASE_K_DRAG;
    q->c_drag[0]  = QR_BASE_B_DRAG;
    q->c_drag[1]  = QR_BASE_B_DRAG;
    q->c_drag[2]  = QR_BASE_B_DRAG;
    q->c_ang_damp = QR_BASE_K_ANG_DAMP;
    q->motor_tau  = QR_BASE_K_MOT;
    q->motor_inertia = QR_BASE_J_MOT;
    q->motor_max  = QR_BASE_MAX_RPM;
    q->max_vel    = QR_BASE_MAX_VEL;
    q->max_omega  = QR_BASE_MAX_OMEGA;
}

static inline void qr_drone_reference_defaults(Quadrotor* q) {
    q->mass       = QR_REF_MASS;
    q->inertia[0] = QR_REF_IXX;
    q->inertia[1] = QR_REF_IYY;
    q->inertia[2] = QR_REF_IZZ;
    q->arm_length = QR_REF_ARM_LEN;
    q->c_thrust   = QR_REF_K_THRUST;
    q->c_torque   = QR_REF_K_DRAG;
    q->c_drag[0]  = QR_REF_B_DRAG;
    q->c_drag[1]  = QR_REF_B_DRAG;
    q->c_drag[2]  = QR_REF_B_DRAG;
    q->c_ang_damp = QR_REF_K_ANG_DAMP;
    q->motor_tau  = QR_REF_K_MOT;
    q->motor_inertia = QR_REF_J_MOT;
    q->motor_max  = QR_REF_MAX_RPM;
    q->max_vel    = QR_REF_MAX_VEL;
    q->max_omega  = QR_REF_MAX_OMEGA;
}

static inline void qr_reset(Quadrotor* q) {
    memset(q->pos, 0, sizeof(q->pos));
    memset(q->vel, 0, sizeof(q->vel));
    q->quat[0] = 1.0f;
    q->quat[1] = 0.0f;
    q->quat[2] = 0.0f;
    q->quat[3] = 0.0f;
    memset(q->omega, 0, sizeof(q->omega));
    for (int i = 0; i < 4; i++)
        q->motor_speed[i] = 0.0f;
    q->crashed = 0;
    q->tick = 0;
}

static inline void qr_body_to_world(const float q[4], const float v[3], float out[3]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float tx = 2.0f * (y * v[2] - z * v[1]);
    float ty = 2.0f * (z * v[0] - x * v[2]);
    float tz = 2.0f * (x * v[1] - y * v[0]);
    out[0] = v[0] + w * tx + (y * tz - z * ty);
    out[1] = v[1] + w * ty + (z * tx - x * tz);
    out[2] = v[2] + w * tz + (x * ty - y * tx);
}

static inline void qr_world_to_body(const float q[4], const float v[3], float out[3]) {
    float conj[4] = {q[0], -q[1], -q[2], -q[3]};
    qr_body_to_world(conj, v, out);
}

static inline void qr_normalize_quat(float q[4]) {
    float n = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n > 1e-8f) {
        float inv = 1.0f / n;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    }
}

static inline void qr_motor_forces(const Quadrotor* q, float* thrust_z,
        float torque[3], const float motor_dot[4]) {
    float f[4], t[4];
    for (int i = 0; i < 4; i++) {
        float w2 = q->motor_speed[i] * q->motor_speed[i];
        f[i] = q->c_thrust * w2;
        t[i] = q->c_torque * f[i];
    }
    *thrust_z = f[0] + f[1] + f[2] + f[3];

    float L = q->arm_length * QR_SQRT2_INV;
    torque[0] = L * ((f[0] + f[1]) - (f[3] + f[2]));
    torque[1] = L * ((f[0] + f[3]) - (f[1] + f[2]));
    torque[2] = -t[0] + t[1] - t[2] + t[3];
    torque[2] += q->motor_inertia *
        (motor_dot[0] - motor_dot[1] + motor_dot[2] - motor_dot[3]);
}

static inline void qr_compute_derivative(const Quadrotor* q,
        const float motor_cmd[4], QuadrotorDerivative* d) {
    for (int i = 0; i < 4; i++) {
        float target = motor_cmd[i];
        if (target < 0.0f) target = 0.0f;
        if (target > q->motor_max) target = q->motor_max;
        d->motor_dot[i] = (target - q->motor_speed[i]) / q->motor_tau;
    }

    float thrust_z, torque[3];
    qr_motor_forces(q, &thrust_z, torque, d->motor_dot);

    float thrust_body[3] = {0.0f, 0.0f, thrust_z};
    float thrust_world[3];
    qr_body_to_world(q->quat, thrust_body, thrust_world);

    float drag_world[3];
    for (int i = 0; i < 3; i++)
        drag_world[i] = -q->c_drag[i] * q->vel[i];

    for (int i = 0; i < 3; i++) {
        d->vel[i] = q->vel[i];
        d->v_dot[i] = (thrust_world[i] + drag_world[i]) / q->mass;
    }
    d->v_dot[2] -= QR_GRAVITY;

    d->q_dot[0] = 0.5f * (-q->quat[1]*q->omega[0] -
                          q->quat[2]*q->omega[1] -
                          q->quat[3]*q->omega[2]);
    d->q_dot[1] = 0.5f * ( q->quat[0]*q->omega[0] +
                          q->quat[2]*q->omega[2] -
                          q->quat[3]*q->omega[1]);
    d->q_dot[2] = 0.5f * ( q->quat[0]*q->omega[1] +
                          q->quat[3]*q->omega[0] -
                          q->quat[1]*q->omega[2]);
    d->q_dot[3] = 0.5f * ( q->quat[0]*q->omega[2] +
                          q->quat[1]*q->omega[1] -
                          q->quat[2]*q->omega[0]);

    float Ix = q->inertia[0], Iy = q->inertia[1], Iz = q->inertia[2];
    float wx = q->omega[0], wy = q->omega[1], wz = q->omega[2];
    d->omega_dot[0] = (torque[0] - q->c_ang_damp * wx +
        (Iy - Iz) * wy * wz) / Ix;
    d->omega_dot[1] = (torque[1] - q->c_ang_damp * wy +
        (Iz - Ix) * wz * wx) / Iy;
    d->omega_dot[2] = (torque[2] - q->c_ang_damp * wz +
        (Ix - Iy) * wx * wy) / Iz;
}

static inline void qr_step_state(const Quadrotor* q,
        const QuadrotorDerivative* d, float dt, Quadrotor* out) {
    *out = *q;
    for (int i = 0; i < 3; i++) {
        out->pos[i] = q->pos[i] + d->vel[i] * dt;
        out->vel[i] = q->vel[i] + d->v_dot[i] * dt;
        out->omega[i] = q->omega[i] + d->omega_dot[i] * dt;
    }
    for (int i = 0; i < 4; i++) {
        out->quat[i] = q->quat[i] + d->q_dot[i] * dt;
        out->motor_speed[i] = q->motor_speed[i] + d->motor_dot[i] * dt;
    }
    qr_normalize_quat(out->quat);
}

static inline void qr_rk4_step(Quadrotor* q, const float motor_cmd[4], float dt) {
    QuadrotorDerivative k1, k2, k3, k4;
    Quadrotor tmp;
    qr_compute_derivative(q, motor_cmd, &k1);
    qr_step_state(q, &k1, 0.5f * dt, &tmp);
    qr_compute_derivative(&tmp, motor_cmd, &k2);
    qr_step_state(q, &k2, 0.5f * dt, &tmp);
    qr_compute_derivative(&tmp, motor_cmd, &k3);
    qr_step_state(q, &k3, dt, &tmp);
    qr_compute_derivative(&tmp, motor_cmd, &k4);

    float dt6 = dt / 6.0f;
    for (int i = 0; i < 3; i++) {
        q->pos[i] += (k1.vel[i] + 2.0f*k2.vel[i] +
            2.0f*k3.vel[i] + k4.vel[i]) * dt6;
        q->vel[i] += (k1.v_dot[i] + 2.0f*k2.v_dot[i] +
            2.0f*k3.v_dot[i] + k4.v_dot[i]) * dt6;
        q->omega[i] += (k1.omega_dot[i] + 2.0f*k2.omega_dot[i] +
            2.0f*k3.omega_dot[i] + k4.omega_dot[i]) * dt6;
    }
    for (int i = 0; i < 4; i++) {
        q->quat[i] += (k1.q_dot[i] + 2.0f*k2.q_dot[i] +
            2.0f*k3.q_dot[i] + k4.q_dot[i]) * dt6;
        q->motor_speed[i] += (k1.motor_dot[i] + 2.0f*k2.motor_dot[i] +
            2.0f*k3.motor_dot[i] + k4.motor_dot[i]) * dt6;
    }
    qr_normalize_quat(q->quat);
}

static inline void qr_finish_step(Quadrotor* q) {
    for (int i = 0; i < 3; i++) {
        q->vel[i] = qr_clampf(q->vel[i], -q->max_vel, q->max_vel);
        q->omega[i] = qr_clampf(q->omega[i], -q->max_omega, q->max_omega);
    }

    q->tick++;
}

static inline void qr_step_dt(Quadrotor* q, const float motor_cmd[4], float dt) {
    if (q->crashed) return;
    qr_rk4_step(q, motor_cmd, dt);
    qr_finish_step(q);
}

static inline void qr_step(Quadrotor* q, const float motor_cmd[4]) {
    qr_step_dt(q, motor_cmd, QR_DT);
}

static inline void qr_step_normalized_dt(Quadrotor* q, const float action[4],
        float dt) {
    float motor_cmd[4];
    for (int i = 0; i < 4; i++) {
        float a = qr_clampf(action[i], -1.0f, 1.0f);
        motor_cmd[i] = 0.5f * (a + 1.0f) * q->motor_max;
    }
    qr_step_dt(q, motor_cmd, dt);
}

static inline void qr_step_normalized(Quadrotor* q, const float action[4]) {
    qr_step_normalized_dt(q, action, QR_DT);
}

static inline float qr_hover_motor_speed(const Quadrotor* q) {
    float thrust_per_motor = q->mass * QR_GRAVITY / 4.0f;
    if (q->c_thrust <= 0.0f || thrust_per_motor <= 0.0f)
        return 0.5f * q->motor_max;
    return qr_clampf(sqrtf(thrust_per_motor / q->c_thrust), 0.0f, q->motor_max);
}

static inline float qr_hover_centered_min_motor_speed(const Quadrotor* q) {
    float hover = qr_hover_motor_speed(q);
    return qr_clampf(2.0f * hover - q->motor_max, 0.0f, q->motor_max);
}

static inline void qr_step_hover_normalized_dt(Quadrotor* q,
        const float action[4], float dt) {
    float motor_cmd[4];
    float hover = qr_hover_motor_speed(q);
    for (int i = 0; i < 4; i++) {
        float a = qr_clampf(action[i], -1.0f, 1.0f);
        float span = a >= 0.0f ? (q->motor_max - hover) : hover;
        motor_cmd[i] = hover + a * span;
    }
    qr_step_dt(q, motor_cmd, dt);
}

static inline void qr_step_hover_normalized(Quadrotor* q, const float action[4]) {
    qr_step_hover_normalized_dt(q, action, QR_DT);
}

static inline void qr_step_hover_centered_normalized_dt(Quadrotor* q,
        const float action[4], float dt) {
    float motor_cmd[4];
    float min_motor = qr_hover_centered_min_motor_speed(q);
    for (int i = 0; i < 4; i++) {
        float a = qr_clampf(action[i], -1.0f, 1.0f);
        float u = 0.5f * (a + 1.0f);
        motor_cmd[i] = min_motor + u * (q->motor_max - min_motor);
    }
    qr_step_dt(q, motor_cmd, dt);
}

#endif // QUADROTOR_PHYSICS_H
