/* Rate and outer-loop controllers for the quadrotor model. */

#ifndef QUADROTOR_CONTROL_H
#define QUADROTOR_CONTROL_H

#include "quadrotor_physics.h"

#ifndef QR_CLAMPF_DEFINED
#define QR_CLAMPF_DEFINED
static inline float qr_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
#endif

typedef struct QuadrotorRateCtrl {
    float kp[3];
    float kd[3];
    float prev_err[3];
    float max_torque_frac;
} QuadrotorRateCtrl;

static inline void qr_rate_defaults(QuadrotorRateCtrl* rc) {
    rc->kp[0] = 20.0f;  rc->kp[1] = 20.0f;  rc->kp[2] = 10.0f;
    rc->kd[0] = 0.5f;   rc->kd[1] = 0.5f;   rc->kd[2] = 0.2f;
    memset(rc->prev_err, 0, sizeof(rc->prev_err));
    rc->max_torque_frac = 0.4f;
}

static inline void qr_rate_reset(QuadrotorRateCtrl* rc) {
    memset(rc->prev_err, 0, sizeof(rc->prev_err));
}

static inline void qr_rate_control(const Quadrotor* q, QuadrotorRateCtrl* rc,
        float thrust, const float rate_des[3], float motor_cmd[4]) {
    float L = q->arm_length * QR_SQRT2_INV;
    float f_avg = thrust * 0.25f;
    if (f_avg < 0.01f) f_avg = 0.01f;
    float rp_lim = rc->max_torque_frac * f_avg;
    float yw_lim = rc->max_torque_frac * 0.5f * f_avg;

    float torque_cmd[3];
    for (int i = 0; i < 3; i++) {
        float err = rate_des[i] - q->omega[i];
        float d_err = (err - rc->prev_err[i]) / QR_DT;
        rc->prev_err[i] = err;
        torque_cmd[i] = rc->kp[i] * err + rc->kd[i] * d_err;
    }

    float roll_f  = qr_clampf(torque_cmd[0] * q->inertia[0] / (4.0f * L), -rp_lim, rp_lim);
    float pitch_f = qr_clampf(torque_cmd[1] * q->inertia[1] / (4.0f * L), -rp_lim, rp_lim);
    float yaw_f   = qr_clampf(torque_cmd[2] * q->inertia[2] / (4.0f * q->c_torque), -yw_lim, yw_lim);

    float f_motor[4];
    f_motor[0] = f_avg + roll_f + pitch_f - yaw_f;
    f_motor[1] = f_avg + roll_f - pitch_f + yaw_f;
    f_motor[2] = f_avg - roll_f - pitch_f - yaw_f;
    f_motor[3] = f_avg - roll_f + pitch_f + yaw_f;

    for (int i = 0; i < 4; i++) {
        if (f_motor[i] < 0.0f) f_motor[i] = 0.0f;
        motor_cmd[i] = sqrtf(f_motor[i] / q->c_thrust);
        if (motor_cmd[i] > q->motor_max) motor_cmd[i] = q->motor_max;
    }
}

typedef struct QuadrotorPosCtrl {
    float kp_pos;
    float max_vel;

    float kp_vel;
    float ki_vel;
    float ki_vel_z;
    float vel_int[3];
    float int_limit;
    float int_limit_z;
    float max_tilt_acc;
    float max_z_acc;

    float kp_att;
    float kd_att;
} QuadrotorPosCtrl;

static inline void qr_pos_defaults(QuadrotorPosCtrl* pc) {
    pc->kp_pos       = 1.0f;
    pc->max_vel      = 3.0f;
    pc->kp_vel       = 1.5f;
    pc->ki_vel       = 0.0f;
    pc->ki_vel_z     = 0.0f;
    memset(pc->vel_int, 0, sizeof(pc->vel_int));
    pc->int_limit    = 1.0f;
    pc->int_limit_z  = 1.0f;
    pc->max_tilt_acc = QR_GRAVITY * 0.364f;
    pc->max_z_acc    = QR_GRAVITY * 0.5f;
    pc->kp_att       = 20.0f;
    pc->kd_att       = 5.0f;
}

static inline void qr_pos_reset(QuadrotorPosCtrl* pc) {
    memset(pc->vel_int, 0, sizeof(pc->vel_int));
}

static inline void qr_acc_to_thrust_rates(const Quadrotor* q, QuadrotorPosCtrl* pc,
        float acc_des[3], float target_yaw, float* thrust_out, float rate_des[3]) {
    for (int i = 0; i < 2; i++)
        acc_des[i] = qr_clampf(acc_des[i], -pc->max_tilt_acc, pc->max_tilt_acc);
    float z_extra = acc_des[2] - QR_GRAVITY;
    acc_des[2] = QR_GRAVITY + qr_clampf(z_extra, -pc->max_z_acc, pc->max_z_acc);

    float acc_norm = sqrtf(acc_des[0]*acc_des[0] + acc_des[1]*acc_des[1] +
                           acc_des[2]*acc_des[2]);
    if (acc_norm < 0.01f) acc_norm = 0.01f;
    *thrust_out = q->mass * acc_norm;

    float zb_des[3] = {acc_des[0]/acc_norm, acc_des[1]/acc_norm, acc_des[2]/acc_norm};

    float xc[3] = {cosf(target_yaw), sinf(target_yaw), 0.0f};
    float yb[3] = {
        zb_des[1]*xc[2] - zb_des[2]*xc[1],
        zb_des[2]*xc[0] - zb_des[0]*xc[2],
        zb_des[0]*xc[1] - zb_des[1]*xc[0]
    };
    float yb_n = sqrtf(yb[0]*yb[0] + yb[1]*yb[1] + yb[2]*yb[2]);
    if (yb_n < 1e-6f) yb_n = 1e-6f;
    yb[0] /= yb_n; yb[1] /= yb_n; yb[2] /= yb_n;

    float xb[3] = {
        yb[1]*zb_des[2] - yb[2]*zb_des[1],
        yb[2]*zb_des[0] - yb[0]*zb_des[2],
        yb[0]*zb_des[1] - yb[1]*zb_des[0]
    };

    float zb_cur[3], xb_cur[3], yb_cur[3];
    float bz[3]={0,0,1}, bx[3]={1,0,0}, by[3]={0,1,0};
    qr_body_to_world(q->quat, bz, zb_cur);
    qr_body_to_world(q->quat, bx, xb_cur);
    qr_body_to_world(q->quat, by, yb_cur);

    float e_R[3];
    e_R[0] = 0.5f * (yb_cur[0]*zb_des[0] + yb_cur[1]*zb_des[1] + yb_cur[2]*zb_des[2]
                    - zb_cur[0]*yb[0]     - zb_cur[1]*yb[1]     - zb_cur[2]*yb[2]);
    e_R[1] = 0.5f * (zb_cur[0]*xb[0]     + zb_cur[1]*xb[1]     + zb_cur[2]*xb[2]
                    - xb_cur[0]*zb_des[0] - xb_cur[1]*zb_des[1] - xb_cur[2]*zb_des[2]);
    e_R[2] = 0.5f * (xb_cur[0]*yb[0]     + xb_cur[1]*yb[1]     + xb_cur[2]*yb[2]
                    - yb_cur[0]*xb[0]     - yb_cur[1]*xb[1]     - yb_cur[2]*xb[2]);

    for (int i = 0; i < 3; i++)
        rate_des[i] = -pc->kp_att * e_R[i] - pc->kd_att * q->omega[i];
}

static inline void qr_vel_control(const Quadrotor* q, QuadrotorPosCtrl* pc,
        const float vel_des_in[3], float target_yaw,
        float* thrust_out, float rate_des[3]) {
    float dt = QR_DT;

    float vel_des[3];
    for (int i = 0; i < 3; i++)
        vel_des[i] = qr_clampf(vel_des_in[i], -pc->max_vel, pc->max_vel);

    float acc_des[3];
    for (int i = 0; i < 3; i++) {
        float err = vel_des[i] - q->vel[i];
        float int_limit = i == 2 ? pc->int_limit_z : pc->int_limit;
        float ki = i == 2 ? pc->ki_vel_z : pc->ki_vel;
        pc->vel_int[i] = qr_clampf(pc->vel_int[i] + err * dt,
                                     -int_limit, int_limit);
        acc_des[i] = pc->kp_vel * err + ki * pc->vel_int[i];
    }
    acc_des[2] += QR_GRAVITY;

    qr_acc_to_thrust_rates(q, pc, acc_des, target_yaw, thrust_out, rate_des);
}

static inline void qr_pos_control(const Quadrotor* q, QuadrotorPosCtrl* pc,
        const float target_pos[3], float target_yaw,
        float* thrust_out, float rate_des[3]) {
    float vel_des[3];
    for (int i = 0; i < 3; i++)
        vel_des[i] = pc->kp_pos * (target_pos[i] - q->pos[i]);
    qr_vel_control(q, pc, vel_des, target_yaw, thrust_out, rate_des);
}

typedef struct QuadrotorController {
    QuadrotorPosCtrl pos;
    QuadrotorRateCtrl rate;
} QuadrotorController;

static inline void qr_ctrl_defaults(QuadrotorController* ctrl) {
    qr_pos_defaults(&ctrl->pos);
    qr_rate_defaults(&ctrl->rate);
}

static inline void qr_ctrl_reset(QuadrotorController* ctrl) {
    qr_pos_reset(&ctrl->pos);
    qr_rate_reset(&ctrl->rate);
}

static inline void qr_ctrl_position(const Quadrotor* q, QuadrotorController* ctrl,
        const float target_pos[3], float target_yaw, float motor_cmd[4]) {
    float thrust;
    float rate_des[3];
    qr_pos_control(q, &ctrl->pos, target_pos, target_yaw, &thrust, rate_des);
    qr_rate_control(q, &ctrl->rate, thrust, rate_des, motor_cmd);
}

#endif // QUADROTOR_CONTROL_H
