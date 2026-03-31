#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "firmwares/rotorcraft/navigation.h"
#include "state.h"
#include "generated/flight_plan.h"

// ---------------- CONFIG ----------------
#define TURN_DEG       15.0f
#define FORWARD_SPEED  0.6f
#define SLOW_SPEED     0.1f

// ---------------- STATE ----------------
static float forward = 0.0f;
static float turn = 0.0f;

static float last_bottom_frac = 0.0f;
static float last_upper_frac = 0.0f;
static bool last_is_obstacle = true;

// ---------------- VISION INPUT ----------------
void orange_avoider_update_from_vision(float bottom_frac, float upper_frac, bool is_obstacle)
{
    last_bottom_frac = bottom_frac;
    last_upper_frac = upper_frac;
    last_is_obstacle = is_obstacle;

    fprintf(stderr,
            "NAV INPUT bottom=%.3f upper=%.3f obstacle=%d\n",
            bottom_frac, upper_frac, is_obstacle ? 1 : 0);

    if (!is_obstacle) {
        // Matches Python detector outcome: safe to proceed
        forward = FORWARD_SPEED;
        turn = 0.0f;
    } else {
        // Python detector only says "obstacle", not left/right.
        // So use a simple fallback maneuver.
        forward = SLOW_SPEED;
        turn = TURN_DEG;
    }
}

// ---------------- MAIN NAV LOOP ----------------
void orange_avoider_periodic(void)
{
    if (!autopilot_in_flight()) {
        return;
    }

    nav.heading += RadOfDeg(turn);
    FLOAT_ANGLE_NORMALIZE(nav.heading);

    if (forward > 0.0f) {
        float h = nav.heading;

        struct EnuCoor_i *pos = stateGetPositionEnu_i();

        int32_t nx = pos->x + POS_BFP_OF_REAL(sinf(h) * forward);
        int32_t ny = pos->y + POS_BFP_OF_REAL(cosf(h) * forward);

        waypoint_move_xy_i(WP_TRAJECTORY, nx, ny);
        waypoint_move_xy_i(WP_GOAL, nx, ny);
    }
}

// ---------------- INIT ----------------
void orange_avoider_init(void)
{
    fprintf(stderr, "[NAV] INIT OK\n");
}