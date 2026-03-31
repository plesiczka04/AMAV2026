#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "modules/orange_avoider/orange_avoider.h"
#include "modules/core/abi.h"
#include "firmwares/rotorcraft/navigation.h"
#include "state.h"
#include "generated/flight_plan.h"

#define FLOW_THRESHOLD 150      // magnitude above which we consider there is an obstacle
#define TURN_ANGLE_DEG 25       // degrees to turn when obstacle detected
#define FORWARD_DIST 1.0f       // meters to move forward each step

#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif

static abi_event flow_ev;
static int32_t flow_magnitude = 0;

/*---------------------------------------------------------------*/
/* Callback when CV module sends optical flow data               */
/*---------------------------------------------------------------*/
static void flow_cb(uint8_t sender_id,
                    int16_t pixel_x,
                    int16_t pixel_y,
                    int16_t pixel_width,
                    int16_t pixel_height,
                    int32_t quality,
                    int16_t extra)
{
    flow_magnitude = quality;
}

/*---------------------------------------------------------------*/
/* Initialization                                                */
/*---------------------------------------------------------------*/
void orange_avoider_init(void)
{
    srand(stateGetTime());  // initialize random seed
    AbiBindMsgVISUAL_DETECTION(
        ORANGE_AVOIDER_VISUAL_DETECTION_ID,
        &flow_ev,
        flow_cb);
}

/*---------------------------------------------------------------*/
/* Main navigation logic                                         */
/*---------------------------------------------------------------*/
void orange_avoider_periodic(void)
{
    if (!autopilot_in_flight())
        return;

    float heading = stateGetNedToBodyEulers_f()->psi;  // current yaw

    // If obstacle detected, turn left or right randomly
    if(flow_magnitude > FLOW_THRESHOLD)
    {
        fprintf(stderr, "Obstacle detected, turning!\n");

        int dir = (rand() % 2) ? 1 : -1;  // 50% chance
        heading += dir * RadOfDeg(TURN_ANGLE_DEG);

        NavHeading(heading);
    }

    // Move waypoint FORWARD along heading
    float x = stateGetPositionEnu_f()->x;
    float y = stateGetPositionEnu_f()->y;

    float new_x = x + FORWARD_DIST * cosf(heading);
    float new_y = y + FORWARD_DIST * sinf(heading);

    waypoint_set_xy_i(WP_GOAL,
                      POS_BFP_OF_REAL(new_x),
                      POS_BFP_OF_REAL(new_y));
}