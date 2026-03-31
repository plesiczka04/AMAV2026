/*
 * orange_avoider.c
 * Reactive obstacle avoidance for orange poles (Cyberzoo)
 * TU Delft AE4317 Autonomous Flight course example
 */

#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "state.h"
#include "modules/core/abi.h"
#include "generated/flight_plan.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ORANGE_AVOIDER_VERBOSE TRUE
#define VERBOSE_PRINT(fmt, ...) \
    do { if (ORANGE_AVOIDER_VERBOSE) fprintf(stderr, "[orange_avoider] " fmt, ##__VA_ARGS__); } while(0)

/* ---------------- Settings ---------------- */
#define COLOR_COUNT_FRAC 0.18f    // Threshold fraction for obstacle detection
#define HEADING_INCREMENT_DEG 5.f // degrees per avoidance step
#define MAX_MOVE_DISTANCE 2.25f    // meters per step
#define MAX_CONFIDENCE 5           // number of negative detections to declare safe

/* ---------------- Internal state ---------------- */
static int32_t color_count = 0;
static int16_t obstacle_free_confidence = 0;
static float heading_increment = 5.f;

enum navigation_state_t {
    SAFE,
    OBSTACLE_FOUND,
    SEARCH_FOR_SAFE_HEADING,
    OUT_OF_BOUNDS
};
static enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;

/* ---------------- ABI callback ---------------- */
#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif
static abi_event color_detection_ev;
static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t __attribute__((unused)) pixel_x,
                               int16_t __attribute__((unused)) pixel_y,
                               int16_t __attribute__((unused)) pixel_width,
                               int16_t __attribute__((unused)) pixel_height,
                               int32_t quality,
                               int16_t __attribute__((unused)) extra)
{
    color_count = quality;
}

/* ---------------- Helper functions ---------------- */

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);

/* ---------------- Initialization ---------------- */
void orange_avoider_init(void)
{
    srand(time(NULL));
    chooseRandomIncrementAvoidance();
    AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);
}

/* ---------------- Periodic update ---------------- */
void orange_avoider_periodic(void)
{
    if (!autopilot_in_flight())
        return;

    // Determine threshold based on camera size
    int32_t color_count_threshold = COLOR_COUNT_FRAC * front_camera.output_size.w * front_camera.output_size.h;
    VERBOSE_PRINT("Color: %d / %d  state: %d\n", color_count, color_count_threshold, navigation_state);

    // Update confidence
    if (color_count < color_count_threshold)
        obstacle_free_confidence++;
    else
        obstacle_free_confidence -= 2;

    if (obstacle_free_confidence < 0) obstacle_free_confidence = 0;
    if (obstacle_free_confidence > MAX_CONFIDENCE) obstacle_free_confidence = MAX_CONFIDENCE;

    float moveDistance = fminf(MAX_MOVE_DISTANCE, 0.2f * obstacle_free_confidence);

    switch (navigation_state) {
        case SAFE:
            moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
            if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
                navigation_state = OUT_OF_BOUNDS;
            } else if (obstacle_free_confidence == 0) {
                navigation_state = OBSTACLE_FOUND;
            } else {
                moveWaypointForward(WP_GOAL, moveDistance);
            }
            break;

        case OBSTACLE_FOUND:
            waypoint_move_here_2d(WP_GOAL);
            waypoint_move_here_2d(WP_TRAJECTORY);
            chooseRandomIncrementAvoidance();
            navigation_state = SEARCH_FOR_SAFE_HEADING;
            break;

        case SEARCH_FOR_SAFE_HEADING:
            increase_nav_heading(heading_increment);
            if (obstacle_free_confidence >= 2)
                navigation_state = SAFE;
            break;

        case OUT_OF_BOUNDS:
            increase_nav_heading(heading_increment);
            moveWaypointForward(WP_TRAJECTORY, 1.5f);
            if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
                increase_nav_heading(heading_increment);
                obstacle_free_confidence = 0;
                navigation_state = SEARCH_FOR_SAFE_HEADING;
            }
            break;

        default:
            break;
    }
}

/* ---------------- Waypoint and heading helpers ---------------- */
uint8_t increase_nav_heading(float incrementDegrees)
{
    float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
    FLOAT_ANGLE_NORMALIZE(new_heading);
    nav.heading = new_heading;
    VERBOSE_PRINT("Heading now: %f deg\n", DegOfRad(new_heading));
    return 0;
}

uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
    struct EnuCoor_i new_coor;
    calculateForwards(&new_coor, distanceMeters);
    moveWaypoint(waypoint, &new_coor);
    return 0;
}

uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
    float heading = stateGetNedToBodyEulers_f()->psi;
    new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * distanceMeters);
    new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * distanceMeters);
    VERBOSE_PRINT("Forward %f m -> x:%f y:%f (heading %f deg)\n",
        distanceMeters, POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y), DegOfRad(heading));
    return 0;
}

uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
    waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
    VERBOSE_PRINT("Waypoint %d set to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y));
    return 0;
}

uint8_t chooseRandomIncrementAvoidance(void)
{
    heading_increment = (rand() % 2 == 0) ? HEADING_INCREMENT_DEG : -HEADING_INCREMENT_DEG;
    VERBOSE_PRINT("Avoidance increment: %f deg\n", heading_increment);
    return 0;
}

/*
#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "state.h"
#include "modules/core/abi.h"
#include "generated/flight_plan.h"

#define FLOW_THR 400       
#define TURN_DEG 20.0f    // faster turn
#define TURN_TIME 2       
#define CRUISE_VEL 0.6f    
#define YAW_COOLDOWN 3    
#define SMOOTH_ALPHA 0.90f  // smoothing factor


#define VECTOR_VERBOSE TRUE
#define VERBOSE_PRINT(fmt, ...) \
   do { if (VECTOR_VERBOSE) fprintf(stderr, "[orange_avoider] " fmt, ##__VA_ARGS__); } while(0)

static float turn_dir=0;
static int timer=0;
static int cooldown=0;
static abi_event ev;

// Flow callback
static int last_dir = 0; // -1 = left, +1 = right

static void flow_cb(uint8_t sender, int16_t x, int16_t y, int16_t w, int32_t q, int16_t e) {
  if(timer>0 || cooldown>0) return;

  if(q>FLOW_THR || (int32_t)e>FLOW_THR){

    float weighted_l = (float)e * 1.2f;
    float weighted_r = (float)q * 1.2f;

    float diff = weighted_r - weighted_l;

    // STRONGER DEAD ZONE
    if(fabsf(diff) < 120.0f){
      turn_dir *= 0.7f;
      return;
    }

    // STRONGER HYSTERESIS (commit harder)
    if(last_dir != 0){
      diff += last_dir * 120.0f;
    }

    float scale = (q>FLOW_THR || (int32_t)e>FLOW_THR) ? 1.5f : 1.0f;
    float new_turn = -diff * 0.10f * scale;

    if(new_turn > TURN_DEG) new_turn = TURN_DEG;
    if(new_turn < -TURN_DEG) new_turn = -TURN_DEG;

    // MORE COMMITMENT
    turn_dir = 0.3f * turn_dir + 0.7f * new_turn;

    // LOCK direction more aggressively
    if(new_turn > 5.0f) last_dir = 1;
    else if(new_turn < -5.0f) last_dir = -1;

    timer = TURN_TIME;
    cooldown = YAW_COOLDOWN;
  }
}

void orange_avoider_init(void){
  AbiBindMsgVISUAL_DETECTION(ABI_BROADCAST,&ev,flow_cb);
}

void orange_avoider_periodic(void){
  if(!autopilot_in_flight()) return;

  if(cooldown>0) cooldown--;

  if(timer == 0 && cooldown == 0){
    if(rand() % 100 < 10){ // 10% chance
      turn_dir = (rand()%2 ? 1 : -1) * TURN_DEG;
      timer = TURN_TIME;
      cooldown = YAW_COOLDOWN;
    }
  }

  float vel = CRUISE_VEL;

  if(timer>0){
    float step = turn_dir/TURN_TIME;
    nav.heading += RadOfDeg(step);
    FLOAT_ANGLE_NORMALIZE(nav.heading);
    vel = 0.5f; 
    timer--;
  }

  float h = nav.heading;
  struct EnuCoor_i* pos = stateGetPositionEnu_i();
  int32_t nx = pos->x + POS_BFP_OF_REAL(sinf(h)*vel);
  int32_t ny = pos->y + POS_BFP_OF_REAL(cosf(h)*vel);

  waypoint_move_xy_i(WP_TRAJECTORY,nx,ny);
  waypoint_move_xy_i(WP_GOAL,nx,ny);

  // Boundary correction
  static bool out_of_bounds = false;
  static int boundary_counter = 0;

  if(!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))){
    out_of_bounds = true;
    boundary_counter = 0;
    // VERBOSE_PRINT("OUT OF OBSTACLE AREA");
  }

}
*/