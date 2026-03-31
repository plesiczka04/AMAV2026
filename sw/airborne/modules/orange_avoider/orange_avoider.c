
#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "state.h"
#include "modules/core/abi.h"
#include "generated/flight_plan.h"

#define FLOW_THR 400       
#define TURN_DEG 20.0f    
#define TURN_TIME 2       
#define CRUISE_VEL 0.6f    
#define YAW_COOLDOWN 3    
#define SMOOTH_ALPHA 0.90f  


#define VECTOR_VERBOSE TRUE
#define VERBOSE_PRINT(fmt, ...) \
   do { if (VECTOR_VERBOSE) fprintf(stderr, "[orange_avoider] " fmt, ##__VA_ARGS__); } while(0)

static float turn_dir=0;
static int timer=0;
static int cooldown=0;
static abi_event ev;

// Flow callback
static int last_dir = 0; 

static void flow_cb(uint8_t sender, int16_t x, int16_t y, int16_t w, int32_t q, int16_t e) {
  if(timer>0 || cooldown>0) return;

  if(q>FLOW_THR || (int32_t)e>FLOW_THR){

    float weighted_l = (float)e * 1.2f;
    float weighted_r = (float)q * 1.2f;

    float diff = weighted_r - weighted_l;

    if(fabsf(diff) < 120.0f){
      turn_dir *= 0.7f;
      return;
    }

    if(last_dir != 0){
      diff += last_dir * 120.0f;
    }

    float scale = (q>FLOW_THR || (int32_t)e>FLOW_THR) ? 1.5f : 1.0f;
    float new_turn = -diff * 0.10f * scale;

    if(new_turn > TURN_DEG) new_turn = TURN_DEG;
    if(new_turn < -TURN_DEG) new_turn = -TURN_DEG;

    turn_dir = 0.3f * turn_dir + 0.7f * new_turn;

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
    if(rand() % 100 < 10){
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