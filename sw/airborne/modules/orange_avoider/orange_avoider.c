/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.c"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 * This module is an example module for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the navigation mode of the autopilot.
 * The avoidance strategy is to simply count the total number of orange pixels. When above a certain percentage threshold,
 * (given by color_count_frac) we assume that there is an obstacle and we turn.
 *
 * The color filter settings are set using the cv_detect_color_object. This module can run multiple filters simultaneously
 * so you have to define which filter to use with the ORANGE_AVOIDER_VISUAL_DETECTION_ID setting.
 */
#include <math.h> //Nikolas
#include "modules/orange_avoider/orange_avoider.h"
#include "modules/cnn_avoider/cnn_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>

#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif




//Computervision outputs

extern float cnn_hx;
extern float cnn_hy;



// Used for low pass filter. Code that remembers the last 3 commands
typedef enum {CMD_FWD, CMD_LEFT, CMD_RIGHT} cmd_t;

static cmd_t hist[3];//Makes array that stores the last 3 commands.
static int h_i = 0;
static int h_n = 0;

static FILE *log_file= NULL; // To save log_file  (STEP1)
static cmd_t majority(cmd_t new_cmd)
{
    hist[h_i] = new_cmd;
    h_i = (h_i + 1) % 3;// allows to overwrite commands (so taht we have continuous real time loop)
    if (h_n < 3) h_n++;

    int f=0,l=0,r=0;
    for(int i=0;i<h_n;i++){
        if(hist[i]==CMD_FWD) f++;
        else if(hist[i]==CMD_LEFT) l++;
        else r++;
    }

    if (f>=l && f>=r) return CMD_FWD;
    if (l>=f && l>=r) return CMD_LEFT;
    return CMD_RIGHT;
}


enum navigation_state_t {
  Forward,
  Diagonal_left,
  Diagonal_right,
  LANDING
};



// define settings, float => stores numbers with DECIMALS.    int => stores numbers only without any decimal!
float oa_color_count_frac = 0.18f;// This means 18% of  color means it is obstacle


//////////////////SEARCH_FOR_SAFE_HEADING//////////////////////
// define and initialise global variables
enum navigation_state_t navigation_state = Forward;// This is where the drone starts from from all cases! IN FUTURE MAKE IT START AT ATAKEOFF



//enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING; // navigation_state is the variable in the category navigation_state_t
int32_t color_count = 0;                // orange color count from color filter for obstacle detection
int16_t obstacle_free_confidence = 0;   // a measure of how certain we are that the way ahead is safe.
float heading_increment = 5.f;          // heading angle increment [deg]
float maxDistance = 2.25;               // max waypoint displacement [m]
//int16_t stores 16 bits, => 32,767.. 
//int32_t stores 32 bits, => 32,767,574,853..   (much larger range)

const int16_t max_trajectory_confidence = 5; // number of consecutive negative object detections to be sure we are obstacle free
// const means constant, CANT CHANGE!

 
 
 static bool goal_outside_OZ(float goal_x, float goal_y)
{
  return !InsideObstacleZone(goal_x, goal_y);  // returns true if point inside OZ polygon or false if outside polygon
}
 
 /*
 
 NIKOLAS
 */
 
void orange_avoider_periodic(void)
{
  if (!autopilot_in_flight()) {
    return;
  }

//SSOOOOOOOOOOOOSSSSS NIKOLAS PUT BACK!!!! /////////////////////////////
  //float current_x = stateGetPositionEnu_f()->x;
 // float current_y = stateGetPositionEnu_f()->y;
 // float heading   = stateGetNedToBodyEulers_f()->psi;
  
float current_x = 0.0f;
float current_y = 0.0f;
float heading   = 0.0f;

  float step = 0.05f;
  float THRESH = -500.0f;     // sensor deadband
  static int rotate_lock = 0;
   float hx = cnn_hx;
    float hy = cnn_hy;

    float angle = DegOfRad(atan2f(hy, hx));

  switch (navigation_state) {

  /* ===================== FORWARD ===================== */

  case Forward: {

    static bool goal_set = false;
    static float goal_x;
    static float goal_y;

    if (!goal_set) {
      goal_x = current_x + step * sinf(heading);
      goal_y = current_y + step * cosf(heading);
      goal_set = true;
    }
    
   if (goal_outside_OZ(goal_x, goal_y)) {
    fprintf(stderr,"OZ BORDER → SMART 90 TURN\n");
    goal_set = false;
   float center_x = WaypointX(WP_HOME);
    float center_y = WaypointY(WP_HOME);
    float dx = center_x - current_x;
    float dy = center_y - current_y;
    /* heading toward center */
    float heading_to_center = atan2f(dx, dy);
    float err = heading_to_center - heading;
    FLOAT_ANGLE_NORMALIZE(err);
    if (err > 0)
        navigation_state = Diagonal_right;   // turning right moves heading CCW in ENU
    else
        navigation_state = Diagonal_left;
    rotate_lock = 20;   // give enough time for ~90° turn
    break;
}




    waypoint_set_xy_i(WP_GOAL,
        POS_BFP_OF_REAL(goal_x),
        POS_BFP_OF_REAL(goal_y));

    float dx = goal_x - current_x;
    float dy = goal_y - current_y;
    float dist = sqrtf(dx*dx + dy*dy);
    

fprintf(stderr,"CNN angle=%.2f deg\n",angle);
    
    if (log_file) {
    fprintf(log_file,
        "%f,%f,%f,%f,%f,%f\n",
        current_x,
        current_y,
        DegOfRad(heading),
        goal_x,
        goal_y,
        angle);
    fflush(log_file);
}

    
    /////////////////////////////////////////////////////////////////////////////////




   if (rotate_lock == 0)
{
    cmd_t c;

    if   (angle > 25 ||    // If positive angle NIKOLAS CHANGE!
          angle < -25 )  // If positive angle NIKOLAS CHANGE!
           {
          goal_set = false;

        if (angle > 0) c = CMD_RIGHT; // If positive angle NIKOLAS CHANGE!
        else c = CMD_LEFT;
                      }
    else
        c = CMD_FWD;

     c = majority(c);
     
     if (h_n < 3) { // This code case to break (exit)  if 3 new commands are not entered (left,right,forward
    break;
}



   if (c == CMD_FWD)   fprintf(stderr,"CMD = FORWARD\n"); //This is just to print to my screen the results from my low pass
   if (c == CMD_LEFT)  fprintf(stderr,"CMD = LEFT\n");
   if (c == CMD_RIGHT) fprintf(stderr,"CMD = RIGHT\n");

    if (c == CMD_LEFT){
           navigation_state = Diagonal_left;
           
           hist[0] = hist[1] = hist[2] = CMD_FWD;// Here I reset my low pass filter
           h_n = 0;
           
           rotate_lock = 1;
          break;
             }
    else if (c == CMD_RIGHT){
          navigation_state = Diagonal_right;
          
          hist[0] = hist[1] = hist[2] = CMD_FWD; // Here I reset my low pass filter
           h_n = 0;
           
          rotate_lock = 1;
           break;
             }
}


    /* waypoint reached → generate next step */
    if (dist < 0.2f) {
        goal_set = false;
        break;
    }

    break;
  }

  /* ================= ROTATE LEFT ================= */

  case Diagonal_left: {

fprintf(stderr,"CNN angle=%.2f deg\n", angle);

    waypoint_set_xy_i(WP_GOAL,
        POS_BFP_OF_REAL(current_x),
        POS_BFP_OF_REAL(current_y));

    float desired_heading = heading - RadOfDeg(9.0f);
    nav_set_heading_rad(desired_heading);
    
     if (log_file) {
    fprintf(log_file,
        "%f,%f,%f,%f,%f,%f\n",
        current_x,
        current_y,
        DegOfRad(heading),
        NAN,
        NAN,
        angle
        );
    fflush(log_file);
}


    if (rotate_lock > 0)
        rotate_lock--;

    if (rotate_lock == 0 &&
        angle < 20 ||
        angle > -20 )
    {
        navigation_state = Forward;
    }
    
    
    cmd_t c = majority(CMD_LEFT); // Here I just print my commands
if (c == CMD_FWD)   fprintf(stderr,"CMD = FORWARD\n");
if (c == CMD_LEFT)  fprintf(stderr,"CMD = LEFT\n");
if (c == CMD_RIGHT) fprintf(stderr,"CMD = RIGHT\n");

    break;
  }

  /* ================= ROTATE RIGHT ================= */

  case Diagonal_right: {

 fprintf(stderr,"CNN angle=%.2f deg\n", angle);

    waypoint_set_xy_i(WP_GOAL,
        POS_BFP_OF_REAL(current_x),
        POS_BFP_OF_REAL(current_y));

    float desired_heading = heading + RadOfDeg(9.0f);
    nav_set_heading_rad(desired_heading);
    
     if (log_file) {
    fprintf(log_file,
        "%f,%f,%f,%f,%f,%f\n",
        current_x,
        current_y,
        DegOfRad(heading),
        NAN,
        NAN,
        angle);
    fflush(log_file);
}

    if (rotate_lock > 0)
        rotate_lock--;

        if (rotate_lock == 0 &&
        angle < 20 ||
        angle > -20 )
    {
        navigation_state = Forward;
    }
    
        cmd_t c = majority(CMD_RIGHT); // Here I just print my commands
if (c == CMD_FWD)   fprintf(stderr,"CMD = FORWARD\n");
if (c == CMD_LEFT)  fprintf(stderr,"CMD = LEFT\n");
if (c == CMD_RIGHT) fprintf(stderr,"CMD = RIGHT\n");

    break;
  }

  case LANDING: {
    autopilot_set_mode(AP_MODE_KILL);
    GotoBlock(10);
    break;
  }

  }
}



void orange_avoider_init(void)
{
    fprintf(stderr,"orange avoider init\n");
}













































