/*
 * CNN-based obstacle avoider for Paparazzi UAV
 *
 * Replaces the color-count heuristic with a 4-layer CNN that takes raw camera
 * frames and outputs a heading vector for obstacle avoidance.
 */

#ifndef CNN_AVOIDER_H
#define CNN_AVOIDER_H

#include <pthread.h>

/* Module frequency for the cv callback (frames per second) */
#ifndef CNN_AVOIDER_FPS
#define CNN_AVOIDER_FPS 10
#endif

/* Tunable settings exposed to GCS via dl_settings */
extern float cnn_max_speed;       /* max forward speed [m/s] */
extern float cnn_heading_gain;    /* output scaling factor */
extern float cnn_confidence_thr;  /* min vector magnitude to act */

/* Paparazzi module interface */
extern void cnn_avoider_init(void);
extern void cnn_avoider_periodic(void);


#endif /* CNN_AVOIDER_H */
