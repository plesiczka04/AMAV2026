/*
 * Copyright (C) 2019 Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of Paparazzi.
 *
 * Paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * Paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * @file modules/computer_vision/cv_detect_object.h
 * Assumes the object consists of a continuous color and checks
 * if you are over the defined object or not
 */

// Own header
#include "modules/computer_vision/cv_detect_color_object.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

#define PRINT(string,...) fprintf(stderr, "[object_detector->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if OBJECT_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static pthread_mutex_t mutex;

#ifndef COLOR_OBJECT_DETECTOR_FPS1
#define COLOR_OBJECT_DETECTOR_FPS1 0 ///< Default FPS (zero means run at camera fps)
#endif
#ifndef COLOR_OBJECT_DETECTOR_FPS2
#define COLOR_OBJECT_DETECTOR_FPS2 0 ///< Default FPS (zero means run at camera fps)
#endif

// Filter Settings
uint8_t cod_lum_min1 = 0;
uint8_t cod_lum_max1 = 0;
uint8_t cod_cb_min1 = 0;
uint8_t cod_cb_max1 = 0;
uint8_t cod_cr_min1 = 0;
uint8_t cod_cr_max1 = 0;

uint8_t cod_lum_min2 = 0;
uint8_t cod_lum_max2 = 0;
uint8_t cod_cb_min2 = 0;
uint8_t cod_cb_max2 = 0;
uint8_t cod_cr_min2 = 0;
uint8_t cod_cr_max2 = 0;

bool cod_draw1 = false;
bool cod_draw2 = false;

// define global variables
struct color_object_t {
  int32_t x_c;
  int32_t y_c;
  uint32_t color_count;
  bool updated;
};
struct color_object_t global_filters[2];

// Function
uint32_t find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
                              uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max);

/*
 * object_detector
 * @param img - input image to process
 * @param filter - which detection filter to process
 * @return img
 */
static struct image_t *object_detector(struct image_t *img, uint8_t filter)
{
  uint8_t lum_min, lum_max;
  uint8_t cb_min, cb_max;
  uint8_t cr_min, cr_max;
  bool draw;

  switch (filter){
    case 1:
      lum_min = cod_lum_min1;
      lum_max = cod_lum_max1;
      cb_min = cod_cb_min1;
      cb_max = cod_cb_max1;
      cr_min = cod_cr_min1;
      cr_max = cod_cr_max1;
      draw = cod_draw1;
      break;
    case 2:
      lum_min = cod_lum_min2;
      lum_max = cod_lum_max2;
      cb_min = cod_cb_min2;
      cb_max = cod_cb_max2;
      cr_min = cod_cr_min2;
      cr_max = cod_cr_max2;
      draw = cod_draw2;
      break;
    default:
      return img;
  };

  int32_t x_c, y_c;

  // Filter and find centroid
  uint32_t count = find_object_centroid(img, &x_c, &y_c, draw, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max);
  VERBOSE_PRINT("Color count %d: %u, threshold %u, x_c %d, y_c %d\n", camera, object_count, count_threshold, x_c, y_c);
  VERBOSE_PRINT("centroid %d: (%d, %d) r: %4.2f a: %4.2f\n", camera, x_c, y_c,
        hypotf(x_c, y_c) / hypotf(img->w * 0.5, img->h * 0.5), RadOfDeg(atan2f(y_c, x_c)));

  pthread_mutex_lock(&mutex);
  global_filters[filter-1].color_count = count;
  global_filters[filter-1].x_c = x_c;
  global_filters[filter-1].y_c = y_c;
  global_filters[filter-1].updated = true;
  pthread_mutex_unlock(&mutex);

  return img;
}

struct image_t *object_detector1(struct image_t *img, uint8_t camera_id);
struct image_t *object_detector1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return object_detector(img, 1);
}

struct image_t *object_detector2(struct image_t *img, uint8_t camera_id);
struct image_t *object_detector2(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return object_detector(img, 2);
}

void color_object_detector_init(void)
{
  memset(global_filters, 0, 2*sizeof(struct color_object_t));
  pthread_mutex_init(&mutex, NULL);
#ifdef COLOR_OBJECT_DETECTOR_CAMERA1
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN1
  cod_lum_min1 = COLOR_OBJECT_DETECTOR_LUM_MIN1;
  cod_lum_max1 = COLOR_OBJECT_DETECTOR_LUM_MAX1;
  cod_cb_min1 = COLOR_OBJECT_DETECTOR_CB_MIN1;
  cod_cb_max1 = COLOR_OBJECT_DETECTOR_CB_MAX1;
  cod_cr_min1 = COLOR_OBJECT_DETECTOR_CR_MIN1;
  cod_cr_max1 = COLOR_OBJECT_DETECTOR_CR_MAX1;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW1
  cod_draw1 = COLOR_OBJECT_DETECTOR_DRAW1;
#endif

  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, object_detector1, COLOR_OBJECT_DETECTOR_FPS1, 0);
#endif

#ifdef COLOR_OBJECT_DETECTOR_CAMERA2
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN2
  cod_lum_min2 = COLOR_OBJECT_DETECTOR_LUM_MIN2;
  cod_lum_max2 = COLOR_OBJECT_DETECTOR_LUM_MAX2;
  cod_cb_min2 = COLOR_OBJECT_DETECTOR_CB_MIN2;
  cod_cb_max2 = COLOR_OBJECT_DETECTOR_CB_MAX2;
  cod_cr_min2 = COLOR_OBJECT_DETECTOR_CR_MIN2;
  cod_cr_max2 = COLOR_OBJECT_DETECTOR_CR_MAX2;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW2
  cod_draw2 = COLOR_OBJECT_DETECTOR_DRAW2;
#endif

  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA2, object_detector2, COLOR_OBJECT_DETECTOR_FPS2, 1);
#endif
}

/*
 * find_object_centroid
 *
 * Finds the centroid of pixels in an image within filter bounds.
 * Also returns the amount of pixels that satisfy these filter bounds.
 *
 * @param img - input image to process formatted as YUV422.
 * @param p_xc - x coordinate of the centroid of color object
 * @param p_yc - y coordinate of the centroid of color object
 * @param lum_min - minimum y value for the filter in YCbCr colorspace
 * @param lum_max - maximum y value for the filter in YCbCr colorspace
 * @param cb_min - minimum cb value for the filter in YCbCr colorspace
 * @param cb_max - maximum cb value for the filter in YCbCr colorspace
 * @param cr_min - minimum cr value for the filter in YCbCr colorspace
 * @param cr_max - maximum cr value for the filter in YCbCr colorspace
 * @param draw - whether or not to draw on image
 * @return number of pixels of image within the filter bounds.
 */
uint32_t find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
                              uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max)
{
  uint32_t cnt = 0;
  uint32_t tot_x = 0;
  uint32_t tot_y = 0;
  uint8_t *buffer = img->buf;

  // Go through all the pixels
  for (uint16_t y = 0; y < img->h; y++) {
    for (uint16_t x = 0; x < img->w; x ++) {
      // Check if the color is inside the specified values
      uint8_t *yp, *up, *vp;
      if (x % 2 == 0) {
        // Even x
        up = &buffer[y * 2 * img->w + 2 * x];      // U
        yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y1
        vp = &buffer[y * 2 * img->w + 2 * x + 2];  // V
        //yp = &buffer[y * 2 * img->w + 2 * x + 3]; // Y2
      } else {
        // Uneven x
        up = &buffer[y * 2 * img->w + 2 * x - 2];  // U
        //yp = &buffer[y * 2 * img->w + 2 * x - 1]; // Y1
        vp = &buffer[y * 2 * img->w + 2 * x];      // V
        yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y2
      }
      if ( (*yp >= lum_min) && (*yp <= lum_max) &&
           (*up >= cb_min ) && (*up <= cb_max ) &&
           (*vp >= cr_min ) && (*vp <= cr_max )) {
        cnt ++;
        tot_x += x;
        tot_y += y;
        if (draw){
          *yp = 255;  // make pixel brighter in image
        }
      }
    }
  }
  if (cnt > 0) {
    *p_xc = (int32_t)roundf(tot_x / ((float) cnt) - img->w * 0.5f);
    *p_yc = (int32_t)roundf(img->h * 0.5f - tot_y / ((float) cnt));
  } else {
    *p_xc = 0;
    *p_yc = 0;
  }
  return cnt;
}

void color_object_detector_periodic(void)
{
  static struct color_object_t local_filters[2];
  pthread_mutex_lock(&mutex);
  memcpy(local_filters, global_filters, 2*sizeof(struct color_object_t));
  pthread_mutex_unlock(&mutex);

  if(local_filters[0].updated){
    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION1_ID, local_filters[0].x_c, local_filters[0].y_c,
        0, 0, local_filters[0].color_count, 0);
    local_filters[0].updated = false;
  }
  if(local_filters[1].updated){
    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION2_ID, local_filters[1].x_c, local_filters[1].y_c,
        0, 0, local_filters[1].color_count, 1);
    local_filters[1].updated = false;
  }
}
/*

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "state.h" // body rates


int cod_lum_min1=0,cod_lum_max1=255,cod_cb_min1=0,cod_cb_max1=255,cod_cr_min1=0,cod_cr_max1=255,cod_draw1=1;
int cod_lum_min2=0,cod_lum_max2=255,cod_cb_min2=0,cod_cb_max2=255,cod_cr_min2=0,cod_cr_max2=255,cod_draw2=0;
float oa_color_count_frac=0.f;

// ---------------- Configuration ----------------
#define DS_W 96  // width of downscaled image
#define DS_H 24  // height of downscaled image
#define WIN 5
#define ITER 6
#define MAX_CORNERS 1000
#define CORNER_THR 40.0f
#define FLOW_SCALE_VIS 20.0f
#define EXPANSION_SCALE 500.0f
#define FOCAL_LENGTH_PX 25.0f

static uint8_t frame_prev[DS_W*DS_H];
static uint8_t frame_curr[DS_W*DS_H];
static float flow_u[DS_H][DS_W];
static float flow_v[DS_H][DS_W];
static bool initialized=false;

typedef struct{ float x,y; } corner_t;
static corner_t corners[MAX_CORNERS];
static int corner_count=0;

// ---------------- Downscale ----------------
static void extract_downscale(struct image_t *img,uint8_t*out)
{
  uint8_t* buf = (uint8_t*)img->buf;
  for(int y=0;y<DS_H;y++) {
    int iy = (y*(img->h-1))/(DS_H-1);
    for(int x=0;x<DS_W;x++) {
      int ix = (x*(img->w-1))/(DS_W-1);
      int idx = (iy*img->w + ix)*2;
      out[y*DS_W + x] = buf[idx]; // luminance only
    }
  }
}

// ---------------- Corner Detection ----------------
static void detect_corners()
{
  corner_count=0;
  for(int y=WIN; y<DS_H-WIN; y++){
    for(int x=WIN; x<DS_W-WIN; x++){
      // Mask central region to avoid false center motion
      if(x > DS_W/2-3 && x < DS_W/2+3) continue;
      float dx = frame_curr[y*DS_W + x+1] - frame_curr[y*DS_W + x-1];
      float dy = frame_curr[(y+1)*DS_W + x] - frame_curr[(y-1)*DS_W + x];
      float score = dx*dx + dy*dy;
      if(score > CORNER_THR && corner_count < MAX_CORNERS){
        corners[corner_count].x = x;
        corners[corner_count].y = y;
        corner_count++;
      }
    }
  }
}

// ---------------- Lucas-Kanade Optical Flow ----------------
static void compute_lk()
{
  memset(flow_u,0,sizeof(flow_u));
  memset(flow_v,0,sizeof(flow_v));
  for(int n=0; n<corner_count; n++){
    int ix=(int)corners[n].x; int iy=(int)corners[n].y;
    float u=0,v=0;
    for(int iter=0; iter<ITER; iter++){
      float G11=0,G12=0,G22=0,b1=0,b2=0;
      for(int wy=-WIN; wy<=WIN; wy++){
        for(int wx=-WIN; wx<=WIN; wx++){
          int x=ix+wx; int y=iy+wy;
          float Ix = frame_curr[y*DS_W+x+1] - frame_curr[y*DS_W+x-1];
          float Iy = frame_curr[(y+1)*DS_W+x] - frame_curr[(y-1)*DS_W+x];
          float It = frame_curr[y*DS_W+x] - frame_prev[y*DS_W+x];
          G11 += Ix*Ix; G12 += Ix*Iy; G22 += Iy*Iy; b1 += Ix*It; b2 += Iy*It;
        }
      }
      float det = G11*G22 - G12*G12;
      if(fabs(det)<0.01f) break;
      u += (G12*b2 - G22*b1)/det;
      v += (G12*b1 - G11*b2)/det;
    }
    flow_u[iy][ix] = u; flow_v[iy][ix] = v;
  }
}

// ---------------- Compute Expansion ----------------
static void compute_expansion(int32_t* left, int32_t* right)
{
  float lsum=0, rsum=0; 
  int lc=0, rc=0;
  float cx = DS_W/2.0f;
  float cy = DS_H/2.0f;

  float yaw_rate = stateGetBodyRates_f()->r;
  float u_rot = -FOCAL_LENGTH_PX * yaw_rate;

  for(int n=0; n<corner_count; n++){
    int x = (int)corners[n].x;
    int y = (int)corners[n].y;
    float dx = x - cx;
    float dy = y - cy;

    float u_corr = flow_u[y][x] - u_rot;
    float v_corr = flow_v[y][x]; // can add pitch/roll correction if needed

    float pitch_rate = stateGetBodyRates_f()->p;
    float roll_rate  = stateGetBodyRates_f()->q;

    v_corr -= FOCAL_LENGTH_PX * pitch_rate;
    u_corr -= FOCAL_LENGTH_PX * roll_rate;

    float exp = u_corr*dx + v_corr*dy;

    if(exp > 0.25f){
      if(x < cx){ lsum += exp; lc++; }
      else { rsum += exp; rc++; }
    }
  }

  *left = lc > 8 ? (int32_t)((lsum/lc)*EXPANSION_SCALE) : 0;
  *right = rc > 8 ? (int32_t)((rsum/rc)*EXPANSION_SCALE) : 0;
}

// ---------------- Visualization ----------------
static void draw_debug_pip(struct image_t *img, int best_slice)
{
  uint8_t* buf = (uint8_t*)img->buf;

  int num_slices = 10;
  int slice_w_ds = DS_W / num_slices;

  int thickness = 3;   // 
  int step = 1;        // increase for performance if needed

  // -------- Draw all slices (white) --------
  for(int s = 0; s < num_slices; s++) {

    int x_ds = s * slice_w_ds + slice_w_ds / 2;

    for(int y_ds = 0; y_ds < DS_H; y_ds += step) {

      int x = img->w - (y_ds * img->w) / DS_H;
      int y = (x_ds * img->h) / DS_W;

      // ---- thickness drawing ----
      for(int t = -thickness; t <= thickness; t++) {

        int xt = x + t;

        if(xt < 0 || xt >= img->w || y < 0 || y >= img->h) continue;

        int idx = (y * img->w + xt) * 2;

        buf[idx]     = 255;
        buf[idx + 1] = 128;
      }
    }
  }

  // -------- Draw best slice (green, thicker) --------
  int bx_ds = best_slice * slice_w_ds + slice_w_ds / 2;

  for(int y_ds = 0; y_ds < DS_H; y_ds += step) {

    int x = (y_ds * img->w) / DS_H;
    int y = img->h - (bx_ds * img->h) / DS_W;

    for(int t = -thickness-1; t <= thickness+1; t++) {

      int xt = x + t;

      if(xt < 0 || xt >= img->w || y < 0 || y >= img->h) continue;

      int idx = (y * img->w + xt) * 2;

      buf[idx]     = 0;
      buf[idx + 1] = 255;
    }
  }
}
// ---------------- Main Detector ----------------
struct image_t* color_object_detector(struct image_t* img, uint8_t cam)
{
  if(!initialized){
    extract_downscale(img,frame_prev);
    initialized = true;
    return img;
  }

  extract_downscale(img,frame_curr);
  detect_corners();
  compute_lk();

  int32_t l=0, r=0;
  compute_expansion(&l,&r);

  // Determine best slice for visualization
  int num_slices = 10;
  int best_slice = num_slices / 2; // center default

  if (l > r + 50) {
    best_slice = 1; // go left
  } else if (r > l + 50) {
  best_slice = num_slices - 2; // go right
}
  draw_debug_pip(img,best_slice);

  AbiSendMsgVISUAL_DETECTION(ABI_BROADCAST,0,0,0,0,r,(int16_t)l);
  memcpy(frame_prev,frame_curr,DS_W*DS_H);

  return img;
}

void color_object_detector_init(void){
  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1,color_object_detector,15,0);
}

void color_object_detector_periodic(void){}
*/