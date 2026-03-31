#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "state.h"


int cod_lum_min1=0,cod_lum_max1=255,cod_cb_min1=0,cod_cb_max1=255,cod_cr_min1=0,cod_cr_max1=255,cod_draw1=1;
int cod_lum_min2=0,cod_lum_max2=255,cod_cb_min2=0,cod_cb_max2=255,cod_cr_min2=0,cod_cr_max2=255,cod_draw2=0;
float oa_color_count_frac=0.f;

// Config 
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

// Downscaling
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

// Corner Detecton
static void detect_corners()
{
  corner_count=0;
  for(int y=WIN; y<DS_H-WIN; y++){
    for(int x=WIN; x<DS_W-WIN; x++){
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

// Lucas Canade Optical FLow
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
    float v_corr = flow_v[y][x];

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

// Visualization
static void draw_debug_pip(struct image_t *img, int best_slice)
{
  uint8_t* buf = (uint8_t*)img->buf;

  int num_slices = 10;
  int slice_w_ds = DS_W / num_slices;

  int thickness = 3;  
  int step = 1;       

  for(int s = 0; s < num_slices; s++) {

    int x_ds = s * slice_w_ds + slice_w_ds / 2;

    for(int y_ds = 0; y_ds < DS_H; y_ds += step) {

      int x = img->w - (y_ds * img->w) / DS_H;
      int y = (x_ds * img->h) / DS_W;

      for(int t = -thickness; t <= thickness; t++) {

        int xt = x + t;

        if(xt < 0 || xt >= img->w || y < 0 || y >= img->h) continue;

        int idx = (y * img->w + xt) * 2;

        buf[idx]     = 255;
        buf[idx + 1] = 128;
      }
    }
  }

  // Draw best slice
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