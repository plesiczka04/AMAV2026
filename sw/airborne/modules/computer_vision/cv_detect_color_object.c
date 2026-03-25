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


///////////////////// Nikolas start ////////////////////
//SO BASICLY TO SCALE IT DOWN FOR COMPUTATION, SEE BLACK PIXELS IN FEED REAL TIME. BUT PAPARAZZI FOR COMPUTATION SKIPS MANY PIXELS TO MATCH SCALING
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "modules/computer_vision/cv.h"


float sum_left= 0.0f;
float sum_middle= 0.0f;
float sum_right = 0.0f;



int cod_lum_min1 = 0;
int cod_lum_max1 = 255;
int cod_cb_min1 = 0;
int cod_cb_max1 = 255;
int cod_cr_min1 = 0;
int cod_cr_max1 = 255;
int cod_draw1 = 0;

int cod_lum_min2 = 0;
int cod_lum_max2 = 255;
int cod_cb_min2 = 0;
int cod_cb_max2 = 255;
int cod_cr_min2 = 0;
int cod_cr_max2 = 255;
int cod_draw2 = 0;



static uint8_t *prev_frame = NULL;
static bool waiting_second = false;
static int frame_size = 0;


#define STEP 20   // distance between sampled pixels  THIS IS SCALING
#define DS_W 12   // sampled width
#define DS_H 26   // sampled height

int flow_dx[DS_H][DS_W];
int flow_dy[DS_H][DS_W];

void detect_surface_obstacles(uint8_t *f1, uint8_t *f2, int w, int h);//NIKOLAS calls detect_surface_obstacles
void color_object_detector_init(void);
void color_object_detector_periodic(void);


static uint8_t small_frame1[DS_W * DS_H];//Frame 1
static uint8_t small_frame2[DS_W * DS_H]; //Frame 2
// TO be used for the rescaling

//Each time the camera captures a frame, Paparazzi calls: my_cv(img, camera_id);
struct image_t *cv_detect_color_object(struct image_t *img, uint8_t __attribute__((unused)) camera_id) //img is a pointer to the image stored somewhere in memory. 
                                                               //I use pointer bcz image is large hence instaead of copying it, paparazzi gives an adress to is
{
    // fprintf(stderr,"NIKOLASSS RUNNING\n");// To check if this file is running
  uint8_t *buffer = img->buf; // this is memeory location where the pixels are stored
  uint8_t *buf = (uint8_t *)img->buf;
  
  
  
 // fprintf(stderr,"img->w=%d img->h=%d samples_x=%d samples_y=%d\n",
 // img->w, img->h,
  //img->w / STEP,
  //img->h / STEP);
        
        
// Here we loop through every pixel in the image.
//Paparazzi uses 2 bytes per pixel (RGB uses 3 bytes per pixel). Format 2 pixels: U Y1 V Y2:
                                                                                             //where U is color info shared between two pixels
                                                                                             //where Y1  is pixel 1 brightness
                                                                                             //where V is color info shared between the two pixels
                                                                                             //where Y2 is pixel 2 brightness
                                                                                        //=> hence pixel 1=(Y1,U,V) , pixel 2=(Y2,U,V)
                                                                                        
 // I store the first frame of the camera:
if (!waiting_second) {//f we are waiting for the first frame of a pair. If false -> we don't have frame1 yet  / if True ->  we already stored frame1 and are waiting for frame2
  frame_size = img->w * img->h * 2; // Frame size
  if (prev_frame == NULL)//If memory for the previous frame has not been allocated yet.
    prev_frame = malloc(frame_size); // then we allocate... lol. Create memory (malloc) to store the previous image. prev_frame is the pointer to that memory and frame_size is how much memory we need
  memcpy(prev_frame, buffer, frame_size);//copy the current image into prev_frame
  
  

  waiting_second = true;   // next frame will be frame2
  return img;//Give the image back to Paparazzi.
}                                                                                    
                                                                                        
                                                                                        
/* we now have frame1 + frame2 */

uint8_t *frame1 = prev_frame;
uint8_t *frame2 = buffer;


/* optical flow calculation using frame1 and frame2 */  
int idx = 0;
static int frame_pairs_printed = 0;

for (int x = img->w - STEP; x >= 0; x -= STEP) {    // columns left to right
 for (int y = 0; y < img->h; y += STEP){  // rows bottom to top  
    //Declare a pointer to one pixel value.It is basicly a pointer to where that pixel is stored in memory
    
//Right below: buffer→raw image memory,then we go look deeper to the correct row(each row occupies 2 bytes*width)(y* 2* img->w)then move to thecorrect col(2*x)    
 // and then select the Y brightness value inside UYVY (+1) . and finally yp=&..it finds the memory adress of the brightness Y pixel at position (x,y)    

   
   int base = y * 2 * img->w + 2 * x;
//SOS FOR OVERLEAF REPORT! I  read U1,Y1,V1 and I combine that into using formula Y1 + 0.25*(U1-128) + 0.25*(V1-128);  to get grayscale!
//uint8_t U1 = frame1[base + 0];
uint8_t Y1 = frame1[base + 1];
//uint8_t V1 = frame1[base + 2];

//uint8_t U2 = frame2[base + 0];
uint8_t Y2 = frame2[base + 1];
//uint8_t V2 = frame2[base + 2];

//int gray1 = Y1 + 0.5*(U1-128) + 0.5*(V1-128);
//int gray2 = Y2 + 0.5*(U2-128) + 0.5*(V2-128);

//if (gray1 < 0) gray1 = 0;
//if (gray1 > 255) gray1 = 255;
//if (gray2 < 0) gray2 = 0;
//if (gray2 > 255) gray2 = 255;

small_frame1[idx] = Y1; // Stores the brightness of the pixel of the downsampled frame 1
small_frame2[idx] = Y2; // Stores the brightness of the pixel of the downsampled frame 2
    
    int sx = x / STEP; //indexes from 11 to 0 rows(hight)
    int sy = y / STEP; //indexes from 0 to 25 colls(width)
    //SOS IMAGE IS PRINT LIEK TRAIN DATA IMAGES INVERTED 90 degrees)  NOW BELOW I PRINT IT TRNASFORMED TO CORRECT FORM (REAL DIRECTION)
    
    if (frame_pairs_printed < 1) {
     fprintf(stderr,"(%d,%d)=%3d ", sx, sy, small_frame1[idx]); // Prints all colors(all colls)  (from white to black 0 to 255)
      }    
        


    idx++;
  }
  if (frame_pairs_printed < 1) {
    fprintf(stderr,"\n"); // Prints all colors(all rows)  (from white to black 0 to 255)
  }
  
}
frame_pairs_printed++;
//////////////////////////////////                                                           ///////////////////////////////// 
//////////////////////////////////                                                           /////////////////////////////////  
/////////////////////////////////    SOSSSSS    WE DRAW OPTICAL FLOW OF IMAGE HEREEEEEEEEEE  //////////////////////////////////
/////////////////////////////////                call function here                          /////////////////////////////////  
/////////////////////////////////                                                            /////////////////////////////////  
         
  detect_surface_obstacles(small_frame1, small_frame2, DS_W, DS_H);   
  
  
    waiting_second = false;   // next frame becomes new frame1
  
  return img;
}



void color_object_detector_init(void)
{
  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1,
                   cv_detect_color_object,
                   COLOR_OBJECT_DETECTOR_FPS1,
                   0);
}


void color_object_detector_periodic(void) {}




// My image is stored to 1D.  2D to 1D conversion --->  :       index = row * width + column
void detect_surface_obstacles(uint8_t *f1, uint8_t *f2, int w, int h)
{


   float Grad_tot_vertical[DS_H] = {0};
    float Grad_tot_surface[DS_W] = {0};

    
    float Grad_tot_vertical_left[8] = {0};
    float Grad_tot_vertical_middle[9] = {0};
    float Grad_tot_vertical_right[9] = {0};

    sum_left = 0;
    sum_middle = 0;
    sum_right = 0;
  float  sum_surface = 0.0;
    
    const float THRESHOLD_ONE = 1.0f; // SOS ADJUST PARAMETER, SURFACE HORIZON DETECTOR!
    
// PRINT RAW IMAGE STORAGE
  //  fprintf(stderr,"Raw 1D memory order:\n");
  //  for (int i = 0; i < w*h; i++) {
    //    fprintf(stderr,"%3d ", f1[i]);
    //    if ((i+1) % w == 0) fprintf(stderr,"\n");
    //}
    //fprintf(stderr,"\n====================\n");
            
            
  //  for (int sy= 0; sy < h; sy++)     
    //    Grad_tot_vertical[sy] = 0; //Matlab: Grad_tot_vertical = zeros(1,cols)
        
    //    for (int sx = 0; sx < w; sx++)
   // Grad_tot_surface[sx] = 0; //Matlab: Grad_tot_surface = zeros(1,rows)

//fprintf(stderr,"///////////////////////////////////\n");




/////////////////////////SURFACE REMOVAL//////////////////////////////////////////////

for (int sx = 0; sx < w - 1; sx++) { //Matlab: Grad_tot_surface = zeros(1,rows)
        float col_sum = 0.0f;

        for (int sy = 0; sy < h - 1; sy++) { //Matlab: Grad_tot_vertical = zeros(1,cols)
            int idx = sx * h + sy;
            col_sum += f1[idx];
        }

        Grad_tot_surface[sx] = col_sum; 
        sum_surface += col_sum;
    }

    float mean_surface = sum_surface / (w - 1);
    float threshold_value = mean_surface * THRESHOLD_ONE;
  
    int limit_sx = w - 1;
    
   // for (int sx = 0; sx < w - 1; sx++) {
    
   //   fprintf(stderr,"%.2f //", Grad_tot_surface[sx]);
   // fprintf(stderr,"\n///////////////////////////////////\n"); 
   // }
    
    for (int sx = 6; sx < w - 1; sx++) {//I start from sx=2 to make sure I dont get any frame edges noise, otherwise it removes my CV completetly thinking top sky is surface
    
        if (Grad_tot_surface[sx] > threshold_value) {
            limit_sx = sx;
            //fprintf(stderr, "limit_sx = %d\n", limit_sx);
            break;
        }
    }
    
    
 
     ////////////////////THIS IS WHERE THE GRADIENT IS COMPUTED////////////////////////////////

    for (int sx = 0; sx < limit_sx; sx++) {       //indexes from 0 to 12 cols
         for (int sy = 0; sy < h-1; sy++) {   //indexes from 0 to 26 rows

          //  int idx      = sy * w + sx;              
         //   int idx_r    = (sy+1) * w + sx;
         //   int idx_c    = sy * w + (sx+1);
            
            int idx   = sx*h + sy;        // (row=sy, col=sx) //CORRECT
            int idx_r = (sx + 1) * h + sy;  // pixel below     //
            int idx_c = sx * h + (sy + 1);  // pixel right    //CORRECT

            int Gx = f1[idx_c] - f1[idx];// Gx in my real axis (what I want) 
            int Gy = f1[idx_r] - f1[idx];

            //float Grad_mag = sqrtf(Gx*Gx + Gy*Gy);
             float Grad_mag = sqrtf((float)(Gx * Gx + Gy * Gy));
             
            //int G_surface= f1[idx];
            
            Grad_tot_vertical[sy] += Grad_mag;
            
            //Grad_tot_surface[sx] += G_surface;
             
      //fprintf(stderr,"sy=%d sx=%d idx_c=%d  f1[idx]=%d f1[idx_c]=%d f1[idx_r]=%d\n",sy,sx,idx_c,f1[idx], f1[idx_c], f1[idx_r]);       
      // fprintf(stderr,"%d ", f1[idx]);   // intensity
       // fprintf(stderr,"| %d // ", Gx);   // gradient
       
            if (sy <= 7) {
                Grad_tot_vertical_left[sy] += Grad_mag;
                sum_left += Grad_mag;
                }
                
            else if (sy <= 16) {
                Grad_tot_vertical_middle[sy - 8] += Grad_mag;
                sum_middle += Grad_mag;
                }
                
            else{
                Grad_tot_vertical_right[sy - 17] += Grad_mag;
                sum_right += Grad_mag;
                }

        }
        
        
        
        
    }
    

     
    
    //fprintf(stderr,"LEFT=%.2f MIDDLE=%.2f RIGHT=%.2f\n",sum_left, sum_middle, sum_right);
    
  //  fprintf(stderr,"\n///////////////////////////////////\n");
    

}


















































