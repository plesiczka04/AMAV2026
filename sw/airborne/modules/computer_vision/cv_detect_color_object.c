#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "modules/computer_vision/cv.h"

// === Config ===
#define STEP 10 // sampling stride

// Top ROI geometry
#define ROI_WIDTH_FRAC             0.22f   
#define ROI_HEIGHT_FRAC            0.20f   
#define ROI_BOTTOM_MARGIN_FRAC     0.45f   
#define ROI_CENTER_OFFSET_FRAC     0.0f  

// Bottom ROI geometry
#define UPPER_ROI_WIDTH_FRAC       0.22f   
#define UPPER_ROI_HEIGHT_FRAC      0.40f   
#define UPPER_ROI_GAP_FRAC         0.015f 

// YUV thresholds
#define Y_MIN                     10
#define Y_MAX                     235
#define U_TARGET                  91.0f
#define V_TARGET                  123.0f
#define UV_RADIUS                 90.0f

// Hue gate
#define USE_HUE_GATE              1
#define HUE_LOW_DEG               160.0f
#define HUE_HIGH_DEG              220.0f

// Decision thresholds
#define FREE_SPACE_GREEN_THRESHOLD 0.55f
#define UPPER_GREEN_THRESHOLD      0.30f

// Globals (Required for compilation because based off orange_avoider)
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

// Initial values
float bottom_green_fraction = 0.0f;
float upper_green_fraction = 0.0f;
bool vision_is_obstacle = true;

// required by autopilot
float oa_color_count_frac = 0.0f;

// Forward declaration 
void orange_avoider_update_from_vision(float bottom_frac, float upper_frac, bool is_obstacle);

// Helper functions
static inline float wrap_angle_deg(float angle)
{
    while (angle < 0.0f)   { angle += 360.0f; }
    while (angle >= 360.0f){ angle -= 360.0f; }
    return angle;
}

static inline bool angle_in_range(float angle, float low, float high)
{
    if (low <= high) {
        return (angle >= low && angle <= high);
    } else {
        return (angle >= low || angle <= high);
    }
}

// Read one pixel from a UYVY image buffer.
static inline void get_uyvy_pixel(uint8_t *buf, int w, int x, int y,
                                  uint8_t *Y, uint8_t *U, uint8_t *V)
{
    int x_even = x & ~1;
    int base = (y * w + x_even) * 2;

    *U = buf[base + 0];
    *V = buf[base + 2];

    if ((x & 1) == 0) {
        *Y = buf[base + 1];
    } else {
        *Y = buf[base + 3];
    }
}

// Check if pixel within manually tuned green range
static inline bool is_green_pixel(uint8_t Y, uint8_t U, uint8_t V)
{
    if (Y < Y_MIN || Y > Y_MAX) {
        return false;
    }

    float du = ((float)U) - U_TARGET;
    float dv = ((float)V) - V_TARGET;
    float uv_dist = sqrtf(du * du + dv * dv);

    if (uv_dist > UV_RADIUS) {
        return false;
    }

#if USE_HUE_GATE
    float u0 = ((float)U) - 128.0f;
    float v0 = ((float)V) - 128.0f;
    float hue_deg = wrap_angle_deg(atan2f(v0, u0) * 180.0f / (float)M_PI);

    if (!angle_in_range(hue_deg, HUE_LOW_DEG, HUE_HIGH_DEG)) {
        return false;
    }
#endif

    return true;
}

// overlay boxes on camera feed
static void draw_roi_border(uint8_t *buf, int w, int h,
                            int x0, int y0, int x1, int y1,
                            uint8_t border_y)
{
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > w) x1 = w;
    if (y1 > h) y1 = h;

    // top + bottom
    for (int x = x0; x < x1; x++) {
        uint8_t Y, U, V;
        int base_top = (y0 * w + (x & ~1)) * 2;
        int base_bot = ((y1 - 1) * w + (x & ~1)) * 2;

        if ((x & 1) == 0) {
            buf[base_top + 1] = border_y;
            buf[base_bot + 1] = border_y;
        } else {
            buf[base_top + 3] = border_y;
            buf[base_bot + 3] = border_y;
        }
    }

    // left + right
    for (int y = y0; y < y1; y++) {
        int xl = x0;
        int xr = x1 - 1;

        int base_l = (y * w + (xl & ~1)) * 2;
        int base_r = (y * w + (xr & ~1)) * 2;

        if ((xl & 1) == 0) buf[base_l + 1] = border_y;
        else               buf[base_l + 3] = border_y;

        if ((xr & 1) == 0) buf[base_r + 1] = border_y;
        else               buf[base_r + 3] = border_y;
    }
}


struct image_t *color_object_detector(struct image_t *img, uint8_t cam)
{
    (void)cam;

    uint8_t *buf = img->buf;
    int w = img->w;
    int h = img->h;

    // Top ROI

    int roi_forward_depth = (int)(w * ROI_HEIGHT_FRAC);   // how far ROI extends into forward direction
    int roi_span = (int)(h * ROI_WIDTH_FRAC);             // vertical span of ROI
    int side_margin = (int)(w * ROI_BOTTOM_MARGIN_FRAC);  // margin from right image edge
    int center_offset = (int)(h * ROI_CENTER_OFFSET_FRAC);

    if (roi_forward_depth < 1) roi_forward_depth = 1;
    if (roi_span < 1) roi_span = 1;

    // vertically centered
    int y_center = (h / 2) + center_offset;
    int y0 = y_center - roi_span / 2;
    int y1 = y0 + roi_span;

    // forward ROI sits near the right edge
    int x0 = side_margin;
    int x1 = x0 + roi_forward_depth;

    // clamp bottom/forward ROI
    if (x0 < 0) { x0 = 0; }
    if (x1 > w) { x1 = w; }
    if (y0 < 0) { y0 = 0; y1 = roi_span; }
    if (y1 > h) { y1 = h; y0 = h - roi_span; }

    // Bottom ROI
    int upper_depth = (int)(w * UPPER_ROI_HEIGHT_FRAC);   // horizontal depth of upper ROI
    int upper_span  = (int)(h * UPPER_ROI_WIDTH_FRAC);    // vertical span of upper ROI
    int upper_gap   = (int)(w * UPPER_ROI_GAP_FRAC);

    if (upper_depth < 1) upper_depth = 1;
    if (upper_span < 1) upper_span = 1;

    int uy0 = y_center - upper_span / 2;
    int uy1 = uy0 + upper_span;

    int ux1 = x0 - upper_gap;
    int ux0 = ux1 - upper_depth;

    // clamp ROI
    if (ux0 < 0) { ux0 = 0; }
    if (ux1 < 0) { ux1 = 0; }
    if (ux1 < ux0) { ux1 = ux0; }

    if (uy0 < 0) { uy0 = 0; uy1 = upper_span; }
    if (uy1 > h) { uy1 = h; uy0 = h - upper_span; }

    // Count green in ROI
    int bottom_total = 0;
    int bottom_green = 0;

    for (int y = y0; y < y1; y += STEP) {
        for (int x = x0; x < x1; x += STEP) {
            uint8_t Y, U, V;
            get_uyvy_pixel(buf, w, x, y, &Y, &U, &V);

            bottom_total++;
            if (is_green_pixel(Y, U, V)) {
                bottom_green++;
            }
        }
    }

    // Count green in ROI
    int upper_total = 0;
    int upper_green = 0;

    for (int y = uy0; y < uy1; y += STEP) {
        for (int x = ux0; x < ux1; x += STEP) {
            uint8_t Y, U, V;
            get_uyvy_pixel(buf, w, x, y, &Y, &U, &V);

            upper_total++;
            if (is_green_pixel(Y, U, V)) {
                upper_green++;
            }
        }
    }

    // Compute green fractions and free area
    bottom_green_fraction = (bottom_total > 0) ? ((float)bottom_green / (float)bottom_total) : 0.0f;
    upper_green_fraction  = (upper_total > 0) ? ((float)upper_green / (float)upper_total) : 0.0f;

    bool bottom_is_free = (upper_green_fraction >= FREE_SPACE_GREEN_THRESHOLD);
    bool upper_has_vertical_green = (bottom_green_fraction >= UPPER_GREEN_THRESHOLD);

    vision_is_obstacle = (!bottom_is_free) || upper_has_vertical_green;

    oa_color_count_frac = bottom_green_fraction;

    // Draw boxes
    draw_roi_border(buf, w, h, x0, y0, x1, y1, 220);   
    draw_roi_border(buf, w, h, ux0, uy0, ux1, uy1, 140);

    // Send to navigation
    orange_avoider_update_from_vision(bottom_green_fraction,
                                      upper_green_fraction,
                                      vision_is_obstacle);

    fprintf(stderr,
            "VISION upper=%.3f bottom=%.3f obstacle=%d\n",
            bottom_green_fraction,
            upper_green_fraction,
            vision_is_obstacle ? 1 : 0);

    return img;
}

void color_object_detector_init(void)
{
    cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1,
                     color_object_detector,
                     15, 0);

    fprintf(stderr, "[VISION] INIT OK\n");
}

void color_object_detector_periodic(void)
{
}