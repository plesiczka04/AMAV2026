/*
 * CNN-based obstacle avoider for Paparazzi UAV
 *
 * This module replaces the simple color-count heuristic of orange_avoider with a
 * 4-layer CNN that processes raw camera frames and outputs a 2D heading vector
 * (vx, vy) representing the recommended direction of travel.
 *
 * Architecture (mirrors the PyTorch training code):
 *   Layer 1: Conv2D 3->16,  kernel 4x4, stride 2, ReLU
 *   Layer 2: Conv2D 16->32, kernel 4x4, stride 2, ReLU
 *   Layer 3: Conv2D 32->64, kernel 4x4, stride 2, ReLU
 *   Layer 4: Conv2D 64->128,kernel 4x4, stride 2, ReLU
 *   Global Average Pooling -> FC 128->2 (heading_x, heading_y)
 *
 * The weights are loaded from a binary file exported by the PyTorch training script.
 * Inference is done in fixed-point or float depending on platform capability.
 *
 * Integration:
 *   - Receives raw images via the ABI VISUAL_DETECTION or a custom image callback
 *   - Runs forward pass each periodic call
 *   - Outputs a heading command consumed by the navigation/guidance layer
 */

#include "modules/cnn_avoider/cnn_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include "modules/computer_vision/cv.h"
#include "mcu_periph/sys_time.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ──────────────────────────────────────────────
 * Compile-time configuration
 * ────────────────────────────────────────────── */

#define CNN_AVOIDER_VERBOSE TRUE

#define PRINT(string, ...) fprintf(stderr, "[cnn_avoider->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if CNN_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

/* Input image is resized/cropped to this fixed size before inference */
#ifndef CNN_INPUT_W
#define CNN_INPUT_W  64
#endif
#ifndef CNN_INPUT_H
#define CNN_INPUT_H  64
#endif
#define CNN_INPUT_C   3   /* RGB channels */

/* Network architecture constants */
#define CONV1_IN_C    3
#define CONV1_OUT_C  16
#define CONV2_IN_C   16
#define CONV2_OUT_C  32
#define CONV3_IN_C   32
#define CONV3_OUT_C  64
#define CONV4_IN_C   64
#define CONV4_OUT_C 128

#define KERNEL_SIZE   4
#define STRIDE        2
#define FC_OUT        2   /* heading vector (vx, vy) */

/*
 * Spatial dimensions after each conv layer (with stride=2, kernel=4, no padding):
 *   After conv1: (64 - 4) / 2 + 1 = 31
 *   After conv2: (31 - 4) / 2 + 1 = 14
 *   After conv3: (14 - 4) / 2 + 1 =  6
 *   After conv4: ( 6 - 4) / 2 + 1 =  2
 * Global average pool over 2x2 -> single 128-dim vector
 */
#define SPATIAL_AFTER_CONV1  31
#define SPATIAL_AFTER_CONV2  14
#define SPATIAL_AFTER_CONV3   6
#define SPATIAL_AFTER_CONV4   2

/* ──────────────────────────────────────────────
 * Weight storage
 *
 * Each conv layer stores weights [out_c][in_c][kH][kW] and biases [out_c].
 * The FC layer stores weights [FC_OUT][CONV4_OUT_C] and biases [FC_OUT].
 * ────────────────────────────────────────────── */

static float conv1_w[CONV1_OUT_C][CONV1_IN_C][KERNEL_SIZE][KERNEL_SIZE];
static float conv1_b[CONV1_OUT_C];

static float conv2_w[CONV2_OUT_C][CONV2_IN_C][KERNEL_SIZE][KERNEL_SIZE];
static float conv2_b[CONV2_OUT_C];

static float conv3_w[CONV3_OUT_C][CONV3_IN_C][KERNEL_SIZE][KERNEL_SIZE];
static float conv3_b[CONV3_OUT_C];

static float conv4_w[CONV4_OUT_C][CONV4_IN_C][KERNEL_SIZE][KERNEL_SIZE];
static float conv4_b[CONV4_OUT_C];

static float fc_w[FC_OUT][CONV4_OUT_C];
static float fc_b[FC_OUT];

float cnn_hx = 0.0f;
float cnn_hy = 0.0f;

/* ──────────────────────────────────────────────
 * Intermediate activation buffers
 *
 * We double-buffer to avoid extra allocations.
 * The largest intermediate tensor is after conv1:
 *   16 * 31 * 31 = 15376 floats ≈ 60 KB
 * Total buffer usage is bounded and statically allocated.
 * ────────────────────────────────────────────── */

/* Input buffer: RGB image normalized to [0,1] */
static float input_buf[CNN_INPUT_C][CNN_INPUT_H][CNN_INPUT_W];

/* Activation buffers — sized for the largest possible intermediate */
#define MAX_C   128
#define MAX_HW   31
static float act_a[MAX_C][MAX_HW][MAX_HW];
static float act_b[MAX_C][MAX_HW][MAX_HW];

/* Output heading vector */
static float cnn_heading_vec[2] = {0.0f, 0.0f};

/* ──────────────────────────────────────────────
 * Module settings (tunable via GCS)
 * ────────────────────────────────────────────── */
float cnn_max_speed       = 0.5f;   /* max forward speed [m/s] */
float cnn_heading_gain    = 1.0f;   /* multiplier on heading output */
float cnn_confidence_thr  = 0.1f;   /* minimum |heading_vec| to act */

/* State */
static bool weights_loaded = false;
static pthread_mutex_t img_mutex = PTHREAD_MUTEX_INITIALIZER;
static volatile bool new_image_available = false;
static uint8_t raw_img_buf[CNN_INPUT_H * CNN_INPUT_W * CNN_INPUT_C];

/* ──────────────────────────────────────────────
 * Image callback — runs in the vision thread
 *
 * This function is registered with the cv module to
 * receive each camera frame. It downscales/crops the
 * image to CNN_INPUT_W x CNN_INPUT_H and stores it
 * in a shared buffer protected by a mutex.
 * ────────────────────────────────────────────── */


static FILE *cnn_log = NULL;

static struct image_t *cnn_image_cb(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
    if (img == NULL) return img;

    /*
     * Simple nearest-neighbor downscale from camera resolution to CNN input.
     * The camera typically outputs UYVY (YUV422); we convert to RGB on the fly.
     * For a production system you'd use a more sophisticated pipeline.
     */
    float scale_x = (float)img->w / CNN_INPUT_W;
    float scale_y = (float)img->h / CNN_INPUT_H;

    pthread_mutex_lock(&img_mutex);
    uint8_t *src = (uint8_t *)img->buf;

    for (int y = 0; y < CNN_INPUT_H; y++) {
        int src_y = (int)(y * scale_y);
        if (src_y >= (int)img->h) src_y = img->h - 1;

        for (int x = 0; x < CNN_INPUT_W; x++) {
            int src_x = (int)(x * scale_x);
            if (src_x >= (int)img->w) src_x = img->w - 1;

            int dst_idx = (y * CNN_INPUT_W + x) * 3;

            if (img->type == IMAGE_YUV422) {
                /* UYVY: U Y0 V Y1 — every 2 pixels share U,V */
                int src_idx = src_y * img->w * 2 + (src_x / 2) * 4;
                uint8_t u  = src[src_idx];
                uint8_t yy = src[src_idx + 1 + (src_x % 2) * 2];
                uint8_t v  = src[src_idx + 2];

                /* YUV -> RGB (clamped to 0-255) */
                int c = yy - 16;
                int d = u - 128;
                int e = v - 128;
                int r = (298 * c + 409 * e + 128) >> 8;
                int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
                int b_val = (298 * c + 516 * d + 128) >> 8;

                raw_img_buf[dst_idx + 0] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
                raw_img_buf[dst_idx + 1] = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g));
                raw_img_buf[dst_idx + 2] = (uint8_t)(b_val < 0 ? 0 : (b_val > 255 ? 255 : b_val));
            } else {
                /* Assume RGB or grayscale — just copy */
                int src_idx = (src_y * img->w + src_x) * img->buf_size / (img->w * img->h);
                raw_img_buf[dst_idx + 0] = src[src_idx];
                raw_img_buf[dst_idx + 1] = src[src_idx + 1];
                raw_img_buf[dst_idx + 2] = src[src_idx + 2];
            }
        }
    }
    new_image_available = true;
    pthread_mutex_unlock(&img_mutex);

    return img;
}

/* ──────────────────────────────────────────────
 * Weight loading
 *
 * Reads a flat binary file with all weights in order:
 *   conv1_w, conv1_b, conv2_w, conv2_b, ..., fc_w, fc_b
 * All stored as 32-bit IEEE float, little-endian.
 * The PyTorch export script produces this format.
 * ────────────────────────────────────────────── */

#ifndef CNN_WEIGHTS_PATH
//for upload on drone
#define CNN_WEIGHTS_PATH "/data/ftp/internal_000/cnn_weights.bin"
//#define CNN_WEIGHTS_PATH "/home/nikolas/paparazzi/sw/airborne/modules/cnn_avoider/cnn_weights.bin"
#endif

static bool load_weights(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        PRINT("ERROR: Cannot open weights file: %s\n", path);
        return false;
    }

    size_t total = 0;
    total += fread(conv1_w, sizeof(float), CONV1_OUT_C * CONV1_IN_C * KERNEL_SIZE * KERNEL_SIZE, f);
    total += fread(conv1_b, sizeof(float), CONV1_OUT_C, f);
    total += fread(conv2_w, sizeof(float), CONV2_OUT_C * CONV2_IN_C * KERNEL_SIZE * KERNEL_SIZE, f);
    total += fread(conv2_b, sizeof(float), CONV2_OUT_C, f);
    total += fread(conv3_w, sizeof(float), CONV3_OUT_C * CONV3_IN_C * KERNEL_SIZE * KERNEL_SIZE, f);
    total += fread(conv3_b, sizeof(float), CONV3_OUT_C, f);
    total += fread(conv4_w, sizeof(float), CONV4_OUT_C * CONV4_IN_C * KERNEL_SIZE * KERNEL_SIZE, f);
    total += fread(conv4_b, sizeof(float), CONV4_OUT_C, f);
    total += fread(fc_w, sizeof(float), FC_OUT * CONV4_OUT_C, f);
    total += fread(fc_b, sizeof(float), FC_OUT, f);

    fclose(f);

    /* Verify expected number of floats */
    size_t expected =
        CONV1_OUT_C * CONV1_IN_C * KERNEL_SIZE * KERNEL_SIZE + CONV1_OUT_C +
        CONV2_OUT_C * CONV2_IN_C * KERNEL_SIZE * KERNEL_SIZE + CONV2_OUT_C +
        CONV3_OUT_C * CONV3_IN_C * KERNEL_SIZE * KERNEL_SIZE + CONV3_OUT_C +
        CONV4_OUT_C * CONV4_IN_C * KERNEL_SIZE * KERNEL_SIZE + CONV4_OUT_C +
        FC_OUT * CONV4_OUT_C + FC_OUT;

    if (total != expected) {
        PRINT("ERROR: Weight file size mismatch. Read %zu, expected %zu floats\n", total, expected);
        return false;
    }

    PRINT("Loaded %zu weight parameters from %s\n", total, path);
    return true;
}

/* ──────────────────────────────────────────────
 * CNN Forward Pass — pure C inference
 * ────────────────────────────────────────────── */

/*
 * conv2d: Applies a single convolution layer.
 *   out[oc][oh][ow] = bias[oc] + sum_{ic,kh,kw} weight[oc][ic][kh][kw] * in[ic][ih*s+kh][iw*s+kw]
 * Followed by ReLU activation.
 *
 * Parameters:
 *   in       — input  tensor [in_c][in_h][in_w]   (pointer into act_a or act_b)
 *   out      — output tensor [out_c][out_h][out_w] (pointer into the OTHER buffer)
 *   weight   — kernel [out_c][in_c][ksize][ksize]
 *   bias     — per-channel bias [out_c]
 *   in_c, out_c — channel counts
 *   in_h, in_w  — spatial input dimensions
 *   ksize       — square kernel side length
 *   stride      — convolution stride
 *
 * Returns: output spatial dimension (assumes square, same for H and W)
 */
static int conv2d_relu(
    const float *in, float *out,
    const float *weight, const float *bias,
    int in_c, int out_c,
    int in_h, int in_w,
    int ksize, int stride)
{
    int out_h = (in_h - ksize) / stride + 1;
    int out_w = (in_w - ksize) / stride + 1;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias[oc];

                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < ksize; kh++) {
                        for (int kw = 0; kw < ksize; kw++) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;

                            /* Index into flattened [C][H][W] tensor */
                            float in_val  = in[ic * (in_h * in_w) + ih * in_w + iw];
                            float w_val   = weight[((oc * in_c + ic) * ksize + kh) * ksize + kw];
                            sum += in_val * w_val;
                        }
                    }
                }

                /* ReLU activation */
                if (sum < 0.0f) sum = 0.0f;

                out[oc * (out_h * out_w) + oh * out_w + ow] = sum;
            }
        }
    }
    return out_h; /* == out_w for square inputs */
}

/*
 * global_avg_pool: Reduces [C][H][W] -> [C] by averaging over spatial dims.
 */
static void global_avg_pool(const float *in, float *out, int channels, int h, int w)
{
    float inv_hw = 1.0f / (float)(h * w);
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int i = 0; i < h * w; i++) {
            sum += in[c * h * w + i];
        }
        out[c] = sum * inv_hw;
    }
}

/*
 * fc_layer: Fully connected layer.  out[j] = bias[j] + sum_i weight[j][i] * in[i]
 */
static void fc_layer(const float *in, float *out,
                     const float *weight, const float *bias,
                     int in_dim, int out_dim)
{
    for (int j = 0; j < out_dim; j++) {
        float sum = bias[j];
        for (int i = 0; i < in_dim; i++) {
            sum += weight[j * in_dim + i] * in[i];
        }
        out[j] = sum;
    }
}

/*
 * cnn_forward: Full forward pass from normalized image to heading vector.
 */
static void cnn_forward(void)
{
    int spatial;

    /* Conv1: [3][64][64] -> [16][31][31] */
    spatial = conv2d_relu(
        (float *)input_buf, (float *)act_a,
        (float *)conv1_w, conv1_b,
        CONV1_IN_C, CONV1_OUT_C,
        CNN_INPUT_H, CNN_INPUT_W,
        KERNEL_SIZE, STRIDE);

    /* Conv2: [16][31][31] -> [32][14][14] */
    spatial = conv2d_relu(
        (float *)act_a, (float *)act_b,
        (float *)conv2_w, conv2_b,
        CONV2_IN_C, CONV2_OUT_C,
        spatial, spatial,
        KERNEL_SIZE, STRIDE);

    /* Conv3: [32][14][14] -> [64][6][6] */
    spatial = conv2d_relu(
        (float *)act_b, (float *)act_a,
        (float *)conv3_w, conv3_b,
        CONV3_IN_C, CONV3_OUT_C,
        spatial, spatial,
        KERNEL_SIZE, STRIDE);

    /* Conv4: [64][6][6] -> [128][2][2] */
    spatial = conv2d_relu(
        (float *)act_a, (float *)act_b,
        (float *)conv4_w, conv4_b,
        CONV4_IN_C, CONV4_OUT_C,
        spatial, spatial,
        KERNEL_SIZE, STRIDE);

    /* Global Average Pool: [128][2][2] -> [128] */
    float pooled[CONV4_OUT_C];
    global_avg_pool((float *)act_b, pooled, CONV4_OUT_C, spatial, spatial);

    /* FC: [128] -> [2] */
    fc_layer(pooled, cnn_heading_vec, (float *)fc_w, fc_b, CONV4_OUT_C, FC_OUT);
}

/* ──────────────────────────────────────────────
 * Paparazzi module interface
 * ────────────────────────────────────────────── */

/*
 * Initialisation: load weights, register camera callback
 */
void cnn_avoider_init(void)
{
    PRINT("Initializing CNN avoider module\n");
    fprintf(stderr, "CNN INIT\n");
    weights_loaded = load_weights(CNN_WEIGHTS_PATH);
    if (!weights_loaded) {
        PRINT("WARNING: Running without trained weights — heading will be random!\n");
    }

    /* Register with the computer vision module to receive front camera frames */
    cv_add_to_device(&front_camera, cnn_image_cb, CNN_AVOIDER_FPS, 0);

    PRINT("CNN avoider initialized (input: %dx%dx%d)\n", CNN_INPUT_W, CNN_INPUT_H, CNN_INPUT_C);

    cnn_log = fopen("/data/ftp/internal_000/cnn_log.txt", "w");
}

/*
 * Periodic function: runs at module frequency (e.g., 10 Hz)
 *
 * 1. Check if a new image is available
 * 2. Normalize the image into the input buffer
 * 3. Run the CNN forward pass
 * 4. Convert the heading vector output into a navigation command
 */
void cnn_avoider_periodic(void)
{
    
   // printf(stderr, "printf test");
    //PRINT("PRINT test");
   // fprintf(stderr, "hello\n");
    static uint32_t call_count = 0;
    static uint32_t inference_count = 0;
 
    call_count++;
    
    /*
    if (call_count % 10 == 0) {
        PRINT("[cnn_avoider] alive: calls=%u inferences=%u weights=%s guided=%s img=%s\n",
                call_count,
                inference_count,
                weights_loaded ? "YES" : "NO",
                (guidance_h.mode == GUIDANCE_H_MODE_GUIDED) ? "YES" : "NO",
                new_image_available ? "YES" : "NO");

        if (cnn_log) {
            fprintf(cnn_log, "calls=%d inferences=%d\n", call_count, inference_count);
            fflush(cnn_log);
        }
    }
    */

    if (call_count % 10 == 0) {
    static float last_t = 0;
    float now = get_sys_time_float();
    if (cnn_log) {
        fprintf(cnn_log, "t=%.3f dt=%.3f calls=%d inferences=%d\n",
                now, now - last_t, call_count, inference_count);
        fflush(cnn_log);
    }
    last_t = now;
    }


    /* Only run if flying in guided mode */
  //  if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {// NIKOLAS comented out
    //    return;
   // }
 
    if (!weights_loaded) {
    fprintf(stderr, "FAIL11111111111111\n");
        return;
    }
 
    /* Grab the latest image if available */
    if (!new_image_available) {
        return;
    }
 
    pthread_mutex_lock(&img_mutex);
    new_image_available = false;
 
    /* Normalize raw uint8 RGB image to float [0, 1] in CHW layout */
    for (int y = 0; y < CNN_INPUT_H; y++) {
        for (int x = 0; x < CNN_INPUT_W; x++) {
            int idx = (y * CNN_INPUT_W + x) * 3;
            input_buf[0][y][x] = raw_img_buf[idx + 0] / 255.0f;
            input_buf[1][y][x] = raw_img_buf[idx + 1] / 255.0f;
            input_buf[2][y][x] = raw_img_buf[idx + 2] / 255.0f;
        }
    }
    pthread_mutex_unlock(&img_mutex);
 
    /* ── Run CNN inference ── */
    cnn_forward();
    inference_count++;
 
    float hx = cnn_heading_vec[0] * cnn_heading_gain;
    float hy = cnn_heading_vec[1] * cnn_heading_gain;
    float magnitude = sqrtf(hx * hx + hy * hy);
    
    cnn_hx = hx;
    cnn_hy = hy;
 
   // fprintf(stderr, "[cnn_avoider] inference #%u: hx=%.3f hy=%.3f mag=%.3f\n",
    //        inference_count, hx, hy, magnitude);
 
    if (magnitude < cnn_confidence_thr) {
        guidance_h_set_body_vel(0, 0);
        return;
    }
 
    float speed = fminf(cnn_max_speed, cnn_max_speed * (hx / magnitude));
    if (speed < 0.0f) speed = 0.0f;
 
    float heading_rate = hy * RadOfDeg(30.0f);
 
 // CNN OUTPUT!!!! NIKOLAS coment out
  //  guidance_h_set_body_vel(speed, 0);
  //  guidance_h_set_heading_rate(heading_rate);
}
