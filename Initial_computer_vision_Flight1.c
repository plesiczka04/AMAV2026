#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <jpeglib.h>

#define MAX_PTS     1000
#define QUALITY     0.005f
#define MIN_DIST    15
#define BLOCK       3

#define PYR_LEVELS  3
#define WINSZ       21
#define ITERS       10
#define EPS         0.01f

typedef struct { float x,y; } P;

/* ---------------- JPEG ---------------- */
static unsigned char* load_jpeg(const char* name,int* w,int* h){
    FILE* f=fopen(name,"rb");
    if(!f){ fprintf(stderr,"Missing %s\n",name); exit(1); }
    struct jpeg_decompress_struct c; struct jpeg_error_mgr j;
    c.err=jpeg_std_error(&j);
    jpeg_create_decompress(&c);
    jpeg_stdio_src(&c,f);
    jpeg_read_header(&c,1);
    jpeg_start_decompress(&c);

    *w=(int)c.output_width; *h=(int)c.output_height;
    int ch=(int)c.output_components;
    if(ch!=3){ fprintf(stderr,"Expected RGB JPEG\n"); exit(1); }

    unsigned char* img=(unsigned char*)malloc((size_t)(*w)*(*h)*3);
    while(c.output_scanline<c.output_height){
        unsigned char* row[1];
        row[0]=img+(size_t)c.output_scanline*(*w)*3;
        jpeg_read_scanlines(&c,row,1);
    }
    jpeg_finish_decompress(&c);
    jpeg_destroy_decompress(&c);
    fclose(f);
    return img;
}

/* ---------------- Gray ---------------- */
static float* to_gray(const unsigned char* rgb,int w,int h){
    float* g=(float*)malloc((size_t)w*h*sizeof(float));
    for(int i=0;i<w*h;i++){
        int k=i*3;
        g[i]=0.299f*rgb[k]+0.587f*rgb[k+1]+0.114f*rgb[k+2];
    }
    return g;
}

/* ---------------- 3x3 Gaussian (separable 1 2 1) ---------------- */
static void gauss3(float* img,int w,int h){
    float* tmp=(float*)malloc((size_t)w*h*sizeof(float));
    memcpy(tmp,img,(size_t)w*h*sizeof(float));

    // horizontal
    for(int y=0;y<h;y++)
        for(int x=1;x<w-1;x++)
            tmp[y*w+x]=(img[y*w+x-1]+2.0f*img[y*w+x]+img[y*w+x+1])*0.25f;

    // vertical
    for(int y=1;y<h-1;y++)
        for(int x=1;x<w-1;x++)
            img[y*w+x]=(tmp[(y-1)*w+x]+2.0f*tmp[y*w+x]+tmp[(y+1)*w+x])*0.25f;

    free(tmp);
}

/* ---------------- Downsample by 2 (after blur) ---------------- */
static float* down2_gauss(const float* img,int w,int h,int* w2,int* h2){
    float* work=(float*)malloc((size_t)w*h*sizeof(float));
    memcpy(work,img,(size_t)w*h*sizeof(float));
    gauss3(work,w,h);

    *w2=w/2; *h2=h/2;
    float* out=(float*)malloc((size_t)(*w2)*(*h2)*sizeof(float));
    for(int y=0;y<*h2;y++)
        for(int x=0;x<*w2;x++)
            out[y*(*w2)+x]=work[(2*y)*w+2*x];

    free(work);
    return out;
}

/* ---------------- Bilinear sample ---------------- */
static inline float samplef(const float* im,int w,int h,float x,float y){
    int x0=(int)floorf(x), y0=(int)floorf(y);
    if(x0<0||y0<0||x0>=w-1||y0>=h-1) return 0.0f;
    float ax=x-x0, ay=y-y0;
    float v00=im[y0*w+x0], v10=im[y0*w+x0+1];
    float v01=im[(y0+1)*w+x0], v11=im[(y0+1)*w+x0+1];
    return (1-ax)*(1-ay)*v00 + ax*(1-ay)*v10 + (1-ax)*ay*v01 + ax*ay*v11;
}

/* ---------------- Gradient ---------------- */
static void grad(const float* img,int w,int h,float* gx,float* gy){
    memset(gx,0,(size_t)w*h*sizeof(float));
    memset(gy,0,(size_t)w*h*sizeof(float));
    for(int y=1;y<h-1;y++)
        for(int x=1;x<w-1;x++){
            int i=y*w+x;
            gx[i]=(img[i+1]-img[i-1])*0.5f;
            gy[i]=(img[i+w]-img[i-w])*0.5f;
        }
}

/* ---------------- Shi–Tomasi + greedy minDistance ---------------- */
static int shi_tomasi(const float* img,int w,int h,P* pts){
    int r=BLOCK/2;
    float* gx=(float*)malloc((size_t)w*h*sizeof(float));
    float* gy=(float*)malloc((size_t)w*h*sizeof(float));
    grad(img,w,h,gx,gy);

    float* score=(float*)calloc((size_t)w*h,sizeof(float));
    float maxS=0.0f;

    for(int y=r;y<h-r;y++)
    for(int x=r;x<w-r;x++){
        float Sxx=0,Syy=0,Sxy=0;
        for(int wy=-r;wy<=r;wy++)
        for(int wx=-r;wx<=r;wx++){
            int k=(y+wy)*w+(x+wx);
            float ix=gx[k], iy=gy[k];
            Sxx+=ix*ix; Syy+=iy*iy; Sxy+=ix*iy;
        }
        float tr=Sxx+Syy;
        float det=Sxx*Syy-Sxy*Sxy;
        float disc=tr*tr-4*det; if(disc<0) disc=0;
        float lmin=(tr-sqrtf(disc))*0.5f;
        score[y*w+x]=lmin;
        if(lmin>maxS) maxS=lmin;
    }

    float thr=QUALITY*maxS;
    int n=0;

    for(int pick=0; pick<MAX_PTS; pick++){
        float best=thr; int bx=-1, by=-1;
        for(int y=r;y<h-r;y++)
        for(int x=r;x<w-r;x++){
            float s=score[y*w+x];
            if(s>best){ best=s; bx=x; by=y; }
        }
        if(bx<0) break;

        pts[n++] = (P){(float)bx,(float)by};

        for(int y=by-MIN_DIST;y<=by+MIN_DIST;y++){
            if(y<r||y>=h-r) continue;
            for(int x=bx-MIN_DIST;x<=bx+MIN_DIST;x++){
                if(x<r||x>=w-r) continue;
                float dx=x-bx, dy=y-by;
                if(dx*dx+dy*dy <= (float)(MIN_DIST*MIN_DIST))
                    score[y*w+x]=0;
            }
        }
    }

    free(gx); free(gy); free(score);
    return n;
}

/* ---------------- LK refine at one pyramid level (uses initial guess dx,dy) ---------------- */
static void lk_refine_level(
    const float* I1,const float* I2,int w,int h,
    const float* gx,const float* gy,
    float u,float v, float* dx,float* dy
){
    int r=WINSZ/2;

    for(int it=0; it<ITERS; it++){
        float A=0,B=0,C=0,D=0,E=0;

        if(u+*dx<r+1 || v+*dy<r+1 || u+*dx>w-r-2 || v+*dy>h-r-2) break;

        for(int wy=-r;wy<=r;wy++)
        for(int wx=-r;wx<=r;wx++){
            float x1=u+wx,     y1=v+wy;
            float x2=u+*dx+wx, y2=v+*dy+wy;

            float ix = samplef(gx,w,h,x1,y1);
            float iy = samplef(gy,w,h,x1,y1);

            float I1v = samplef(I1,w,h,x1,y1);
            float I2v = samplef(I2,w,h,x2,y2);
            float iterm = I2v - I1v;

            A+=ix*ix; B+=iy*iy; C+=ix*iy;
            D+=ix*iterm; E+=iy*iterm;
        }

        float det=A*B-C*C;
        if(fabsf(det)<1e-6f) break;

        float ddx = (-B*D + C*E)/det;
        float ddy = ( C*D - A*E)/det;

        *dx += ddx;
        *dy += ddy;

        if(fabsf(ddx)<EPS && fabsf(ddy)<EPS) break;
    }
}

/* ---------------- Pyramidal LK (coarse->fine) ---------------- */
static void lk_pyramidal(
    float* pyr1[PYR_LEVELS], float* pyr2[PYR_LEVELS],
    int W[PYR_LEVELS], int H[PYR_LEVELS],
    const P* pts, int n,
    P* flow, unsigned char* status
){
    // gradients for pyr1 at each level
    float* gx[PYR_LEVELS];
    float* gy[PYR_LEVELS];
    for(int l=0;l<PYR_LEVELS;l++){
        gx[l]=(float*)malloc((size_t)W[l]*H[l]*sizeof(float));
        gy[l]=(float*)malloc((size_t)W[l]*H[l]*sizeof(float));
        grad(pyr1[l],W[l],H[l],gx[l],gy[l]);
    }

    for(int i=0;i<n;i++){
        float dx=0.0f, dy=0.0f;
        status[i]=1;

        for(int l=PYR_LEVELS-1; l>=0; l--){
            float scale=(float)(1<<l);
            float u=pts[i].x/scale;
            float v=pts[i].y/scale;

            // upscale guess when going to finer level
            if(l != PYR_LEVELS-1){ dx*=2.0f; dy*=2.0f; }

            int r=WINSZ/2;
            if(u<r+2 || v<r+2 || u>W[l]-r-3 || v>H[l]-r-3){
                status[i]=0; break;
            }

            lk_refine_level(pyr1[l],pyr2[l],W[l],H[l],gx[l],gy[l],u,v,&dx,&dy);
        }

        flow[i]=(P){dx,dy}; // full-res flow
    }

    for(int l=0;l<PYR_LEVELS;l++){ free(gx[l]); free(gy[l]); }
}

/* ---------------- Apply flow to points ---------------- */
static void apply_flow(const P* pts,const P* flow,int n,P* out){
    for(int i=0;i<n;i++){
        out[i].x = pts[i].x + flow[i].x;
        out[i].y = pts[i].y + flow[i].y;
    }
}

/* ---------------- Arrow drawing (thick + head + capped length) ---------------- */
static void draw_pixel(unsigned char* img,int w,int h,int x,int y){
    for(int dy=-1;dy<=1;dy++)
    for(int dx=-1;dx<=1;dx++){
        int xx=x+dx, yy=y+dy;
        if((unsigned)xx<(unsigned)w && (unsigned)yy<(unsigned)h){
            int k=(yy*w+xx)*3;
            img[k]=255; img[k+1]=0; img[k+2]=0;
        }
    }
}

static void draw_arrow(unsigned char* img,int w,int h,
                       float x0,float y0,float x1,float y1)
{
    float vx = x1-x0, vy = y1-y0;
    float L = hypotf(vx,vy);
    if(L < 0.5f) return;

    // cap arrow length so outliers don't wreck the plot
    float max_len = 12.0f;
    if(L > max_len){
        vx = vx / L * max_len;
        vy = vy / L * max_len;
        x1 = x0 + vx;
        y1 = y0 + vy;
        L = max_len;
    }

    int steps = (int)L;
    for(int i=0;i<=steps;i++){
        float t=(float)i/(float)steps;
        int x=(int)(x0 + t*(x1-x0));
        int y=(int)(y0 + t*(y1-y0));
        draw_pixel(img,w,h,x,y);
    }

    float ang = atan2f(y1-y0, x1-x0);
    float head = 6.0f;
    float a1 = ang + 2.6f;
    float a2 = ang - 2.6f;

    for(int i=0;i<=(int)head;i++){
        float t=(float)i/head;
        int xA=(int)(x1 + t*(head*cosf(a1)));
        int yA=(int)(y1 + t*(head*sinf(a1)));
        int xB=(int)(x1 + t*(head*cosf(a2)));
        int yB=(int)(y1 + t*(head*sinf(a2)));
        draw_pixel(img,w,h,xA,yA);
        draw_pixel(img,w,h,xB,yB);
    }
}

/* ---------------- MAIN ---------------- */
int main(int argc,char** argv){
    if(argc<3){ printf("Usage: %s img1 img2\n",argv[0]); return 1; }

    int w1,h1,w2,h2;
    unsigned char* rgb1=load_jpeg(argv[1],&w1,&h1);
    unsigned char* rgb2=load_jpeg(argv[2],&w2,&h2);
    if(w1!=w2 || h1!=h2){ printf("Size mismatch\n"); return 1; }

    float* g1=to_gray(rgb1,w1,h1);
    float* g2=to_gray(rgb2,w1,h1);

    // build pyramids
    float* pyr1[PYR_LEVELS];
    float* pyr2[PYR_LEVELS];
    int W[PYR_LEVELS], H[PYR_LEVELS];

    pyr1[0]=g1; pyr2[0]=g2; W[0]=w1; H[0]=h1;
    gauss3(pyr1[0],W[0],H[0]);
    gauss3(pyr2[0],W[0],H[0]);

    for(int l=1;l<PYR_LEVELS;l++){
        pyr1[l]=down2_gauss(pyr1[l-1],W[l-1],H[l-1],&W[l],&H[l]);
        pyr2[l]=down2_gauss(pyr2[l-1],W[l-1],H[l-1],&W[l],&H[l]);
    }

    // corners at full res
    P pts[MAX_PTS];
    int n=shi_tomasi(pyr1[0],W[0],H[0],pts);
    printf("Corners: %d\n", n);

    // forward flow
    P flow12[MAX_PTS];
    unsigned char st12[MAX_PTS];
    lk_pyramidal(pyr1,pyr2,W,H,pts,n,flow12,st12);

    // backward flow from predicted points
    P pts2[MAX_PTS];
    apply_flow(pts,flow12,n,pts2);

    P flow21[MAX_PTS];
    unsigned char st21[MAX_PTS];
    lk_pyramidal(pyr2,pyr1,W,H,pts2,n,flow21,st21);

    // draw
    float fb_thresh = 5.0f;    // loosened to match non-OpenCV accuracy
    float draw_scale = 1.0f;   // increase/decrease visualization

    int kept=0;
    for(int i=0;i<n;i++){
        int x=(int)(pts[i].x+0.5f), y=(int)(pts[i].y+0.5f);
        if((unsigned)x<(unsigned)w1 && (unsigned)y<(unsigned)h1){
            int k=(y*w1+x)*3;
            rgb1[k]=0; rgb1[k+1]=255; rgb1[k+2]=0; // green point
        }

        if(!st12[i] || !st21[i]) continue;

        float mag = hypotf(flow12[i].x, flow12[i].y);
        if(mag < 0.2f || mag > 30.0f) continue;

        // forward-backward error
        float rx = pts2[i].x + flow21[i].x;
        float ry = pts2[i].y + flow21[i].y;
        float err = hypotf(rx - pts[i].x, ry - pts[i].y);
        if(err > fb_thresh) continue;

        draw_arrow(rgb1,w1,h1,
                   pts[i].x, pts[i].y,
                   pts[i].x + flow12[i].x * draw_scale,
                   pts[i].y + flow12[i].y * draw_scale);
        kept++;
    }

    printf("Kept arrows: %d\n", kept);

    FILE* f=fopen("flow.ppm","wb");
    fprintf(f,"P6\n%d %d\n255\n",w1,h1);
    fwrite(rgb1,1,(size_t)w1*h1*3,f);
    fclose(f);

    for(int l=1;l<PYR_LEVELS;l++){ free(pyr1[l]); free(pyr2[l]); }
    free(rgb1); free(rgb2); free(g1); free(g2);

    printf("Saved flow.ppm\n");
    return 0;
}