#include "lab1.h"
#include <iostream>
#include <cuComplex.h>
static const unsigned DIMX = 1000;
static const unsigned DIMY = 1000;
static const int BLOCKX = 32;
static const int BLOCKY = 32;
static const unsigned NFRAME = 500;
using namespace std ;
#define DIM 1000

struct Lab1VideoGenerator::Impl {
    int iter=0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = DIMX;
	info.h = DIMY;
	info.n_frame = NFRAME;
	info.fps_n = 10;
	info.fps_d = 1;
};


__device__ cuDoubleComplex my_complex_exp(cuDoubleComplex arg)
{
    cuDoubleComplex res;
    double s, c;
    double e = exp(arg.x);
    sincos(arg.y, &s, &c);
    res.x = c * e;
    res.y = s * e;
    return res;
}


__device__ void RGBtoYUV(float *RGBcolor, float *YUVcolor){
    
    YUVcolor[0] = (0.229 * RGBcolor[0]) + (0.587 * RGBcolor[1]) + (0.114 * RGBcolor[2]);
    YUVcolor[1] = -(0.169 * RGBcolor[0]) - (0.331 * RGBcolor[1]) + (0.500 * RGBcolor[2]) + 128;
    YUVcolor[2] =  (0.500 * RGBcolor[0]) - (0.419 * RGBcolor[1]) - (0.081 * RGBcolor[2]) + 128;
}

__global__ void kernel(uint8_t *yuv,cuDoubleComplex c,double zoom,int iTime) {

    cuDoubleComplex ec = my_complex_exp(c);
    ec = cuCmul(make_cuDoubleComplex(0.7885,0),ec);
    int maxiter=256;
    // map from blockIdx to pixel position
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    //int y1 = DIM-y;

    
    if(x<DIM && y<DIM){
        double newRe = 1.5*(x - DIM / 2) / (0.5 * zoom * DIM);
        double newIm = 1.5*(y - DIM/ 2) / (0.5 * zoom * DIM);
        int it;
        
        cuDoubleComplex z = make_cuDoubleComplex(newRe, newIm);
        double smoothcolor = exp(-cuCabs(z));
        for(it = 0; it <maxiter; it++)
        {
            cuDoubleComplex temp =cuCmul(z,z);
            z =cuCmul(z,temp);
            z = cuCadd(z,ec);
            if(cuCabs(z)>4) break;
        }
        
        float  RGBcolor[3];
        float YUVcolor[3];
        if(it==maxiter){
                RGBcolor[2]= 0;
                RGBcolor[1]= 0;
                RGBcolor[0] = 0;
                RGBtoYUV(RGBcolor,YUVcolor);
                yuv[y*DIM+x]=YUVcolor[0];
                if(x%2==0 && y%2==0){
                    yuv[DIM*DIM+(y*DIM)/4+x/2]= YUVcolor[1];
                    yuv[DIM*DIM+DIM*DIM/4+y*DIM/4+x/2]= YUVcolor[2]; 
                }           
        }
        if(it==0){
                RGBcolor[2]= 0;
                RGBcolor[1]= 255;
                RGBcolor[0] = 255;
                RGBtoYUV(RGBcolor,YUVcolor);
                yuv[y*DIM+x]=YUVcolor[0];
                if(x%2==0 && y%2==0){
                    yuv[DIM*DIM+(y*DIM)/4+x/2]= YUVcolor[1];
                    yuv[DIM*DIM+DIM*DIM/4+y*DIM/4+x/2]= YUVcolor[2]; 
                }
        }else if(it==1){
                RGBcolor[2]= 0;
                RGBcolor[1]= 203;
                RGBcolor[0] = 136;
                RGBtoYUV(RGBcolor,YUVcolor);
                yuv[y*DIM+x]=YUVcolor[0];
                if(x%2==0 && y%2==0){
                    yuv[DIM*DIM+(y*DIM)/4+x/2]= YUVcolor[1];
                    yuv[DIM*DIM+DIM*DIM/4+y*DIM/4+x/2]= YUVcolor[2]; 
                }
        }else if(it==2){
                RGBcolor[2]= 0;
                RGBcolor[1]= 128;
                RGBcolor[0] = 0;
                RGBtoYUV(RGBcolor,YUVcolor);
                yuv[y*DIM+x]=YUVcolor[0];
                if(x%2==0 && y%2==0){
                    yuv[DIM*DIM+(y*DIM)/4+x/2]= YUVcolor[1];
                    yuv[DIM*DIM+DIM*DIM/4+y*DIM/4+x/2]= YUVcolor[2]; 
                }
        }else if(it<=4){
                RGBcolor[2]= 0;
                RGBcolor[1]= 255;
                RGBcolor[0] = 0;
                RGBtoYUV(RGBcolor,YUVcolor);
                yuv[y*DIM+x]=YUVcolor[0];
                if(x%2==0 && y%2==0){
                    yuv[DIM*DIM+(y*DIM)/4+x/2]= YUVcolor[1];
                    yuv[DIM*DIM+DIM*DIM/4+y*DIM/4+x/2]= YUVcolor[2]; 
                }
        }else if(it<=8){
                RGBcolor[2]= 255;
                RGBcolor[1]= 255;
                RGBcolor[0] = 255;
                RGBtoYUV(RGBcolor,YUVcolor);
                yuv[y*DIM+x]=YUVcolor[0];
                if(x%2==0 && y%2==0){
                    yuv[DIM*DIM+(y*DIM)/4+x/2]= YUVcolor[1];
                    yuv[DIM*DIM+DIM*DIM/4+y*DIM/4+x/2]= YUVcolor[2]; 
                }
        }else if(it<=16){
                RGBcolor[2]= 256;
                RGBcolor[1]= 149;
                RGBcolor[0] = 245;
                RGBtoYUV(RGBcolor,YUVcolor);
                yuv[y*DIM+x]=YUVcolor[0];
                if(x%2==0 && y%2==0){
                    yuv[DIM*DIM+(y*DIM)/4+x/2]= YUVcolor[1];
                    yuv[DIM*DIM+DIM*DIM/4+y*DIM/4+x/2]= YUVcolor[2]; 
                }
        }else if(it< maxiter/4){
                RGBcolor[2]= 128;
                RGBcolor[1]= 0;
                RGBcolor[0] = 128;
                RGBtoYUV(RGBcolor,YUVcolor);
                yuv[y*DIM+x]=YUVcolor[0];
                if(x%2==0 && y%2==0){
                    yuv[DIM*DIM+(y*DIM)/4+x/2]= YUVcolor[1];
                    yuv[DIM*DIM+DIM*DIM/4+y*DIM/4+x/2]= YUVcolor[2]; 
                }
        }else if(it< maxiter/2){
                RGBcolor[2]= 128;
                RGBcolor[1]= 0;
                RGBcolor[0] = 0;
                RGBtoYUV(RGBcolor,YUVcolor);
                yuv[y*DIM+x]=YUVcolor[0];
                if(x%2==0 && y%2==0){
                    yuv[DIM*DIM+(y*DIM)/4+x/2]= YUVcolor[1];
                    yuv[DIM*DIM+DIM*DIM/4+y*DIM/4+x/2]= YUVcolor[2]; 
                }
        }else if(it< maxiter){
                RGBcolor[2]= 0;
                RGBcolor[1]= 0;
                RGBcolor[0] = 255;
                RGBtoYUV(RGBcolor,YUVcolor);
                yuv[y*DIM+x]=YUVcolor[0];
                if(x%2==0 && y%2==0){
                    yuv[DIM*DIM+(y*DIM)/4+x/2]= YUVcolor[1];
                   yuv[DIM*DIM+DIM*DIM/4+y*DIM/4+x/2]= YUVcolor[2]; 
                }
        }

    }
        
}

int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

void Lab1VideoGenerator::Generate(uint8_t *yuv) {

    cudaMemset(yuv,255, DIM*DIM);
    cudaMemset(yuv+DIM*DIM, 128, DIM*DIM/2);
    //double cRe=-0.7;
    double cIm=0.5;
    double zoom=1;
    zoom = zoom + impl->iter * (0.001);

    //cRe = cRe+impl->iter*(0.000003);
    cIm = cIm+impl->iter*(0.012566);
    
    cuDoubleComplex c = make_cuDoubleComplex(0, cIm);
    
    dim3 bs(BLOCKX,BLOCKY);
    dim3 gs(divup(DIM, bs.x), divup(DIM, bs.y));
    kernel <<<gs,bs>>>(yuv,c,zoom,impl->iter);

    ++impl->iter;
}
