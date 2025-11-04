"""
CUDA Kernels for JetsonSky Image Processing

This module contains all CUDA/CuPy kernel definitions used for GPU-accelerated
image processing operations. These kernels are written in CUDA C and compiled
at runtime by CuPy.

Kernel Categories:
- Frame Noise Reduction (FNR, FNR2)
- HDR Processing
- Binning
- RGB Alignment
- Debayering
- Dead Pixel Removal
- Contrast Enhancement
- Noise Reduction
- Saturation Adjustment
- Gaussian Filtering
- Color Conversions
- Amplification
- Star Amplification
- Adaptive Denoising (AANR)
- Variation Reduction
- Histogram Operations
- Non-Local Means (NLM2)
- K-Nearest Neighbors (KNN)
"""

import cupy as cp

FNR_Mono = cp.RawKernel(r'''
extern "C" __global__
void FNR_Mono_C(unsigned char *dest_b, unsigned char *im1_b, unsigned char *im2_b, unsigned char *im3_b,long int width, long int height, float val_3FNR_Thres)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  char inv_pente_b;
  char inflex_b;
  char old_inflex_b;
 
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
    if ((im1_b[index]-im2_b[index])*(im2_b[index]-im3_b[index]) <= 0) {
        inv_pente_b = 1;
    }
    else {
        inv_pente_b = 0;
    }

    if (abs(im3_b[index]-im2_b[index]) > (val_3FNR_Thres * im3_b[index])) {
        inflex_b = 1;
    }
    else {
        inflex_b = 0;
    }


    if (abs(im2_b[index]-im1_b[index]) > (val_3FNR_Thres * im2_b[index])) {
        old_inflex_b = 1;
    }
    else {
        old_inflex_b = 0;
    }


    if (inflex_b == 1) {
        dest_b[index] = im3_b[index];
    }


    if (inflex_b == 0 && old_inflex_b == 1) {
        dest_b[index] = (int)((im3_b[index]+im2_b[index]) / 2);
    }

    if (inflex_b == 0 && old_inflex_b == 0) {
        if (inv_pente_b == 1) {
            dest_b[index] = (int)((im3_b[index] + im2_b[index] + im1_b[index]) / 3);
        }
        else {
            dest_b[index] = (int)(im1_b[index] + (im2_b[index] - im1_b[index]) / 2 + (im3_b[index] - im2_b[index]) / 2);
        }
    }    
  }
}
''', 'FNR_Mono_C')


FNR_Color = cp.RawKernel(r'''
extern "C" __global__
void FNR_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *im1_r, unsigned char *im1_g, unsigned char *im1_b, unsigned char *im2_r, unsigned char *im2_g, unsigned char *im2_b,
unsigned char *im3_r, unsigned char *im3_g, unsigned char *im3_b,long int width, long int height, float val_3FNR_Thres)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  char inv_pente_r,inv_pente_g,inv_pente_b;
  char inflex_r,inflex_g,inflex_b;
  char old_inflex_r,old_inflex_g,old_inflex_b;
 
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
  
    if ((im1_r[index]-im2_r[index])*(im2_r[index]-im3_r[index]) <= 0) {
        inv_pente_r = 1;
    }
    else {
        inv_pente_r = 0;
    }
    if ((im1_g[index]-im2_g[index])*(im2_g[index]-im3_g[index]) <= 0) {
        inv_pente_g = 1;
    }
    else {
        inv_pente_g = 0;
    }
    if ((im1_b[index]-im2_b[index])*(im2_b[index]-im3_b[index]) <= 0) {
        inv_pente_b = 1;
    }
    else {
        inv_pente_b = 0;
    }

    if (abs(im3_r[index]-im2_r[index]) > (val_3FNR_Thres * im3_r[index])) {
        inflex_r = 1;
    }
    else {
        inflex_r = 0;
    }
    if (abs(im3_g[index]-im2_g[index]) > (val_3FNR_Thres * im3_g[index])) {
        inflex_g = 1;
    }
    else {
        inflex_g = 0;
    }
    if (abs(im3_b[index]-im2_b[index]) > (val_3FNR_Thres * im3_b[index])) {
        inflex_b = 1;
    }
    else {
        inflex_b = 0;
    }


    if (abs(im2_r[index]-im1_r[index]) > (val_3FNR_Thres * im2_r[index])) {
        old_inflex_r = 1;
    }
    else {
        old_inflex_r = 0;
    }
    if (abs(im2_g[index]-im1_g[index]) > (val_3FNR_Thres * im2_g[index])) {
        old_inflex_g = 1;
    }
    else {
        old_inflex_g = 0;
    }
    if (abs(im2_b[index]-im1_b[index]) > (val_3FNR_Thres * im2_b[index])) {
        old_inflex_b = 1;
    }
    else {
        old_inflex_b = 0;
    }


    if (inflex_r == 1) {
        dest_r[index] = im3_r[index];
    }
    if (inflex_g == 1) {
        dest_g[index] = im3_g[index];
    }
    if (inflex_r == 1) {
        dest_b[index] = im3_b[index];
    }


    if (inflex_r == 0 && old_inflex_r == 1) {
        dest_r[index] = (int)((im3_r[index]+im2_r[index]) / 2);
    }
    if (inflex_g == 0 && old_inflex_g == 1) {
        dest_g[index] = (int)((im3_g[index]+im2_g[index]) / 2);
    }
    if (inflex_b == 0 && old_inflex_b == 1) {
        dest_b[index] = (int)((im3_b[index]+im2_b[index]) / 2);
    }


    if (inflex_r == 0 && old_inflex_r == 0) {
        if (inv_pente_r == 1) {
            dest_r[index] = (int)((im3_r[index] + im2_r[index] + im1_r[index]) / 3);
        }
        else {
            dest_r[index] = (int)(im1_r[index] + (im2_r[index] - im1_r[index]) / 2 + (im3_r[index] - im2_r[index]) / 2);
        }
    }
    if (inflex_g == 0 && old_inflex_g == 0) {
        if (inv_pente_g == 1) {
            dest_g[index] = (int)((im3_g[index] + im2_g[index] + im1_g[index]) / 3);
        }
        else {
            dest_g[index] = (int)(im1_g[index] + (im2_g[index] - im1_g[index]) / 2 + (im3_g[index] - im2_g[index]) / 2);
        }
    }
    if (inflex_b == 0 && old_inflex_b == 0) {
        if (inv_pente_b == 1) {
            dest_b[index] = (int)((im3_b[index] + im2_b[index] + im1_b[index]) / 3);
        }
        else {
            dest_b[index] = (int)(im1_b[index] + (im2_b[index] - im1_b[index]) / 2 + (im3_b[index] - im2_b[index]) / 2);
        }
    }    
  }
}
''', 'FNR_Color_C')


FNR2_Color = cp.RawKernel(r'''
extern "C" __global__
void FNR2_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *im1_r, unsigned char *im1_g, unsigned char *im1_b, unsigned char *im2_r, unsigned char *im2_g, unsigned char *im2_b,
unsigned char *im3_r, unsigned char *im3_g, unsigned char *im3_b,long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int D1r,D1g,D1b;
  int D2r,D2g,D2b;
  int Delta_r,Delta_g,Delta_b;
  
 
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
    D1r = im2_r[index] - im1_r[index];
    D1g = im2_g[index] - im1_g[index];
    D1b = im2_b[index] - im1_b[index];
    D2r = im3_r[index] - im2_r[index];
    D2g = im3_g[index] - im2_g[index];
    D2b = im3_b[index] - im2_b[index];
  
    if ((D1r*D2r) < 0) {
        Delta_r = (D1r + D2r) / (2.5 - abs(D2r)/255.0);
    }
    else {
        Delta_r = (D1r + D2r) / 2.0;
    }
    if ((D1g*D2g) < 0) {
        Delta_g = (D1g + D2g) / (2.5 - abs(D2g)/255.0);
    }
    else {
        Delta_g = (D1g + D2g) / 2.0;
    }
    if ((D1b*D2b) < 0) {
        Delta_b = (D1b + D2b) / (2.5 - abs(D2b)/255.0);
    }
    else {
        Delta_b = (D1b + D2b) / 2.0;
    }
    if (abs(D2r) > 40) {
        dest_r[index] = im3_r[index];
    }
    else {
        dest_r[index] = (int)(min(max(int((im1_r[index] + im2_r[index]) / 2.0 + Delta_r), 0), 255));
    }
    if (abs(D2g) > 40) {
        dest_g[index] = im3_g[index];
    }
    else {
        dest_g[index] = (int)(min(max(int((im1_g[index] + im2_g[index]) / 2.0 + Delta_g), 0), 255));
    }
    if (abs(D2b) > 40) {
        dest_b[index] = im3_b[index];
    }
    else {
        dest_b[index] = (int)(min(max(int((im1_b[index] + im2_b[index]) / 2.0 + Delta_b), 0), 255));
    }
  }
}
''', 'FNR2_Color_C')

FNR2_Mono = cp.RawKernel(r'''
extern "C" __global__
void FNR2_Mono_C(unsigned char *dest_b, unsigned char *im1_b, unsigned char *im2_b, unsigned char *im3_b,long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int D1b;
  int D2b;
  int Delta_b;
  
 
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
    D1b = im2_b[index] - im1_b[index];
    D2b = im3_b[index] - im2_b[index];
  
    if ((D1b*D2b) < 0) {
        Delta_b = (D1b + D2b) / (2.5 - abs(D2b)/255.0);
    }
    else {
        Delta_b = (D1b + D2b) / 2.0;
    }
    if (abs(D2b) > 40) {
        dest_b[index] = im3_b[index];
    }
    else {
        dest_b[index] = (int)(min(max(int((im1_b[index] + im2_b[index]) / 2.0 + Delta_b), 0), 255));
    }
  }
}
''', 'FNR2_Mono_C')


HDR_compute_GPU = cp.RawKernel(r'''
extern "C" __global__
void HDR_compute_GPU_C(unsigned char *dest_r, unsigned short int *img_r, long int width, long int height, float thres_16b, char method, char BIN_mode, char Hard_BIN)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int image_brute_16;
  unsigned int image_brute1,image_brute2,image_brute3,image_brute4;
  float delta_th,thres1,thres2,thres3,thres4;
  
  index = i * width + j;

  if (i < height && j < width) {
  
      if ((16.0 - thres_16b) <= 5.0) {
          delta_th = (16.0 - thres_16b) / 3.0;
      }
      else {
          delta_th = 5.0 / 3.0;
      }
      thres4 = __powf(2,thres_16b) - 1;
      thres3 = __powf(2,thres_16b + delta_th) - 1;
      thres2 = __powf(2,thres_16b + delta_th * 2) - 1;
      thres1 = __powf(2,thres_16b + delta_th * 3) - 1;

      image_brute_16 = img_r[index];
      if (image_brute_16 > thres1) {
          image_brute_16 = thres1;
      }
      if (BIN_mode == 2) {
          if (Hard_BIN == 0) {
              image_brute1 = (int)((image_brute_16 / thres1 * 255.0)*4.0);
          }
          else {
              image_brute1 = (int)(image_brute_16 / thres1 * 255.0);
          }
      }
      else {
          image_brute1 = (int)(image_brute_16 / thres1 * 255.0);
      }
      image_brute1 = (int)(min(max(image_brute1, 0), 255));


      image_brute_16 = img_r[index];
      if (image_brute_16 > thres2) {
          image_brute_16 = thres2;
      }
      if (BIN_mode == 2) {
          if (Hard_BIN == 0) {
              image_brute2 = (int)((image_brute_16 / thres2 * 255.0)*4.0);
              image_brute2 = (int)(min(max(image_brute2, 0), 255));
          }
          else {
              image_brute2 = (int)(image_brute_16 / thres2 * 255.0);
          }
      }
      else {
          image_brute2 = (int)(image_brute_16 / thres2 * 255.0);
      }
      image_brute2 = (int)(min(max(image_brute2, 0), 255));

      image_brute_16 = img_r[index];
      if (image_brute_16 > thres3) {
          image_brute_16 = thres3;
      }
      if (BIN_mode == 2) {
          if (Hard_BIN == 0) {
              image_brute3 = (int)((image_brute_16 / thres3 * 255.0)*4.0);
              image_brute3 = (int)(min(max(image_brute3, 0), 255));
          }
          else {
              image_brute3 = (int)(image_brute_16 / thres3 * 255.0);
          }
      }
      else {
          image_brute3 = (int)(image_brute_16 / thres3 * 255.0);
      }
      image_brute3 = (int)(min(max(image_brute3, 0), 255));

      image_brute_16 = img_r[index];
      if (image_brute_16 > thres4) {
          image_brute_16 = thres4;
      }
      if (BIN_mode == 2) {
          if (Hard_BIN == 0) {
              image_brute4 = (int)((image_brute_16 / thres4 * 255.0)*4.0);
              image_brute4 = (int)(min(max(image_brute4, 0), 255));
          }
          else {
              image_brute4 = (int)(image_brute_16 / thres4 * 255.0);
          }
      }
      else {
          image_brute4 = (int)(image_brute_16 / thres4 * 255.0);
      }
      image_brute4 = (int)(min(max(image_brute4, 0), 255));

    dest_r[index] = (int)((image_brute1 + image_brute2 + image_brute3 + image_brute4)/4.0);
    dest_r[index] = (int)(min(max(dest_r[index], 0), 255));
    } 
}
''', 'HDR_compute_GPU_C')



BIN_Color_GPU = cp.RawKernel(r'''
extern "C" __global__
void BIN_Color_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, int BIN_mode)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index,i1,i2,i3,i4;
  int tmp_r,tmp_g,tmp_b;

  index = i * width + j;
  
  i1 = (i * 2) * (width * 2) + (j * 2);
  i2 = (i * 2) * (width * 2) + (j * 2 + 1);
  i3 = (i * 2 + 1) * (width * 2) + (j * 2);
  i4 = (i * 2 + 1) * (width * 2) + (j * 2 + 1);
  
  if (i < height && i > 0 && j < width && j > 0) {
      if (BIN_mode == 0) {
          tmp_r = img_r[i1] + img_r[i2] + img_r[i3] + img_r[i4];  
          tmp_g = img_g[i1] + img_g[i2] + img_g[i3] + img_g[i4];  
          tmp_b = img_b[i1] + img_b[i2] + img_b[i3] + img_b[i4];

          dest_r[index] = (int)(min(max(tmp_r, 0), 255));
          dest_g[index] = (int)(min(max(tmp_g, 0), 255));
          dest_b[index] = (int)(min(max(tmp_b, 0), 255));
          }
      else {
          dest_r[index] = (img_r[i1] + img_r[i2] + img_r[i3] + img_r[i4]) / 4;  
          dest_g[index] = (img_g[i1] + img_g[i2] + img_g[i3] + img_g[i4]) / 4;  
          dest_b[index] = (img_b[i1] + img_b[i2] + img_b[i3] + img_b[i4]) / 4;
          }
    }
}
''', 'BIN_Color_GPU_C')

BIN_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void BIN_Mono_GPU_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height, int BIN_mode)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index,i1,i2,i3,i4;
  int tmp_r;

  index = i * width + j;
  
  i1 = (i * 2) * (width * 2) + (j * 2);
  i2 = (i * 2) * (width * 2) + (j * 2 + 1);
  i3 = (i * 2 + 1) * (width * 2) + (j * 2);
  i4 = (i * 2 + 1) * (width * 2) + (j * 2 + 1);
  
  if (i < height && i > 0 && j < width && j > 0) {
      if (BIN_mode == 0) {
          tmp_r = img_r[i1] + img_r[i2] + img_r[i3] + img_r[i4];  

          dest_r[index] = (int)(min(max(tmp_r, 0), 255));
          }
      else {
          dest_r[index] = (img_r[i1] + img_r[i2] + img_r[i3] + img_r[i4]) / 4;  
          }
    }
}
''', 'BIN_Mono_GPU_C')


RGB_Align_GPU = cp.RawKernel(r'''
extern "C" __global__
void RGB_Align_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height,
long int delta_RX, long int delta_RY, long int delta_BX, long int delta_BY)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index,indexR,indexB,iR,jR,iB,jB;
  
  index = i * width + j;
  indexR = (i + delta_RY) * width + j + delta_RX;
  indexB = (i + delta_BY) * width + j + delta_BX;
  iR = i + delta_RY;
  jR = j + delta_RX;
  iB = i + delta_BY;
  jB = j + delta_BX;

  if (i < height && j < width) {
      if (iR > 0 && iR< height && jR > 0 && jR < width && iB > 0 && iB< height && jB > 0 && jB < width) {
        dest_r[index] = img_r[indexR];
        dest_g[index] = img_g[index];
        dest_b[index] = img_b[indexB];
        }
      else {
        dest_r[index] = 0;
        dest_g[index] = 0;
        dest_b[index] = 0;
        }
    } 
}
''', 'RGB_Align_GPU_C')

Image_Debayer_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void Image_Debayer_Mono_GPU_C(unsigned char *dest_r, unsigned char *img, long int width, long int height, int GPU_BAYER)
{

  long int j = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
  long int i = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
  long int i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16;
  int r1,r2,r3,r4,g1,g2,g3,g4,b1,b2,b3,b4;
  float att;
  
  i1 = i * width + j;
  i2 = i * width + j+1;
  i3 = (i+1) * width + j;
  i4 = (i+1) * width + j+1;
  i5 = (i-1) * width + j-1;
  i6 = (i-1) * width + j;
  i7 = (i-1) * width + j+1;
  i8 = (i-1) * width + j+2;
  i9 = i * width + j+2;
  i10 = (i+1) * width + j+2;
  i11 = (i+2) * width + j+2;
  i12 = (i+2) * width + j+1;
  i13 = (i+2) * width + j;
  i14 = (i+2) * width + j-1;
  i15 = (i+1) * width + j-1;
  i16 = i * width + j-1;
  att = 1 / 4.0;
  
  if (i < (height-1) && i > 0 && j < (width-1) && j > 0) {
      if (GPU_BAYER == 1) {
// RGGB
          r1=img[i1];  
          g1=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
          b1=min(img[i4],img[i5])+int(att*abs(img[i4]-img[i5]));
      
          r2=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          g2=img[i2];
          b2=min(img[i4],img[i7])+int(att*abs(img[i4]-img[i7]));

          r3=min(img[i1],img[i13])+int(att*abs(img[i1]-img[i13]));
          g3=img[i3];
          b3=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));

          r4=min(img[i1],img[i11])+int(att*abs(img[i1]-img[i11]));
          g4=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          b4=img[i4];
          }
// BGGR
      if (GPU_BAYER == 2) {
          b1=img[i1];  
          g1=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
          r1=min(img[i4],img[i5])+int(att*abs(img[i4]-img[i5]));
      
          b2=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          g2=img[i2];
          r2=min(img[i4],img[i7])+int(att*abs(img[i4]-img[i7]));

          b3=min(img[i1],img[i13])+int(att*abs(img[i1]-img[i13]));
          g3=img[i3];
          r3=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));

          b4=min(img[i1],img[i11])+int(att*abs(img[i1]-img[i11]));
          g4=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          r4=img[i4];
          }
// GBRG
      if (GPU_BAYER == 3) {
          r1=min(img[i3],img[i6])+int(att*abs(img[i3]-img[i6]));
          g1=img[i1];
          b1=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
      
          r2=min(img[i3],img[i8])+int(att*abs(img[i3]-img[i8]));
          g2=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          b2=img[i2];

          r3=img[i3];
          g3=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));
          b3=min(img[i14],img[i2])+int(att*abs(img[i14]-img[i2]));

          r4=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          g4=img[i4];
          b4=min(img[i12],img[i2])+int(att*abs(img[i12]-img[i2]));
          }
// GRBG
      if (GPU_BAYER == 4) {
          b1=min(img[i3],img[i6])+int(att*abs(img[i3]-img[i6]));
          g1=img[i1];
          r1=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
      
          b2=min(img[i3],img[i8])+int(att*abs(img[i3]-img[i8]));
          g2=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          r2=img[i2];

          b3=img[i3];
          g3=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));
          r3=min(img[i14],img[i2])+int(att*abs(img[i14]-img[i2]));

          b4=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          g4=img[i4];
          r4=min(img[i12],img[i2])+int(att*abs(img[i12]-img[i2]));
          }
      dest_r[i1] = (int)(min(max(int(0.299*r1 + 0.587*g1 + 0.114*b1), 0), 255));
      dest_r[i2] = (int)(min(max(int(0.299*r2 + 0.587*g2 + 0.114*b2), 0), 255));
      dest_r[i3] = (int)(min(max(int(0.299*r3 + 0.587*g3 + 0.114*b3), 0), 255));
      dest_r[i4] = (int)(min(max(int(0.299*r4 + 0.587*g4 + 0.114*b4), 0), 255));
    }
}
''', 'Image_Debayer_Mono_GPU_C')

Image_Debayer_GPU = cp.RawKernel(r'''
extern "C" __global__
void Image_Debayer_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img, long int width, long int height, int GPU_BAYER)
{

  long int j = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
  long int i = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
  long int i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16;
  float att;
  
  i1 = i * width + j;
  i2 = i * width + j+1;
  i3 = (i+1) * width + j;
  i4 = (i+1) * width + j+1;
  i5 = (i-1) * width + j-1;
  i6 = (i-1) * width + j;
  i7 = (i-1) * width + j+1;
  i8 = (i-1) * width + j+2;
  i9 = i * width + j+2;
  i10 = (i+1) * width + j+2;
  i11 = (i+2) * width + j+2;
  i12 = (i+2) * width + j+1;
  i13 = (i+2) * width + j;
  i14 = (i+2) * width + j-1;
  i15 = (i+1) * width + j-1;
  i16 = i * width + j-1;
  att = 1 / 4.0;
  
  if (i < (height-1) && i > 0 && j < (width-1) && j > 0) {
// RGGB
      if (GPU_BAYER == 1) {
          dest_r[i1]=img[i1];  
          dest_g[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
          dest_b[i1]=min(img[i4],img[i5])+int(att*abs(img[i4]-img[i5]));
      
          dest_r[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_g[i2]=img[i2];
          dest_b[i2]=min(img[i4],img[i7])+int(att*abs(img[i4]-img[i7]));

          dest_r[i3]=min(img[i1],img[i13])+int(att*abs(img[i1]-img[i13]));
          dest_g[i3]=img[i3];
          dest_b[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));

          dest_r[i4]=min(img[i1],img[i11])+int(att*abs(img[i1]-img[i11]));
          dest_g[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_b[i4]=img[i4];
          }
// BGGR
      if (GPU_BAYER == 2) {
          dest_b[i1]=img[i1];  
          dest_g[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
          dest_r[i1]=min(img[i4],img[i5])+int(att*abs(img[i4]-img[i5]));
      
          dest_b[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_g[i2]=img[i2];
          dest_r[i2]=min(img[i4],img[i7])+int(att*abs(img[i4]-img[i7]));

          dest_b[i3]=min(img[i1],img[i13])+int(att*abs(img[i1]-img[i13]));
          dest_g[i3]=img[i3];
          dest_r[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));

          dest_b[i4]=min(img[i1],img[i11])+int(att*abs(img[i1]-img[i11]));
          dest_g[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_r[i4]=img[i4];
          }
// GBRG
      if (GPU_BAYER == 3) {
          dest_r[i1]=min(img[i3],img[i6])+int(att*abs(img[i3]-img[i6]));
          dest_g[i1]=img[i1];
          dest_b[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
      
          dest_r[i2]=min(img[i3],img[i8])+int(att*abs(img[i3]-img[i8]));
          dest_g[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_b[i2]=img[i2];

          dest_r[i3]=img[i3];
          dest_g[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));
          dest_b[i3]=min(img[i14],img[i2])+int(att*abs(img[i14]-img[i2]));

          dest_r[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_g[i4]=img[i4];
          dest_b[i4]=min(img[i12],img[i2])+int(att*abs(img[i12]-img[i2]));
          }
// GRBG
      if (GPU_BAYER == 4) {
          dest_b[i1]=min(img[i3],img[i6])+int(att*abs(img[i3]-img[i6]));
          dest_g[i1]=img[i1];
          dest_r[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
      
          dest_b[i2]=min(img[i3],img[i8])+int(att*abs(img[i3]-img[i8]));
          dest_g[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_r[i2]=img[i2];

          dest_b[i3]=img[i3];
          dest_g[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));
          dest_r[i3]=min(img[i14],img[i2])+int(att*abs(img[i14]-img[i2]));

          dest_b[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_g[i4]=img[i4];
          dest_r[i4]=min(img[i12],img[i2])+int(att*abs(img[i12]-img[i2]));
          }
    }
}
''', 'Image_Debayer_GPU_C')

Dead_Pixels_Remove_Colour_GPU = cp.RawKernel(r'''
extern "C" __global__
void Dead_Pixels_Remove_Colour_C(unsigned char *dest, unsigned char *img, long int width, long int height, unsigned char Threshold, int GPU_BAYER)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16;
  int Delta1r, Delta2r;
  int Delta1g1, Delta2g1;
  int Delta1g2, Delta2g2;
  int Delta1b, Delta2b;

  i1 = i * width + j;
  i2 = i * width + j+1;
  i3 = (i+1) * width + j;
  i4 = (i+1) * width + j+1;

  if (i < (height-2) && i > 1 && j < (width-2) && j > 1) {
    if (GPU_BAYER == 1) {
        Delta1r = abs(img[i1] - img[i1-2]);  
        Delta2r = abs(img[i1] - img[i1+2]);
        Delta1g1 = abs(img[i2] - img[i2-2]);  
        Delta2g1 = abs(img[i2] - img[i2+2]);
        Delta1g2 = abs(img[i3] - img[i3-2]);  
        Delta2g2 = abs(img[i3] - img[i3+2]);
        Delta1b = abs(img[i4] - img[i4-2]);  
        Delta2b = abs(img[i4] - img[i4+2]);
        }
    if (GPU_BAYER == 2) {
        Delta1b = abs(img[i1] - img[i1-2]);  
        Delta2b = abs(img[i1] - img[i1+2]);
        Delta1g1 = abs(img[i2] - img[i2-2]);  
        Delta2g1 = abs(img[i2] - img[i2+2]);
        Delta1g2 = abs(img[i3] - img[i3-2]);  
        Delta2g2 = abs(img[i3] - img[i3+2]);
        Delta1r = abs(img[i4] - img[i4-2]);  
        Delta2r = abs(img[i4] - img[i4+2]);
        }
    if (GPU_BAYER == 3) {
        Delta1g1 = abs(img[i1] - img[i1-2]);  
        Delta2g1 = abs(img[i1] - img[i1+2]);
        Delta1b = abs(img[i2] - img[i2-2]);  
        Delta2b = abs(img[i2] - img[i2+2]);
        Delta1r = abs(img[i3] - img[i3-2]);  
        Delta2r = abs(img[i3] - img[i3+2]);
        Delta1g2 = abs(img[i4] - img[i4-2]);  
        Delta2g2 = abs(img[i4] - img[i4+2]);
        }
    if (GPU_BAYER == 4) {
        Delta1g1 = abs(img[i1] - img[i1-2]);  
        Delta2g1 = abs(img[i1] - img[i1+2]);
        Delta1r = abs(img[i2] - img[i2-2]);  
        Delta2r = abs(img[i2] - img[i2+2]);
        Delta1b = abs(img[i3] - img[i3-2]);  
        Delta2b = abs(img[i3] - img[i3+2]);
        Delta1g2 = abs(img[i4] - img[i4-2]);  
        Delta2g2 = abs(img[i4] - img[i4+2]);
        }
    if (GPU_BAYER > 0) {
        if ((Delta1r > Threshold) && (Delta2r > Threshold)) { 
            dest[i1] = int((img[i1-2] + img[i1+2]) / 2.0);
        }
        else {
            dest[i1] = img[i1];
        }

        if ((Delta1g1 > Threshold) && (Delta2g1 > Threshold)) { 
            dest[i2] = int((img[i2-2] + img[i2+2]) / 2.0);
        }
        else {
            dest[i2] = img[i2];
        }

        if ((Delta1g2 > Threshold) && (Delta2g2 > Threshold)) { 
            dest[i3] = int((img[i3-2] + img[i3+2]) / 2.0);
        }
        else {
            dest[i3] = img[i3];
        }

        if ((Delta1b > Threshold) && (Delta2b > Threshold)) { 
            dest[i4] = int((img[i4-2] + img[i4+2]) / 2.0);
        }
        else {
            dest[i4] = img[i4];
        }
      }
    else {
        dest[i1] = img[i1];
        dest[i2] = img[i2];
        dest[i3] = img[i3];
        dest[i4] = img[i4];
        }
    }
}
''', 'Dead_Pixels_Remove_Colour_C')

Dead_Pixels_Remove_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void Dead_Pixels_Remove_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height, unsigned char Threshold)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int Delta1r, Delta2r;
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      Delta1r = abs(img_r[index] - img_r[index-1]);  
      Delta2r = abs(img_r[index] - img_r[index+1]);
      
      if ((Delta1r > Threshold) && (Delta2r > Threshold)) { 
      dest_r[index] = int((img_r[index-1] + img_r[index+1]) / 2.0);
      }
      else {
      dest_r[index] = img_r[index];
      }      
    } 
}
''', 'Dead_Pixels_Remove_Mono_C')

Contrast_Low_Light_Colour_GPU = cp.RawKernel(r'''
extern "C" __global__
void Contrast_Low_Light_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, unsigned char *Corr_CLL)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int vr,vg,vb;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];  
      vg = img_g[index];  
      vb = img_b[index];  
      dest_r[index] = Corr_CLL[vr];
      dest_g[index] = Corr_CLL[vg];
      dest_b[index] = Corr_CLL[vb];
    } 
}
''', 'Contrast_Low_Light_Colour_C')

Contrast_Low_Light_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void Contrast_Low_Light_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height, unsigned char *Corr_CLL)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int vr;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];  
      dest_r[index] = Corr_CLL[vr];
    } 
}
''', 'Contrast_Low_Light_Mono_C')

Contrast_Combine_Colour = cp.RawKernel(r'''
extern "C" __global__
void Contrast_Combine_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *luminance, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float X;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      if (luminance[index] > 1.1 *(0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index])) {
          X = luminance[index] / (0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]);
          dest_r[index] = (int)(min(max(int(img_r[index]*X * 0.7), 0), 255));
          dest_g[index] = (int)(min(max(int(img_g[index]*X * 0.7), 0), 255));
          dest_b[index] = (int)(min(max(int(img_b[index]*X * 0.7), 0), 255));
          }
    } 
}
''', 'Contrast_Combine_Colour_C')

reduce_noise_Color = cp.RawKernel(r'''
extern "C" __global__
void reduce_noise_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b,
long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  int i1,i2,i3,i4,i5,i6,i7,i8,i9;
  int ind1,ind2,ind3,ind4;
  int mini,maxi;
      
  if (i < height && i > 1 && j < width && j >1) {
      i1 = (i-1) * width + (j-1);
      i2 = (i-1) * width + j;
      i3 = (i-1) * width + (j+1);
      i4 = i * width + (j-1);
      i5 = i * width + j;
      i6 = i * width + (j+1);
      i7 = (i+1) * width + (j-1);
      i8 = (i+1) * width + j;
      i9 = (i+1) * width + (j+1);
	  
	  if ((img_r[i5] - img_r[i1]) * (img_r[i9] - img_r[i5]) > 0)
		ind1 = (img_r[i1] + img_r[i5]*5 + img_r[i9]) / 7;
	  else
		ind1 = (img_r[i1] + img_r[i5] + img_r[i9]) / 3;
		
	  if ((img_r[i5] - img_r[i2]) * (img_r[i8] - img_r[i5]) > 0)
		ind2 = (img_r[i2] + img_r[i5]*5 + img_r[i8]) / 7;
	  else
		ind2 = (img_r[i2] + img_r[i5] + img_r[i8]) / 3;
  
	  if ((img_r[i5] - img_r[i3]) * (img_r[i7] - img_r[i5]) > 0)
		ind3 = (img_r[i3] + img_r[i5]*5 + img_r[i7]) / 7;
	  else
		ind3 = (img_r[i3] + img_r[i5] + img_r[i7]) / 3;
		
	  if ((img_r[i5] - img_r[i6]) * (img_r[i4] - img_r[i5]) > 0)
		ind4 = (img_r[i4] + img_r[i5]*5 + img_r[i6]) / 7;
	  else
		ind4 = (img_r[i4] + img_r[i5] + img_r[i6]) / 3;
       
      mini = int(min(ind1,min(ind2,min(ind3,min(ind4,255)))));
      maxi = int(max(ind1,max(ind2,max(ind3,max(ind4,0)))));
      dest_r[i5] = (int)(min(max(int(mini + (maxi-mini) / 2), 0), 255));
 

	  if ((img_g[i5] - img_g[i1]) * (img_g[i9] - img_g[i5]) > 0)
		ind1 = (img_g[i1] + img_g[i5]*5 + img_g[i9]) / 7;
	  else
		ind1 = (img_g[i1] + img_g[i5] + img_g[i9]) / 3;
		
	  if ((img_g[i5] - img_g[i2]) * (img_g[i8] - img_g[i5]) > 0)
		ind2 = (img_g[i2] + img_g[i5]*5 + img_g[i8]) / 7;
	  else
		ind2 = (img_g[i2] + img_g[i5] + img_g[i8]) / 3;
  
	  if ((img_g[i5] - img_g[i3]) * (img_g[i7] - img_g[i5]) > 0)
		ind3 = (img_g[i3] + img_g[i5]*5 + img_g[i7]) / 7;
	  else
		ind3 = (img_g[i3] + img_g[i5] + img_g[i7]) / 3;
		
	  if ((img_g[i5] - img_g[i6]) * (img_g[i4] - img_g[i5]) > 0)
		ind4 = (img_g[i4] + img_g[i5]*5 + img_g[i6]) / 7;
	  else
		ind4 = (img_g[i4] + img_g[i5] + img_g[i6]) / 3;
       
      mini = int(min(ind1,min(ind2,min(ind3,min(ind4,255)))));
      maxi = int(max(ind1,max(ind2,max(ind3,max(ind4,0)))));
      dest_g[i5] = (int)(min(max(int(mini + (maxi-mini) / 2), 0), 255));


	  if ((img_b[i5] - img_b[i1]) * (img_b[i9] - img_b[i5]) > 0)
		ind1 = (img_b[i1] + img_b[i5]*5 + img_b[i9]) / 7;
	  else
		ind1 = (img_b[i1] + img_b[i5] + img_b[i9]) / 3;
		
	  if ((img_b[i5] - img_b[i2]) * (img_b[i8] - img_b[i5]) > 0)
		ind2 = (img_b[i2] + img_b[i5]*5 + img_b[i8]) / 7;
	  else
		ind2 = (img_b[i2] + img_b[i5] + img_b[i8]) / 3;
  
	  if ((img_b[i5] - img_b[i3]) * (img_b[i7] - img_b[i5]) > 0)
		ind3 = (img_b[i3] + img_b[i5]*5 + img_b[i7]) / 7;
	  else
		ind3 = (img_b[i3] + img_b[i5] + img_b[i7]) / 3;
		
	  if ((img_b[i5] - img_b[i6]) * (img_b[i4] - img_b[i5]) > 0)
		ind4 = (img_b[i4] + img_b[i5]*5 + img_b[i6]) / 7;
	  else
		ind4 = (img_b[i4] + img_b[i5] + img_b[i6]) / 3;
       
      mini = int(min(ind1,min(ind2,min(ind3,min(ind4,255)))));
      maxi = int(max(ind1,max(ind2,max(ind3,max(ind4,0)))));
      dest_b[i5] = (int)(min(max(int(mini + (maxi-mini) / 2), 0), 255));
      }
}
''', 'reduce_noise_Color_C')

reduce_noise_Mono = cp.RawKernel(r'''
extern "C" __global__
void reduce_noise_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  int i1,i2,i3,i4,i5,i6,i7,i8,i9;
  int ind1,ind2,ind3,ind4;
  int mini,maxi;
      
  if (i < height && i > 1 && j < width && j >1) {
      i1 = (i-1) * width + (j-1);
      i2 = (i-1) * width + j;
      i3 = (i-1) * width + (j+1);
      i4 = i * width + (j-1);
      i5 = i * width + j;
      i6 = i * width + (j+1);
      i7 = (i+1) * width + (j-1);
      i8 = (i+1) * width + j;
      i9 = (i+1) * width + (j+1);
	  
	  if ((img_r[i5] - img_r[i1]) * (img_r[i9] - img_r[i5]) > 0)
		ind1 = (img_r[i1] + img_r[i5]*5 + img_r[i9]) / 7;
	  else
		ind1 = (img_r[i1] + img_r[i5] + img_r[i9]) / 3;
		
	  if ((img_r[i5] - img_r[i2]) * (img_r[i8] - img_r[i5]) > 0)
		ind2 = (img_r[i2] + img_r[i5]*5 + img_r[i8]) / 7;
	  else
		ind2 = (img_r[i2] + img_r[i5] + img_r[i8]) / 3;
  
	  if ((img_r[i5] - img_r[i3]) * (img_r[i7] - img_r[i5]) > 0)
		ind3 = (img_r[i3] + img_r[i5]*5 + img_r[i7]) / 7;
	  else
		ind3 = (img_r[i3] + img_r[i5] + img_r[i7]) / 3;
		
	  if ((img_r[i5] - img_r[i6]) * (img_r[i4] - img_r[i5]) > 0)
		ind4 = (img_r[i4] + img_r[i5]*5 + img_r[i6]) / 7;
	  else
		ind4 = (img_r[i4] + img_r[i5] + img_r[i6]) / 3;
       
      mini = int(min(ind1,min(ind2,min(ind3,min(ind4,255)))));
      maxi = int(max(ind1,max(ind2,max(ind3,max(ind4,0)))));
      dest_r[i5] = (int)(min(max(int(mini + (maxi-mini) / 2), 0), 255));
      }
}
''', 'reduce_noise_Mono_C')

Saturation_Combine_Colour = cp.RawKernel(r'''
extern "C" __global__
void Saturation_Combine_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *ext_r, unsigned char *ext_g, unsigned char *ext_b, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float X;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      X = (0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]) / (0.299*ext_r[index] + 0.587*ext_g[index] + 0.114*ext_b[index]);
      dest_r[index] = (int)(min(max(int(ext_r[index]*X), 0), 255));
      dest_g[index] = (int)(min(max(int(ext_g[index]*X), 0), 255));
      dest_b[index] = (int)(min(max(int(ext_b[index]*X), 0), 255));
    } 
}
''', 'Saturation_Combine_Colour_C')

Gaussian_CUDA_Colour = cp.RawKernel(r'''
extern "C" __global__
void Gaussian_CUDA_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, int sigma)
{
    long int j = threadIdx.x + blockIdx.x * blockDim.x;
    long int i = threadIdx.y + blockIdx.y * blockDim.y;
    long int index;
    float red,green,blue;
    float factor;
    int filterX;
    int filterY;
    int imageX;
    int imageY;
    #define filterWidth 7
    #define filterHeight 7

    index = i * width + j;

    float filter[filterHeight][filterWidth] =
    {
      0, 0, 1, 2, 1, 0, 0,
      0, 3, 13, 22, 11, 3, 0,
      1, 13, 59, 97, 59, 13, 1,
      2, 22, 97, 159, 97, 22, 2,
      1, 13, 59, 97, 59, 13, 1,
      0, 3, 13, 22, 11, 3, 0,
      0, 0, 1, 2, 1, 0, 0,
    };
    
    factor = 1.0 / 1003.0;

    red = 0.0;
    green = 0.0;
    blue = 0.0;

    if (i < height && j < width) {
    
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (int)((j - filterWidth / 2 + filterX + width) % width);
        imageY = (int)((i - filterHeight / 2 + filterY + height) % height);
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
        green += img_g[imageY * width + imageX] * filter[filterY][filterX];
        blue += img_b[imageY * width + imageX] * filter[filterY][filterX];
     }
    dest_r[index] = (int)(min(int(factor * red), 255));
    dest_g[index] = (int)(min(int(factor * green), 255));
    dest_b[index] = (int)(min(int(factor * blue), 255));
    }
}
''', 'Gaussian_CUDA_Colour_C')

Colour_2_Grey_GPU = cp.RawKernel(r'''
extern "C" __global__
void Colour_2_Grey_GPU_C(unsigned char *dest_r, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      dest_r[index] = (int)(0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]);
    } 
}
''', 'Colour_2_Grey_GPU_C')

grey_estimate_Mono = cp.RawKernel(r'''
extern "C" __global__
void grey_estimate_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index1,index2,index3,index4;
  float colonne,ligne;
  
  index1 = i * width + j;
  index2 = i * width + j+1;
  index3 = (i+1) * width + j;
  index4 = (i+1) * width + (j+1);

  if (i < height && j < width) {
    colonne = (j/2-int(j/2))*2;
    ligne = (i/2-int(i/2))*2;

    if ((colonne == 0 && ligne == 0) || (colonne == 1 && ligne == 1)) {
        dest_r[index1] = (int)(min(max(int(img_r[index1]+(img_r[index2]+img_r[index3])/2+img_r[index4]), 0), 255));  
    }
    else {
        dest_r[index1] = (int)(min(max(int(img_r[index2]+(img_r[index1]+img_r[index4])/2+img_r[index3]), 0), 255));  
    }
  }
}
''', 'grey_estimate_Mono_C')

color_estimate_Mono = cp.RawKernel(r'''
extern "C" __global__
void color_estimate_Mono_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float Rcalc,Gcalc,Bcalc;
  float Lum_grey,Lum_color,Lum_factor;
  
  index = i * width + j;

  if (i < height && j < width) {
    Lum_grey = (0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]);
    Lum_color = img_r[index] + img_g[index] + img_b[index];
    Lum_factor = Lum_color / Lum_grey;
    Rcalc = img_r[index] * Lum_factor;
    Gcalc = img_g[index] * Lum_factor;
    Bcalc = img_b[index] * Lum_factor;
    dest_r[index] = (int)(min(max(int(Rcalc), 0), 255));
    dest_g[index] = (int)(min(max(int(Gcalc), 0), 255));
    dest_b[index] = (int)(min(max(int(Bcalc), 0), 255));
  }
}
''', 'color_estimate_Mono_C')

Mono_ampsoft_GPU = cp.RawKernel(r'''
extern "C" __global__
void Mono_ampsoft_GPU_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height, float val_ampl, float *Corr_GS)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int cor,vr;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];
      cor = (int)(img_r[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vr]));
      dest_r[index] = min(max(cor, 0), 255);
    } 
}
''', 'Mono_ampsoft_GPU_C')

Colour_ampsoft_GPU = cp.RawKernel(r'''
extern "C" __global__
void Colour_ampsoft_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, float val_ampl, float *Corr_GS)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int cor_r,cor_g,cor_b;
  int vr,vg,vb;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];  
      vg = img_g[index];  
      vb = img_b[index];
      cor_r = (int)(img_r[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vr]));
      cor_g = (int)(img_g[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vg]));
      cor_b = (int)(img_b[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vb]));
      dest_r[index] = min(max(cor_r, 0), 255);
      dest_g[index] = min(max(cor_g, 0), 255);
      dest_b[index] = min(max(cor_b, 0), 255);
    } 
}
''', 'Colour_ampsoft_GPU_C')

Colour_contrast_GPU = cp.RawKernel(r'''
extern "C" __global__
void Colour_contrast_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, float val_ampl, float *Corr_cont)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int vr,vg,vb;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];  
      vg = img_g[index];  
      vb = img_b[index];  
      dest_r[index] = (int)(img_r[index] *Corr_cont[vr]);
      dest_g[index] = (int)(img_g[index] *Corr_cont[vg]);
      dest_b[index] = (int)(img_b[index] *Corr_cont[vb]);
    } 
}
''', 'Colour_contrast_GPU_C')

Saturation_Color = cp.RawKernel(r'''
extern "C" __global__
void Saturation_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, float val_sat)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float P;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      P = __fsqrt_rd(img_r[index]*img_r[index]*0.299+img_g[index]*img_g[index]*0.587+img_b[index]*img_b[index]*0.114);
      dest_r[index] = (int)(min(max(int(P+(img_r[index]-P)*val_sat), 0), 255));
      dest_g[index] = (int)(min(max(int(P+(img_g[index]-P)*val_sat), 0), 255));
      dest_b[index] = (int)(min(max(int(P+(img_b[index]-P)*val_sat), 0), 255));
    } 
}
''', 'Saturation_Color_C')

Saturation_Combine_Colour = cp.RawKernel(r'''
extern "C" __global__
void Saturation_Combine_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *ext_r, unsigned char *ext_g, unsigned char *ext_b, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float X;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      X = (0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]) / (0.299*ext_r[index] + 0.587*ext_g[index] + 0.114*ext_b[index]);
      dest_r[index] = (int)(min(max(int(ext_r[index]*X), 0), 255));
      dest_g[index] = (int)(min(max(int(ext_g[index]*X), 0), 255));
      dest_b[index] = (int)(min(max(int(ext_b[index]*X), 0), 255));
    } 
}
''', 'Saturation_Combine_Colour_C')

Saturation_Colour = cp.RawKernel(r'''
extern "C" __global__
void Saturation_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b,
long int width, long int height, float val_sat, int flag_neg_sat)
{

    long int j = threadIdx.x + blockIdx.x * blockDim.x;
    long int i = threadIdx.y + blockIdx.y * blockDim.y;
    long int index;
    double R1,G1,B1;
    double X1;
    double r,g,b;
    double C,X,m;
    double cmax,cmin,diff,h,s,v;
    double radian;
    double cosA,sinA;
    double m1,m2,m3,m4,m5,m6,m7,m8,m9;

    index = i * width + j;
  
    if (i < height && j < width) {
        r = img_r[index] / 255.0;
        g = img_g[index] / 255.0;
        b = img_b[index] / 255.0;
        cmax = max(r, max(g, b));
        cmin = min(r, min(g, b));
        diff = cmax - cmin;
        h = -1.0;
        s = -1.0;
        if (cmax == cmin) 
            h = 0; 
        else if (cmax == r) 
            h = fmod(60 * ((g - b) / diff) + 360, 360); 
        else if (cmax == g) 
            h = fmod(60 * ((b - r) / diff) + 120, 360); 
        else if (cmax == b) 
            h = fmod(60 * ((r - g) / diff) + 240, 360); 
  
        if (cmax == 0) 
            s = 0; 
        else
            s = (diff / cmax); 

        v = cmax;

        s = s * val_sat;

            
        if (h > 360)
            h = 360;
        if (h < 0)
            h = 0;
        if (s > 1.0)
            s = 1.0;
        if (s < 0)
            s = 0;

        C = s*v;
        X = C*(1-abs(fmod(h/60.0, 2)-1));
        m = v-C;

        if(h >= 0 && h < 60){
            r = C,g = X,b = 0;
        }
        else if(h >= 60 && h < 120){
            r = X,g = C,b = 0;
        }
        else if(h >= 120 && h < 180){
            r = 0,g = C,b = X;
        }
        else if(h >= 180 && h < 240){
            r = 0,g = X,b = C;
        }
        else if(h >= 240 && h < 300){
            r = X,g = 0,b = C;
        }
        else{
            r = C,g = 0,b = X;
        }

        R1 = (int)((r+m)*255);
        G1 = (int)((g+m)*255);
        B1 = (int)((b+m)*255);

        if (flag_neg_sat == 1) {
            radian = 3.141592;
            cosA = cos(radian);
            sinA = sin(radian);
            m1 = cosA + (1.0 - cosA) / 3.0;
            m2 = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA;
            m3 = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA;
            m4 = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA;
            m5 = cosA + 1./3.*(1.0 - cosA);
            m6 = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA;
            m7 = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA;
            m8 = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA;
            m9 = cosA + 1./3. * (1.0 - cosA);
            dest_r[index] = (int)(min(max(int(R1 * m1 + G1 * m2 + B1 * m3), 0), 255));
            dest_g[index] = (int)(min(max(int(R1 * m4 + G1 * m5 + B1 * m6), 0), 255));
            dest_b[index] = (int)(min(max(int(R1 * m7 + G1 * m8 + B1 * m9), 0), 255));
        }
        else {
            dest_r[index] = (int)(min(max(int(R1), 0), 255));
            dest_g[index] = (int)(min(max(int(G1), 0), 255));
            dest_b[index] = (int)(min(max(int(B1), 0), 255));
        }
    }
}
''', 'Saturation_Colour_C')

Colour_staramp_GPU = cp.RawKernel(r'''
extern "C" __global__
void Colour_staramp_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b,
unsigned char *grey_gpu, unsigned char *grey_blur_gpu,long int width, long int height, float val_Mu, float val_Ro, float val_ampl, float *Corr_GS)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta;
  unsigned char index_grey;
  float factor;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      delta =(int)(min(max(int( grey_gpu[index] - grey_blur_gpu[index]), 0), 255));
      index_grey = grey_gpu[index];
      factor = delta*Corr_GS[index_grey]*val_ampl;
      dest_r[index] = (int)(min(max(int(img_r[index] + factor), 0), 255));
      dest_g[index] = (int)(min(max(int(img_g[index] + factor), 0), 255));
      dest_b[index] = (int)(min(max(int(img_b[index] + factor), 0), 255));
  }
}
''', 'Colour_staramp_GPU_C')

Mono_staramp_GPU = cp.RawKernel(r'''
extern "C" __global__
void Mono_staramp_GPU_C(unsigned char *dest_r, unsigned char *img_r, unsigned char *grey_blur_gpu,long int width, long int height, float val_Mu, float val_Ro, float val_ampl, float *Corr_GS)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta;
  unsigned char index_grey;
  float factor;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      delta =(int)(min(max(int(img_r[index] - grey_blur_gpu[index]), 0), 255));
      index_grey = img_r[index];
      factor = delta*Corr_GS[index_grey]*val_ampl;
      dest_r[index] = (int)(min(max(int(img_r[index] + factor), 0), 255));
  }
}
''', 'Mono_staramp_GPU_C')

Smooth_Mono_high = cp.RawKernel(r'''
extern "C" __global__
void Smooth_Mono_high_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float red;
  float factor;
  int filterX;
  int filterY;
  int imageX;
  int imageY;

  #define filterWidth 7
  #define filterHeight 7

  index = i * width + j;

  if (i < height && j < width) {
    float filter[filterHeight][filterWidth] =
    {
      0, 0, 1, 2, 1, 0, 0,
      0, 3, 13, 22, 11, 3, 0,
      1, 13, 59, 97, 59, 13, 1,
      2, 22, 97, 159, 97, 22, 2,
      1, 13, 59, 97, 59, 13, 1,
      0, 3, 13, 22, 11, 3, 0,
      0, 0, 1, 2, 1, 0, 0,
    };
    
    factor = 1.0 / 1003.0;
      
    red = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (int)((j - filterWidth / 2 + filterX + width) % width);
        imageY = (int)((i - filterHeight / 2 + filterY + height) % height);
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
  }
}
''', 'Smooth_Mono_high_C')


adaptative_absorber_denoise_Color = cp.RawKernel(r'''
extern "C" __global__
void adaptative_absorber_denoise_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *old_r, unsigned char *old_g, unsigned char *old_b,
long int width, long int height, int flag_dyn_AANR, int flag_ghost_reducer, int val_ghost_reducer)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta_r,delta_g,delta_b;
  int flag_r,flag_g,flag_b;
  float coef_r,coef_g,coef_b;
  
  flag_r = 0;
  flag_g = 0;
  flag_b = 0;
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      delta_r = old_r[index] - img_r[index];
      delta_g = old_g[index] - img_g[index];
      delta_b = old_b[index] - img_b[index];
      if (flag_dyn_AANR == 1) {
          flag_ghost_reducer = 0;
      }
      if (flag_ghost_reducer == 1) {
          if (abs(delta_r) > val_ghost_reducer) {
              flag_r = 1;
              dest_r[index] = img_r[index];
          }
          if (abs(delta_g) > val_ghost_reducer) {
              flag_g = 1;
              dest_g[index] = img_g[index];
          }
          if (abs(delta_b) > val_ghost_reducer) {
              flag_b = 1;
              dest_b[index] = img_b[index];
          }
          if (delta_r > 0 && flag_dyn_AANR == 1 && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.025995987)*1.2669433195)));
          }
          if ((delta_r < 0 || flag_dyn_AANR == 0) && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.54405)*20.8425))); 
          }
          if (delta_g > 0 && flag_dyn_AANR == 1 && flag_g == 0) {
              dest_g[index] = (int)((old_g[index] - delta_g / (__powf(abs(delta_g),-0.025995987)*1.2669433195)));
          }
          if ((delta_g < 0 || flag_dyn_AANR == 0) && flag_g == 0) {
              dest_g[index] = (int)((old_g[index] - delta_g / (__powf(abs(delta_g),-0.54405)*20.8425))); 
          }
          if (delta_b > 0 && flag_dyn_AANR == 1 && flag_b == 0) {
              dest_b[index] = (int)((old_b[index] - delta_b / (__powf(abs(delta_b),-0.025995987)*1.2669433195)));
          }
          if ((delta_b < 0 || flag_dyn_AANR == 0) && flag_b == 0) {
              dest_b[index] = (int)((old_b[index] - delta_b / (__powf(abs(delta_b),-0.54405)*20.8425))); 
          }
          }
      if (flag_ghost_reducer == 0) {
          if (delta_r > 0 && flag_dyn_AANR == 1) {
              coef_r = __powf(abs(delta_r),-0.025995987)*1.2669433195;
          }
          else {
              coef_r = __powf(abs(delta_r),-0.54405)*20.8425; 
          }
          if (delta_g > 0 && flag_dyn_AANR == 1) {
              coef_g = __powf(abs(delta_g),-0.025995987)*1.2669433195;
          }
          else {
              coef_g = __powf(abs(delta_g),-0.54405)*20.8425; 
          }
          if (delta_b > 0 && flag_dyn_AANR == 1) {
              coef_b = __powf(abs(delta_b),-0.025995987)*1.2669433195;
          }
          else {
              coef_b = __powf(abs(delta_b),-0.54405)*20.8425;
          }
          dest_r[index] = (int)((old_r[index] - delta_r / coef_r));
          dest_g[index] = (int)((old_g[index] - delta_g / coef_g));
          dest_b[index] = (int)((old_b[index] - delta_b / coef_b));
      } 
      }
}
''', 'adaptative_absorber_denoise_Color_C')

adaptative_absorber_denoise_Mono = cp.RawKernel(r'''
extern "C" __global__
void adaptative_absorber_denoise_Mono_C(unsigned char *dest_r, unsigned char *img_r, unsigned char *old_r, long int width, long int height, int flag_dyn_AANR,
int flag_ghost_reducer, int val_ghost_reducer)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta_r;
  int flag_r;
  float coef_r;
  
  flag_r = 0;
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      delta_r = old_r[index] - img_r[index];
      if (flag_dyn_AANR == 1) {
          flag_ghost_reducer = 0;
      }
      if (flag_ghost_reducer == 1) {
          if (abs(delta_r) > val_ghost_reducer) {
              flag_r = 1;
              dest_r[index] = img_r[index];
          }
          if (delta_r > 0 && flag_dyn_AANR == 1 && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.025995987)*1.2669433195)));
          }
          if ((delta_r < 0 || flag_dyn_AANR == 0) && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.54405)*20.8425))); 
          }
          }
      if (flag_ghost_reducer == 0) {
          if (delta_r > 0 && flag_dyn_AANR == 1) {
              coef_r = __powf(abs(delta_r),-0.025995987)*1.2669433195;
          }
          else {
              coef_r = __powf(abs(delta_r),-0.54405)*20.8425; 
          }
          dest_r[index] = (int)((old_r[index] - delta_r / coef_r));
      } 
      }
}
''', 'adaptative_absorber_denoise_Mono_C')

reduce_variation_Color = cp.RawKernel(r'''
extern "C" __global__
void reduce_variation_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *old_r, unsigned char *old_g, unsigned char *old_b,
long int width, long int height, int variation)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta_r,delta_g,delta_b;  
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      delta_r = old_r[index] - img_r[index];
      delta_g = old_g[index] - img_g[index];
      delta_b = old_b[index] - img_b[index];

      if (abs(delta_r) > variation) {
          if (delta_r >= 0) {
              dest_r[index] = min(max(old_r[index] - variation, 0), 255);
          }
          else {
              dest_r[index] = min(max(old_r[index] + variation, 0), 255);          
          }
      }
      else {
          dest_r[index] = img_r[index];
      }
      
      if (abs(delta_g) > variation) {
          if (delta_g >= 0) {
              dest_g[index] = min(max(old_g[index] - variation, 0), 255);
          }
          else {
              dest_g[index] = min(max(old_g[index] + variation, 0), 255);          
          }
      }
      else {
          dest_g[index] = img_g[index];
      }

      if (abs(delta_b) > variation) {
          if (delta_b >= 0) {
              dest_b[index] = min(max(old_b[index] - variation, 0), 255);
          }
          else {
              dest_b[index] = min(max(old_b[index] + variation, 0), 255);          
          }
      }
      else {
          dest_b[index] = img_b[index];
      }
      }
}
''', 'reduce_variation_Color_C')

reduce_variation_Mono = cp.RawKernel(r'''
extern "C" __global__
void reduce_variation_Mono_C(unsigned char *dest_r,
unsigned char *img_r, unsigned char *old_r, long int width, long int height, int variation)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta_r;  
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      delta_r = old_r[index] - img_r[index];

      if (abs(delta_r) > variation) {
          if (delta_r >= 0) {
              dest_r[index] = min(max(old_r[index] - variation, 0), 255);
          }
          else {
              dest_r[index] = min(max(old_r[index] + variation, 0), 255);          
          }
      }
      else {
          dest_r[index] = img_r[index];
      }
      }
}
''', 'reduce_variation_Mono_C')

Denoise_Paillou_Colour = cp.RawKernel(r'''
extern "C" __global__
void Denoise_Paillou_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, int cell_size, int sqr_cell_size)
{    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;
     
    long int index1;
    long int index2;
    int delta;
    //float3 corr = {0, 0, 0};
    float3 Grd = {0, 0, 0};
    float3 Mean = {0, 0, 0};
    float3 Delta =  {0, 0, 0};

    delta = (int)(abs(cell_size/2));
    index1 = ix + iy * imageW;
    
    if(ix<=(imageW-cell_size) && ix > delta && iy<=(imageH-cell_size) && iy > delta){
        for(float n = -delta; n <= delta; n++)
            for(float m = -delta; m <= delta; m++) {
                index2 = ix + m + (iy + n) * imageW;
                Grd.x += img_r[index1]-img_r[index2];
                Grd.y += img_g[index1]-img_g[index2];
                Grd.z += img_b[index1]-img_b[index2];
                Mean.x += img_r[index2];
                Mean.y += img_g[index2];
                Mean.z += img_b[index2];
                }
        Delta.x = (Grd.x / (sqr_cell_size * (1.0 + Grd.x/Mean.x))*(-0.00392157 * img_r[index1] +1.0));
        Delta.y = (Grd.y / (sqr_cell_size * (1.0 + Grd.y/Mean.y))*(-0.00392157 * img_g[index1] +1.0));
        Delta.z = (Grd.z / (sqr_cell_size * (1.0 + Grd.z/Mean.z))*(-0.00392157 * img_b[index1] +1.0));
        if (dest_r[index1] > abs(Delta.x) && dest_g[index1] > abs(Delta.y) && dest_b[index1] > abs(Delta.z)) {
            dest_r[index1] = (int)(min(max(int(img_r[index1] - Delta.x), 0), 255));
            dest_g[index1] = (int)(min(max(int(img_g[index1] - Delta.y), 0), 255));
            dest_b[index1] = (int)(min(max(int(img_b[index1] - Delta.z), 0), 255));
            }
        else {
            dest_r[index1] = int((img_r[ix - 1 + iy * imageW] + img_r[ix + 1 + iy * imageW] + img_r[ix + (iy-1) * imageW] + img_r[ix + (iy+1) * imageW])/4.0);
            dest_g[index1] = int((img_g[ix - 1 + iy * imageW] + img_g[ix + 1 + iy * imageW] + img_g[ix + (iy-1) * imageW] + img_g[ix + (iy+1) * imageW])/4.0);
            dest_b[index1] = int((img_b[ix - 1 + iy * imageW] + img_b[ix + 1 + iy * imageW] + img_b[ix + (iy-1) * imageW] + img_b[ix + (iy+1) * imageW])/4.0);
        }
    }
}
''', 'Denoise_Paillou_Colour_C')

Denoise_Paillou_Mono = cp.RawKernel(r'''
extern "C" __global__
void Denoise_Paillou_Mono_C(unsigned char *dest_r, unsigned char *img_r, int imageW, int imageH, int cell_size, int sqr_cell_size)
{ 
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;
     
    long int index1;
    long int index2;
    int delta;
    //float3 corr = {0, 0, 0};
    float Grd = 0;
    float Mean = 0;
    float Delta =  0;

    delta = (int)(abs(cell_size/2));
    index1 = ix + iy * imageW;
    
    if(ix<=(imageW-cell_size) && ix > delta && iy<=(imageH-cell_size) && iy > delta){
        // Dead pixels detection and correction
        for(float n = -delta; n <= delta; n++)
            for(float m = -delta; m <= delta; m++) {
                index2 = ix + m + (iy + n) * imageW;
                Grd += img_r[index1]-img_r[index2];
                Mean += img_r[index2];
                }
        Delta = (Grd / (sqr_cell_size * (1.0 + Grd/Mean))*(-0.00392157 * img_r[index1] +1.0));
        if (dest_r[index1] > abs(Delta)) {
            dest_r[index1] = (int)(min(max(int(img_r[index1] - Delta), 0), 255));
            }
        else {
            dest_r[index1] = int((img_r[ix - 1 + iy * imageW] + img_r[ix + 1 + iy * imageW] + img_r[ix + (iy-1) * imageW] + img_r[ix + (iy+1) * imageW])/4.0);
        }
    }
}
''', 'Denoise_Paillou_Mono_C')

Histo_Mono = cp.RawKernel(r'''
extern "C" __global__
void Histo_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height,
int flag_histogram_stretch, float val_histo_min, float val_histo_max, int flag_histogram_equalize2, float val_heq2, int flag_histogram_phitheta,
float val_phi, float val_theta)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  
  index = i * width + j;
  
  if (i < height && j < width) {
  if (flag_histogram_phitheta == 1) {
      dest_r[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_r[index]-val_theta)/32.0))));
      img_r[index] = dest_r[index];
    }
  if (flag_histogram_equalize2 == 1 ) {
      dest_r[index] = (int)(255.0*__powf(((img_r[index]) / 255.0),val_heq2));
      img_r[index] = dest_r[index];
    }
  if (flag_histogram_stretch == 1 ) {
      dest_r[index] = (int)(min(max(int((img_r[index]-val_histo_min)*(255.0/(val_histo_max-val_histo_min))), 0), 255));
      img_r[index] = dest_r[index];
    }    
  }
}
''', 'Histo_Mono_C')


Histo_Color = cp.RawKernel(r'''
extern "C" __global__
void Histo_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, long int width, long int height, int flag_histogram_stretch, float val_histo_min, float val_histo_max, int flag_histogram_equalize2,
float val_heq2, int flag_histogram_phitheta, float val_phi, float val_theta)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int delta_histo = val_histo_max-val_histo_min;
  
  index = i * width + j;
  
  if (i < height && j < width) {
  if (flag_histogram_phitheta == 1) {
      dest_r[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_r[index]-val_theta)/32.0))));
      dest_g[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_g[index]-val_theta)/32.0))));
      dest_b[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_b[index]-val_theta)/32.0))));
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    }
  if (flag_histogram_equalize2 == 1 ) {
      dest_r[index] = (int)(255.0*__powf(((img_r[index]) / 255.0),val_heq2));
      dest_g[index] = (int)(255.0*__powf(((img_g[index]) / 255.0),val_heq2));
      dest_b[index] = (int)(255.0*__powf(((img_b[index]) / 255.0),val_heq2));
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    } 
  if (flag_histogram_stretch == 1 ) {
      dest_r[index] = (int)(min(max(int((img_r[index]-val_histo_min)*(255.0/delta_histo)), 0), 255));
      dest_g[index] = (int)(min(max(int((img_g[index]-val_histo_min)*(255.0/delta_histo)), 0), 255));
      dest_b[index] = (int)(min(max(int((img_b[index]-val_histo_min)*(255.0/delta_histo)), 0), 255));
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    }
  }
}
''', 'Histo_Color_C')

Set_RGB = cp.RawKernel(r'''
extern "C" __global__
void Set_RGB_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, long int width, long int height, float mod_red, float mod_green, float mod_blue)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      if (mod_blue != 1.0) {
          dest_r[index] = (int)(min(max(int(img_r[index] * mod_blue), 0), 255));
          }
      else {
          dest_r[index] = img_r[index];
          }
      if (mod_green != 1.0) {        
          dest_g[index] = (int)(min(max(int(img_g[index] * mod_green), 0), 255));
          }
      else {
          dest_g[index] = img_g[index];
          }
      if (mod_red != 1.0) {  
          dest_b[index] = (int)(min(max(int(img_b[index] * mod_red), 0), 255));
          }
      else {
          dest_b[index] = img_b[index];
          }
    } 
}
''', 'Set_RGB_C')


NLM2_Colour_GPU = cp.RawKernel(r'''
extern "C" __global__
void NLM2_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define NLM_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define NLM_WEIGHT_THRESHOLD    0.00039f
    #define NLM_LERP_THRESHOLD      0.10f
    
    __shared__ float fWeights[64];

    const float NLM_WINDOW_AREA = (2.0 * NLM_WINDOW_RADIUS + 1.0) * (2.0 * NLM_WINDOW_RADIUS + 1.0) ;
    const float INV_NLM_WINDOW_AREA = (1.0 / NLM_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 1.0f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 1.0f;
    const float limxmin = 6;
    const float limxmax = imageW - 6;
    const float limymin = 6;
    const float limymax = imageH - 6;
   
    long int index4;
    long int index5;

    if(x>limxmin && x<limxmax && y>limymin && y<limymax){
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for(float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for(float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++) {
                long int index1 = cx + m + (cy + n) * imageW;
                long int index2 = x + m + (y + n) * imageW;
                weight += ((img_r[index2] - img_r[index1]) * (img_r[index2] - img_r[index1])
                + (img_g[index2] - img_g[index1]) * (img_g[index2] - img_g[index1])
                + (img_b[index2] - img_b[index1]) * (img_b[index2] - img_b[index1])) / (256.0 * 256.0);
                }

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * 8 + threadIdx.x] = weight / 256.0;
        //Wait until all the weights are ready
        __syncthreads();


        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0.0, 0.0, 0.0};

        int idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        
        for(float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for(float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Accumulate (x + j, y + i) texel color with computed weight
                float3 clrIJ ; // Ligne code modifiée
                int index3 = x + j + (y + i) * imageW;
                clrIJ.x = img_r[index3];
                clrIJ.y = img_g[index3];
                clrIJ.z = img_b[index3];
                
                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJ;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotent basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        float3 clr00 = {0.0, 0.0, 0.0};
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00.x = img_r[index4] / 256.0;
        clr00.y = img_g[index4] / 256.0;
        clr00.z = img_b[index4] / 256.0;
        
        clr.x = clr.x + (clr00.x - clr.x) * lerpQ;
        clr.y = clr.y + (clr00.y - clr.y) * lerpQ;
        clr.z = clr.z + (clr00.z - clr.z) * lerpQ;
        
        dest_r[index5] = (int)(clr.x * 256.0);
        dest_g[index5] = (int)(clr.y * 256.0);
        dest_b[index5] = (int)(clr.z * 256.0);
    }
}
''', 'NLM2_Colour_C')

NLM2_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void NLM2_Mono_C(unsigned char *dest_r, unsigned char *img_r,
int imageW, int imageH, float Noise, float lerpC)
{
    
    #define NLM_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define NLM_WEIGHT_THRESHOLD    0.00039f
    #define NLM_LERP_THRESHOLD      0.10f
    
    __shared__ float fWeights[64];

    const float NLM_WINDOW_AREA = (2.0 * NLM_WINDOW_RADIUS + 1.0) * (2.0 * NLM_WINDOW_RADIUS + 1.0) ;
    const float INV_NLM_WINDOW_AREA = (1.0 / NLM_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 1.0f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 1.0f;
    const float limxmin = NLM_BLOCK_RADIUS + 3;
    const float limxmax = imageW - NLM_BLOCK_RADIUS - 3;
    const float limymin = NLM_BLOCK_RADIUS + 3;
    const float limymax = imageH - NLM_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for(float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for(float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++) {
                long int index1 = cx + m + (cy + n) * imageW;
                long int index2 = x + m + (y + n) * imageW;
                weight += ((img_r[index2] - img_r[index1]) * (img_r[index2] - img_r[index1])) / (256.0 * 256.0);
                }

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * 8 + threadIdx.x] = weight / 256.0;
        //Wait until all the weights are ready
        __syncthreads();


        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float clr = 0.0;

        int idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        
        for(float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for(float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Accumulate (x + j, y + i) texel color with computed weight
                float clrIJ ; // Ligne code modifiée
                int index3 = x + j + (y + i) * imageW;
                clrIJ = img_r[index3];
                
                clr += clrIJ * weightIJ;
 
                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJ;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr *= sumWeights;

        //Choose LERP quotent basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        float clr00 = 0.0;
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00 = img_r[index4] / 256.0;
        
        clr = clr + (clr00 - clr) * lerpQ;
       
        dest_r[index5] = (int)(clr * 256.0);
    }
}
''', 'NLM2_Mono_C')

KNN_Colour_GPU = cp.RawKernel(r'''
extern "C" __global__
void KNN_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define KNN_WINDOW_RADIUS   3
    #define KNN_BLOCK_RADIUS    3

    #define KNN_WEIGHT_THRESHOLD    0.00078125f
    #define KNN_LERP_THRESHOLD      0.79f

    const float KNN_WINDOW_AREA = (2.0 * KNN_WINDOW_RADIUS + 1.0) * (2.0 * KNN_WINDOW_RADIUS + 1.0) ;
    const float INV_KNN_WINDOW_AREA = (1.0 / KNN_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float limxmin = KNN_BLOCK_RADIUS + 3;
    const float limxmax = imageW - KNN_BLOCK_RADIUS - 3;
    const float limymin = KNN_BLOCK_RADIUS + 3;
    const float limymax = imageH - KNN_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0, 0, 0};
        float3 clr00 = {0, 0, 0};
        float3 clrIJ = {0, 0, 0};
        //Center of the KNN window
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00.x = img_r[index4];
        clr00.y = img_g[index4];
        clr00.z = img_b[index4];
    
        for(float i = -KNN_BLOCK_RADIUS; i <= KNN_BLOCK_RADIUS; i++)
            for(float j = -KNN_BLOCK_RADIUS; j <= KNN_BLOCK_RADIUS; j++) {
                long int index2 = x + j + (y + i) * imageW;
                clrIJ.x = img_r[index2];
                clrIJ.y = img_g[index2];
                clrIJ.z = img_b[index2];
                float distanceIJ = ((clrIJ.x - clr00.x) * (clrIJ.x - clr00.x)
                + (clrIJ.y - clr00.y) * (clrIJ.y - clr00.y)
                + (clrIJ.z - clr00.z) * (clrIJ.z - clr00.z)) / 65536.0;

                //Derive final weight from color and geometric distance
                float   weightIJ = (__expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA))) / 256.0;

                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
        }
        
        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
        
        clr.x = clr.x + (clr00.x / 256.0 - clr.x) * lerpQ;
        clr.y = clr.y + (clr00.y / 256.0 - clr.y) * lerpQ;
        clr.z = clr.z + (clr00.z / 256.0 - clr.z) * lerpQ;
        
        dest_r[index5] = (int)(clr.x * 256.0);
        dest_g[index5] = (int)(clr.y * 256.0);
        dest_b[index5] = (int)(clr.z * 256.0);
    }
}
''', 'KNN_Colour_C')

KNN_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void KNN_Mono_C(unsigned char *dest_r, unsigned char *img_r, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define KNN_WINDOW_RADIUS   3
    #define KNN_BLOCK_RADIUS    3

    #define KNN_WEIGHT_THRESHOLD    0.00078125f
    #define KNN_LERP_THRESHOLD      0.79f

    const float KNN_WINDOW_AREA = (2.0 * KNN_WINDOW_RADIUS + 1.0) * (2.0 * KNN_WINDOW_RADIUS + 1.0) ;
    const float INV_KNN_WINDOW_AREA = (1.0 / KNN_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float limxmin = KNN_BLOCK_RADIUS + 3;
    const float limxmax = imageW - KNN_BLOCK_RADIUS - 3;
    const float limymin = KNN_BLOCK_RADIUS + 3;
    const float limymax = imageH - KNN_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float clr = 0.0;
        float clr00 = 0.0;
        float clrIJ = 0.0;
        //Center of the KNN window
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00 = img_r[index4];

        for(float i = -KNN_BLOCK_RADIUS; i <= KNN_BLOCK_RADIUS; i++)
            for(float j = -KNN_BLOCK_RADIUS; j <= KNN_BLOCK_RADIUS; j++) {
                long int index2 = x + j + (y + i) * imageW;
                clrIJ = img_r[index2];
                float distanceIJ = ((clrIJ - clr00) * (clrIJ - clr00)) / 65536.0;

                //Derive final weight from color and geometric distance
                float   weightIJ = (__expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA))) / 256.0;

                clr += clrIJ * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
        }
        
        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
        
        clr = clr + (clr00 / 256.0 - clr) * lerpQ;
        
        dest_r[index5] = (int)(clr * 256.0);
    }
}
''', 'KNN_Mono_C')
