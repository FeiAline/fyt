#include <cuda.h>
#include <cutil.h>
#include "resize_GPU.cu"
texture<float, 2, cudaReadModeElementType> Texref;
texture<unsigned char, 2, cudaReadModeNormalizedFloat> Texrefresz;
__constant__ float d_ker_h_r[25];
__constant__ float d_ker_h_i[25];
__constant__ float d_ker_d_r[25];
__constant__ float d_ker_d_i[25];
__constant__ float d_ker_v_r[25];
__constant__ float d_ker_v_i[25];
__constant__ float d_ker_ld_r[25];
__constant__ float d_ker_ld_i[25];

__global__ void Textureconvol_RP5_Arr_ori(short int* RP_out, int width)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	const float x = (float)xIndex + 0.5;
	const float y = (float)yIndex + 0.5;

	//if (xIndex < width && yIndex < height)
	{
	float sum_h_r = 0.0f;
	float sum_h_i = 0.0f;
	float sum_d_r = 0.0f;
	float sum_d_i = 0.0f;
	float sum_v_r = 0.0f;
	float sum_v_i = 0.0f;
	float sum_ld_r= 0.0f;
	float sum_ld_i= 0.0f;
	float val;
	short int id;
	for (int i=-2; i<3; i++)
		for (int j=-2; j<3; j++)
		{
		val =255.0f*tex2D(Texrefresz, x+j, y+i);
		id  = (i+2)*5+j+2;
		sum_h_r += val*d_ker_h_r[id];
		sum_h_i += val*d_ker_h_i[id];
		sum_d_r += val*d_ker_d_r[id];
		sum_d_i += val*d_ker_d_i[id];
		sum_v_r += val*d_ker_v_r[id];
		sum_v_i += val*d_ker_v_i[id];
		sum_ld_r += val*d_ker_ld_r[id];
		sum_ld_i += val*d_ker_ld_i[id];
		}
	val = 0.0f;
	float ene_h = (sum_h_r*sum_h_r+sum_h_i*sum_h_i);
	val += ene_h;
	float ene_d = (sum_d_r*sum_d_r+sum_d_i*sum_d_i);
	val += ene_d;
	float ene_v = (sum_v_r*sum_v_r+sum_v_i*sum_v_i);
	val += ene_v;
	float ene_ld= (sum_ld_r*sum_ld_r+sum_ld_i*sum_ld_i);
	val += ene_ld;
	val = val/4.0f;
	//sum_h_r /= val; sum_h_i/=val;sum_d_r /=val; sum_d_i /=val;
	//sum_v_r /= val; sum_v_i/=val;sum_ld_r/=val; sum_ld_i/=val;
	id =0;
	id = 4*(!(ene_h>val));
	id += (ene_h>val)*(((sum_h_r+sum_h_i)>0.0f)*2 + ((sum_h_r-sum_h_i)>0.0f));

	id = id*5;
	id += 4*(!(ene_d>val));
	id += (ene_d>val)*(((sum_d_r+sum_d_i)>0.0f)*2 + ((sum_d_r-sum_d_i)>0.0f));


	id = id*5;
	id += 4*(!(ene_v>val));
	id +=(ene_v>val) *(((sum_v_r+sum_v_i)>0.0f)*2 + ((sum_v_r-sum_v_i)>0.0f));


	id = id*5;
	id += 4*(!(ene_ld>val));
	id += (ene_ld>val)*(((sum_ld_r+sum_ld_i)>0.0f)*2 + ((sum_ld_r-sum_ld_i)>0.0f));

	RP_out[yIndex*width+xIndex]=id;
	}
}

__global__ void Textureconvol_RP5_Arr(short int* RP_out, int width)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	const float x = (float)xIndex + 0.5;
	const float y = (float)yIndex + 0.5;

	//if (xIndex < width && yIndex < height)
	{
	float sum_h_r = 0.0f;
	float sum_h_i = 0.0f;
	float sum_d_r = 0.0f;
	float sum_d_i = 0.0f;
	float sum_v_r = 0.0f;
	float sum_v_i = 0.0f;
	float sum_ld_r= 0.0f;
	float sum_ld_i= 0.0f;
	float val;
	short int id;
	for (int i=-2; i<3; i++)
		for (int j=-2; j<3; j++)
		{
		val = tex2D(Texref, x+j, y+i);
		id  = (i+2)*5+j+2;
		sum_h_r += val*d_ker_h_r[id];
		sum_h_i += val*d_ker_h_i[id];
		sum_d_r += val*d_ker_d_r[id];
		sum_d_i += val*d_ker_d_i[id];
		sum_v_r += val*d_ker_v_r[id];
		sum_v_i += val*d_ker_v_i[id];
		sum_ld_r += val*d_ker_ld_r[id];
		sum_ld_i += val*d_ker_ld_i[id];
		}
	val = 0.0f;
	float ene_h = (sum_h_r*sum_h_r+sum_h_i*sum_h_i);
	val += ene_h;
	float ene_d = (sum_d_r*sum_d_r+sum_d_i*sum_d_i);
	val += ene_d;
	float ene_v = (sum_v_r*sum_v_r+sum_v_i*sum_v_i);
	val += ene_v;
	float ene_ld= (sum_ld_r*sum_ld_r+sum_ld_i*sum_ld_i);
	val += ene_ld;
	val = val/4.0f;
	//sum_h_r /= val; sum_h_i/=val;sum_d_r /=val; sum_d_i /=val;
	//sum_v_r /= val; sum_v_i/=val;sum_ld_r/=val; sum_ld_i/=val;
	id =0;
	id = 4*(!(ene_h>val));
	id += (ene_h>val)*(((sum_h_r+sum_h_i)>0.0f)*2 + ((sum_h_r-sum_h_i)>0.0f));

	id = id*5;
	id += 4*(!(ene_d>val));
	id += (ene_d>val)*(((sum_d_r+sum_d_i)>0.0f)*2 + ((sum_d_r-sum_d_i)>0.0f));


	id = id*5;
	id += 4*(!(ene_v>val));
	id +=(ene_v>val) *(((sum_v_r+sum_v_i)>0.0f)*2 + ((sum_v_r-sum_v_i)>0.0f));


	id = id*5;
	id += 4*(!(ene_ld>val));
	id += (ene_ld>val)*(((sum_ld_r+sum_ld_i)>0.0f)*2 + ((sum_ld_r-sum_ld_i)>0.0f));

	RP_out[yIndex*width+xIndex]=id;
	}
}

__global__ void resize_bilinear(float* odata, float w_ratio, float h_ratio, int width)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int Index = yIndex * width + xIndex;
	float index_row, index_col;
	//if (xIndex < (width) && yIndex < (height))
	{
		index_row = float(yIndex) * h_ratio;
		index_col = float(xIndex) * w_ratio;
		odata[Index] = 255.0f*tex2D(Texrefresz, index_col, index_row); 
	}
}

__global__ void resize_4N(float* odata, float w_ratio, float h_ratio, int width)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int Index = yIndex * width + xIndex;
	float index_row, index_col;
	//if (xIndex < (width) && yIndex < (height))
	{
		index_row = float(yIndex) * h_ratio;
		index_col = float(xIndex) * w_ratio;
		odata[Index] = 255.0f*tex2D4N< unsigned char, float>(Texrefresz, index_col, index_row);//
	}
}

__global__ void resize_16N(float* odata, float w_ratio, float h_ratio, int width)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int Index = yIndex * width + xIndex;
	float index_row, index_col;
	//if (xIndex < (width) && yIndex < (height))
	{
		index_row = float(yIndex) * h_ratio;
		index_col = float(xIndex) * w_ratio;
		odata[Index] = 255.0f*tex2D16N<unsigned char, float>(Texrefresz, index_col, index_row);//
	}
}

__global__ void Textureconvol_ene_Arr(float* ene_out, int width)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	const float x = (float)xIndex + 0.5;
	const float y = (float)yIndex + 0.5;

	//if (xIndex < width && yIndex < height)
	{
	float sum_h_r = 0.0f;
	float sum_h_i = 0.0f;
	float sum_d_r = 0.0f;
	float sum_d_i = 0.0f;
	float sum_v_r = 0.0f;
	float sum_v_i = 0.0f;
	float sum_ld_r= 0.0f;
	float sum_ld_i= 0.0f;
	float val;
	int id;
	for (int i=-2; i<3; i++)
		for (int j=-2; j<3; j++)
		{
		val = tex2D(Texref, x+j, y+i);
		id  = (i+2)*5+j+2;
		sum_h_r += val*d_ker_h_r[id];
		sum_h_i += val*d_ker_h_i[id];
		sum_d_r += val*d_ker_d_r[id];
		sum_d_i += val*d_ker_d_i[id];
		sum_v_r += val*d_ker_v_r[id];
		sum_v_i += val*d_ker_v_i[id];
		sum_ld_r += val*d_ker_ld_r[id];
		sum_ld_i += val*d_ker_ld_i[id];
		}
	val = 0.0f;
	float ene_h = (sum_h_r*sum_h_r+sum_h_i*sum_h_i);
	val += ene_h;
	float ene_d = (sum_d_r*sum_d_r+sum_d_i*sum_d_i);
	val += ene_d;
	float ene_v = (sum_v_r*sum_v_r+sum_v_i*sum_v_i);
	val += ene_v;
	float ene_ld= (sum_ld_r*sum_ld_r+sum_ld_i*sum_ld_i);
	val += ene_ld;

	id=(yIndex*width+xIndex)*4;
	ene_out[id++]=sqrtf(ene_h/val);
	ene_out[id++]=sqrtf(ene_d/val);
	ene_out[id++]=sqrtf(ene_v/val);
	ene_out[id]=sqrtf(ene_ld/val);
	}
}

__global__ void Textureconvol_ene_Arr_thr(float* ene_out, float beta, int width)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	const float x = (float)xIndex + 0.5;
	const float y = (float)yIndex + 0.5;

	//if (xIndex < width && yIndex < height)
	{
	float sum_h_r = 0.0f;
	float sum_h_i = 0.0f;
	float sum_d_r = 0.0f;
	float sum_d_i = 0.0f;
	float sum_v_r = 0.0f;
	float sum_v_i = 0.0f;
	float sum_ld_r= 0.0f;
	float sum_ld_i= 0.0f;
	float val;
	int id;
	for (int i=-2; i<3; i++)
		for (int j=-2; j<3; j++)
		{
		val = tex2D(Texref, x+j, y+i);
		id  = (i+2)*5+j+2;
		sum_h_r += val*d_ker_h_r[id];
		sum_h_i += val*d_ker_h_i[id];
		sum_d_r += val*d_ker_d_r[id];
		sum_d_i += val*d_ker_d_i[id];
		sum_v_r += val*d_ker_v_r[id];
		sum_v_i += val*d_ker_v_i[id];
		sum_ld_r += val*d_ker_ld_r[id];
		sum_ld_i += val*d_ker_ld_i[id];
		}
	val = 0.0f;
	float ene_h = (sum_h_r*sum_h_r+sum_h_i*sum_h_i);
	val += ene_h;
	float ene_d = (sum_d_r*sum_d_r+sum_d_i*sum_d_i);
	val += ene_d;
	float ene_v = (sum_v_r*sum_v_r+sum_v_i*sum_v_i);
	val += ene_v;
	float ene_ld= (sum_ld_r*sum_ld_r+sum_ld_i*sum_ld_i);
	val += ene_ld;
	val=max(val, beta*beta);

	id=(yIndex*width+xIndex)*4;
	ene_out[id++]=sqrtf(ene_h/val);
	ene_out[id++]=sqrtf(ene_d/val);
	ene_out[id++]=sqrtf(ene_v/val);
	ene_out[id]=sqrtf(ene_ld/val);
	}
}
