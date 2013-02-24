#include "convol.cu"
#include "fifconvol_pyr.hpp"


fifconvol_pyr::fifconvol_pyr(unsigned int w, unsigned int h)
{
	d_w = w;
	d_h = h;
	p_w = iAlignUp(d_w, BLOCK_DIM);
	p_h = iAlignUp(d_h, BLOCK_HEI);
	U8Tex = cudaCreateChannelDesc<unsigned char>();	
	F32Tex= cudaCreateChannelDesc<float>();	
	cudaMallocArray(&d_input, &U8Tex, p_w, p_h); 
	cudaMallocArray(&d_inputf, &F32Tex, p_w, p_h); 
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_resz, p_w*p_h*sizeof(float)));  
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_RP, p_w*p_h*sizeof(short int)));
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_ene, 4*p_w*p_h*sizeof(float)));  
	grid.x = iDivUp(d_w, BLOCK_DIM);
	grid.y = iDivUp(d_h, BLOCK_HEI);
	threads.x = BLOCK_DIM;
	threads.y = BLOCK_HEI;
	CUDA_SAFE_CALL( cudaMallocHost( (void **)&host_pad, p_w*p_h*sizeof(unsigned char)));  
	//setConvolutionKernels
	cudaMemcpyToSymbol(d_ker_h_r, ker_h_r, 25 * sizeof(float));
	cudaMemcpyToSymbol(d_ker_h_i, ker_h_i, 25 * sizeof(float));
	cudaMemcpyToSymbol(d_ker_d_r, ker_d_r, 25 * sizeof(float));
	cudaMemcpyToSymbol(d_ker_d_i, ker_d_i, 25 * sizeof(float));
	cudaMemcpyToSymbol(d_ker_v_r, ker_v_r, 25 * sizeof(float));
	cudaMemcpyToSymbol(d_ker_v_i, ker_v_i, 25 * sizeof(float));
	cudaMemcpyToSymbol(d_ker_ld_r, ker_ld_r, 25 * sizeof(float));
	cudaMemcpyToSymbol(d_ker_ld_i, ker_ld_i, 25 * sizeof(float));	
	Texref.filterMode = cudaFilterModeLinear;	
	Texrefresz.filterMode = cudaFilterModeLinear;	
}

fifconvol_pyr::~fifconvol_pyr()
{
	cudaFreeArray(d_input);
	cudaFreeArray(d_inputf);
	cudaFree(d_resz);
	cudaFree(d_RP);
	cudaFree(host_pad);
	cudaFree(d_ene);
}

void fifconvol_pyr::convol_RP5(const unsigned char* h_input, short int* h_RP)
{
	const unsigned char* host_in;
	host_in = h_input;
	if (d_w != p_w || d_h !=p_h)
	{
	pad_onhost(h_input);
	host_in = host_pad;
	}
	cudaMemcpyToArray(d_input, 0, 0, host_in,p_w * p_h, cudaMemcpyHostToDevice);
	// Set up the texture parameters for bilinear interpolation & clamping
	//cudaUnbindTexture(Texref);
	cudaUnbindTexture(Texrefresz);
	//cudaBindTextureToArray(Texref, d_input);
	cudaBindTextureToArray(Texrefresz, d_input);
	Textureconvol_RP5_Arr_ori<<<grid, threads>>>(d_RP, p_w);	
	CUDA_SAFE_CALL(cudaMemcpy(h_RP, d_RP, p_w*p_h*sizeof(short int), cudaMemcpyDeviceToHost));
}

void fifconvol_pyr::convol_RP5_d(short int* RP_out, int ori_width, int ori_height)
{
	float w_ratio=float(ori_width)/float(d_w);
	float h_ratio=float(ori_height)/float(d_h);
	if (w_ratio<2 && h_ratio<2)
		resize_bilinear<<<grid, threads>>>(d_resz, w_ratio, h_ratio, p_w);
	else if (w_ratio<4 && h_ratio<4)
		resize_4N<<<grid, threads>>>(d_resz, w_ratio, h_ratio, p_w);
	else
		resize_16N<<<grid, threads>>>(d_resz, w_ratio, h_ratio, p_w);
	cudaMemcpyToArray(d_inputf, 0, 0, d_resz, p_w*p_h*4, cudaMemcpyDeviceToDevice);
	cudaUnbindTexture(Texref);
	cudaBindTextureToArray(Texref, d_inputf);
	Textureconvol_RP5_Arr<<<grid, threads>>>(d_RP, p_w);	
	CUDA_SAFE_CALL(cudaMemcpy(RP_out, d_RP, p_w*p_h*sizeof(short int), cudaMemcpyDeviceToHost));
}

void fifconvol_pyr::convol_ene_d(float* ene_out, int ori_width, int ori_height)
{
	float w_ratio=float(ori_width)/float(d_w);
	float h_ratio=float(ori_height)/float(d_h);
	if (w_ratio<2 && h_ratio<2)
		resize_bilinear<<<grid, threads>>>(d_resz, w_ratio, h_ratio, p_w);
	else if (w_ratio<4 && h_ratio<4)
		resize_4N<<<grid, threads>>>(d_resz, w_ratio, h_ratio, p_w);
	else
		resize_16N<<<grid, threads>>>(d_resz, w_ratio, h_ratio, p_w);
	cudaMemcpyToArray(d_inputf, 0, 0, d_resz, p_w*p_h*4, cudaMemcpyDeviceToDevice);
	cudaUnbindTexture(Texref);
	cudaBindTextureToArray(Texref, d_inputf);
	Textureconvol_ene_Arr<<<grid, threads>>>(d_ene, p_w);
	CUDA_SAFE_CALL(cudaMemcpy(ene_out, d_ene, 4*p_w*p_h*sizeof(float), cudaMemcpyDeviceToHost));
}

void fifconvol_pyr::convol_ene_d_thr(float* ene_out, int ori_width, int ori_height, float beta)
{
	float w_ratio=float(ori_width)/float(d_w);
	float h_ratio=float(ori_height)/float(d_h);
	if (w_ratio<2 && h_ratio<2)
		resize_bilinear<<<grid, threads>>>(d_resz, w_ratio, h_ratio, p_w);
	else if (w_ratio<4 && h_ratio<4)
		resize_4N<<<grid, threads>>>(d_resz, w_ratio, h_ratio, p_w);
	else
		resize_16N<<<grid, threads>>>(d_resz, w_ratio, h_ratio, p_w);
	cudaMemcpyToArray(d_inputf, 0, 0, d_resz, p_w*p_h*4, cudaMemcpyDeviceToDevice);
	cudaUnbindTexture(Texref);
	cudaBindTextureToArray(Texref, d_inputf);
	Textureconvol_ene_Arr_thr<<<grid, threads>>>(d_ene, beta, p_w);
	CUDA_SAFE_CALL(cudaMemcpy(ene_out, d_ene, 4*p_w*p_h*sizeof(float), cudaMemcpyDeviceToHost));
}

void fifconvol_pyr::bind_input(const unsigned char* h_input)
{	
	const unsigned char* host_in = h_input;
	if (d_w != p_w || d_h !=p_h)
	{
	pad_onhost(h_input);
	host_in = host_pad;
	}
	cudaMemcpyToArray(d_input, 0, 0, host_in,p_w * p_h, cudaMemcpyHostToDevice);
	// Set up the texture parameters for bilinear interpolation & clamping
	cudaUnbindTexture(Texrefresz);
	cudaBindTextureToArray(Texrefresz, d_input);
}

void fifconvol_pyr::pad_onhost(const unsigned char* h_input)
{
	for (int i=0; i<p_h; i++)
		for (int j=0; j<p_w; j++)
		{
		if (i<d_h)
			{
			if (j<d_w)
			host_pad[i*p_w+j]=h_input[i*d_w+j];
			else
			host_pad[i*p_w+j]=h_input[i*d_w+d_w-1];
			}
		else
		host_pad[i*p_w+j]=host_pad[(d_h-1)*p_w+j];
		}
}

void fifconvol_pyr::get_padsz(int& w, int& h)
{
	w = p_w;
	h = p_h;
}

