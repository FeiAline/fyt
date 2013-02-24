#include <cutil_inline.h>

//texture<float, 2, cudaReadModeElementType> texref;

//////////////////////
//my area filter
//////////////////////

template<class T, class R>  // return type, texture type
__device__
R tex2D4N(const texture<T, 2, cudaReadModeNormalizedFloat> tex_ref, float x, float y)
{
	R r;
	r = 0.25f * (tex2D(tex_ref, x-0.5f, y-0.5f)+tex2D(tex_ref, x-0.5f, y+0.5f)+
		     tex2D(tex_ref, x+0.5f, y-0.5f)+tex2D(tex_ref, x+0.5f, y+0.5f)); 

    return r;
}

template<class T, class R>  // return type, texture type
__device__
R tex2D16N(const texture<T, 2, cudaReadModeNormalizedFloat> tex_ref, float x, float y)
{
	R r;
	r = 0.0625f * (tex2D(tex_ref, x-1.5f, y-1.5f) + tex2D(tex_ref, x-0.5f, y-1.5f) + tex2D(tex_ref, x+0.5f, y-1.5f) + tex2D(tex_ref, x+1.5f, y-1.5f)+
		       tex2D(tex_ref, x-1.5f, y-0.5f) + tex2D(tex_ref, x-0.5f, y-0.5f) + tex2D(tex_ref, x+0.5f, y-0.5f) + tex2D(tex_ref, x+1.5f, y-0.5f)+
		       tex2D(tex_ref, x-1.5f, y+0.5f) + tex2D(tex_ref, x-0.5f, y+0.5f) + tex2D(tex_ref, x+0.5f, y+0.5f) + tex2D(tex_ref, x+1.5f, y+0.5f)+
		       tex2D(tex_ref, x-1.5f, y+1.5f) + tex2D(tex_ref, x-0.5f, y+1.5f) + tex2D(tex_ref, x+0.5f, y+1.5f) + tex2D(tex_ref, x+1.5f, y+1.5f)); 
    return r;
}


/*__device__
float tex2D4N(const texture<unsigned char, 2, cudaReadModeElementType> tex_ref, float x, float y)
{
	float r;
	r = 0.25f * (tex2D(tex_ref, x-0.5f, y-0.5f)+tex2D(tex_ref, x-0.5f, y+0.5f)+
		     tex2D(tex_ref, x+0.5f, y-0.5f)+tex2D(tex_ref, x+0.5f, y+0.5f)); 

    return r;
}

__device__
float tex2D16N(const texture<unsigned char, 2, cudaReadModeElementType> tex_ref, float x, float y)
{
	float r;
	r = 0.0625f * (tex2D(tex_ref, x-1.5f, y-1.5f) + tex2D(tex_ref, x-0.5f, y-1.5f) + tex2D(tex_ref, x+0.5f, y-1.5f) + tex2D(tex_ref, x+1.5f, y-1.5f)+
		       tex2D(tex_ref, x-1.5f, y-0.5f) + tex2D(tex_ref, x-0.5f, y-0.5f) + tex2D(tex_ref, x+0.5f, y-0.5f) + tex2D(tex_ref, x+1.5f, y-0.5f)+
		       tex2D(tex_ref, x-1.5f, y+0.5f) + tex2D(tex_ref, x-0.5f, y+0.5f) + tex2D(tex_ref, x+0.5f, y+0.5f) + tex2D(tex_ref, x+1.5f, y+0.5f)+
		       tex2D(tex_ref, x-1.5f, y+1.5f) + tex2D(tex_ref, x-0.5f, y+1.5f) + tex2D(tex_ref, x+0.5f, y+1.5f) + tex2D(tex_ref, x+1.5f, y+1.5f)); 
    return r;
}*/

