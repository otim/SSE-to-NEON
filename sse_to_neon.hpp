//
//  sse_to_neon.hpp
//  neon_test
//
//  Created by Tim Oberhauser on 11/16/13.
//  Copyright (c) 2013 Tim Oberhauser. All rights reserved.
//

#ifndef neon_test_sse_to_neon_hpp
#define neon_test_sse_to_neon_hpp

#include <arm_neon.h>

typedef int16x8_t __m128i;
typedef float32x4_t __m128;

//#define __constrange(min,max)  const
//#define __transfersize(size)

// ADDITION
inline __m128i _mm_add_epi16(const __m128i& a, const __m128i& b){
    return vaddq_s16(reinterpret_cast<int16x8_t>(a),reinterpret_cast<int16x8_t>(b));
}

inline __m128 _mm_add_ps(const __m128& a, const __m128& b){
    return vaddq_f32(a,b);
}


// SUBTRACTION
inline __m128i _mm_sub_epi16(const __m128i& a, const __m128i& b){
    return vsubq_s16(reinterpret_cast<int16x8_t>(a),reinterpret_cast<int16x8_t>(b));
}

inline __m128 _mm_sub_ps(const __m128& a, const __m128& b){
    return vsubq_f32(a,b);
}


// MULTIPLICATION
inline __m128i _mm_mullo_epi16(const __m128i& a, const __m128i& b){
    return vqrdmulhq_s16(reinterpret_cast<int16x8_t>(a),reinterpret_cast<int16x8_t>(b));
}

inline __m128 _mm_mul_ps(const __m128& a, const __m128& b){
    return vmulq_f32(a,b);
}


// SET VALUE
inline __m128i _mm_set1_epi16(const int16_t w){
    return vmovq_n_s16(w);
}

inline __m128i _mm_setzero_si128(){
    return vmovq_n_s16(0);
}

inline __m128 _mm_set1_ps(const float32_t& w){
    return vmovq_n_f32(w);
}


// STORE
inline void _mm_storeu_si128(__m128i* p, __m128i& a){
//    a = *p; // TODO vst1q_s16
    vst1q_s16(reinterpret_cast<int16_t*>(p),reinterpret_cast<int16x8_t>(a));
}

inline void _mm_store_ps(float32_t* p, __m128&a){
    vst1q_f32(p,a);
}


// LOAD
inline __m128i _mm_loadu_si128(__m128i* p){//For SSE address p does not need be 16-byte aligned
    return reinterpret_cast<int16x8_t>(vld1q_s16(reinterpret_cast<int16_t*>(p)));
}

inline __m128i _mm_load_si128(__m128i* p){//For SSE address p must be 16-byte aligned
    return reinterpret_cast<int16x8_t>(vld1q_s16(reinterpret_cast<int16_t*>(p)));
}

inline __m128 _mm_load_ps(float32_t* p){
    return vld1q_f32(p);
}


// SHIFT OPERATIONS
//__m128i _mm_srai_epi16(__m128i a, __constrange(1,16) int count);
inline __m128i _mm_srai_epi16(const __m128i& a, const int count){
//    return (int16x8_t){a[0]>>count,usw.};
    int16x8_t b = vmovq_n_s16(-count);
    return vshlq_s16(a,b);
//    return vrshrq_n_s16(a, count);// TODO Argument to '__builtin_neon_vrshrq_n_v' must be a constant integer
}


// MIN/MAX OPERATIONS
inline __m128i _mm_max_ps(const __m128i& a, const __m128i& b){
    return reinterpret_cast<__m128i>(vmaxq_f32(reinterpret_cast<float32x4_t>(a),reinterpret_cast<float32x4_t>(b)));
}


// SINGLE ELEMENT ACCESS
inline int16_t _mm_extract_epi16(__m128i& a, int index){
    return (reinterpret_cast<int16_t*>(&a))[index];
//    return vgetq_lane_s16(a,index);// TODO Argument to '__builtin_neon_vgetq_lane_i16' must be a constant integer
}


// MISCELLANOUS
inline __m128i _mm_sad_epu8 (const __m128i& a, __m128i& b){
    uint64x2_t sad = reinterpret_cast<uint64x2_t>(vabdq_u8(a,b));
    sad = reinterpret_cast<uint64x2_t>(vpaddlq_u8(reinterpret_cast<uint8x16_t>(sad)));
    sad = reinterpret_cast<uint64x2_t>(vpaddlq_u16(reinterpret_cast<uint16x8_t>(sad)));
    sad = vpaddlq_u32(reinterpret_cast<uint32x4_t>(sad));
    return reinterpret_cast<__m128i>(sad);
}


// LOGICAL OPERATIONS
inline __m128 _mm_and_ps(__m128& a, __m128& b){
    __m128 result;
    float32_t* result_ptr = reinterpret_cast<float32_t*>(&result);
    float32_t* a_ptr = reinterpret_cast<float32_t*>(&a);
    float32_t* b_ptr = reinterpret_cast<float32_t*>(&b);
    result_ptr[0] = a_ptr[0] && b_ptr[0];
    result_ptr[1] = a_ptr[1] && b_ptr[1];
    result_ptr[2] = a_ptr[2] && b_ptr[2];
    result_ptr[3] = a_ptr[3] && b_ptr[3];
    return result;
}


// CONVERSIONS
inline __m128i _mm_packus_epi16 (const __m128i a, const __m128i b){
    __m128i result = _mm_setzero_si128();// = new __m128i;
    int8x8_t* a_narrow = reinterpret_cast<int8x8_t*>(&result);
    int8x8_t* b_narrow = &a_narrow[1];
    *a_narrow = vqmovun_s16(a);//vqmovn_s16(a);
    *b_narrow = vqmovun_s16(b);//vqmovn_s16(b);
//    __m128i result;
//    vst1_s8(&a_narrow,reinterpret_cast<int8x8_t*>(&result)[0]);
//    vst1_s8(&b_narrow,reinterpret_cast<int8x8_t*>(&result)[8]);
    return result;
}

inline __m128i _mm_unpacklo_epi8(__m128i a, const __m128i dummy_zero){
    // dummy_zero is a dummy variable
    uint8x8_t* a_low = reinterpret_cast<uint8x8_t*>(&a);
    return reinterpret_cast<__m128i>(vmovl_u8(*a_low));
}

inline __m128i _mm_unpackhi_epi8(__m128i a, const __m128i dummy_zero){
    // dummy_zero is a dummy variable
    uint8x8_t* a_low = reinterpret_cast<uint8x8_t*>(&a);
    return reinterpret_cast<__m128i>(vmovl_u8(a_low[1]));
}




#endif
