/**
********************************************************************************
*
* @file: adi_sharcfx_fully_connected.cpp
*
* @brief: contains optimized version of fully connected layer
*
* @details: contains optimized version of fully connected layer for 8bit and 16bit integer input
*
*******************************************************************************
 Copyright(c) 2024 Analog Devices, Inc. All Rights Reserved. This software is
 proprietary & confidential to Analog Devices, Inc. and its licensors. By using
 this software you agree to the terms of the associated Analog Devices License
 Agreement.
*******************************************************************************
*/

/*============= I N C L U D E S =============*/
#include "adi_sharcfx_nn.h"

/* Number of outputs processed per vectorised quantisation step; equals PDX_M */
#define FC_VEC_OUTPUT_CHUNK  8

/*============= C O D E =============*/

/**
 *******************************************************************************
 * Function: adi_sharcfx_fully_connected_int16
 * @brief Optimised fully-connected layer for int16 input.
 *
 * @details Vectorised implementation of a fully-connected (dense) layer for
 *          16-bit integer activations and 8-bit weights. Handles both
 *          filter-depth multiples of 16 and non-multiples via separate paths.
 *
 * Parameters:
 * @param [in]  pInputBuffer         Input activation buffer (int16).
 * @param [in]  pWeightsBuffer       Weight buffer (int8).
 * @param [in]  pBiasBuffer          Bias buffer (int64); may be NULL.
 * @param [out] pOutputBuffer        Output activation buffer (int16).
 * @param [in]  nFilterDepth         Number of input features per neuron.
 * @param [in]  nOutsize             Number of output neurons.
 * @param [in]  nBatches             Batch count.
 * @param [in]  nQuantizedMultiplier Quantization multiplier (TFLM scheme).
 * @param [in]  nQuantizedShift      Quantization shift (TFLM scheme).
 * @param [in]  nInputOffset         Input zero-point offset.
 * @param [in]  nFilterOffset        Filter zero-point offset.
 * @param [in]  nOutputOffset        Output zero-point offset.
 * @param [in]  output_activation_min Activation minimum clamp value.
 * @param [in]  output_activation_max Activation maximum clamp value.
 * @return None
 *******************************************************************************
 */
void adi_sharcfx_fully_connected_int16(const int16_t* pInputBuffer,
                                       const int8_t* pWeightsBuffer,
                                       const int64_t* pBiasBuffer,
                                       int16_t* pOutputBuffer,
                                       int32_t nFilterDepth,
                                       int32_t nOutsize,
                                       int32_t nBatches,
                                       uint32_t nQuantizedMultiplier,
                                       int32_t nQuantizedShift,
                                       int32_t nInputOffset,
                                       int32_t nFilterOffset,
                                       int32_t nOutputOffset,
                                       int32_t output_activation_min,
                                       int32_t output_activation_max)
{
    int16_t* __restrict outp = (int16_t*) pOutputBuffer;

    const immediate Lane=0;
    //Defining the input and filter offsets
    xb_vec2Mx16 vInZP = PDX_REP_2MX16((xb_vec2Mx16)nInputOffset,Lane);//Replicates the lane of data specified, across all lanes of a vector register
    xb_vec2Mx16 vFilterZP = PDX_REP_2MX16((xb_vec2Mx16)nFilterOffset,Lane);//Replicates the lane of data specified, across all lanes of a vector register

    xb_vec2Mx16 vin,vwt;
    xb_int32 temp;
    xb_int40 sat_sum;
    xb_int80 product;

    vbool4M temp_mask;
    vbool2M acc_mask;

    int32_t nPixProcessed=0;
    xb_vec2Mx40 acc = 0;

    if(nFilterDepth % 16)
    {
        for (int b = 0; b < nBatches; b++)
        {
            //No of filter is equals to the nOutsize
            for (int32_t nChannelCnt = 0; nChannelCnt < nOutsize; nChannelCnt++)
            {
                acc = 0;//reset accumulator

                xb_vec2Mx16 *inp = (xb_vec2Mx16 *)(pInputBuffer + b*nFilterDepth);    //Reinitialise input pointer for each output pixel
                valign ina; // define align vector
                ina=PDX_LA_2MX16_PP (inp); // prime, NOP if a[] is aligned

                xb_vec2Mx8 *wtp = (xb_vec2Mx8 *) (pWeightsBuffer + nFilterDepth*nChannelCnt); //Move to next filter for each output pixel
                valign wta;    // define align vector
                wta=PDX_LA_2MX8_PP (wtp);

                //for all the input pixels
                for (nPixProcessed= 0; nFilterDepth - nPixProcessed > 2*PDX_M; nPixProcessed += (2*PDX_M))
                {
                    PDX_LA_2MX16_XP (vin, ina, inp, 2*PDX_M*sizeof(int16_t)); // load aligned, extend
                    PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M); // load aligned, extend
                    vin+=vInZP;        //Add input offset
                    vwt+=vFilterZP;    //Add filter offset

                    PDX_MULAQW_2MX16(acc,vwt,vin);
                }

                PDX_LA_2MX16_XP (vin, ina, inp, 0); // load aligned, extend
                PDX_LA16_2MX8_XP (vwt, wta, wtp, 0); // load aligned, extend
                vin+=vInZP;        //Add input offset
                vwt+=vFilterZP;    //Add filter offset

                temp_mask = PDX_MOVB_AU32((0b1<<(nFilterDepth % 16)) - 1);
                acc_mask = PDX_CVTBB2M_B4M_L(temp_mask);
                //multiply and accumulate
                PDX_MULAQW_2MX16_T(acc,vwt,vin,acc_mask);

                sat_sum = PDX_RADD_2MX40(acc);                                    //PDX_RADD_2MX40: Adds across all 16lanes of acc and returns sum
                //adding bias<<1 to compensate for sign bit during multiplication
                //doubling index for bias buffer to account for 32bit bias instead of 64bit
                if(pBiasBuffer)
                {
                    sat_sum += (xb_int40)(pBiasBuffer[nChannelCnt]<<1);
                }
                temp = (xb_int32)((int32_t)((int64_t)PDX_CVT64_40(sat_sum)));    //convert 40bit var into 32bit to perform multiplication

                product = PDX_MULW_32(temp, (uint32_t)nQuantizedMultiplier);    //multiply with the quantization multiplier; product(80bit) = temp(32bit) * nQuantizedMultiplier(32bit)
                product =  PDX_SLA_80(product, (xb_int32)nQuantizedShift);        //shift result by quantization multiplier
                temp = PDX_PACKQSRV_80(product,2);                                //packs 80bit product into 32bit var with saturation and rounding
                temp+= (xb_int32)nOutputOffset;                                    //add output offset
                temp = MIN(temp, (xb_int32)output_activation_max);
                temp = MAX(temp, (xb_int32)output_activation_min);//saturation check to store result
                *outp++ =(int16_t)((int32_t)temp);                                //store result as 16-bit data
            }
        }
    }
    else{
        for (int b = 0; b < nBatches; b++)
        {
            //No of filter is equals to the nOutsize
            for (int32_t nChannelCnt = 0; nChannelCnt < nOutsize; nChannelCnt++)
            {
                acc = 0;//reset accumulator

                xb_vec2Mx16 * inp = (xb_vec2Mx16 *)(pInputBuffer + b*nFilterDepth);    //Reinitialise input pointer for each output pixel
                valign ina; // define align vector
                ina=PDX_LA_2MX16_PP (inp); // prime, NOP if a[] is aligned

                xb_vec2Mx8 *wtp = (xb_vec2Mx8 *) (pWeightsBuffer + nFilterDepth*nChannelCnt); //Move to next filter for each output pixel
                valign wta;    // define align vector
                wta=PDX_LA_2MX8_PP (wtp);

                //for all the input pixels
                for (int32_t nFilterCnt = 0; nFilterCnt < nFilterDepth; nFilterCnt += (2*PDX_M))
                {
                    PDX_LA_2MX16_XP (vin, ina, inp, 2*PDX_M*sizeof(int16_t)); // load aligned, extend
                    PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M);
                    vin+=vInZP;        //Add input offset
                    vwt+=vFilterZP;    //Add filter offset

                    PDX_MULAQW_2MX16(acc,vwt,vin);
                }
                sat_sum = PDX_RADD_2MX40(acc);                                    //PDX_RADD_2MX40: Adds across all 16lanes of acc and returns sum
                //adding bias<<1 to compensate for sign bit during multiplication
                //doubling index for bias buffer to account for 32bit bias instead of 64bit
                if(pBiasBuffer)
                {
                    sat_sum += (xb_int40)(pBiasBuffer[nChannelCnt]<<1);
                }
                temp = (xb_int32)((int32_t)((int64_t)PDX_CVT64_40(sat_sum)));    //convert 40bit var into 32bit to perform multiplication

                product = PDX_MULW_32(temp, (uint32_t)nQuantizedMultiplier);    //multiply with the quantization multiplier; product(80bit) = temp(32bit) * nQuantizedMultiplier(32bit)
                product =  PDX_SLA_80(product, (xb_int32)nQuantizedShift);        //shift result by quantization multiplier
                temp = PDX_PACKQSRV_80(product,2);                                //packs 80bit product into 32bit var with saturation and rounding
                temp+= (xb_int32)nOutputOffset;                                    //add output offset
                temp = MIN(temp, (xb_int32)output_activation_max);
                temp = MAX(temp, (xb_int32)output_activation_min);                //saturation check to store result
                *outp++ =(int16_t)((int32_t)temp);                                //store result as 16-bit data
            }
        }
    }
}


#ifdef PROFILE_FC_INTERNALS
#define __PRE_FX_COMPATIBILITY
#define DO_CYCLE_COUNTS
#include <cycle_count.h>
	static int8_t fc_count = 0;
#endif

/**
 *******************************************************************************
 * Function: adi_sharcfx_fully_connected_int8
 * @brief Optimised fully-connected layer for int8 input.
 *
 * @details Vectorised implementation of a fully-connected (dense) layer for
 *          8-bit integer activations and weights. Handles both
 *          filter-depth multiples of 16 and non-multiples via separate paths.
 *
 * Parameters:
 * @param [in]  pInputBuffer         Input activation buffer (int8).
 * @param [in]  pWeightsBuffer       Weight buffer (int8).
 * @param [in]  pBiasBuffer          Bias buffer (int32); may be NULL.
 * @param [out] pOutputBuffer        Output activation buffer (int8).
 * @param [in]  in_feat              Number of input features per neuron.
 * @param [in]  out_feat             Number of output neurons.
 * @param [in]  nBatches             Batch count.
 * @param [in]  nQuantizedMultiplier Quantization multiplier (TFLM scheme).
 * @param [in]  nQuantizedShift      Quantization shift (TFLM scheme).
 * @param [in]  nInputOffset         Input zero-point offset.
 * @param [in]  nFilterOffset        Filter zero-point offset.
 * @param [in]  nOutputOffset        Output zero-point offset.
 * @param [in]  output_activation_min Activation minimum clamp value.
 * @param [in]  output_activation_max Activation maximum clamp value.
 * @return None
 *******************************************************************************
 */
void adi_sharcfx_fully_connected_int8(const int8_t* pInputBuffer,
                                      const int8_t* pWeightsBuffer,
                                      const int32_t* pBiasBuffer,
                                      int8_t* pOutputBuffer,
                                      int32_t in_feat,
                                      int32_t out_feat,
                                      int32_t nBatches,
                                      uint32_t nQuantizedMultiplier,
                                      int32_t nQuantizedShift,
                                      int32_t nInputOffset,
                                      int32_t nFilterOffset,
                                      int32_t nOutputOffset,
                                      int32_t output_activation_min,
                                      int32_t output_activation_max)
{
#ifdef PROFILE_FC_INTERNALS
	cycle_t var_x=0;cycle_t cyc_x=0;
	START_CYCLE_COUNT (var_x);
#endif
    int8_t* __restrict outp = (int8_t*) pOutputBuffer;

    const immediate Lane=0;
    //Defining the input and filter offsets
    xb_vec2Mx16 vInZP = PDX_REP_2MX16((xb_vec2Mx16)nInputOffset,Lane);//Replicates the lane of data specified, across all lanes of a vector register
    xb_vec2Mx16 vFilterZP = PDX_REP_2MX16((xb_vec2Mx16)nFilterOffset,Lane);//Replicates the lane of data specified, across all lanes of a vector register
    xb_vec2Mx16 vin,vwt;
    xb_int32 temp;
    xb_int40 sat_sum;
    xb_int80 product;

    vbool4M temp_mask;
    vbool2M acc_mask;

    int32_t nPixProcessed=0;

    xb_vec2Mx40 acc = 0;
    xb_vec2Mx8 *inp = (xb_vec2Mx8 *)pInputBuffer;
#ifdef PROFILE_FC_INTERNALS
    STOP_CYCLE_COUNT (cyc_x, var_x);
	PRINT_INFO("Number of cycles for pre-loop setup init: \t%lu\n",cyc_x);
#endif
    if(in_feat % 16){
        int32_t in_feat_16mul = in_feat - (in_feat % 16);
        for (int b = 0; b < nBatches; b++){
            int32_t nChannelCnt = 0;

            // Process FC_VEC_OUTPUT_CHUNK outputs at a time for vectorized quantization
            for (; nChannelCnt + FC_VEC_OUTPUT_CHUNK <= out_feat; nChannelCnt += FC_VEC_OUTPUT_CHUNK)
            {
                xb_vecMx32 sums_vec;
                xb_int32* sums_ptr = (xb_int32*)&sums_vec;

                // Accumulate FC_VEC_OUTPUT_CHUNK RADD results into vector
                for(int32_t i = 0; i < FC_VEC_OUTPUT_CHUNK; i++)
                {
                    acc = 0;

                    inp = (xb_vec2Mx8 *)(pInputBuffer + b*in_feat);
                    valign ina;
                    ina=PDX_LA_2MX8_PP (inp);

                    xb_vec2Mx8 *wtp = (xb_vec2Mx8 *) (pWeightsBuffer + in_feat*(nChannelCnt+i));
                    valign wta;
                    wta=PDX_LA_2MX8_PP (wtp);

                    for (nPixProcessed= 0; nPixProcessed < in_feat_16mul; nPixProcessed += (2*PDX_M))
                    {
                        PDX_LA16_2MX8_XP (vin, ina, inp, 2*PDX_M);
                        PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M);
                        vin+=vInZP;
                        vwt+=vFilterZP;
                        PDX_MULAQW_2MX16(acc,vwt,vin);
                    }

                    PDX_LA16_2MX8_XP (vin, ina, inp, 2*PDX_M);
                    PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M);
                    vin+=vInZP;
                    vwt+=vFilterZP;
                    temp_mask = PDX_MOVB_AU32((0b1<<(in_feat % 16)) - 1);
                    acc_mask = PDX_CVTBB2M_B4M_L(temp_mask);
                    PDX_MULAQW_2MX16_T(acc,vwt,vin,acc_mask);

                    sat_sum = PDX_RADD_2MX40(acc);
                    *(sums_ptr + i) = (xb_int32)((int32_t)((int64_t)PDX_CVT64_40(sat_sum)));
                }

                // Add 8 biases in parallel if bias buffer exists
                if(pBiasBuffer)
                {
                    xb_vecMx32 bias_vec;
                    xb_int32* bias_ptr = (xb_int32*)&bias_vec;
                    for(int32_t i = 0; i < FC_VEC_OUTPUT_CHUNK; i++)
                    {
                        *(bias_ptr + i) = pBiasBuffer[nChannelCnt + i];
                    }
                    sums_vec = PDX_ADD_MX32(sums_vec, bias_vec);
                    sums_vec = PDX_ADD_MX32(sums_vec, bias_vec); // Add twice for <<1
                }

                // Vectorized quantization for 8 outputs
                xb_vecMx32 multiplier_vec = PDX_REP_MX32((xb_vecMx32)nQuantizedMultiplier, Lane);
                xb_vecMx80 products_vec = multiplier_vec * sums_vec; // 8-way 32x32→80 multiply
                products_vec = PDX_SLS_MX80(products_vec, (xb_vecMx32)nQuantizedShift); // 8-way shift
                xb_vecMx32 results_vec = PDX_PACKQSRV_MX80(products_vec, 2); // 8-way pack with rounding

                // Add output offset and clamp
                xb_vecMx32 offset_vec = PDX_REP_MX32((xb_vecMx32)nOutputOffset, Lane);
                results_vec = PDX_ADD_MX32(results_vec, offset_vec);
                xb_vecMx32 vmax = PDX_REP_MX32((xb_vecMx32)output_activation_max, Lane);
                xb_vecMx32 vmin = PDX_REP_MX32((xb_vecMx32)output_activation_min, Lane);
                results_vec = PDX_MIN_MX32(results_vec, vmax);
                results_vec = PDX_MAX_MX32(results_vec, vmin);

                // Store 8 int8 outputs at once
                xb_vecMx8* outp_vec = (xb_vecMx8*)outp;
                valign outa = PDX_LA_MX8_PP(outp_vec);
                PDX_SAV32_MX8_XP(results_vec, outa, outp_vec, FC_VEC_OUTPUT_CHUNK);
                PDX_SAPOS_MX8_FP(outa, outp_vec);
                outp = (int8_t*)outp_vec;
            }

            // Handle remaining outputs (< 8)
            for (; nChannelCnt < out_feat; nChannelCnt++)
            {
                acc = 0;

                inp = (xb_vec2Mx8 *)(pInputBuffer + b*in_feat);
                valign ina;
                ina=PDX_LA_2MX8_PP (inp);

                xb_vec2Mx8 *wtp = (xb_vec2Mx8 *) (pWeightsBuffer + in_feat*nChannelCnt);
                valign wta;
                wta=PDX_LA_2MX8_PP (wtp);

                for (nPixProcessed= 0; nPixProcessed < in_feat_16mul; nPixProcessed += (2*PDX_M))
                {
                    PDX_LA16_2MX8_XP (vin, ina, inp, 2*PDX_M);
                    PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M);
                    vin+=vInZP;
                    vwt+=vFilterZP;
                    PDX_MULAQW_2MX16(acc,vwt,vin);
                }

                PDX_LA16_2MX8_XP (vin, ina, inp, 2*PDX_M);
                PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M);
                vin+=vInZP;
                vwt+=vFilterZP;
                temp_mask = PDX_MOVB_AU32((0b1<<(in_feat % 16)) - 1);
                acc_mask = PDX_CVTBB2M_B4M_L(temp_mask);
                PDX_MULAQW_2MX16_T(acc,vwt,vin,acc_mask);

                sat_sum = PDX_RADD_2MX40(acc);
                if(pBiasBuffer)
                {
                    sat_sum += (xb_int40)(pBiasBuffer[nChannelCnt]<<1);
                }
                temp = (xb_int32)((int32_t)((int64_t)PDX_CVT64_40(sat_sum)));

                product = PDX_MULW_32(temp, (uint32_t)nQuantizedMultiplier);
                product =  PDX_SLA_80(product, (xb_int32)nQuantizedShift);
                temp = PDX_PACKQSRV_80(product,2);
                temp+= (xb_int32)nOutputOffset;
                temp = MIN(temp, (xb_int32)output_activation_max);
                temp = MAX(temp, (xb_int32)output_activation_min);
                *outp++ =(int8_t)((int32_t)temp);
            }
        }
    }
    else{
        for (int b = 0; b < nBatches; b++){
        	int32_t nChannelCnt = 0;
            // Process FC_VEC_OUTPUT_CHUNK outputs at a time for vectorized quantization
            for (; nChannelCnt + FC_VEC_OUTPUT_CHUNK <= out_feat; nChannelCnt += FC_VEC_OUTPUT_CHUNK)
            {
                xb_vecMx32 sums_vec;
                xb_int32* sums_ptr = (xb_int32*)&sums_vec;

                // Accumulate FC_VEC_OUTPUT_CHUNK RADD results into vector
                for(int32_t i = 0; i < FC_VEC_OUTPUT_CHUNK; i++)
                {
#ifdef PROFILE_FC_INTERNALS
	var_x=0;cyc_x=0;
	START_CYCLE_COUNT (var_x);
#endif
                    acc = 0;

                    inp = (xb_vec2Mx8 *)(pInputBuffer + b*in_feat);
                    valign ina;
                    ina=PDX_LA_2MX8_PP (inp);

                    xb_vec2Mx8 *wtp = (xb_vec2Mx8 *) (pWeightsBuffer + in_feat*(nChannelCnt+i));
                    valign wta;
                    wta=PDX_LA_2MX8_PP (wtp);
#ifdef PROFILE_FC_INTERNALS
    STOP_CYCLE_COUNT (cyc_x, var_x);
	PRINT_INFO("Number of cycles for MAC setup for %d/8 element: \t%lu\n",i, cyc_x);
#endif
#ifdef PROFILE_FC_INTERNALS
	var_x=0;cyc_x=0;
	START_CYCLE_COUNT (var_x);
#endif
#if 0
    xb_vec2Mx16 vin2,vwt2;
#endif
                    for (int32_t nFilterCnt = 0; nFilterCnt < in_feat; nFilterCnt += (2*PDX_M))
                    {
                        PDX_LA16_2MX8_XP (vin, ina, inp, 2*PDX_M);
                        PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M);
                        vin+=vInZP;
                        vwt+=vFilterZP;
                        PDX_MULAQW_2MX16(acc,vwt,vin);

                    }
#ifdef PROFILE_FC_INTERNALS
    STOP_CYCLE_COUNT (cyc_x, var_x);
	PRINT_INFO("Number of cycles for MAC loop for %d/8 element: \t%lu\n",i, cyc_x);
#endif
#ifdef PROFILE_FC_INTERNALS
	var_x=0;cyc_x=0;
	START_CYCLE_COUNT (var_x);
#endif
                    sat_sum = PDX_RADD_2MX40(acc);
#ifdef PROFILE_FC_INTERNALS
    STOP_CYCLE_COUNT (cyc_x, var_x);
	PRINT_INFO("Number of cycles for RADD intrinsic for %d/8 element: \t%lu\n",i,cyc_x);
#endif
#ifdef PROFILE_FC_INTERNALS
	var_x=0;cyc_x=0;
	START_CYCLE_COUNT (var_x);
#endif
                    *(sums_ptr + i) = (xb_int32)((int32_t)((int64_t)PDX_CVT64_40(sat_sum)));
#ifdef PROFILE_FC_INTERNALS
    STOP_CYCLE_COUNT (cyc_x, var_x);
	PRINT_INFO("Number of cycles for PDX_CVT64_40 intrinsic for %d/8 element: \t%lu\n",i,cyc_x);
#endif
                }


#ifdef PROFILE_FC_INTERNALS
	var_x=0;cyc_x=0;
	START_CYCLE_COUNT (var_x);
#endif
                // Add 8 biases in parallel if bias buffer exists
                if(pBiasBuffer)
                {
                    xb_vecMx32 bias_vec;
                    xb_int32* bias_ptr = (xb_int32*)&bias_vec;
                    for(int32_t i = 0; i < FC_VEC_OUTPUT_CHUNK; i++)
                    {
                        *(bias_ptr + i) = pBiasBuffer[nChannelCnt + i];
                    }
                    sums_vec = PDX_ADD_MX32(sums_vec, bias_vec);
                    sums_vec = PDX_ADD_MX32(sums_vec, bias_vec); // Add twice for <<1
                }
#ifdef PROFILE_FC_INTERNALS
    STOP_CYCLE_COUNT (cyc_x, var_x);
	PRINT_INFO("Number of cycles for bias add: \t%lu\n",cyc_x);
#endif
#ifdef PROFILE_FC_INTERNALS
	var_x=0;cyc_x=0;
	START_CYCLE_COUNT (var_x);
#endif
                // Vectorized quantization for 8 outputs
                xb_vecMx32 multiplier_vec = PDX_REP_MX32((xb_vecMx32)nQuantizedMultiplier, Lane);
                xb_vecMx80 products_vec = multiplier_vec * sums_vec; // 8-way 32x32→80 multiply
                products_vec = PDX_SLS_MX80(products_vec, (xb_vecMx32)nQuantizedShift); // 8-way shift
                xb_vecMx32 results_vec = PDX_PACKQSRV_MX80(products_vec, 2); // 8-way pack with rounding

                // Add output offset and clamp
                xb_vecMx32 offset_vec = PDX_REP_MX32((xb_vecMx32)nOutputOffset, Lane);
                results_vec = PDX_ADD_MX32(results_vec, offset_vec);
                xb_vecMx32 vmax = PDX_REP_MX32((xb_vecMx32)output_activation_max, Lane);
                xb_vecMx32 vmin = PDX_REP_MX32((xb_vecMx32)output_activation_min, Lane);
                results_vec = PDX_MIN_MX32(results_vec, vmax);
                results_vec = PDX_MAX_MX32(results_vec, vmin);

                // Store 8 int8 outputs at once
                xb_vecMx8* outp_vec = (xb_vecMx8*)outp;
                valign outa = PDX_LA_MX8_PP(outp_vec);
                PDX_SAV32_MX8_XP(results_vec, outa, outp_vec, FC_VEC_OUTPUT_CHUNK);
                PDX_SAPOS_MX8_FP(outa, outp_vec);
                outp = (int8_t*)outp_vec;
#ifdef PROFILE_FC_INTERNALS
    STOP_CYCLE_COUNT (cyc_x, var_x);
	PRINT_INFO("Number of cycles for quantization for 8 elements: \t%lu\n",cyc_x);
#endif
            }
#ifdef PROFILE_FC_INTERNALS
	var_x=0;cyc_x=0;
	START_CYCLE_COUNT (var_x);
#endif
            // Handle remaining outputs (< 8)
            for (; nChannelCnt < out_feat; nChannelCnt++)
            {
                acc = 0;

                inp = (xb_vec2Mx8 *)(pInputBuffer + b*in_feat);
                valign ina;
                ina=PDX_LA_2MX8_PP (inp);

                xb_vec2Mx8 *wtp = (xb_vec2Mx8 *) (pWeightsBuffer + in_feat*nChannelCnt);
                valign wta;
                wta=PDX_LA_2MX8_PP (wtp);

                for (int32_t nFilterCnt = 0; nFilterCnt < in_feat; nFilterCnt += (2*PDX_M))
                {
                    PDX_LA16_2MX8_XP (vin, ina, inp, 2*PDX_M);
                    PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M);
                    vin+=vInZP;
                    vwt+=vFilterZP;
                    PDX_MULAQW_2MX16(acc,vwt,vin);
                }
                sat_sum = PDX_RADD_2MX40(acc);
                if(pBiasBuffer)
                {
                    sat_sum += (xb_int40)(pBiasBuffer[nChannelCnt]<<1);
                }
                temp = (xb_int32)((int32_t)((int64_t)PDX_CVT64_40(sat_sum)));

                product = PDX_MULW_32(temp, (uint32_t)nQuantizedMultiplier);
                product =  PDX_SLA_80(product, (xb_int32)nQuantizedShift);
                temp = PDX_PACKQSRV_80(product,2);
                temp+= (xb_int32)nOutputOffset;
                temp = MIN(temp, (xb_int32)output_activation_max);
                temp = MAX(temp, (xb_int32)output_activation_min);
                *outp++ =(int8_t)((int32_t)temp);
            }
#ifdef PROFILE_FC_INTERNALS
    STOP_CYCLE_COUNT (cyc_x, var_x);
	PRINT_INFO("Number of cycles for remaining 8 elements: \t%lu\n",cyc_x);
#endif
        }
    }
}

/**
 *******************************************************************************
 * Function: adi_sharcfx_fully_connected_int8_reordered_weights
 * @brief Optimised fully-connected layer for int8 input with pre-reordered weights.
 *
 * @details Vectorised implementation of a fully-connected (dense) layer for
 *          8-bit activations using weights that have been pre-reordered for
 *          improved memory-access locality. Handles both filter-depth multiples
 *          of 16 and non-multiples.
 *
 * Parameters:
 * @param [in]  pInputBuffer         Input activation buffer (int8).
 * @param [in]  pWeightsBuffer       Pre-reordered weight buffer (int8).
 * @param [in]  pBiasBuffer          Bias buffer (int32); may be NULL.
 * @param [out] pOutputBuffer        Output activation buffer (int8).
 * @param [in]  nFilterDepth         Number of input features per neuron.
 * @param [in]  nOutsize             Number of output neurons.
 * @param [in]  nBatches             Batch count.
 * @param [in]  nQuantizedMultiplier Quantization multiplier (TFLM scheme).
 * @param [in]  nQuantizedShift      Quantization shift (TFLM scheme).
 * @param [in]  nInputOffset         Input zero-point offset.
 * @param [in]  nFilterOffset        Filter zero-point offset.
 * @param [in]  nOutputOffset        Output zero-point offset.
 * @param [in]  output_activation_min Activation minimum clamp value.
 * @param [in]  output_activation_max Activation maximum clamp value.
 * @return None
 *******************************************************************************
 */
void adi_sharcfx_fully_connected_int8_reordered_weights(const int8_t* pInputBuffer,
                                      const int8_t* pWeightsBuffer,
                                      const int32_t* pBiasBuffer,
                                      int8_t* pOutputBuffer,
                                      int32_t nFilterDepth,
                                      int32_t nOutsize,
                                      int32_t nBatches,
                                      uint32_t nQuantizedMultiplier,
                                      int32_t nQuantizedShift,
                                      int32_t nInputOffset,
                                      int32_t nFilterOffset,
                                      int32_t nOutputOffset,
                                      int32_t output_activation_min,
                                      int32_t output_activation_max)
{
    xb_vecMx8* __restrict outp = (xb_vecMx8 *)pOutputBuffer;
    valign outa = PDX_LA_MX8_PP(outp);

    xb_vec2Mx16 vInZP = nInputOffset;
    xb_vec2Mx16 vFilterZP = nFilterOffset;
    xb_vecMx32 vOutZP = nOutputOffset;

    xb_vecMx32 vmin = output_activation_min;
    xb_vecMx32 vmax = output_activation_max;

    xb_vec2Mx16 vin, vwt;

    int32_t nPixProcessed = 0;
    int32_t nPixLeft = nOutsize;

    xb_vec2Mx40 acc = 0;

    xb_vecMx32 first8, last8;
    xb_vecMx80 quant_acc, quant_acc2;
    const immediate round_mode = 2;
    xb_vecMx32 mult = nQuantizedMultiplier;
    xb_vecMx32 shift = nQuantizedShift;
    xb_vecMx32 fc_out, fc_out2;
    xb_vecMx32 vbias_l, vbias_h;

    int16_t nPixToWrite;
    int8_t* inp;

    for (int32_t b = 0; b < nBatches; b++)
    {
        nPixProcessed = 0;
        nPixLeft = nOutsize;

        /* ============================================================
         * Main loop: process 2*PDX_M (16) outputs per iteration
         * ============================================================ */
        while (nPixLeft >= 2 * PDX_M)
        {
            acc = 0;

            inp = (int8_t *)(pInputBuffer + b * nFilterDepth);

            xb_vec2Mx8 *wtp = (xb_vec2Mx8 *)(pWeightsBuffer + nPixProcessed * nFilterDepth);
            valign wta = PDX_LA_2MX8_PP(wtp);

            if (pBiasBuffer) {
                xb_vecMx32 *biasp = (xb_vecMx32 *)(pBiasBuffer + nPixProcessed);
                valign biasa = PDX_LA_MX32_PP(biasp);

                PDX_LA_MX32_XP(vbias_l, biasa, biasp, PDX_M * sizeof(int32_t));
                PDX_LA_MX32_XP(vbias_h, biasa, biasp, 0);

                vbias_l = PDX_SLS_MX32(vbias_l, 1);
                vbias_h = PDX_SLS_MX32(vbias_h, 1);
            }

            for (int32_t filterD = 0; filterD < nFilterDepth; filterD++)
            {
                vin = (*inp++);
                PDX_LA16_2MX8_XP(vwt, wta, wtp, 2 * PDX_M);
                vin += vInZP;
                vwt += vFilterZP;

                PDX_MULAQW_2MX16(acc, vwt, vin);
            }

            PDX_CVT32D_2MX40(last8, first8, acc);
            if (pBiasBuffer) {
                first8 += vbias_l;
                last8 += vbias_h;
            }

            quant_acc = mult * first8;
            quant_acc2 = mult * last8;
            quant_acc = PDX_SLS_MX80(quant_acc, shift);
            quant_acc2 = PDX_SLS_MX80(quant_acc2, shift);
            fc_out = PDX_PACKQSRV_MX80(quant_acc, round_mode);
            fc_out2 = PDX_PACKQSRV_MX80(quant_acc2, round_mode);

            fc_out += vOutZP;
            fc_out2 += vOutZP;

            fc_out = PDX_MIN_MX32(fc_out, vmax);
            fc_out = PDX_MAX_MX32(fc_out, vmin);
            fc_out2 = PDX_MIN_MX32(fc_out2, vmax);
            fc_out2 = PDX_MAX_MX32(fc_out2, vmin);

            PDX_SAV32_MX8_XP(fc_out, outa, outp, PDX_M);
            PDX_SAPOS_MX8_FP(outa, outp);
            PDX_SAV32_MX8_XP(fc_out2, outa, outp, PDX_M);
            PDX_SAPOS_MX8_FP(outa, outp);

            nPixProcessed += 2 * PDX_M;
            nPixLeft -= 2 * PDX_M;
        }

        /* ============================================================
         * Tail: remaining outputs (< 2*PDX_M)
         * Weight tail is packed with stride = nPixLeft (the remainder),
         * matching the blocked layout from reorder_weights.
         * ============================================================ */
        if (nPixLeft > 0)
        {
            acc = 0;

            inp = (int8_t *)(pInputBuffer + b * nFilterDepth);

            xb_vec2Mx8 *wtp = (xb_vec2Mx8 *)(pWeightsBuffer + nPixProcessed * nFilterDepth);
            valign wta = PDX_LA_2MX8_PP(wtp);

            if (pBiasBuffer) {
                xb_vecMx32 *biasp = (xb_vecMx32 *)(pBiasBuffer + nPixProcessed);
                valign biasa = PDX_LA_MX32_PP(biasp);

                PDX_LA_MX32_XP(vbias_l, biasa, biasp, PDX_M * sizeof(int32_t));
                PDX_LA_MX32_XP(vbias_h, biasa, biasp, 0);

                vbias_l = PDX_SLS_MX32(vbias_l, 1);
                vbias_h = PDX_SLS_MX32(vbias_h, 1);
            }

            for (int32_t filterD = 0; filterD < nFilterDepth; filterD++)
            {
                vin = (*inp++);
                PDX_LA16_2MX8_XP(vwt, wta, wtp, nPixLeft);
                wta = PDX_LA_2MX8_PP(wtp);   /* reprime: stride != natural width */
                vin += vInZP;
                vwt += vFilterZP;

                PDX_MULAQW_2MX16(acc, vwt, vin);
            }

            PDX_CVT32D_2MX40(last8, first8, acc);
            if (pBiasBuffer)
            {
                first8 += vbias_l;
            }

            quant_acc = mult * first8;
            quant_acc = PDX_SLS_MX80(quant_acc, shift);
            fc_out = PDX_PACKQSRV_MX80(quant_acc, round_mode);

            fc_out += vOutZP;
            fc_out = PDX_MIN_MX32(fc_out, vmax);
            fc_out = PDX_MAX_MX32(fc_out, vmin);

            nPixToWrite = MIN(nPixLeft, PDX_M);
            PDX_SAV32_MX8_XP(fc_out, outa, outp, nPixToWrite);
            PDX_SAPOS_MX8_FP(outa, outp);
            nPixLeft -= nPixToWrite;

            if (nPixLeft > 0)
            {
                if (pBiasBuffer)
                {
                    last8 += vbias_h;
                }

                quant_acc = mult * last8;
                quant_acc = PDX_SLS_MX80(quant_acc, shift);
                fc_out2 = PDX_PACKQSRV_MX80(quant_acc, round_mode);

                fc_out2 += vOutZP;
                fc_out2 = PDX_MIN_MX32(fc_out2, vmax);
                fc_out2 = PDX_MAX_MX32(fc_out2, vmin);

                nPixToWrite = MIN(nPixLeft, PDX_M);
                PDX_SAV32_MX8_XP(fc_out2, outa, outp, nPixToWrite);
                PDX_SAPOS_MX8_FP(outa, outp);
                nPixLeft -= nPixToWrite;
            }
        }
    }
}
