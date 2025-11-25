/**
********************************************************************************
*
* @file: adi_sharcfx_fully_connected.cpp
*
* @brief: contains optimized version of fully connected layer
*
* @details: contains optimized version of fully connected layer or 8bit and 16bit integer input
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

/*============= C O D E =============*/


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
    vbool2M acc_mask, ffff;

    temp_mask = PDX_MOVB_AU32(0xFFFFFFFF);
    PDX_CVTBB2M_B4M(ffff, ffff, temp_mask);

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
                for (int32_t nFilterCnt = 0; nFilterCnt <= nFilterDepth; nFilterCnt += (2*PDX_M))
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

void adi_sharcfx_fully_connected_int8(const int8_t* pInputBuffer,
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
    vbool2M acc_mask, ffff;

    temp_mask = PDX_MOVB_AU32(0xFFFFFFFF);
    PDX_CVTBB2M_B4M(ffff, ffff, temp_mask);

    int32_t nPixProcessed=0;

    xb_vec2Mx40 acc = 0;
    xb_vec2Mx8 *inp = (xb_vec2Mx8 *)pInputBuffer;
    if(nFilterDepth % 16){
        int32_t nFilterDepth_16mul = nFilterDepth - ((nFilterDepth % 16)*nFilterDepth);
        for (int b = 0; b < nBatches; b++){
            //No of filter is equals to the nOutsize
            for (int32_t nChannelCnt = 0; nChannelCnt < nOutsize; nChannelCnt++)
            {
                acc = 0;//reset accumulator

                inp = (xb_vec2Mx8 *)(pInputBuffer + b*nFilterDepth);    //Reinitialise input pointer for each output pixel
                valign ina; // define align vector
                ina=PDX_LA_2MX8_PP (inp); // prime, NOP if a[] is aligned

                xb_vec2Mx8 *wtp = (xb_vec2Mx8 *) (pWeightsBuffer + nFilterDepth*nChannelCnt); //Move to next filter for each output pixel
                valign wta;    // define align vector
                wta=PDX_LA_2MX8_PP (wtp);

                //for all the input pixels
                for (nPixProcessed= 0; nPixProcessed < nFilterDepth_16mul; nPixProcessed += (2*PDX_M))
                {
                    PDX_LA16_2MX8_XP (vin, ina, inp, 2*PDX_M); // load aligned, extend
                    PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M); // load aligned, extend
                    vin+=vInZP;        //Add input offset
                    vwt+=vFilterZP;    //Add filter offset

                    PDX_MULAQW_2MX16(acc,vwt,vin);

                }

                PDX_LA16_2MX8_XP (vin, ina, inp, 2*PDX_M); // load aligned, extend
                PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M); // load aligned, extend
                vin+=vInZP;        //Add input offset
                vwt+=vFilterZP;    //Add filter offset

                temp_mask = PDX_MOVB_AU32((0b1<<(nFilterDepth % 16)) - 1);
                acc_mask = PDX_CVTBB2M_B4M_L(temp_mask);
                //multiply and accumulate
                PDX_MULAQW_2MX16_T(acc,vwt,vin,acc_mask);

                sat_sum = PDX_RADD_2MX40(acc);                                    //PDX_RADD_2MX40: Adds across all 16lanes of acc and returns sum
                if(pBiasBuffer)
                {
                    //adding bias>>1 to compensate for sign bit during multiplication
                    sat_sum += (xb_int40)(pBiasBuffer[nChannelCnt]<<1);
                }
                temp = (xb_int32)((int32_t)((int64_t)PDX_CVT64_40(sat_sum)));    //convert 40bit var into 32bit to perform multiplication

                product = PDX_MULW_32(temp, (uint32_t)nQuantizedMultiplier);    //multiply with the quantization multiplier; product(80bit) = temp(32bit) * nQuantizedMultiplier(32bit)
                product =  PDX_SLA_80(product, (xb_int32)nQuantizedShift);        //shift result by quantization multiplier
                temp = PDX_PACKQSRV_80(product,2);                                //packs 80bit product into 32bit var with saturation and rounding
                temp+= (xb_int32)nOutputOffset;                                    //add output offset
                temp = MIN(temp, (xb_int32)output_activation_max);
                temp = MAX(temp, (xb_int32)output_activation_min);//8bit saturation check to store result
                *outp++ =(int8_t)((int32_t)temp);                                //store result as 8-bit data
            }
        }
    }
    else{
        for (int b = 0; b < nBatches; b++){
            //No of filter is equals to the nOutsize
            for (int32_t nChannelCnt = 0; nChannelCnt < nOutsize; nChannelCnt++)
            {
                acc = 0;//reset accumulator

                inp = (xb_vec2Mx8 *)(pInputBuffer + b*nFilterDepth);    //Reinitialise input pointer for each output pixel
                valign ina; // define align vector
                ina=PDX_LA_2MX8_PP (inp); // prime, NOP if a[] is aligned

                xb_vec2Mx8 *wtp = (xb_vec2Mx8 *) (pWeightsBuffer + nFilterDepth*nChannelCnt); //Move to next filter for each output pixel
                valign wta;    // define align vector
                wta=PDX_LA_2MX8_PP (wtp);

                //for all the input pixels
                for (int32_t nFilterCnt = 0; nFilterCnt < nFilterDepth; nFilterCnt += (2*PDX_M))
                {
                    PDX_LA16_2MX8_XP (vin, ina, inp, 2*PDX_M); // load aligned, extend
                    PDX_LA16_2MX8_XP (vwt, wta, wtp, 2*PDX_M); // load aligned, extend
                    vin+=vInZP;        //Add input offset
                    vwt+=vFilterZP;    //Add filter offset

                    PDX_MULAQW_2MX16(acc,vwt,vin);

                }
                sat_sum = PDX_RADD_2MX40(acc);                                    //PDX_RADD_2MX40: Adds across all 16lanes of acc and returns sum
                if(pBiasBuffer)
                {
                    //adding bias>>1 to compensate for sign bit during multiplication
                    sat_sum += (xb_int40)(pBiasBuffer[nChannelCnt]<<1);
                }
                temp = (xb_int32)((int32_t)((int64_t)PDX_CVT64_40(sat_sum)));    //convert 40bit var into 32bit to perform multiplication

                product = PDX_MULW_32(temp, (uint32_t)nQuantizedMultiplier);    //multiply with the quantization multiplier; product(80bit) = temp(32bit) * nQuantizedMultiplier(32bit)
                product =  PDX_SLA_80(product, (xb_int32)nQuantizedShift);        //shift result by quantization multiplier
                temp = PDX_PACKQSRV_80(product,2);                                //packs 80bit product into 32bit var with saturation and rounding
                temp+= (xb_int32)nOutputOffset;                                    //add output offset
                temp = MIN(temp, (xb_int32)output_activation_max);
                temp = MAX(temp, (xb_int32)output_activation_min);//8bit saturation check to store result
                *outp++ =(int8_t)((int32_t)temp);                                //store result as 8-bit data
            }
        }
    }
}

void transform_matrices(const int8_t * inputMat, int32_t M, int32_t N, int8_t * outputMat)
{
    size_t block = 4;
    for (size_t i = 0; i < M; i += block) {
        for(size_t j = 0; j < N; ++j) {
            for(size_t b = 0; b < block && i + b < M; ++b) {
                outputMat[j*M + i + b] = inputMat[(i + b)*N + j];
            }
        }
    }
}

void adi_sharcfx_fully_connected_int8_new(const int8_t* pInputBuffer,
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
    xb_vecMx8* __restrict outp  = (xb_vecMx8 *)pOutputBuffer;
    valign outa = PDX_LA_MX8_PP (outp); // prime, NOP if a[] is aligned

    const immediate Lane=0;
    //Defining the input and filter offsets
    xb_vec2Mx16 vInZP = nInputOffset;//PDX_REP_2MX16((xb_vec2Mx16)nInputOffset,Lane);//Replicates the lane of data specified, across all lanes of a vector register
    xb_vec2Mx16 vFilterZP = nFilterOffset;//PDX_REP_2MX16((xb_vec2Mx16)nFilterOffset,Lane);//Replicates the lane of data specified, across all lanes of a vector register
    xb_vecMx32 vOutZP = nOutputOffset;

    xb_vecMx32 vmin = output_activation_min;
    xb_vecMx32 vmax = output_activation_max;

    xb_vec2Mx16 vin,vwt;
    xb_int32 temp;

    int32_t nPixProcessed=0;
    int32_t nPixLeft=nOutsize;

    xb_vec2Mx40 acc = 0;
    int8_t* inp;
    vbool4M temp_mask;
    vbool2M acc_mask;

    xb_vecMx32 first8, last8;
    xb_vec2Mx40 outacc;
    xb_vecMx80 quant_acc, quant_acc2;
    const immediate round_mode = ROUNDING_MODE;
    xb_vecMx32 mult = nQuantizedMultiplier;
    xb_vecMx32 shift = nQuantizedShift;
    xb_vecMx32 vBias;
    xb_vecMx32 fc_out, fc_out2;
    xb_vecMx32 vbias_l,vbias_h;

    int16_t nPixToWrite;

    //Store transposed weight matrix in pTemp
    //Use pTemp as the weights buff
    transform_matrices(pWeightsBuffer, nOutsize, nFilterDepth, pTemp);

    for(int32_t b = 0; b < nBatches; b++)
    {
        nPixLeft = nOutsize;
        //for every batch, reset input pointer
        inp = (int8_t *)(pInputBuffer+b*nFilterDepth);
        if(nPixLeft>2*PDX_M)
        {
            for(int32_t outP = 0; outP < nOutsize; outP+=2*PDX_M)
            {
                //main loop for output pixels, 16 at a time
                //acc=0
                acc = 0;//reset accumulator

                inp = (int8_t *)(pInputBuffer+b*nFilterDepth + outP);    //Reinitialise input pointer for each output pixel

                xb_vec2Mx8 *wtp = (xb_vec2Mx8 *) (pTemp + outP); //Move to next filter for each output pixel
                valign wta;    // define align vector
                wta=PDX_LA_2MX8_PP (wtp);
                if (pBiasBuffer){
                    xb_vecMx32 *biasp = (xb_vecMx32 *) (pBiasBuffer + outP);
                    valign biasa=PDX_LA_MX32_PP (biasp);

                    PDX_LA_MX32_XP (vbias_l, biasa, biasp, PDX_M);
                    PDX_LA_MX32_XP (vbias_h, biasa, biasp, 0);

                    vbias_l = PDX_SLS_MX32(vbias_l,1);//*2 to match with acc
                    vbias_h = PDX_SLS_MX32(vbias_h,1);//*2 to match with acc
                }

                for(int32_t filterD = 0; filterD < nFilterDepth; filterD++)
                {
                    //sub-loop to repeat for how many input features are present
                    //accumulate, quantize and save
                    //decrement nPixLeft by 16 after saving. incremenet nPixProcessed by 16 after loop;
                    vin=(*inp++);//repeat input for all output lanes
                    PDX_LA16_2MX8_XP (vwt, wta, wtp, nOutsize); // load aligned, extend
                    vin+=vInZP;        //Add input offset
                    vwt+=vFilterZP;    //Add filter offset

                    PDX_MULAQW_2MX16(acc,vwt,vin);
                }
                //Quantize and store
                //quantization compensation
                PDX_CVT32D_2MX40(last8, first8, acc);    //Converting 40bit results to 32bit to prevent loss of accuracy from packing of 40bit -> 16bit
                if (pBiasBuffer){
                    first8 += vbias_l;
                    last8 += vbias_h;
                }

                //first8
                quant_acc = mult * first8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
                //last8
                quant_acc2 = mult * last8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
                //shift and round
                quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
                //shift and round
                quant_acc2 = PDX_SLS_MX80(quant_acc2,shift);//saturating left shift, right shift if negative
                //round and shift and saturate
                fc_out = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
                //round and shift and saturate
                fc_out2 = PDX_PACKQSRV_MX80   (quant_acc2, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
                //add output zero point
                fc_out += vOutZP;
                //add output zero point
                fc_out2 += vOutZP;
                //Saturate to 8 bit range output_activation_min to 127
                fc_out = PDX_MIN_MX32(fc_out,vmax);
                fc_out2 = PDX_MAX_MX32(fc_out2,vmin);
                fc_out = PDX_MAX_MX32(fc_out,vmin);
                fc_out2 = PDX_MIN_MX32(fc_out2,vmax);

                nPixLeft -= 2*PDX_M;
                nPixProcessed += 2*PDX_M;
                PDX_SAV32_MX8_XP(fc_out, outa, outp, PDX_M);//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
                PDX_SAPOS_MX8_FP(outa,outp);//flush
                PDX_SAV32_MX8_XP(fc_out2, outa, outp, PDX_M);//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
                PDX_SAPOS_MX8_FP(outa,outp);//flush
            }
        }
        if(nPixLeft>0)
        {
            temp_mask = PDX_MOVB_AU32((0b1<<(nPixLeft)) - 1);
            acc_mask = PDX_CVTBB2M_B4M_L(temp_mask);
            //acc=0

            acc = 0;//reset accumulator

            inp = (int8_t *)(pInputBuffer+b*nFilterDepth + nPixProcessed);

            xb_vec2Mx8 *wtp = (xb_vec2Mx8 *) (pTemp + nPixProcessed); //Move to next filter for each output pixel
            valign wta;    // define align vector
            wta=PDX_LA_2MX8_PP (wtp);

            if (pBiasBuffer){
                xb_vecMx32 *biasp = (xb_vecMx32 *) (pBiasBuffer + nPixProcessed);
                valign biasa=PDX_LA_MX32_PP (biasp);

                PDX_LA_MX32_XP (vbias_l, biasa, biasp, PDX_M);
                PDX_LA_MX32_XP (vbias_h, biasa, biasp, 0);

                vbias_l = PDX_SLS_MX32(vbias_l,1);//*2 to match with acc
                vbias_h = PDX_SLS_MX32(vbias_h,1);//*2 to match with acc
            }
            for(int32_t filterD = 0; filterD < nFilterDepth; filterD++)
            {
                //sub-loop to repeat for how many input features are present
                //accumulate, quantize and save
                //decrement nPixLeft by 16 after saving. incremenet nPixProcessed by 16 after loop;
                vin=(*inp++);//repeat input for all output lanes
                PDX_LA16_2MX8_XP (vwt, wta, wtp, nOutsize); // load aligned, extend
                vin+=vInZP;        //Add input offset
                vwt+=vFilterZP;    //Add filter offset

                PDX_MULAQW_2MX16(acc,vwt,vin);
            }
            //Quantize and store
            //Use bool mask to store non-16 outsizes
            //quantization compensation

            PDX_CVT32D_2MX40(last8, first8, acc);    //Converting 40bit results to 32bit to prevent loss of accuracy from packing of 40bit -> 16bit
            if (pBiasBuffer)
            {
                first8 += vbias_l;
            }
            //first8
            quant_acc = mult * first8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //shift and round
            quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
            //round and shift and saturate
            fc_out = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //add output zero point
            fc_out += vOutZP;
            //Saturate to 8 bit range output_activation_min to 127
            fc_out = PDX_MIN_MX32(fc_out,vmax);
            fc_out = PDX_MAX_MX32(fc_out,vmin);
            nPixToWrite = MIN(nPixLeft,PDX_M);
            PDX_SAV32_MX8_XP(fc_out, outa, outp, nPixToWrite);//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX8_FP(outa,outp);//flush
            nPixLeft -= nPixToWrite;

            if (nPixLeft>0)
            {
                if (pBiasBuffer)
                {
                    last8 += vbias_h;
                }
                //last8
                quant_acc = mult * last8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
                //shift and round
                quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
                //round and shift and saturate
                fc_out2 = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
                //add output zero point
                fc_out2 += vOutZP;
                //Saturate to 8 bit range output_activation_min to 127
                fc_out2 = PDX_MIN_MX32(fc_out2,vmax);
                fc_out2 = PDX_MAX_MX32(fc_out2,vmin);
                nPixToWrite = MIN(nPixLeft,PDX_M);
                PDX_SAV32_MX8_XP(fc_out2, outa, outp, nPixToWrite);//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
                PDX_SAPOS_MX8_FP(outa,outp);//flush
                nPixLeft -= nPixToWrite;
            }
        }
    }
}
