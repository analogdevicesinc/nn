/**
********************************************************************************
*
* @file: adi_sharcfx_activations.cpp
*
* @brief: Contains optimized Relu, Tanh and Logistic functions
*
* @details: Contains the optimized Relu and Logistic activation functions for int8 input data and optimized Logistic and Tanh activation functions for int16 input data
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

/*============= D E F I N E S =============*/
#define LUT_SIZE 256

/*============= C O D E =============*/
static uint16_t sigmoid_table_uint16[LUT_SIZE]__attribute__((section(".L1.data"), aligned(8))) = {
    32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498,
    40149, 40794, 41432, 42064, 42688, 43304, 43912, 44511, 45102, 45683, 46255,
    46817, 47369, 47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409, 51865,
    52311, 52745, 53169, 53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174,
    56503, 56823, 57133, 57433, 57724, 58007, 58280, 58544, 58800, 59048, 59288,
    59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110, 61279, 61441,
    61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886,
    62990, 63090, 63186, 63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835,
    63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308, 64357, 64405, 64450,
    64494, 64536, 64576, 64614, 64652, 64687, 64721, 64754, 64786, 64816, 64845,
    64873, 64900, 64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079, 65097,
    65115, 65132, 65149, 65164, 65179, 65194, 65208, 65221, 65234, 65246, 65258,
    65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360,
    65367, 65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425,
    65429, 65433, 65438, 65442, 65445, 65449, 65453, 65456, 65459, 65462, 65465,
    65468, 65471, 65474, 65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491,
    65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508,
    65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516, 65517, 65517, 65518,
    65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524, 65525,
    65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529,
    65529, 65529, 65530, 65530, 65530, 65530, 65531, 65531, 65531, 65531, 65531,
    65532, 65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533, 65533, 65533,
    65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534,
    65534, 65534, 65535};

/**
*******************************************************************************
* Function: adi_sharcfx_relu_int8
* @brief optimized implementation of Relu activation function
*
* @details optimized implementation of Relu activation function for int8 data
*
* Parameters:
* @param [in] pInput - input buffer
* @param [in] nSize - input height
* @param [in] nQuantizedMultiplier - multiplier, corresponds to TFLM quantization scheme
* @param [in] nQuantizedShift - shift, corresponds to TFLM quantization scheme
* @param [in] nInOffset - input offset, corresponds to TFLM quantization scheme
* @param [in] nOutOffset - output offset, corresponds to TFLM quantization scheme
* @param [in] output_activation_min - min value of activation function
* @param [in] output_activation_max - max value of activation function
*
* @param [out] pOutput - output buffer
*
* @return None
*
*******************************************************************************
*/ 
void adi_sharcfx_relu_int8(const int8_t* pInput,
                           int8_t* pOutput,
                           const uint32_t nSize,
                           uint32_t nQuantizedMultiplier,
                           int32_t nQuantizedShift,
                           int32_t nInOffset,
                           int32_t nOutOffset,
                           int32_t output_activation_min,
                           int32_t output_activation_max)
{
    const immediate Lane = 0;
    const immediate round_mode = 2;

    xb_vec2Mx16 vInOff = PDX_REP_2MX16((xb_vec2Mx16)nInOffset,Lane);
    xb_vec2Mx16 vOutOff = PDX_REP_2MX16((xb_vec2Mx16)nOutOffset,Lane);

    xb_vec2Mx16 vmin = PDX_REP_2MX16((xb_vec2Mx16)output_activation_min,Lane);//Replicates the lane of data specified, across all lanes of a vector register
    xb_vec2Mx16 vmax = PDX_REP_2MX16((xb_vec2Mx16)output_activation_max,Lane);//Replicates the lane of data specified, across all lanes of a vector register

    xb_vec2Mx8 *inp = (xb_vec2Mx8 *)pInput;
    valign ina=PDX_LA_2MX8_PP (inp);// define align vector

    valign outa = PDX_Z_ALIGN();
    xb_vec2Mx8 *outp = (xb_vec2Mx8 *)pOutput;

    xb_vec2Mx16 shift = PDX_REP_2MX16((xb_vec2Mx16)(nQuantizedShift+1),Lane);
    xb_vec2Mx16 multiplier = PDX_REP_2MX16((xb_vec2Mx16)((nQuantizedMultiplier + (1 << 15)) >> 16),Lane);

    xb_vec2Mx16 vin, result;
    xb_vec2Mx40 acc;

    int32_t nElementsProcessed=0;
    for (; (nSize - nElementsProcessed) >= (PDX_2M);) {
        //load inputs
        PDX_LA16_2MX8_XP (vin, ina, inp, PDX_2M );
        vin-=vInOff;
        //reset acc
        acc=0;
        //multiply by multipler
        PDX_MULAW_2MX16(acc, multiplier, vin);
        //shift
        acc = PDX_SLS_2MX40(acc,shift);
        //saturate and save to 16bit
        result = PDX_PACKQSRV_2MX40(acc, round_mode);
        //add offset
        result +=vOutOff;
        //saturate
        result = PDX_MIN_2MX16(result,vmax);
        result = PDX_MAX_2MX16(result,vmin);
        //save output as 8-bit data
        PDX_SAV16_2MX8_XP(result, outa, outp,PDX_2M);
        PDX_SAPOS_2MX8_FP(outa,outp);//flush
        nElementsProcessed += (PDX_2M);
    }
    if((nSize - nElementsProcessed)>0){
        //load inputs
        PDX_LA16_2MX8_XP (vin, ina, inp, 0);
        vin-=vInOff;
        //reset acc
        acc=0;
        //multiply by multipler
        PDX_MULAW_2MX16(acc, multiplier, vin);
        //shift
        acc = PDX_SLS_2MX40(acc,shift);
        //saturate and save to 16bit
        result = PDX_PACKQSRV_2MX40(acc, round_mode);
        //add offset
        result +=vOutOff;
        //saturate
        result = PDX_MIN_2MX16(result,vmax);
        result = PDX_MAX_2MX16(result,vmin);
        //save output as 8-bit data
        PDX_SAV16_2MX8_XP(result, outa, outp,(nSize - nElementsProcessed));
        PDX_SAPOS_2MX8_FP(outa,outp);//flush
    }
}


// Q of input data
#define QIN          12  /* Q3.12 input format: 1 sign + 3 integer + 12 fractional bits */
// Q of output
#define QOUT         15  /* Q0.15 output format: 1 sign + 0 integer + 15 fractional bits */
/* Number of address bits into the table: 2^5 = 32 table entries per half */
#define INDEX_BITS_TH 5
/* Width of one table entry interval: (1 << QOUT) / 32 = 1024 */
#define INDEX_STEP_TH (1 << (QOUT - INDEX_BITS_TH))
/* Empirical fractional offset for xi within each table entry.
 * xi = (i + 0.21) * INDEX_STEP_TH, i in 0..31.
 * The value 0.21 was found empirically to minimise average and RMS approximation error. */
#define INDEX_OFFSET_TH (INDEX_STEP_TH * 21 / 100)


/* Tables used by adi_vectanh_16b
 * These are values of tanh function and its first three derivatives in QOUT form
 * See equations below for values
 */
// For a range of 0 to 32767
// For offset = 0.21, max LSB diff = 7 at 2036. Average diff = 0.168426513671875, RMS error in dB = 180.639773050057
static int16_t tanh_table_qout [2][16] = {
    { 1718, 9620, 16462, 21804, 25650, 28257, 29956, 31033, 31704, 32119, 32372, 32527, 32622, 32679, 32714, 32735 },
    { 32748, 32755, 32760, 32763, 32765, 32766, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767 },
};
static int16_t tanh_first_derivative_table_qout [2][16] = {
    { 32678, 29943, 24497, 18258, 12690, 8399, 5382, 3377, 2092, 1285, 786, 479, 291, 177, 107, 65 },
    { 40, 24, 15, 9, 5, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0 },
};
static int16_t tanh_second_derivative_table_qout [2][16] = {
    { -1713, -8790, -12306, -12149, -9932, -7242, -4919, -3198, -2023, -1259, -775, -474, -289, -175, -106, -64 },
    { -39, -23, -14, -8, -4, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};
static int16_t tanh_third_derivative_table_qout [2][16] = {
    { -10802, -7399, -1982, 1999, 3546, 3447, 2704, 1903, 1261, 806, 505, 312, 192, 117, 71, 43 },
    { 26, 16, 10, 6, 4, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
};

/*-------------------------------------------------------------------------
Vectorized Tanh
The function returns the tanh ((exp(x) - exp(-x)/(exp(x)+exp(-x))) of x. 16-bit fixed-point
function accepts input in Q3.12 and form output in Q0.15 format.

Algorithm:

Third-order Taylor series:

f(x) = f(xi) + f'(xi) (x - xi) + f''(xi) * (x - xi)^2 / 2 + f'''(xi) * (x - xi)^3 / 6

where
xi = (i + 0.21) * 2^Q, i in 0..31.
The 0.21 was found empirically to give the least average error and least RMS error

f(x) = 2^Q (e^x - e^-x) / (e^x + e^-x)
f'(x) = 2^Q * 4 / (e^x - e^-x)^2
f''(x)/2 = 2^Q * 8 * (e^-x - e^x) / (e^x + e^-x)^3
f'''(x)/6 = 2^Q * (-2 + 4 * (e^x - e^-x)^2 / (e^x + e^-x)^4

Input:
  x  input value (Q3.12)
Output:
  z  result (Q0.15)
Returned value:
  None
Domain:
  Whole range
---------------------------------------------------------------------------*/
/**
 *******************************************************************************
 * Function: adi_vectanh_16b
 * @brief Vectorized tanh for 16-bit fixed-point input (Q3.12 in, Q0.15 out).
 *
 * @details Uses a 32-entry lookup table with third-order Taylor series.
 *          xi = (i + 0.21) * 2^QOUT, i in 0..31 (offset chosen empirically
 *          to minimise average and RMS error). Internal helper; called by
 *          adi_sharcfx_tanh_int16 and adi_sharcfx_tanh_int8.
 *
 * Parameters:
 * @param [in]  pInputData  Input array (int16, Q3.12 format).
 * @param [out] pOutputData Output array (int16, Q0.15 format).
 * @param [in]  nLength     Number of elements.
 * @return None
 *******************************************************************************
 */
void adi_vectanh_16b (
        const int16_t *pInputData,     /* [in] array of N 16-bit fixed point values with 1 sign, 3 int, and 12 fractional bits  */
        int16_t *  pOutputData,        /* [out] array of N 16-bit fixed point values with 1 sign and 15 fractional bits */
        int nLength)                   /* [in] length of array */
{
    int n;
    int left = nLength * sizeof (*pInputData);
    // Input pointers
    xb_vec2Mx16 *vx = (xb_vec2Mx16 *) pInputData;
    valign vxa = PDX_LA_2MX16_PP (vx);
    // Output pointers
    xb_vec2Mx16 * vz = (xb_vec2Mx16 *) pOutputData;
    valign vza = PDX_Z_ALIGN ();

    // Tables
    xb_vec2Mx16 sig_lo    = *((xb_vec2Mx16 *) tanh_table_qout[0]);
    xb_vec2Mx16 sig_hi    = *((xb_vec2Mx16 *) tanh_table_qout[1]);
    xb_vec2Mx16 sig_d1_lo = *((xb_vec2Mx16 *) tanh_first_derivative_table_qout[0]);
    xb_vec2Mx16 sig_d1_hi = *((xb_vec2Mx16 *) tanh_first_derivative_table_qout[1]);
    xb_vec2Mx16 sig_d2_lo = *((xb_vec2Mx16 *) tanh_second_derivative_table_qout[0]);
    xb_vec2Mx16 sig_d2_hi = *((xb_vec2Mx16 *) tanh_second_derivative_table_qout[1]);
    xb_vec2Mx16 sig_d3_lo = *((xb_vec2Mx16 *) tanh_third_derivative_table_qout[0]);
    xb_vec2Mx16 sig_d3_hi = *((xb_vec2Mx16 *) tanh_third_derivative_table_qout[1]);

    xb_vec2Mx16 vvx_qin, avvx_qin, vxi_qin, index, frac_qin, frac2_qin, frac3_qin;
    xb_vec2Mx16 vvz_qout, vvz_neg_qout, sig_qout, sig_d1_qout, sig_d2_qout, sig_d3_qout;
    xb_vec2Mx40 acc;
    xb_vec2Mx16 vzero = 0;
    xb_vec2Mx16 qout_max = ((1 << QOUT) - 1);
    xb_vec2Mx16 indexOffset = INDEX_OFFSET_TH;

    for (n = 0; n < nLength; n += 2*PDX_M)
    {
        // Get input, padding with zeros if beyond the end of the array
        PDX_LAV_2MX16_XP (vvx_qin, vxa, vx, left);
        // Find its absolute value.  The tanh function is mirrored around zero
        avvx_qin = PDX_ABS_2MX16 (vvx_qin);
        // Get table index
        index = PDX_SRAI_2MX16 (avvx_qin, QOUT - INDEX_BITS_TH);
        index = PDX_MIN_2MX16(index, 31);
        index = PDX_MAX_2MX16(index, 0);
        // Look up f(xi)
        sig_qout    = PDX_SEL_2MX16 (sig_hi, sig_lo, index);
        // Look up f'(xi)
        sig_d1_qout = PDX_SEL_2MX16 (sig_d1_hi, sig_d1_lo, index);
        // Look up f''(xi)/2
        sig_d2_qout = PDX_SEL_2MX16 (sig_d2_hi, sig_d2_lo, index);
        // Look up f'''(xi)/6
        sig_d3_qout = PDX_SEL_2MX16 (sig_d3_hi, sig_d3_lo, index);
        // Find the value of x used by the tables
        vxi_qin = ((index << (QOUT - INDEX_BITS_TH)) + indexOffset);
        // Get x - xi
        frac_qin = avvx_qin - vxi_qin;
        // Get (x - xi)^2 >> QIN
        acc = frac_qin * frac_qin;
        frac2_qin = PDX_PACKIV_2MX40  (acc, QIN);
        // Get (x - xi)^3 >> QIN
        acc = frac2_qin * frac_qin;
        frac3_qin = PDX_PACKIV_2MX40  (acc, QIN);
        // Accumulate output
        acc  = PDX_MULW_2MX16 (sig_qout, 1 << QIN);
        acc += sig_d1_qout * frac_qin;
        acc += sig_d2_qout * frac2_qin;
        acc += sig_d3_qout * frac3_qin;
        // Adjust to Q31
        acc <<= (QOUT + 16) - (QOUT + QIN);
        // Convert to Q15 with nearest-even rounding and saturation
        vvz_qout = PDX_PACKQSRV_2MX40  (acc, 2);
        // Negate for negative inputs
        vvz_neg_qout = - vvz_qout;
        // Find if input is negative
        vbool2M x_neg = PDX_LT_2MX16 (vvx_qin, vzero);
        // Select the negative output if input is negative
        vvz_qout = PDX_MOV_2MX16_T (vvz_neg_qout, vvz_qout, x_neg);
        // Save away, not modifying if beyond the end of the array
        PDX_SAV_2MX16_XP (vvz_qout, vza, vz, left);
        // Get number of bytes left
        left -= 4*PDX_M;
    }
    PDX_SAPOS_2MX16_FP (vza, vz);  // Save tails
}

/**
*******************************************************************************
* Function: adi_sharcfx_tanh_int16
* @brief optimized implementation of tanh activation function
*
* @details optimized implementation of tanh activation function for int16 data. The function returns the hyperbolic Tan of x. 16-bit fixed-point function
* accepts input in Q3.12 and form output in Q0.15 format.
*
* Parameters:
* @param [in] nInputMultiplier - multiplier, corresponds to TFLM quantization scheme
* @param [in] nInputLeftShift - shift, corresponds to TFLM quantization scheme
* @param [in] nLength - input size
* @param [in] pInputData - input buffer (Q3.12)
*
* @param [out] pOutputData - output buffer(Q0.15)
*
* @return None
*
*******************************************************************************
*/ 
void adi_sharcfx_tanh_int16(int32_t nInputMultiplier, 
                            int32_t nInputLeftShift, 
                            int32_t nLength,
                            const int16_t* pInputData, 
                            int16_t* pOutputData)
{
    int16_t* pInput_in_q3_12 = (int16_t*)pTempL3;
    xb_vec2Mx16 *inp = (xb_vec2Mx16 *)pInputData;
    xb_vec2Mx16 *          outp;
    outp=(      xb_vec2Mx16 *)pInput_in_q3_12;
    xb_vec2Mx16 vin;
    valign ina,outa; // define align vector
    ina=PDX_LA_2MX16_PP (inp); // prime, NOP if a[] is aligned
    outa = PDX_Z_ALIGN();
    int32_t nPixLeft = nLength;
    //scaling input to fit into Q3.12.
    if (nInputMultiplier == 0)
    {
        for (int i = 0; i < nLength; i += (PDX_2M))
        {
            xb_vec2Mx16 vTempInput ;
            PDX_LA_2MX16_XP (vin, ina, inp, PDX_4M); // load aligned, extend
            vTempInput = PDX_SLS_2MX16(vin, nInputLeftShift);
            PDX_SAV_2MX16_XP(vTempInput,outa,outp, PDX_4M);
            PDX_SAPOS_2MX16_FP( outa, outp );
            nPixLeft -=PDX_2M;
        }
        if(nPixLeft>0)
        {
            xb_vec2Mx16 vTempInput ;
            PDX_LA_2MX16_XP (vin, ina, inp, 0); // load aligned, extend
            vTempInput = PDX_SLS_2MX16(vin, nInputLeftShift);
            PDX_SAV_2MX16_XP(vTempInput,outa,outp, nPixLeft*2);
            PDX_SAPOS_2MX16_FP( outa, outp );
        }
    }
    else
    {
        xb_vec2Mx40 vTempOut;
        int32_t nTempRound = nInputLeftShift > 0 ? (1<<(nInputLeftShift-1)) : 0;
        for (int i = 0; i < nLength; i += PDX_2M)
        {
            PDX_LA_2MX16_XP (vin, ina, inp, PDX_4M); // load aligned, extend
            vTempOut = PDX_MULW_2MX16(vin,nInputMultiplier);
            vTempOut = PDX_ADD_2MX40(vTempOut,nTempRound);  //rounding multiplied scaled input with round data nTempRound for 16 bit data;
            vTempOut = PDX_SRA_2MX40 (vTempOut,nInputLeftShift);
            PDX_SAV_2MX16_XP(PDX_PACKSIV_2MX40(vTempOut,0),outa,outp, PDX_4M);
            PDX_SAPOS_2MX16_FP( outa, outp );
            nPixLeft -=PDX_2M;
        }
        if(nPixLeft>0)
        {
            PDX_LA_2MX16_XP (vin, ina, inp, 0); // load aligned, extend
            vTempOut = PDX_MULW_2MX16(vin,nInputMultiplier);
            vTempOut = PDX_ADD_2MX40(vTempOut,nTempRound);  //rounding multiplied scaled input with round data nTempRound for 16 bit data;
            vTempOut = PDX_SRA_2MX40 (vTempOut,nInputLeftShift);
            PDX_SAV_2MX16_XP(PDX_PACKSIV_2MX40(vTempOut,0),outa,outp, nPixLeft*2);
            PDX_SAPOS_2MX16_FP( outa, outp );
        }
    }
    adi_vectanh_16b(pInput_in_q3_12, pOutputData, nLength );
}
/**
*******************************************************************************
* Function: adi_sharcfx_logistic_int8
* @brief optimized implementation of logistic/sigmoid activation function
*
* @details optimized implementation of tanh activation function for int8 data. The function returns the Logistic (1/(1+exp(-x))) of x. 8-bit
* fixed-point function accepts input in Q3.4 and form output in Q0.7 format.
*
* Parameters:
* @param [in] nInputZeroPoint - zero point, corresponds to TFLM quantization scheme
* @param [in] nInputMultiplier - multiplier, corresponds to TFLM quantization scheme
* @param [in] nInputLeftShift - shift, corresponds to TFLM quantization scheme
* @param [in] nInputSize - input size
* @param [in] pInputData - input buffer (Q3.4)
*
* @param [out] pOutputData - output buffer(Q0.7)
*
* @return None
*
*******************************************************************************
*/ 
void adi_sharcfx_logistic_int8(int32_t nInputZeroPoint, 
                               int32_t nInputMultiplier, 
                               int32_t nInputLeftShift, 
                               int32_t nInputSize, 
                               const int8_t* pInputData, 
                               int8_t* pOutputData)
{
    // Integer bits must be in sync with Prepare() function.
    static constexpr int32_t kOutputZeroPoint = -128;

    //scaling input to fit into Q3.4
    int8_t* pInput_in_q3_4 = (int8_t*)pTempL3;
    nInputLeftShift = 4-(27 - nInputLeftShift);
    nInputMultiplier = nInputMultiplier>>24;
    xb_vec4Mx8 vInZP = PDX_REP_4MX8((xb_vec4Mx8)nInputZeroPoint,0);
    xb_vec4Mx8 vTempShift = PDX_REP_4MX8((xb_vec4Mx8)nInputLeftShift,0);
    xb_vec4Mx8 vTempMult = PDX_REP_4MX8((xb_vec4Mx8)nInputMultiplier,0);
    xb_vec4Mx8 *inp = (xb_vec4Mx8 *)pInputData;
    xb_vec4Mx8 *          outp;
    outp=(      xb_vec4Mx8 *)pInput_in_q3_4;
    xb_vec4Mx8 vin;
    valign ina,outa; // define align vector
    ina=PDX_LA_4MX8_PP (inp); // prime, NOP if a[] is aligned
    outa = PDX_Z_ALIGN();
    for (int i = 0; i < nInputSize; i += 4*PDX_M)
    {
        xb_vec4Mx20 vTempOut;
        PDX_LA_4MX8_XP (vin, ina, inp, 4*PDX_M); // load aligned, extend;
        vin += vInZP;
        vTempOut = PDX_MULW_4MX8(vin, vTempMult);
        vTempOut = PDX_SLS_4MX20(vTempOut, vTempShift);
        vTempOut = PDX_ADD_4MX20(vTempOut,64);  //rounding multiplied scaled input with round data 1<<6 for 8 bit data;
        vTempOut = PDX_SRAI_4MX20(vTempOut,7);
        PDX_SAV_4MX8_XP(PDX_PACKSIV_4MX20(vTempOut,0),outa,outp, 4*PDX_M);
        PDX_SAPOS_4MX8_FP( outa, outp );
    }

    vecsigmoid_8b((int8_t *)pInput_in_q3_4, pOutputData, nInputSize );

    //scaling output and applying sign
    int16_t* output_in_q7_8 = (int16_t*)pTempL3;
    xb_vec2Mx16 vOutZP = PDX_REP_2MX16((xb_vec2Mx16)kOutputZeroPoint,0);
    xb_vec2Mx8 *out = (xb_vec2Mx8 *)pOutputData;
    xb_vec2Mx16 vout;
    xb_vec2Mx8 *          outpSig;
    outpSig=(      xb_vec2Mx8 *)pOutputData;
    valign inaSig,outaSig; // define align vector
    inaSig=PDX_LA_2MX8_PP (out); // prime, NOP if a[] is aligned
    outaSig = PDX_Z_ALIGN();
    for (int i = 0; i < nInputSize; i += PDX_2M)
    {
        PDX_LA16_2MX8_XP (vout, inaSig, out, PDX_4M); // load aligned, extend
        vout = PDX_SLLI_2MX16(vout, 1);
        vout = PDX_ADD_2MX16(vout, vOutZP);
        PDX_SAV16_2MX8_XP(vout,outaSig,outpSig, PDX_4M);
        PDX_SAPOS_2MX8_FP( outaSig, outpSig );
    }
}




// Q of input data
#define QIN        12  /* Q3.12 input format: 1 sign + 3 integer + 12 fractional bits */
// Q of output
#define QOUT       15  /* Q0.15 output format: 1 sign + 0 integer + 15 fractional bits */
/* Number of address bits into the table: 2^5 = 32 table entries per half */
#define INDEX_BITS  5
/* Width of one table entry interval: (1 << QOUT) / 32 = 1024 */
#define INDEX_STEP (1 << (QOUT - INDEX_BITS))
/* Empirical fractional offset for xi within each table entry.
 * xi = (i + 0.35) * INDEX_STEP, i in 0..31.
 * The value 0.35 was found empirically to minimise average and RMS approximation error. */
#define INDEX_OFFSET (INDEX_STEP * 35 / 100)


/* Tables used by adi_vecsigmoid_16b
 * These are values of sigmoid function and its first two derivatives in QOUT form
 * See equations below for values
 */
static int16_t sigmoid_table_qout [2][16] = {
    { 17100, 19122, 21062, 22870, 24507, 25954, 27206, 28267, 29153, 29882, 30475, 30954, 31338, 31643, 31885, 32076 },
    { 32227, 32345, 32437, 32510, 32566, 32611, 32645, 32672, 32693, 32710, 32722, 32732, 32740, 32746, 32751, 32755 },
};
static int16_t sigmoid_first_derivative_table_qout [2][16] = {
    { 8176, 7963, 7523, 6908, 6178, 5396, 4617, 3882, 3216, 2631, 2131, 1712, 1367, 1085, 858, 676 },
    { 531, 417, 326, 255, 199, 156, 121, 95, 74, 57, 45, 35, 27, 21, 16, 12 },
};
static int16_t sigmoid_second_derivative_table_qout [2][16] = {
    { -178, -665, -1074, -1367, -1531, -1576, -1525, -1407, -1253, -1084, -916, -761, -623, -505, -406, -324 },
    { -257, -203, -160, -125, -98, -77, -60, -47, -36, -28, -22, -17, -13, -10, -8, -6 },
};

/*-------------------------------------------------------------------------
Vectorized Sigmoid - Helper function for 16bit Logistic
The function returns the sigmoid (1/(1+exp(-x))) of x. 16-bit fixed-point
function accepts input in Q3.12 and form output in Q0.15 format.

Algorithm:

Second-order Taylor series:

f(x) = f(xi) + f'(xi) (x - xi) + f''(xi) * (x - xi)^2 / 2

where
xi = (i + 0.35) * 2^Q, i in 0..31.
The 0.35 was found empirically to give the least average error and least RMS error

f(x) = 2^Q / (1 + e^-x)
f'(x) = 2^Q * e^-x /  (1 + e^-x)^2
f''(x)/2 = 2^Q * (e^-2x / (1 + e^-x)^3 - e^-x /  (1 + e^-x)^2)

Input:
  x  input value (Q3.12)
Output:
  z  result (Q0.15)
Returned value:
  None
Domain:
  Whole range
---------------------------------------------------------------------------*/
/**
 *******************************************************************************
 * Function: adi_vecsigmoid_16b
 * @brief Vectorized sigmoid for 16-bit fixed-point input (Q3.12 in, Q0.15 out).
 *
 * @details Uses a 32-entry lookup table with second-order Taylor series.
 *          xi = (i + 0.35) * 2^QOUT, i in 0..31 (offset chosen empirically
 *          to minimise average and RMS error). Internal helper; called by
 *          adi_sharcfx_logistic_int16 and adi_sharcfx_logistic_int8.
 *
 * Parameters:
 * @param [in]  pInputData  Input array (int16, Q3.12 format).
 * @param [out] pOutputData Output array (int16, Q0.15 format).
 * @param [in]  nInputSize  Number of elements.
 * @return None
 *******************************************************************************
 */
void adi_vecsigmoid_16b (
        const int16_t *pInputData,     /* [in] array of N 16-bit fixed point values with 1 sign, 3 int, and 12 fractional bits  */
        int16_t *  pOutputData,         /* [out] array of N 16-bit fixed point values with 1 sign and 15 fractional bits */
        int nInputSize)                /* [in] length of array */
{
    int n;
    int left = nInputSize * sizeof (*pInputData);
    // Input pointers
    xb_vec2Mx16 *vx = (xb_vec2Mx16 *) pInputData;
    valign vxa = PDX_LA_2MX16_PP (vx);
    // Output pointers
    xb_vec2Mx16 * vz = (xb_vec2Mx16 *) pOutputData;
    valign vza = PDX_Z_ALIGN ();

    // Tables
    xb_vec2Mx16 sig_lo    = *((xb_vec2Mx16 *) sigmoid_table_qout[0]);
    xb_vec2Mx16 sig_hi    = *((xb_vec2Mx16 *) sigmoid_table_qout[1]);
    xb_vec2Mx16 sig_d1_lo = *((xb_vec2Mx16 *) sigmoid_first_derivative_table_qout[0]);
    xb_vec2Mx16 sig_d1_hi = *((xb_vec2Mx16 *) sigmoid_first_derivative_table_qout[1]);
    xb_vec2Mx16 sig_d2_lo = *((xb_vec2Mx16 *) sigmoid_second_derivative_table_qout[0]);
    xb_vec2Mx16 sig_d2_hi = *((xb_vec2Mx16 *) sigmoid_second_derivative_table_qout[1]);

    xb_vec2Mx16 vvx_qin, avvx_qin, vxi_qin, index, frac_qin, frac2_qin;
    xb_vec2Mx16 vvz_qout, vvz_neg_qout, sig_qout, sig_d1_qout, sig_d2_qout;
    xb_vec2Mx40 acc;
    xb_vec2Mx16 vzero = 0;
    xb_vec2Mx16 qout_max = (1 << QOUT);
    xb_vec2Mx16 indexOffset = INDEX_OFFSET;

    for (n = 0; n < nInputSize; n += PDX_2M)
    {
        // Get input, padding with zeros if beyond the end of the array
        PDX_LAV_2MX16_XP (vvx_qin, vxa, vx, left);
        // Find its absolute value.  The sigmoid function is mirrored around zero
        avvx_qin = PDX_ABS_2MX16 (vvx_qin);
        // Get table index
        index = PDX_SRAI_2MX16 (avvx_qin, QOUT - INDEX_BITS);
        index = PDX_MIN_2MX16(index, 31);
        index = PDX_MAX_2MX16(index, 0);
        // Look up f(xi)
        sig_qout    = PDX_SEL_2MX16 (sig_hi, sig_lo, index);
        // Look up f'(xi)
        sig_d1_qout = PDX_SEL_2MX16 (sig_d1_hi, sig_d1_lo, index);
        // Look up f''(xi)/2
        sig_d2_qout = PDX_SEL_2MX16 (sig_d2_hi, sig_d2_lo, index);
        // Find the value of x used by the tables
        vxi_qin = ((index << (QOUT - INDEX_BITS)) + indexOffset);
        // Get x - xi
        frac_qin = avvx_qin - vxi_qin;
        // Get (x - xi)^2 >> QIN
        acc = frac_qin * frac_qin;
        frac2_qin = PDX_PACKIV_2MX40  (acc, QIN);
        // Accumulate output
        acc  = PDX_MULW_2MX16 (sig_qout, 1 << QIN);
        acc += sig_d1_qout * frac_qin;
        acc += sig_d2_qout * frac2_qin;
        // Adjust to Q31
        acc <<= (QOUT + 16) - (QOUT + QIN);
        // Convert to Q15 with nearest-even rounding and saturation
        vvz_qout = PDX_PACKQSRV_2MX40  (acc, 2);
        // Complement for negative inputs
        vvz_neg_qout = qout_max - vvz_qout;
        // Find if input is negative
        vbool2M x_neg = PDX_LT_2MX16 (vvx_qin, vzero);
        // Select the negative output if input is negative
        vvz_qout = PDX_MOV_2MX16_T (vvz_neg_qout, vvz_qout, x_neg);
        // Save away, not modifying if beyond the end of the array
        PDX_SAV_2MX16_XP (vvz_qout, vza, vz, left);
        // Get number of bytes left
        left -= PDX_4M;
    }
    PDX_SAPOS_2MX16_FP (vza, vz);  // Save tails
}/*adi_vecsigmoid_16b*/

/**
*******************************************************************************
* Function: adi_sharcfx_logistic_int16
* @brief optimized implementation of logistic/sigmoid activation function
*
* @details optimized implementation of tanh activation function for int16 data. The function returns the sigmoid (1/(1+exp(-x))) of x. 16-bit fixed-point
* function accepts input in Q3.12 and form output in Q0.15 format.
*
* Parameters:
* @param [in] nInputMultiplier - multiplier, corresponds to TFLM quantization scheme
* @param [in] nInputLeftShift - shift, corresponds to TFLM quantization scheme
* @param [in] nInputSize - input size
* @param [in] pInputData - input buffer (Q3.12)
*
* @param [out] pOutputData - output buffer(Q0.15)
*
* @return None
*
*******************************************************************************
*/ 
void adi_sharcfx_logistic_int16(int32_t nInputMultiplier, 
                                int32_t nInputLeftShift, 
                                int32_t nInputSize, 
                                const int16_t* pInputData,
                                int16_t* pOutputData)
{
    int16_t* pInput_in_q3_12 = (int16_t*)pTempL3;
    xb_vec2Mx16 *inp = (xb_vec2Mx16 *)pInputData;
    xb_vec2Mx16 *outp = (xb_vec2Mx16 *)pInput_in_q3_12;
    valign ina,outa; // define align vector
    ina=PDX_LA_2MX16_PP (inp); // prime, NOP if a[] is aligned
    outa = PDX_Z_ALIGN();

    xb_vec2Mx16 vin;
    int32_t nPixLeft = nInputSize;
    int32_t nTempRound = nInputLeftShift > 0 ? (1<<(nInputLeftShift-1)) : 0;
    //scaling input to fit into Q3.12.
    if (nInputMultiplier == 0)
    {
        for (int i = 0; i < nInputSize; i += (PDX_2M))
        {
            xb_vec2Mx16 vTempInput ;
            PDX_LA_2MX16_XP (vin, ina, inp, PDX_4M); // load aligned, extend
            vTempInput = PDX_SLS_2MX16(vin, nInputLeftShift);
            PDX_SAV_2MX16_XP(vTempInput,outa,outp, PDX_4M);
            PDX_SAPOS_2MX16_FP( outa, outp );
            nPixLeft -=PDX_2M;
        }
        if(nPixLeft>0)
        {
            xb_vec2Mx16 vTempInput ;
            PDX_LA_2MX16_XP (vin, ina, inp, 0); // load aligned, extend
            vTempInput = PDX_SLS_2MX16(vin, nInputLeftShift);
            PDX_SAV_2MX16_XP(vTempInput,outa,outp, nPixLeft*2);
            PDX_SAPOS_2MX16_FP( outa, outp );
        }
    }
    else
    {
        xb_vec2Mx40 vTempOut;
        for (int i = 0; i < nInputSize; i += PDX_2M)
        {
            PDX_LA_2MX16_XP (vin, ina, inp, PDX_4M); // load aligned, extend
            vTempOut = PDX_MULW_2MX16(vin,nInputMultiplier);
            vTempOut = PDX_ADD_2MX40(vTempOut,nTempRound);
            vTempOut = PDX_SRA_2MX40(vTempOut,nInputLeftShift);
            PDX_SAV_2MX16_XP(PDX_PACKSIV_2MX40(vTempOut,0),outa,outp, PDX_4M);
            PDX_SAPOS_2MX16_FP( outa, outp );
            nPixLeft -=PDX_2M;
        }
        if(nPixLeft>0)
        {
            PDX_LA_2MX16_XP (vin, ina, inp, 0); // load aligned, extend
            vTempOut = PDX_MULW_2MX16(vin,nInputMultiplier);
            vTempOut = PDX_ADD_2MX40(vTempOut,nTempRound);  //rounding multiplied scaled input with round data nTempRound for 16 bit data;
            vTempOut = PDX_SRA_2MX40 (vTempOut,nInputLeftShift);
            PDX_SAV_2MX16_XP(PDX_PACKSIV_2MX40(vTempOut,0),outa,outp, nPixLeft*2);
            PDX_SAPOS_2MX16_FP( outa, outp );
        }
    }
    adi_vecsigmoid_16b(pInput_in_q3_12, pOutputData , nInputSize);
}

/**
*******************************************************************************
* Function: adi_sharcfx_logistic_int16_optimized_LUT
* @brief optimized implementation of logistic/sigmoid activation function with Look-up tables
*
* @details optimized implementation of tanh activation function for int16 data. The function returns the sigmoid (1/(1+exp(-x))) of x. 16-bit fixed-point
* function accepts input in Q3.12 and form output in Q0.15 format. Uses the TFLM 16-bit Logistic function as reference.
*
* Parameters:
* @param [in] nInputMultiplier - multiplier, corresponds to TFLM quantization scheme
* @param [in] nInputLeftShift - shift, corresponds to TFLM quantization scheme
* @param [in] nInputSize - input size
* @param [in] pInputData - input buffer (Q3.12)
*
* @param [out] pOutputData - output buffer(Q0.15)
*
* @return None
*
*******************************************************************************
*/ 
void adi_sharcfx_logistic_int16_optimized_LUT(int32_t nInputMultiplier,
                                              int32_t nInputLeftShift,
                                              int32_t nInputSize,
                                              const int16_t* pInputData,
                                              int16_t* pOutputData)
{

    if (nInputMultiplier == 0) {  // power of two case
        nInputMultiplier = 3 << nInputLeftShift;
      nInputLeftShift = 0;
    }

    int32_t round = (nInputLeftShift > 0) ? 1 << (nInputLeftShift - 1) : 0;

//    PRINT_INFO("%s: %d, %s: %d, %s: %d ",STRINGIZE(round),round,STRINGIZE(nInputLeftShift), nInputLeftShift, STRINGIZE(nInputMultiplier),nInputMultiplier );
    const xb_vecMx16 * inp = (xb_vecMx16 *)pInputData;
    valign ina=PDX_LA_MX16_PP (inp);
    xb_vecMx32 vin;
    xb_vecMx32 vround = round;
    xb_vecMx32 vmul = nInputMultiplier;
    xb_vecMx32 vshift = nInputLeftShift;


    xb_vecMxu32 uh,ut;
    vboolM  uh_lt_255, uh_gte_255,vneg;
    xb_vecMxu32 ua, ub;
    xb_vecMxu32 vsat =0x7FFF << 10;
    xb_vecMxu32 vmask = 0x1ff;
    xb_vecMxu32 vresult;
    xb_vecMxu32 vP,vN;

    xb_vecMx16 * outp = (xb_vecMx16 *)pOutputData;
    valign outa=PDX_Z_ALIGN();

    int32_t nElementsLeft = nInputSize;
    int32_t nToWrite =0;
    for(int32_t i=0; i< nInputSize; i+=PDX_M)
    {
        PDX_LA32_MX16_XP(vin, ina, (const xb_vecMx16*)inp, PDX_M*2);                                 //loaded input into 32 bit vector
        vin = PDX_MUL_MX32(vin, vmul);
        vin += vround;                                                      // add offset
        vin = PDX_SRA_MX32(vin, vshift);                       //divide by 256
        vneg = PDX_LT_MX32(vin,0);
        vin = PDX_ABS_MX32(vin);

        uh = PDX_SRAI_MX32(vin, 9);

        uh_lt_255 = PDX_LT_MX32( uh, (xb_vecMx32)255);
        uh_gte_255 = PDX_NOT_BM(uh_lt_255); //handle separately

        uh = PDX_MIN_MX32(uh, LUT_SIZE-2);    // LUT_SIZE -2 to avoid overflow for index +1.
        uh = PDX_MAX_MX32(uh, 0);             // to avoid negative index

        //get base address of LUT
        xb_vecMxu32 v_base_addr = (uint32_t)(uintptr_t)sigmoid_table_uint16;
        xb_vecMxu32 v_final_addr = 0;
        //compute final address of LUT by adding base address and index with type casting and store in v_final_addr
        v_final_addr = PDX_ADD_MXU32(v_base_addr, uh<<1);   // index * 2 for uint16_t

        uint32_t* addr = (uint32_t*)(xb_vecMxu32*)&v_final_addr;
        uint16_t* lut_data_ptr;

        uint32_t *result_ptr1 = (uint32_t*)&ua;
        uint32_t *result_ptr2 = (uint32_t*)&ub;
        for(int i = 0 ; i< PDX_M; i++)
        {
            lut_data_ptr = (uint16_t*)(uintptr_t)*addr;

            *result_ptr1 = *lut_data_ptr;
            *result_ptr2 = *(lut_data_ptr + 1);

            addr++;
            result_ptr1++;
            result_ptr2++;
        }


        ut = PDX_AND_MX32(vin,vmask);//ut = abs_input_data & 0x1ff;
        vresult = PDX_ADD_MX32(PDX_MUL_MX32(ut, PDX_SUB_MXU32(ub,ua)), PDX_SLA_MX32(ua,9));//(ua << 9) + ut * (ub - ua);
        vresult = PDX_MOV_MX32_T(vsat,vresult,uh_gte_255);

        //(result + (1 << 9)) -> vP
        //((1 << (16 + 9)) - result + (1 << 9) - 1) -> vN

        vP = PDX_ADD_MX32(vresult, (xb_vecMxu32)(1<<9));
        vN = PDX_ADD_MX32(PDX_SUB_MXU32((xb_vecMxu32)(1 << 25), vresult), (xb_vecMxu32)((1 << 9) - 1));
        vresult = PDX_MOV_MX32_T(vN,vP,vneg);
        //result >>= 10;
        vresult = PDX_SRLI_MX32(vresult, 10);

        nToWrite = MIN(PDX_M,nElementsLeft);
        //Save outputs to out buf
        PDX_SAV32_MX16_XP(vresult, outa, outp, nToWrite*2);
        PDX_SAPOS_MX16_FP(outa,outp);//flush
        nElementsLeft-=nToWrite;
    }
}


/**
*******************************************************************************
* Function: adi_sharcfx_tanh_int8
* @brief Vectorized implementation of tanh activation function for int8 data
*
* @details Uses Taylor series approximation with LUT, following the int16
* adi_vectanh_16b approach. Input is widened to int16, scaled to Q3.12,
* processed via the existing int16 Taylor series, and output downscaled
* from Q0.15 to Q0.7.
*
* This approach reuses the proven int16 Taylor coefficient tables for maximum
* accuracy while achieving full vectorization.
*
* Parameters:
* @param [in] nInputZeroPoint - zero point of input quantization
* @param [in] nInputMultiplier - multiplier for Q3.12 conversion (16-bit effective)
* @param [in] nInputLeftShift - shift for Q3.12 conversion
* @param [in] nInputSize - number of elements
* @param [in] pInputData - input buffer (int8)
*
* @param [out] pOutputData - output buffer (int8, Q0.7)
*
* @return None
*
*******************************************************************************
*/
void adi_sharcfx_tanh_int8(int32_t nInputZeroPoint,
                           int32_t nInputMultiplier,
                           int32_t nInputLeftShift,
                           int32_t nInputSize,
                           const int8_t* pInputData,
                           int8_t* pOutputData)
{
    // L1 buffer
    int16_t* pInput_q3_12 = (int16_t*)pTempL1;

    //Convert int8 to Q3.12
    {
        // Reduce 32-bit multiplier to 16-bit with rounding
        int16_t mult16 = static_cast<int16_t>((nInputMultiplier + (1 << 15)) >> 16);
        int32_t shift_amount = 30 - nInputLeftShift;
        // Rounding constant for right shift
        int32_t rounding = (shift_amount > 0) ? (1 << (shift_amount - 1)) : 0;
        // Vector constants
        xb_vec2Mx16 vZeroPoint = static_cast<int16_t>(nInputZeroPoint);
        // Input pointer (int8)
        xb_vec2Mx8* inp = (xb_vec2Mx8*)pInputData;
        valign ina = PDX_LA_2MX8_PP(inp);
        // Output pointer (int16)
        xb_vec2Mx16* outp = (xb_vec2Mx16*)pInput_q3_12;
        valign outa = PDX_Z_ALIGN();

        xb_vec2Mx16 vin;
        xb_vec2Mx40 acc;

        int32_t nElementsProcessed = 0;

        // process 16 elements (PDX_2M) at a time
        for (; (nInputSize - nElementsProcessed) >= PDX_2M; nElementsProcessed += PDX_2M)
        {
            // Load 16 int8 values, sign-extend to int16
            PDX_LA16_2MX8_XP(vin, ina, inp, PDX_2M);
            // Subtract zero point
            vin = PDX_SUB_2MX16(vin, vZeroPoint);
            // Multiply int16 * int16 -> 40-bit accumulator
            acc = PDX_MULW_2MX16(vin, mult16);
            // Add rounding constant
            acc = PDX_ADD_2MX40(acc, rounding);
            // Right shift
            acc = PDX_SRA_2MX40(acc, shift_amount);
            // Pack 40-bit to 16-bit with saturation and store
            PDX_SAV_2MX16_XP(PDX_PACKSIV_2MX40(acc, 0), outa, outp, PDX_4M);
            PDX_SAPOS_2MX16_FP(outa, outp);
        }

        // Handle remaining elements
        if ((nInputSize - nElementsProcessed) > 0)
        {
            int32_t remaining = nInputSize - nElementsProcessed;
            // Load remaining int8 values
            PDX_LA16_2MX8_XP(vin, ina, inp, 0);
            // Subtract zero point
            vin = PDX_SUB_2MX16(vin, vZeroPoint);
            // Multiply int16 * int16 -> 40-bit accumulator
            acc = PDX_MULW_2MX16(vin, mult16);
            // Add rounding constant
            acc = PDX_ADD_2MX40(acc, rounding);
            // Right shift
            acc = PDX_SRA_2MX40(acc, shift_amount);
            // Pack 40-bit to 16-bit with saturation and store remaining elements
            PDX_SAV_2MX16_XP(PDX_PACKSIV_2MX40(acc, 0), outa, outp, remaining * 2);
            PDX_SAPOS_2MX16_FP(outa, outp);
        }
    }

    // Apply int16 vectorized tanh (Q3.12 -> Q0.15)
    int16_t* pOutput_q0_15 = pInput_q3_12 + ((nInputSize + 15) & ~15);
    adi_vectanh_16b(pInput_q3_12, pOutput_q0_15, nInputSize);

    //Downscale Q0.15 to Q0.7 and pack to int8
    {
        int left = nInputSize * sizeof(int16_t);  // bytes left to process

        // Input pointer (int16)
        xb_vec2Mx16* vIn = (xb_vec2Mx16*)pOutput_q0_15;
        valign vInAlign = PDX_LA_2MX16_PP(vIn);

        // Output pointer (int8) - use xb_vec2Mx8 for 16-element int8 stores
        xb_vec2Mx8* vOut = (xb_vec2Mx8*)pOutputData;
        valign vOutAlign = PDX_Z_ALIGN();

        // Constants for rounding and clamping
        xb_vec2Mx16 vRound = 128;       // Rounding constant
        xb_vec2Mx16 vMin = -128;        // int8 min
        xb_vec2Mx16 vMax = 127;         // int8 max
        xb_vec2Mx16 vQ15, vQ7;

        for (int n = 0; n < nInputSize; n += 2 * PDX_M)
        {
            // Load 16 Q0.15 values
            PDX_LAV_2MX16_XP(vQ15, vInAlign, vIn, left);
            // Add rounding constant (128) with SATURATION to prevent overflow
            // (32767 + 128 would overflow to negative without saturation)
            vQ7 = PDX_ADDS_2MX16(vQ15, vRound);
            // Shift right by 8 to convert Q0.15 -> Q0.7
            vQ7 = PDX_SRAI_2MX16(vQ7, 8);
            // Clamp to int8 range [-128, 127]
            vQ7 = PDX_MIN_2MX16(vQ7, vMax);
            vQ7 = PDX_MAX_2MX16(vQ7, vMin);
            // Store as int8 (width-converting store: takes lower byte of each int16)
            // Note: left/2 gives number of int8 elements remaining
            PDX_SAV16_2MX8_XP(vQ7, vOutAlign, vOut, left / 2);
            left -= 4 * PDX_M;  // 2*PDX_M int16 elements = 4*PDX_M bytes
        }
        PDX_SAPOS_2MX8_FP(vOutAlign, vOut);  // Flush alignment tail
    }
}

