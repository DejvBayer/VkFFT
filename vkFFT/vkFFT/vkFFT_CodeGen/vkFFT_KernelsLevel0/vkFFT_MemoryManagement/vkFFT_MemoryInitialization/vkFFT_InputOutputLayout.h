// This file is part of VkFFT
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef VKFFT_INPUTOUTPUTLAYOUT_H
#define VKFFT_INPUTOUTPUTLAYOUT_H
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h"
static inline void appendLayoutVkFFT(VkFFTSpecializationConstantsLayout* sc) {
	if (sc->res != VKFFT_SUCCESS) return;
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "layout (local_size_x = %" PRIi64 ", local_size_y = %" PRIi64 ", local_size_z = %" PRIi64 ") in;\n", sc->localSize[0].data.i, sc->localSize[1].data.i, sc->localSize[2].data.i);
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==1)
#elif(VKFFT_BACKEND==2)
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#endif
	return;
}
static inline void appendInputLayoutVkFFT(VkFFTSpecializationConstantsLayout* sc, int id) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer* inputMemoryType;
	int typeSize;
	if(sc->inputBufferSeparateComplexComponents) {
		PfGetTypeFromCode(sc, sc->floatTypeInputMemoryCode, &inputMemoryType);
		typeSize = PfSizeTypeFromCode(sc, sc->floatTypeInputMemoryCode);
	}	
	else{
		PfGetTypeFromCode(sc, sc->inputMemoryCode, &inputMemoryType);
		typeSize =  PfSizeTypeFromCode(sc, sc->inputMemoryCode);
	}
#if(VKFFT_BACKEND==0)
	if (sc->inputBufferBlockNum == 1) {
		sc->tempLen = sprintf(sc->tempStr, "\
layout(std430, binding = %d) buffer DataIn{\n\
	%s %s[%" PRIu64 "];\n\
};\n\n", id, inputMemoryType->name, sc->inputsStruct.name, sc->inputBufferBlockSize / typeSize);
		PfAppendLine(sc);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "\
layout(std430, binding = %d) buffer DataIn{\n\
	%s %s[%" PRIu64 "];\n\
} %sBlocks[%" PRIu64 "];\n\n", id, inputMemoryType->name, sc->inputsStruct.name, sc->inputBufferBlockSize / typeSize, sc->inputsStruct.name, sc->inputBufferBlockNum);
		PfAppendLine(sc);
	}
#elif(VKFFT_BACKEND==1)
#elif(VKFFT_BACKEND==2)
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#elif(VKFFT_BACKEND==5)
#endif
	return;
}
static inline void appendOutputLayoutVkFFT(VkFFTSpecializationConstantsLayout* sc, int id) {
	if (sc->res != VKFFT_SUCCESS) return; 
	PfContainer* outputMemoryType;
	int typeSize;
	if(sc->outputBufferSeparateComplexComponents) {
		PfGetTypeFromCode(sc, sc->floatTypeOutputMemoryCode, &outputMemoryType);
		typeSize = PfSizeTypeFromCode(sc, sc->floatTypeOutputMemoryCode);
	}	
	else{
		PfGetTypeFromCode(sc, sc->outputMemoryCode, &outputMemoryType);
		typeSize =  PfSizeTypeFromCode(sc, sc->outputMemoryCode);
	}
#if(VKFFT_BACKEND==0)
	if (sc->outputBufferBlockNum == 1) {
		sc->tempLen = sprintf(sc->tempStr, "\
layout(std430, binding = %d) buffer DataOut{\n\
	%s %s[%" PRIu64 "];\n\
};\n\n", id, outputMemoryType->name, sc->outputsStruct.name, sc->outputBufferBlockSize / typeSize);
		PfAppendLine(sc);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "\
layout(std430, binding = %d) buffer DataOut{\n\
	%s %s[%" PRIu64 "];\n\
} %sBlocks[%" PRIu64 "];\n\n", id, outputMemoryType->name, sc->outputsStruct.name, sc->outputBufferBlockSize / typeSize, sc->outputsStruct.name, sc->outputBufferBlockNum);
		PfAppendLine(sc);
	}
#elif(VKFFT_BACKEND==1)
#elif(VKFFT_BACKEND==2)
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#elif(VKFFT_BACKEND==5)
#endif
	return;
}
static inline void appendKernelLayoutVkFFT(VkFFTSpecializationConstantsLayout* sc, int id) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer* kernelMemoryType;
	int typeSize;
	if(sc->kernelSeparateComplexComponents) {
		PfGetTypeFromCode(sc, sc->floatTypeKernelMemoryCode, &kernelMemoryType);
		typeSize = PfSizeTypeFromCode(sc, sc->floatTypeKernelMemoryCode);
	}	
	else{
		PfGetTypeFromCode(sc, sc->vecTypeKernelMemoryCode, &kernelMemoryType);
		typeSize = PfSizeTypeFromCode(sc, sc->vecTypeKernelMemoryCode);
	}
#if(VKFFT_BACKEND==0)
	if (sc->kernelBlockNum == 1) {
		sc->tempLen = sprintf(sc->tempStr, "\
layout(std430, binding = %d) buffer Kernel_FFT{\n\
	%s %s[%" PRIu64 "];\n\
};\n\n", id, kernelMemoryType->name, sc->kernelStruct.name, sc->kernelBlockSize / typeSize);
		PfAppendLine(sc);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "\
layout(std430, binding = %d) buffer Kernel_FFT{\n\
	%s %s[%" PRIu64 "];\n\
} %sBlocks[%" PRIu64 "];\n\n", id, kernelMemoryType->name, sc->kernelStruct.name, sc->kernelBlockSize / typeSize, sc->kernelStruct.name, sc->kernelBlockNum);
		PfAppendLine(sc);
		
	}
#elif(VKFFT_BACKEND==1)
#elif(VKFFT_BACKEND==2)
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#elif(VKFFT_BACKEND==5)
#endif
	return;
}
static inline void appendLUTLayoutVkFFT(VkFFTSpecializationConstantsLayout* sc, int id) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);

#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "\
layout(std430, binding = %d) readonly buffer DataLUT {\n\
%s twiddleLUT[];\n\
};\n", id, vecType->name);
	PfAppendLine(sc);
	
#elif(VKFFT_BACKEND==1)
#elif(VKFFT_BACKEND==2)
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#elif(VKFFT_BACKEND==5)
#endif
	return;
}
static inline void appendRaderUintLUTLayoutVkFFT(VkFFTSpecializationConstantsLayout* sc, int id) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer* uintType32;
	PfGetTypeFromCode(sc, sc->uintType32Code, &uintType32);

#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "\
layout(std430, binding = %d) readonly buffer DataRaderUintLUT {\n\
%s g_pow[];\n\
};\n", id, uintType32->name);
	PfAppendLine(sc);
	
#elif(VKFFT_BACKEND==1)
#elif(VKFFT_BACKEND==2)
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#elif(VKFFT_BACKEND==5)
#endif
	return;
}
static inline void appendBluesteinLayoutVkFFT(VkFFTSpecializationConstantsLayout* sc, int id) {
	if (sc->res != VKFFT_SUCCESS) return;
	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);

#if(VKFFT_BACKEND==0)
	int loc_id = id;
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, "\
layout(std430, binding = %d) readonly buffer DataBluesteinConvolutionKernel {\n\
%s BluesteinConvolutionKernel[];\n\
};\n", loc_id, vecType->name);
		PfAppendLine(sc);
		loc_id++;
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, "\
layout(std430, binding = %d) readonly buffer DataBluesteinMultiplication {\n\
%s BluesteinMultiplication[];\n\
};\n", loc_id, vecType->name);
		PfAppendLine(sc);
		loc_id++;
	}
#elif(VKFFT_BACKEND==1)
#elif(VKFFT_BACKEND==2)
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#elif(VKFFT_BACKEND==5)
#endif
	return;
}

#endif
