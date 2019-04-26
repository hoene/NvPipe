/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <NvPipe.h>

#include "utils.h"

#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

constexpr int MAX_CODECS  = 10;
constexpr uint32_t width = 100;
constexpr uint32_t height = 100;
constexpr float bitrateMbps = 1;
constexpr uint32_t targetFPS = 10;

struct Instance {
        std::vector<uint8_t> *compressed;
	uint64_t size;
        std::vector<uint8_t> *decompressed;
        NvPipe *encoder;
        NvPipe *decoder;
    } instances[MAX_CODECS];

int main(int argc, char *argv[])
{
    std::cout << "NvPipe example application: How many codecs can run parallel?" << std::endl
              << std::endl;

    const uint32_t width = 100;
    const uint32_t height = 100;

    const NvPipe_Codec codec = NVPIPE_H264;
    const NvPipe_Compression compression = NVPIPE_LOSSY;


    std::cout << "Resolution: " << width << " x " << height << std::endl;
    std::cout << "Codec: " << (codec == NVPIPE_H264 ? "H.264" : "HEVC") << std::endl;
    if (compression == NVPIPE_LOSSY)
        std::cout << "Bitrate: " << bitrateMbps << " Mbps @ " << targetFPS << " Hz" << std::endl;

/*
   * query the number of graphic cards 
   */
  int device_count = 1;
  cudaError_t result = cudaGetDeviceCount(&device_count);
  if (result != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount error " << NvPipe_GetError(NULL) << std::endl;
  }
        std::cout << "Graphic cards: " << device_count << std::endl;


    // Construct dummy frame
    std::vector<uint8_t> rgba(width * height * 4);
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
            rgba[4 * (y * width + x) + 1] = (255.0f * x * y) / (width * height) * (y % 100 < 50);

    std::cout << "Resolution: " << width << " x " << height << std::endl;

    Timer timer;

    // Create encoders
    std::cout << std::endl
              << "--- Create encoders ---" << std::endl;
 
    int device = 0;

    for(int i=0;i<MAX_CODECS;i++) {

	 result = cudaSetDevice(device);
         if (result != cudaSuccess) {
            std::cerr << "cudaSetDevice error " << NvPipe_GetError(NULL) << std::endl;
         }

 	instances[i].encoder = NvPipe_CreateEncoder(NVPIPE_BGRA32, codec, compression, bitrateMbps * 1000 * 1000, targetFPS);
        if (!instances[i].encoder) {
	    device++;
	    if(device < device_count) {
		 result = cudaSetDevice(device);
	         if (result != cudaSuccess) {
        	    std::cerr << "cudaSetDevice error " << NvPipe_GetError(NULL) << std::endl;
	         }
	
 		  instances[i].encoder = NvPipe_CreateEncoder(NVPIPE_BGRA32, codec, compression, bitrateMbps * 1000 * 1000, targetFPS);
             }
	}
        if (!instances[i].encoder) {
		 std::cout << "Failed to create " << (i+1) << ". encoder on graphic card " << device << ": " <<  NvPipe_GetError(NULL) << std::endl;
		break;
        }
        instances[i].compressed = new std::vector<uint8_t>(rgba.size());
    }


    // Create decoder
    std::cout << std::endl
              << "--- Create decoders ---" << std::endl;
 
    for(int i=0;i<MAX_CODECS;i++) {
 	instances[i].decoder = NvPipe_CreateDecoder(NVPIPE_BGRA32, codec);
	    if (!instances[i].decoder) {
                  std::cout << "Failed to create " << (i+1) << ". decoder: " << NvPipe_GetError(NULL) << std::endl;
		  break;
	    }
	    instances[i].decompressed = new std::vector<uint8_t>(rgba.size());
    }

    // Device memory benchmark
    std::cout << std::endl
              << "--- Encode from device memory / Decode to device memory ---" << std::endl;
    std::cout << "Frame | Encode (ms) | Decode (ms) | Size (KB)" << std::endl;

    // A few frames ...
    for (uint32_t i = 0; i < 10; ++i)
    {
        // Encode
        timer.reset();

	for(int i=0;i<MAX_CODECS && instances[i].encoder;i++) {
	        instances[i].size = NvPipe_Encode(instances[i].encoder, rgba.data(), width * 4, instances[i].compressed->data(), instances[i].compressed->size(), width, height, false);
		   if (0 == instances[i].size)
		            std::cerr << "Encode error at " << i << ": " << NvPipe_GetError(instances[i].encoder) << std::endl;
	}
        double encodeMs = timer.getElapsedMilliseconds();
   
        // Decode
        timer.reset();

	for(int i=0;i<MAX_CODECS && instances[i].decoder;i++) {
		int j=i;
		if(instances[i].encoder == NULL)
			j=0;
		uint64_t r = NvPipe_Decode(instances[i].decoder, instances[j].compressed->data(), instances[j].size, instances[i].decompressed->data(), width, height);
                if (0 == r)
                    std::cerr << "Decode error at " << i << ": " << NvPipe_GetError(instances[i].decoder) << std::endl;
	}
        double decodeMs = timer.getElapsedMilliseconds();


        double sizeKB = instances[0].size / 1000.0;
        std::cout << std::fixed << std::setprecision(1) << std::setw(5) << i << " | " << std::setw(11) << encodeMs << " | " << std::setw(11) << decodeMs << " | " << std::setw(8) << sizeKB << std::endl;
    }

    // Clean up
	for(int i=0;i<MAX_CODECS && instances[i].encoder;i++) 
    		NvPipe_Destroy(instances[i].encoder);
	for(int i=0;i<MAX_CODECS && instances[i].decoder;i++) 
    		NvPipe_Destroy(instances[i].decoder);

    return 0;
}
