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

int main(int argc, char *argv[])
{
    std::cout << "NvPipe example application: Comparison of using host/device memory." << std::endl
              << std::endl;

    const uint32_t width = 3840;
    const uint32_t height = 2160;

    const NvPipe_Codec codec = NVPIPE_H264;
    const NvPipe_Compression compression = NVPIPE_LOSSY;
    const float bitrateMbps = 32;
    const uint32_t targetFPS = 90;

    std::cout << "Resolution: " << width << " x " << height << std::endl;
    std::cout << "Codec: " << (codec == NVPIPE_H264 ? "H.264" : "HEVC") << std::endl;
    if (compression == NVPIPE_LOSSY)
        std::cout << "Bitrate: " << bitrateMbps << " Mbps @ " << targetFPS << " Hz" << std::endl;

    // Construct dummy frame
    std::vector<uint8_t> rgba(width * height * 4);
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
            rgba[4 * (y * width + x) + 1] = (255.0f * x * y) / (width * height) * (y % 100 < 50);

    std::cout << "Resolution: " << width << " x " << height << std::endl;

    std::vector<uint8_t> compressed1(rgba.size());
    std::vector<uint8_t> decompressed1(rgba.size());
    std::vector<uint8_t> compressed2(rgba.size());
    std::vector<uint8_t> decompressed2(rgba.size());

    Timer timer;

    // Host memory benchmark

    std::cout << std::endl
              << "--- Encode from host memory / Decode to host memory ---" << std::endl;
    std::cout << "Frame | Encode (ms) | Decode (ms) | Size (KB)" << std::endl;

    // Create encoder
    NvPipe *encoder1 = NvPipe_CreateEncoder(NVPIPE_BGRA32, codec, compression, bitrateMbps * 1000 * 1000, targetFPS);
    if (!encoder1)
        std::cerr << "Failed to create encoder1: " << NvPipe_GetError(NULL) << std::endl;

    // Create decoder
    NvPipe *decoder1 = NvPipe_CreateDecoder(NVPIPE_BGRA32, codec);
    if (!decoder1)
        std::cerr << "Failed to create decoder1: " << NvPipe_GetError(NULL) << std::endl;

    // Device memory benchmark
    std::cout << std::endl
              << "--- Encode from device memory / Decode to device memory ---" << std::endl;
    std::cout << "Frame | Encode (ms) | Decode (ms) | Size (KB)" << std::endl;

    // Create encoder
    NvPipe *encoder2 = NvPipe_CreateEncoder(NVPIPE_BGRA32, codec, compression, bitrateMbps * 1000 * 1000, targetFPS);
    if (!encoder2)
        std::cerr << "Failed to create encoder2: " << NvPipe_GetError(NULL) << std::endl;

    // Create decoder
    NvPipe *decoder2 = NvPipe_CreateDecoder(NVPIPE_BGRA32, codec);
    if (!decoder2)
        std::cerr << "Failed to create decoder2: " << NvPipe_GetError(NULL) << std::endl;

    // Allocate device memory and copy input
    void *rgbaDevice;
    cudaMalloc(&rgbaDevice, rgba.size());
    cudaMemcpy(rgbaDevice, rgba.data(), rgba.size(), cudaMemcpyHostToDevice);

    void *decompressedDevice;
    cudaMalloc(&decompressedDevice, rgba.size());

    // A few frames ...
    for (uint32_t i = 0; i < 10; ++i)
    {
        // Encode
        timer.reset();
        uint64_t size1 = NvPipe_Encode(encoder1, rgba.data(), width * 4, compressed1.data(), compressed1.size(), width, height, false);
        double encodeMs1 = timer.getElapsedMilliseconds();

        if (0 == size1)
            std::cerr << "Encode error: " << NvPipe_GetError(encoder1) << std::endl;

        // Encode
        timer.reset();
        uint64_t size2 = NvPipe_Encode(encoder2, rgbaDevice, width * 4, compressed2.data(), compressed2.size(), width, height, false);
        double encodeMs2 = timer.getElapsedMilliseconds();

        if (0 == size2)
            std::cerr << "Encode error: " << NvPipe_GetError(encoder2) << std::endl;

        // Decode
        timer.reset();
        uint64_t r = NvPipe_Decode(decoder1, compressed1.data(), size1, decompressed1.data(), width, height);
        double decodeMs = timer.getElapsedMilliseconds();

        if (0 == r)
            std::cerr << "Decode error: " << NvPipe_GetError(decoder1) << std::endl;

        double sizeKB = size1 / 1000.0;
        std::cout << std::fixed << std::setprecision(1) << std::setw(5) << i << " | " << std::setw(11) << encodeMs1 << " | " << std::setw(11) << decodeMs << " | " << std::setw(8) << sizeKB << std::endl;

        // Decode
        timer.reset();
        r = NvPipe_Decode(decoder2, compressed2.data(), size2, decompressedDevice, width, height);
        decodeMs = timer.getElapsedMilliseconds();

        if (0 == r)
            std::cerr << "Decode error: " << NvPipe_GetError(decoder2) << std::endl;

        sizeKB = size2 / 1000.0;
        std::cout << std::fixed << std::setprecision(1) << std::setw(5) << i << " | " << std::setw(11) << encodeMs2 << " | " << std::setw(11) << decodeMs << " | " << std::setw(8) << sizeKB << std::endl;
    }

    // Clean up
    NvPipe_Destroy(encoder1);
    NvPipe_Destroy(decoder1);

    cudaFree(rgbaDevice);
    cudaFree(decompressedDevice);

    // Clean up
    NvPipe_Destroy(encoder2);
    NvPipe_Destroy(decoder2);

    return 0;
}
