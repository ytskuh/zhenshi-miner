#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <iomanip>
#include <openssl/sha.h>
#include <random>
#include <chrono>

// SHA-256 常量
const __constant__ uint32_t c_H256[8] = {
    0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
    0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};

const __constant__ uint32_t c_K[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

// SHA-256 操作函数
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

__device__ __forceinline__ uint32_t xandx(uint32_t e, uint32_t f, uint32_t g) {
    return (((f) ^ (g)) & (e)) ^ (g);
}

__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x) {
    return (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22));
}

__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x) {
    return (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25));
}

__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x) {
    return (ROTR32(x, 7) ^ ROTR32(x, 18) ^ (x >> 3));
}

__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x) {
    return (ROTR32(x, 17) ^ ROTR32(x, 19) ^ (x >> 10));
}

__device__ __forceinline__ uint32_t andor32(const uint32_t a, const uint32_t b, const uint32_t c) {
    return ((b) & (c)) | (((b) | (c)) & (a));
}

__device__ static void sha2_step1(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
                                  uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
                                  uint32_t in, const uint32_t Kshared) {
    uint32_t t1 = h + bsg2_1(e) + xandx(e, f, g) + Kshared + in;
    uint32_t t2 = bsg2_0(a) + andor32(a, b, c);
    d += t1;
    h = t1 + t2;
}

__device__ static void sha2_step2(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
                                  uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
                                  uint32_t* in, uint32_t pc, const uint32_t Kshared) {
    uint32_t t1, t2;
    int pcidx1 = (pc - 2) & 0xF;
    int pcidx2 = (pc - 7) & 0xF;
    int pcidx3 = (pc - 15) & 0xF;

    uint32_t inx0 = in[pc];
    uint32_t inx1 = in[pcidx1];
    uint32_t inx2 = in[pcidx2];
    uint32_t inx3 = in[pcidx3];

    in[pc] = ssg2_1(inx1) + inx2 + ssg2_0(inx3) + inx0;

    t1 = h + bsg2_1(e) + xandx(e, f, g) + Kshared + in[pc];
    t2 = bsg2_0(a) + andor32(a, b, c);
    d += t1;
    h = t1 + t2;
}

__device__ static void sha256_round_body(uint32_t* in, uint32_t* state, uint32_t* const Kshared) {
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    sha2_step1(a, b, c, d, e, f, g, h, in[0], Kshared[0]);
    sha2_step1(h, a, b, c, d, e, f, g, in[1], Kshared[1]);
    sha2_step1(g, h, a, b, c, d, e, f, in[2], Kshared[2]);
    sha2_step1(f, g, h, a, b, c, d, e, in[3], Kshared[3]);
    sha2_step1(e, f, g, h, a, b, c, d, in[4], Kshared[4]);
    sha2_step1(d, e, f, g, h, a, b, c, in[5], Kshared[5]);
    sha2_step1(c, d, e, f, g, h, a, b, in[6], Kshared[6]);
    sha2_step1(b, c, d, e, f, g, h, a, in[7], Kshared[7]);
    sha2_step1(a, b, c, d, e, f, g, h, in[8], Kshared[8]);
    sha2_step1(h, a, b, c, d, e, f, g, in[9], Kshared[9]);
    sha2_step1(g, h, a, b, c, d, e, f, in[10], Kshared[10]);
    sha2_step1(f, g, h, a, b, c, d, e, in[11], Kshared[11]);
    sha2_step1(e, f, g, h, a, b, c, d, in[12], Kshared[12]);
    sha2_step1(d, e, f, g, h, a, b, c, in[13], Kshared[13]);
    sha2_step1(c, d, e, f, g, h, a, b, in[14], Kshared[14]);
    sha2_step1(b, c, d, e, f, g, h, a, in[15], Kshared[15]);

#pragma unroll
    for (int i = 0; i < 3; i++) {
        sha2_step2(a, b, c, d, e, f, g, h, in, 0, Kshared[16 + 16 * i]);
        sha2_step2(h, a, b, c, d, e, f, g, in, 1, Kshared[17 + 16 * i]);
        sha2_step2(g, h, a, b, c, d, e, f, in, 2, Kshared[18 + 16 * i]);
        sha2_step2(f, g, h, a, b, c, d, e, in, 3, Kshared[19 + 16 * i]);
        sha2_step2(e, f, g, h, a, b, c, d, in, 4, Kshared[20 + 16 * i]);
        sha2_step2(d, e, f, g, h, a, b, c, in, 5, Kshared[21 + 16 * i]);
        sha2_step2(c, d, e, f, g, h, a, b, in, 6, Kshared[22 + 16 * i]);
        sha2_step2(b, c, d, e, f, g, h, a, in, 7, Kshared[23 + 16 * i]);
        sha2_step2(a, b, c, d, e, f, g, h, in, 8, Kshared[24 + 16 * i]);
        sha2_step2(h, a, b, c, d, e, f, g, in, 9, Kshared[25 + 16 * i]);
        sha2_step2(g, h, a, b, c, d, e, f, in, 10, Kshared[26 + 16 * i]);
        sha2_step2(f, g, h, a, b, c, d, e, in, 11, Kshared[27 + 16 * i]);
        sha2_step2(e, f, g, h, a, b, c, d, in, 12, Kshared[28 + 16 * i]);
        sha2_step2(d, e, f, g, h, a, b, c, in, 13, Kshared[29 + 16 * i]);
        sha2_step2(c, d, e, f, g, h, a, b, in, 14, Kshared[30 + 16 * i]);
        sha2_step2(b, c, d, e, f, g, h, a, in, 15, Kshared[31 + 16 * i]);
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// SHA-256 填充
__device__ void sha256_pad(uint8_t* input, size_t len, uint32_t* padded, size_t& padded_len) {
    for (size_t i = 0; i < len; ++i) {
        padded[i / 4] |= ((uint32_t)input[i]) << ((3 - (i % 4)) * 8);
    }
    padded[len / 4] |= 0x80U << ((3 - (len % 4)) * 8);
    padded_len = ((len + 8 + 63) / 64) * 64;
    for (size_t i = len + 1; i < padded_len - 8; ++i) {
        padded[i / 4] &= ~(0xFFU << ((3 - (i % 4)) * 8));
    }
    uint64_t bit_len = len * 8;
    padded[(padded_len / 4) - 2] = (uint32_t)(bit_len >> 32);
    padded[(padded_len / 4) - 1] = (uint32_t)bit_len;
}

// 检查前 k 位
__device__ bool check_leading_zeros(const uint32_t* hash, int k) {
    int full_bytes = k / 8;
    int extra_bits = k % 8;

    for (int i = 0; i < full_bytes; ++i) {
        int byte_idx = i / 4;
        int byte_offset = 3 - (i % 4);
        if (((uint8_t*)&hash[byte_idx])[byte_offset] != 0) {
            return false;
        }
    }

    if (extra_bits > 0) {
        int byte_idx = full_bytes / 4;
        int byte_offset = 3 - (full_bytes % 4);
        uint8_t byte = ((uint8_t*)&hash[byte_idx])[byte_offset];
        if (byte >> (8 - extra_bits) != 0) {
            return false;
        }
    }
    return true;
}

// 生成 64 进制字符串
__device__ void generate_x(uint64_t index, char* x, const int x_len) {
    for (int i = x_len - 1; i >= 0; --i) {
        x[i] = '0' + (index % 64);
        index /= 64;
    }
}

// CUDA 内核
__global__ void find_nonce(const char* q, size_t q_len, uint64_t start_index, int k, int x_len,
                          char* result_x, uint32_t* result_hash, int* counter) {
    __shared__ uint32_t s_K[64];
    if (threadIdx.x < 64) s_K[threadIdx.x] = c_K[threadIdx.x];
    __syncthreads();

    uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t index = start_index + thread;

    // 生成 x
    char x[16];
    generate_x(index, x, x_len);

    // 拼接 q 和 x
    size_t input_len = q_len + x_len;
    uint8_t input[256];
    for (size_t i = 0; i < q_len; ++i) {
        input[i] = q[i];
    }
    for (int i = 0; i < x_len; ++i) {
        input[q_len + i] = x[i];
    }

    // 填充
    uint32_t padded[64] = {0};
    size_t padded_len;
    sha256_pad(input, input_len, padded, padded_len);

    // SHA-256
    uint32_t state[8];
    for (int i = 0; i < 8; ++i) {
        state[i] = c_H256[i];
    }
    for (size_t block = 0; block < padded_len / 64; ++block) {
        sha256_round_body(&padded[block * 16], state, s_K);
    }

    // 检查
    if (check_leading_zeros(state, k)) {
        int res_idx = atomicAdd(counter, 1);
        if (res_idx < 1024) {
            for (int i = 0; i < x_len; ++i) {
                result_x[res_idx * 16 + i] = x[i];
            }
            for (int i = 0; i < 8; ++i) {
                result_hash[res_idx * 8 + i] = state[i];
            }
        }
    }
}

// OpenSSL SHA-256
void compute_sha256_openssl(const std::string& input, unsigned char* hash) {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, input.c_str(), input.length());
    SHA256_Final(hash, &sha256);
}

int main(int argc, char* argv[]) {
    // 命令行参数
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <q> <k> <output: full|minimal>" << std::endl;
        return 1;
    }

    std::string q = argv[1];
    if (q.empty()) {
        std::cerr << "Error: q cannot be empty" << std::endl;
        return 1;
    }

    int k;
    try {
        k = std::stoi(argv[2]);
        if (k <= 0) throw std::invalid_argument("k must be positive");
    } catch (...) {
        std::cerr << "Error: k must be a positive integer" << std::endl;
        return 1;
    }

    std::string output_mode = argv[3];
    bool full_output = true;
    if (output_mode == "minimal") {
        full_output = false;
    } else if (output_mode != "full") {
        std::cerr << "Error: output must be 'full' or 'minimal'" << std::endl;
        return 1;
    }

    size_t q_len = q.length();
    const int x_len = 8;

    // GPU 内存
    char* d_q;
    cudaMalloc(&d_q, q_len);
    cudaMemcpy(d_q, q.c_str(), q_len, cudaMemcpyHostToDevice);

    char* d_result_x;
    cudaMalloc(&d_result_x, 1024 * 16 * sizeof(char));
    cudaMemset(d_result_x, 0, 1024 * 16 * sizeof(char));

    uint32_t* d_result_hash;
    cudaMalloc(&d_result_hash, 1024 * 8 * sizeof(uint32_t));
    cudaMemset(d_result_hash, 0, 1024 * 8 * sizeof(uint32_t));

    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    // 线程配置
    const int block_size = 256;
    const int grid_size = 4096;
    const int num_threads = block_size * grid_size;

    // 随机起点
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 48) - num_threads);
    uint64_t start_index = dist(gen);

    // 哈希率统计
    uint64_t total_hashes = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    bool found = false;

    if (full_output) {
        std::cout << "Searching for x with " << k << " leading zeros for q = \"" << q << "\"" 
                  << " (random start: " << start_index << ")..." << std::endl;
    }

    // 主循环
    while (!found) {
        find_nonce<<<grid_size, block_size>>>(d_q, q_len, start_index, k, x_len, d_result_x, d_result_hash, d_counter);
        cudaDeviceSynchronize();

        total_hashes += num_threads;

        int counter;
        cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
        if (counter > 0) {
            std::vector<char> result_x(counter * 16);
            cudaMemcpy(result_x.data(), d_result_x, counter * 16 * sizeof(char), cudaMemcpyDeviceToHost);
            std::vector<uint32_t> result_hash(counter * 8);
            cudaMemcpy(result_hash.data(), d_result_hash, counter * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            for (int i = 0; i < counter; ++i) {
                std::string x(result_x.data() + i * 16, x_len);

                if (!full_output) {
                    std::cout << x << std::endl;
                } else {
                    std::cout << "Found x: " << x << std::endl;

                    // 构造 q + x
                    std::string q_plus_x = q + x;
                    std::cout << "q + x: " << q_plus_x << std::endl;

                    // CUDA 哈希
                    std::cout << "CUDA SHA-256(q + x) = ";
                    for (int j = 0; j < 8; ++j) {
                        uint32_t word = result_hash[i * 8 + j];
                        for (int b = 3; b >= 0; --b) {
                            std::cout << std::hex << std::setw(2) << std::setfill('0') 
                                      << ((word >> (b * 8)) & 0xFF);
                        }
                    }
                    std::cout << std::dec << std::endl;
                }

                // OpenSSL 计算
                std::string q_plus_x = q + x;
                unsigned char hash[SHA256_DIGEST_LENGTH];
                compute_sha256_openssl(q_plus_x, hash);

                if (full_output) {
                    // 输出哈希
                    std::cout << "OpenSSL SHA-256(q + x) = ";
                    for (int j = 0; j < SHA256_DIGEST_LENGTH; ++j) {
                        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)hash[j];
                    }
                    std::cout << std::dec << std::endl;
                }

                // 验证前导零
                int leading_zeros = 0;
                for (int j = 0; j < SHA256_DIGEST_LENGTH; ++j) {
                    if (hash[j] == 0) {
                        leading_zeros += 8;
                    } else {
                        uint8_t byte = hash[j];
                        for (int b = 7; b >= 0; --b) {
                            if (byte & (1 << b)) break;
                            leading_zeros++;
                        }
                        break;
                    }
                }

                if (full_output) {
                    std::cout << "Leading zeros: " << leading_zeros << " (expected: " << k << ")" << std::endl;
                }

                if (leading_zeros >= k) {
                    found = true;

                    if (full_output) {
                        // 计算哈希率
                        auto end_time = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                        double seconds = duration.count() / 1e6;
                        double hash_rate = total_hashes / seconds / 1e6;
                        std::cout << "Hash rate: " << std::fixed << std::setprecision(2) 
                                  << hash_rate << " MH/s" << std::endl;
                    }
                } else if (full_output) {
                    std::cout << "Invalid result, continuing search..." << std::endl;
                    cudaMemset(d_counter, 0, sizeof(int));
                    start_index = dist(gen);
                }
            }
        } else {
            start_index = dist(gen);
            cudaMemset(d_counter, 0, sizeof(int));
        }
    }

    // 清理
    cudaFree(d_q);
    cudaFree(d_result_x);
    cudaFree(d_result_hash);
    cudaFree(d_counter);

    return 0;
}