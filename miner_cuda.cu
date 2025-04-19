#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cstring>
#include <iomanip>
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <random>
#include <chrono>

#include "sha256.cu"

#define X_LEN 8

// SHA-256 填充, input_len <= 55
__device__
void sha256_pad_64(const uint8_t* input, const size_t input_len, uint32_t* padded) {
    for (size_t i = 0; i < input_len; ++i) 
        padded[i / 4] |= ((uint32_t)input[i]) << ((3 - (i % 4)) * 8);

    padded[input_len / 4] |= 0x80 << ((3 - (input_len % 4)) * 8);
    padded[15] = (uint32_t)(input_len * 8);
}

// 检查前 k 位
__device__
bool check_leading_zeros(const uint32_t* hash, const int k) {
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
__device__
void generate_x(uint64_t index, char* x, const int x_len) {
    for (int i = x_len - 1; i >= 0; --i) {
        x[i] = '0' + (index & 0x3F);
        index >>= 6;
    }
}

// CUDA 内核
__global__
void find_nonce(const char* q, const size_t q_len, const uint64_t start_index,
                    const int k, char* result_x, int* found) {
    const uint64_t index = (start_index + blockIdx.x * blockDim.x + threadIdx.x) * 64;
    const size_t input_len = q_len + X_LEN;
    uint8_t input[55];

    memcpy(input, q, q_len);
    generate_x(index, (char*)input + q_len, X_LEN);

    uint32_t state[8], padded[64], padded_2[64]={};
    sha256_pad_64(input, input_len, padded_2);

    const int word_idx = (input_len-1)/4;
    const uint32_t byte_offset = 0x01U<<((3-((input_len-1)%4))*8);

    for (int i = 0; i < 64; i++) {
        memcpy(state, c_H256, 32);
        memcpy(padded, padded_2, 64);
        sha256_round_body(padded, state, c_K);

        if (check_leading_zeros(state, k) && !atomicCAS(found, 0, 1)) {
            memcpy(result_x, input + q_len, X_LEN);
            result_x[X_LEN-1] = '0'+i;
            return;
        }
        padded_2[word_idx] += byte_offset;
    }
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
    bool minimal_output = true;
    if (output_mode == "full")
        minimal_output = false;

    size_t q_len = q.length();
    const int x_len = 8;

    // GPU 内存
    char* d_q;
    cudaMalloc(&d_q, q_len);
    cudaMemcpy(d_q, q.c_str(), q_len, cudaMemcpyHostToDevice);

    char* d_result_x;
    cudaMalloc(&d_result_x, X_LEN * sizeof(char));
    cudaMemset(d_result_x, 0, X_LEN * sizeof(char));

    int* d_found;
    cudaMalloc(&d_found, sizeof(int));
    cudaMemset(d_found, 0, sizeof(int));

    // 线程配置
    const unsigned int grid_size = 96;
    const unsigned int block_size = 512;
    const unsigned int num_threads = block_size * grid_size;

    // 随机起点
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 48) - num_threads);
    uint64_t start_index = dist(gen);

    // 哈希率统计
    uint64_t total_hashes = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    if (!minimal_output) {
        std::cout << "Searching for x with " << k << " leading zeros for q = \"" << q << "\"" 
                  << " (random start: " << start_index << ")..." << std::endl;
    }

    // 主循环
    int h_found = 0;
    while (!h_found) {
        find_nonce<<<grid_size, block_size>>>(d_q, q_len, start_index, k, d_result_x, d_found);
        cudaDeviceSynchronize();

        total_hashes += num_threads*64;
        start_index += num_threads;
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    }

    // 获取结果
    std::vector<char> result_x(X_LEN);
    cudaMemcpy(result_x.data(), d_result_x, X_LEN * sizeof(char), cudaMemcpyDeviceToHost);

    std::string x(result_x.data(), x_len);
    if (minimal_output) {
        std::cout << x << std::endl;
    } else {

    std::cout << "Found x: " << x << std::endl;
    std::string q_plus_x = q + x;
    std::cout << "q + x: " << q_plus_x << std::endl;

    // OpenSSL 验证
    unsigned char hash[SHA256_DIGEST_LENGTH];
    EVP_Digest(q_plus_x.c_str(), q_plus_x.length(), hash, nullptr, EVP_sha256(), nullptr);

    std::cout << "OpenSSL SHA-256(q + x) = ";
    for (int j = 0; j < SHA256_DIGEST_LENGTH; ++j)
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)hash[j];
    std::cout << std::dec << std::endl;

    // 验证前导零
    int leading_zeros = 0;
    for (int j = 0; j < SHA256_DIGEST_LENGTH; ++j) {
        if (hash[j] == 0)
            leading_zeros += 8;
        else {
            uint8_t byte = hash[j];
            for (int b = 7; b >= 0; --b) {
                if (byte & (1 << b)) break;
                leading_zeros++;
            }
            break;
        }
    }

    std::cout << "Leading zeros: " << leading_zeros << " (expected: " << k << ")" << std::endl;
    if (leading_zeros >= k) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double seconds = duration.count() / 1e6;
        double hash_rate = total_hashes / seconds / 1e6;
        std::cout << "Hash rate: " << std::fixed << std::setprecision(2) 
                    << hash_rate << " MH/s" << std::endl;
        std::cout << "Total hashes: " << total_hashes << std::endl;
    } 
    else 
        std::cout << "Invalid result" << std::endl;
    }

    cudaFree(d_q);
    cudaFree(d_result_x);
    cudaFree(d_found);

    return 0;
}