#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <openssl/sha.h>
#include <cstring>

using namespace std;

const char base_char = '0';
const int x_length = 8;
const int batch_size = 64;
const bool output_flag = true;
const int num_threads = 8;

struct Result {
    string x;
    string hash;
};

void int_to_base64(uint64_t n, char* out) {
    for (int i = x_length - 1; i >= 0; i--) {
        out[i] = base_char + (n & 0x3F);
        n >>= 6;
    }
}

inline void increment_base_x(char* base_x) {
    for (int i = x_length - 2; i >= 0; i--) { // Start from second-to-last char
        if (base_x[i] < base_char + 63) {
            base_x[i]++;
            base_x[x_length - 1] = base_char; // Reset last char to '0'
            return;
        }
        base_x[i] = base_char; // Carry over
    }
}

void search(const string& q, const int num_zeros, atomic<bool>& found, Result& result, atomic<uint64_t>& counter) {
    mt19937_64 rng(random_device{}());
    const uint64_t start = (rng() & ~0x3F);
    const size_t q_len = q.length();
    
    // 关键优化1: 预先分配所有批次的输入缓冲区
    vector<char> inputs(batch_size * (q_len + x_length));
    vector<unsigned char> hashes(batch_size * SHA256_DIGEST_LENGTH);
    vector<SHA256_CTX> ctx(batch_size);
    
    // 关键优化2: 预先填充所有批次的前缀q
    for (int i = 0; i < batch_size; i++) {
        char* input = inputs.data() + i * (q_len + x_length);
        memcpy(input, q.c_str(), q_len);
    }

    char base_x[x_length];
    int_to_base64(start, base_x);

    const int full_bytes = num_zeros / 2;
    const int extra_nibble = num_zeros % 2;

    while (!found) {
        for (int i = 0; i < batch_size; i++) {
            char* input = inputs.data() + i * (q_len + x_length);
            // 关键优化3: 只更新x部分，不重复复制q
            memcpy(input + q_len, base_x, x_length);
            // 每个批次使用不同的最后一个字符
            input[q_len + x_length - 1] = base_char + i;

            SHA256_Init(&ctx[i]);
            SHA256_Update(&ctx[i], input, q_len + x_length);
            SHA256_Final(hashes.data() + i * SHA256_DIGEST_LENGTH, &ctx[i]);
        }

        for (int i = 0; i < batch_size; i++) {
            unsigned char* hash = hashes.data() + i * SHA256_DIGEST_LENGTH;
            bool valid = true;
            
            // 检查前导零
            for (int j = 0; j < full_bytes; j++) {
                if (hash[j] != 0) {
                    valid = false;
                    break;
                }
            }
            if (valid && extra_nibble && (hash[full_bytes] & 0xF0) != 0) valid = false;
            
            if (valid) {
                if (!found.exchange(true)) {
                    char* input = inputs.data() + i * (q_len + x_length);
                    result.x = string(input + q_len, x_length);
                    char hex[65];
                    for (int j = 0; j < SHA256_DIGEST_LENGTH; j++) {
                        sprintf(hex + j * 2, "%02x", hash[j]);
                    }
                    hex[64] = 0;
                    result.hash = hex;
                }
                return;
            }
        }
        counter += batch_size;
        increment_base_x(base_x);
    }
}
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <q> <num_zeros>" << endl;
        return 1;
    }
    string q = argv[1];
    int num_zeros = atoi(argv[2]);
    if (num_zeros <= 0 || num_zeros > 64) {
        cerr << "Number of zeros must be between 1 and 64" << endl;
        return 1;
    }
    if (output_flag) {
        cout << "Searching for x where sha256(" << q << " + x) starts with " << string(num_zeros, '0') << endl;
    }
    auto start_time = chrono::high_resolution_clock::now();

    vector<thread> threads;
    atomic<bool> found(false);
    atomic<uint64_t> counter(0);
    Result result;

    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        threads.push_back(thread(static_cast<void(*)(const string&, int, atomic<bool>&, Result&, atomic<uint64_t>&)>(search),
                                 ref(q), num_zeros, ref(found), ref(result), ref(counter)));
    }

    while (output_flag and !found) {
        cout << "Tried " << counter << " combinations...\r" << flush;
        this_thread::sleep_for(chrono::seconds(1));
    }

    for (auto& t : threads) t.join();

    auto end_time = chrono::high_resolution_clock::now();
    double duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() / 1000.0;

    if (output_flag) {
    cout << "\nFound solution!" << endl;
    cout << "x = " << result.x << endl;
    cout << "sha256(" << q << result.x << ") = " << result.hash << endl;
    cout << "Time taken: " << duration << " seconds" << endl;
    cout << "Hash rate: " << (counter.load() / duration) / 1000000 << " Mhashes/s" << endl;
    }
    else {
        cout << result.x << endl;
    }

    return 0;
}