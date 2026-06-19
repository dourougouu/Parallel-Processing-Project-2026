//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Συνδεση στον λογαριασμό για το μηχανημα krylov100
// προσβάσιμο μέσω scgroup3.ceid.upatras.gr
// General info για το μηχανημα: A server with 24 Intel Xeon cores, 128 GB RAM, and a single NVIDIA Tesla V100 GPU (5120 CUDA cores, 32 GB of graphics memory)"
// Ανοιγουμε το ubuntu wsl terminal και τρέχω:
// ssh hpcgrpXX@scgroup3.ceid.upatras.gr XX=αριθμος της ομαδας μου
// και επειτα τρεχω τις εντολες:
// nvidia-smi
// nvcc --version για να δω την GPU και τις διεργασίες που τρέχουν σε αυτήν

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <cuda_runtime.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
//  Kernel για τον υπολογισμό των στατιστικών (1st Pass)
//  καθε thread επεξεργαζεται μια γραμμη του block των δεδομένων(στη GPU) και υπολογιζει τοπικα στατιστικα για καθε feature(D)
__global__ void compute_statistics_krnl(const double *block, double *sum, double *sum_squares, double *min_value, double *max_value, size_t rows_in_block, size_t D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // index του feature που επεξεργάζεται το τρέχον thread
    if (j >= D)
        return;

    double loc_sum = 0.0;
    double loc_sum_squares = 0.0;
    double loc_min = DBL_MAX;
    double loc_max = -DBL_MAX;

    // Επεξεργασια γραμμων για το εκαστοτε feature j
    for (size_t i = 0; i < rows_in_block; i++)
    {
        double x = block[i * D + j]; // προσβαση στο στοιχειο (i,j) του block
        loc_sum += x;
        loc_sum_squares += x * x;
        if (x < loc_min)
            loc_min = x;

        if (x > loc_max)
            loc_max = x;
    }

    // Ατομικές λειτουργιες για safe συγχωνευση των local αποτελεσματων στα global arrays
    atomicAdd(&sum[j], loc_sum);
    atomicAdd(&sum_squares[j], loc_sum_squares);

    // Για min και max, δεν υπάρχει built-in atomic λειτουργία, οπότε χρησιμοποιώ έναν απλό μηχανισμό με mutex
    // οπότε χρησιμοποιούμε compare-and-swap loop
    // atomicMin για double
    unsigned long long old_value, new_value;

    unsigned long long int *min_addr = (unsigned long long int *)&min_value[j];
    do
    {
        old_value = *min_addr;
        double old_min = __longlong_as_double(old_value);
        if (loc_min >= old_min)
            break; // Αν το τοπικό min δεν είναι μικρότερο, δεν χρειάζεται ενημέρωση
        new_value = __double_as_longlong(loc_min);
    } while (atomicCAS(min_addr, old_value, new_value) != old_value);

    unsigned long long int *max_addr = (unsigned long long int *)&max_value[j];
    do
    {
        old_value = *max_addr;
        double old_max = __longlong_as_double(old_value);
        if (loc_max <= old_max)
            break; // Αν το τοπικό max δεν είναι μεγαλύτερο, δεν χρειάζεται ενημέρωση
        new_value = __double_as_longlong(loc_max);
    } while (atomicCAS(max_addr, old_value, new_value) != old_value);
}
//
// Kernel για StandardScaler (2nd Pass)
// καθε thread επεξεργαζεται ενα στοιχειο του block των δεδομένων block[i][j]
__global__ void standard_scaler_krnl(double *block, const double *mean, const double *stddev, size_t rows_in_block, size_t D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // index του στοιχείου που επεξεργάζεται το τρέχον thread
    int i = blockIdx.y * blockDim.y + threadIdx.y; // γραμμή του index

    if (i >= rows_in_block || j >= D)
        return;

    double x = block[i * D + j];
    if (stddev[j] > 1e-12) // Χρησιμοποιώ ένα μικρό threshold αντί για 0 για να αποφύγω προβλήματα με πολύ μικρά εύρη λόγω floating-point σφάλματος
    {
        x = (x - mean[j]) / stddev[j];
    }
    else
    {
        x = 0.0; // Αν η τυπική απόκλιση είναι μηδέν, τότε όλα τα δεδομένα είναι ίδια, οπότε το κανονικοποιημένο αποτέλεσμα είναι 0
    }
    block[i * D + j] = x;
}
//
// Kernel για MinMaxScaler (2nd Pass)
__global__ void minmax_scaler_krnl(double *block, const double *min_value, const double *range, size_t rows_in_block, size_t D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // index του στοιχείου που επεξεργάζεται το τρέχον thread
    int i = blockIdx.y * blockDim.y + threadIdx.y; // γραμμή του index

    if (i >= rows_in_block || j >= D)
        return;

    double x = block[i * D + j];
    if (range[j] > 1e-12) // Χρησιμοποιώ ένα μικρό threshold αντί για 0 για να αποφύγω προβλήματα με πολύ μικρά εύρη λόγω floating-point σφάλματος
    {
        x = (x - min_value[j]) / range[j];
    }
    else
    {
        x = 0.0;
    }
    block[i * D + j] = x;
}
//
// Για να κάνουμε handle τους CUDA errors
#define CUDA_CHECKING(call)                                                                                                          \
    do                                                                                                                               \
    {                                                                                                                                \
        cudaError_t err = (call);                                                                                                    \
        if (err != cudaSuccess)                                                                                                      \
        {                                                                                                                            \
            std::cerr << "We have a CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1);                                                                                                                 \
        }                                                                                                                            \
    } while (0)

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN
int main(int argc, char *argv[])
{

    if (argc < 6)
    {
        std::cerr << "Usage-Help:\n";
        std::cerr << "./scaler_cuda input.bin output.bin N D mode\n";
        std::cerr << "mode: standard or minmax\n";
        return 1;
    }

    std::string input_file = argv[1];  // Input binary file containing the dataset
    std::string output_file = argv[2]; // Output binary file to store the scaled dataset

    size_t N = std::stoul(argv[3]); // Number of samples in the dataset  N
    size_t D = std::stoul(argv[4]); // Number of features in the dataset D

    std::string mode = argv[5]; // Scaling mode: "standard" για Κανονικοποίηση
                                //              ή "minmax" για MinMaxScaler

    const size_t block_rows = 10000;

    // Check για invalid mode από τον χρήστη
    if (mode != "standard" && mode != "minmax")
    {
        std::cerr << "Error! Invalid mode: " << mode << "!\n";
        std::cerr << "Use: standard or minmax , and try again :)\n";
        return 1;
    }

    // Δηλώσεις-Αρχικοποιησεις για την Φάση Υπολογισμού των Στατιστικών (1st Pass)
    // CPU (host) arrays
    std::vector<double> h_sum(D, 0.0);            // Άθροισμα δεδομένων εισόδου
    std::vector<double> h_sum_squares(D, 0.0);    // Άθροισμα των τετραγώνων των δεδομένων εισόδου
    std::vector<double> h_min_value(D, DBL_MAX);  // Έλαχιστη τιμή
    std::vector<double> h_max_value(D, -DBL_MAX); // Μέγιστη τιμή
    std::vector<double> h_mean(D);                // Μέση τιμή
    std::vector<double> h_stddev(D);              // Τυπική απόκλιση
    std::vector<double> h_block(block_rows * D);  // Buffer to hold a block of data (Προσωρινός χώρος μνήμης-πίνακας για να διαβάζω το file block by block)
                                                  // 3.4 Out of Core Επεξεργασία: block_size = B · D · sizeof(double)
    std::vector<double> h_range(D);               // Εύρος (για MinMaxScaler)

    // GPU device arrays:
    double *d_block = nullptr;       // block δεδομένων στη GPU
    double *d_sum = nullptr;         // άθροισμα ανά feature στη GPU
    double *d_sum_squares = nullptr; // άθροισμα τετραγώνων ανά feature στη GPU
    double *d_min_value = nullptr;   // ελάχιστη τιμή ανά feature στη GPU
    double *d_max_value = nullptr;   // μέγιστη τιμή ανά feature στη GPU
    double *d_mean = nullptr;        // μέση τιμή ανά feature στη GPU
    double *d_stddev = nullptr;      // τυπική απόκλιση ανά feature στη GPU
    double *d_range = nullptr;       // εύρος ανά feature στη GPU

    // Δέσμευση μνήμης GPU
    CUDA_CHECKING(cudaMalloc(&d_block, block_rows * D * sizeof(double)));
    CUDA_CHECKING(cudaMalloc(&d_sum, D * sizeof(double)));
    CUDA_CHECKING(cudaMalloc(&d_sum_squares, D * sizeof(double)));
    CUDA_CHECKING(cudaMalloc(&d_min_value, D * sizeof(double)));
    CUDA_CHECKING(cudaMalloc(&d_max_value, D * sizeof(double)));
    CUDA_CHECKING(cudaMalloc(&d_mean, D * sizeof(double)));
    CUDA_CHECKING(cudaMalloc(&d_stddev, D * sizeof(double)));
    CUDA_CHECKING(cudaMalloc(&d_range, D * sizeof(double)));

    // Αρχικοποίηση GPU arrays για στατιστικά
    CUDA_CHECKING(cudaMemcpy(d_sum, h_sum.data(), D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECKING(cudaMemcpy(d_sum_squares, h_sum_squares.data(), D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECKING(cudaMemcpy(d_min_value, h_min_value.data(), D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECKING(cudaMemcpy(d_max_value, h_max_value.data(), D * sizeof(double), cudaMemcpyHostToDevice));

    // CUDA grid/block dimensions για stats kernel (1 thread ανά feature)
    int threads_per_block_1d = 256;
    int blocks_1d = (D + threads_per_block_1d - 1) / threads_per_block_1d;

    // CUDA grid/block dimensions για scaling kernel (2D: threads για rows × features)
    dim3 threads_2d(32, 32); // 32×32 = 1024 threads ανά block
    dim3 blocks_2d(
        (D + threads_2d.x - 1) / threads_2d.x,
        (block_rows + threads_2d.y - 1) / threads_2d.y);

    auto start_time = std::chrono::high_resolution_clock::now(); // Ξεκινάω το χρονόμετρο

    //================================================================================
    //================================================================================
    // 3.1                1st PASS: ΥΠΟΛΟΓΙΣΜΟΣ ΣΤΑΤΙΣΤΙΚΩΝ
    //================================================================================
    //================================================================================
    // Διαβάζουμε block-by-block από το αρχείο (CPU),
    // μεταφέρουμε στη GPU και υπολογίζουμε στατιστικά εκεί
    {
        std::ifstream fin(input_file, std::ios::binary);
        if (!fin)
        {
            std::cerr << "Error! Could not read the input file!\n";
            return 1;
        }

        size_t total_rows_processed = 0;
        while (total_rows_processed < N)
        {
            size_t rows_to_read = std::min(block_rows, N - total_rows_processed);
            size_t elements_to_read = rows_to_read * D;

            // Ανάγνωση block από αρχείο (CPU) — I/O δεν μπορεί να γίνει στη GPU
            fin.read(reinterpret_cast<char *>(h_block.data()), elements_to_read * sizeof(double));
            if (!fin)
            {
                std::cerr << "Error reading input file!\n";
                return 1;
            }

            // Μεταφορα vlock CPU στη GPU
            CUDA_CHECKING(cudaMemcpy(d_block, h_block.data(), elements_to_read * sizeof(double), cudaMemcpyHostToDevice));

            // Εκτέλεση kernel για υπολογισμό στατιστικών
            // Κάθε thread αναλαμβάνει ένα feature
            compute_statistics_krnl<<<blocks_1d, threads_per_block_1d>>>(d_block, d_sum, d_sum_squares, d_min_value, d_max_value, rows_to_read, D);

            CUDA_CHECKING(cudaGetLastError());
            CUDA_CHECKING(cudaDeviceSynchronize());

            total_rows_processed += rows_to_read;
        }

        fin.close();
    }

    // Μεταφορά στατιστικών GPU → CPU
    CUDA_CHECKING(cudaMemcpy(h_sum.data(), d_sum, D * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECKING(cudaMemcpy(h_sum_squares.data(), d_sum_squares, D * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECKING(cudaMemcpy(h_min_value.data(), d_min_value, D * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECKING(cudaMemcpy(h_max_value.data(), d_max_value, D * sizeof(double), cudaMemcpyDeviceToHost));

    // Υπολογισμός mean and standard deviation for each feature (για Κανονικοποίηση)
    for (size_t j = 0; j < D; j++)
    {
        // Μέση τιμή
        h_mean[j] = h_sum[j] / N;

        // Διακύμανση , χρήση της εναλλακτικής σχέσης
        double variance = (h_sum_squares[j] / N) - (h_mean[j] * h_mean[j]);
        if (variance < 0.0)
            variance = 0.0; // Λόγω floating-point σφάλματος, η διακύμανση μπορεί να είναι αρνητική, οπότε την θέτω σε 0

        // Τυπική απόκλιση
        h_stddev[j] = std::sqrt(variance);

        h_range[j] = h_max_value[j] - h_min_value[j]; // Υπολογισμός εύρους για κάθε χαρακτηριστικό , πριν το 2nd Pass, για να αποφύγω επαναλαμβανόμενους υπολογισμούς μέσα στο loop του 2nd Pass
    }

    // Μεταφορά mean/stddev/range/min CPU → GPU (χρειάζονται στο 2nd pass kernel)
    CUDA_CHECKING(cudaMemcpy(d_mean, h_mean.data(), D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECKING(cudaMemcpy(d_stddev, h_stddev.data(), D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECKING(cudaMemcpy(d_range, h_range.data(), D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECKING(cudaMemcpy(d_min_value, h_min_value.data(), D * sizeof(double), cudaMemcpyHostToDevice));

    //================================================================================
    //================================================================================
    // 3.2                2nd PASS: ΜΕΤΑΣΧΗΜΑΤΙΣΜΟΣ ΔΕΔΟΜΕΝΩΝ
    //================================================================================
    //================================================================================
    // Διαβάζουμε block-by-block (CPU), μεταφέρουμε στη GPU,
    // εφαρμόζουμε scaling εκεί και γράφουμε το αποτέλεσμα στο αρχείο (CPU)
    {
        std::ifstream fin(input_file, std::ios::binary);
        if (!fin)
        {
            std::cerr << "Error opening input file: " << output_file << "!" << std::endl;
            return 1;
        }
        std::ofstream fout(output_file, std::ios::binary);
        if (!fout)
        {
            std::cerr << "Error opening output file: " << output_file << "!\n";
            return 1;
        }

        size_t total_rows_processed = 0;
        while (total_rows_processed < N)
        {

            size_t rows_to_read = std::min(block_rows, N - total_rows_processed);
            size_t elements_to_read = rows_to_read * D;

            // Ανάγνωση block από αρχείο (CPU)
            fin.read(reinterpret_cast<char *>(h_block.data()), elements_to_read * sizeof(double));
            if (!fin)
            {
                std::cerr << "Error! Could not read the input file!\n";
                return 1;
            }

            // Μεταφορά block CPU → GPU
            CUDA_CHECKING(cudaMemcpy(d_block, h_block.data(), elements_to_read * sizeof(double), cudaMemcpyHostToDevice));

            // Εκτέλεση scaling kernel
            // 2D grid: blockIdx.x → features, blockIdx.y → rows
            dim3 actual_blocks_2d(
                (D + threads_2d.x - 1) / threads_2d.x,
                (rows_to_read + threads_2d.y - 1) / threads_2d.y);

            if (mode == "standard")
            {
                standard_scaler_krnl<<<actual_blocks_2d, threads_2d>>>(d_block, d_mean, d_stddev, rows_to_read, D);
            }
            else
            {
                minmax_scaler_krnl<<<actual_blocks_2d, threads_2d>>>(d_block, d_min_value, d_range, rows_to_read, D);
            }

            CUDA_CHECKING(cudaGetLastError());
            CUDA_CHECKING(cudaDeviceSynchronize());

            // Μεταφορά scaled block GPU → CPU
            CUDA_CHECKING(cudaMemcpy(h_block.data(), d_block, elements_to_read * sizeof(double), cudaMemcpyDeviceToHost));

            // Εγγραφή στο αρχείο εξόδου (CPU — I/O δεν γίνεται από GPU)
            fout.write(reinterpret_cast<char *>(h_block.data()), elements_to_read * sizeof(double));

            total_rows_processed += rows_to_read;
        }

        fin.close();
        fout.close();
    }

    // Αποδέσμευση μνήμης GPU
    cudaFree(d_block);
    cudaFree(d_sum);
    cudaFree(d_sum_squares);
    cudaFree(d_min_value);
    cudaFree(d_max_value);
    cudaFree(d_mean);
    cudaFree(d_stddev);
    cudaFree(d_range);

    auto end = std::chrono::high_resolution_clock::now(); // Σταματάω το χρονόμετρο

    double elapsed_seconds = std::chrono::duration<double>(end - start_time).count(); // Υπολογίζω τον χρόνο εκτέλεσης σε δευτερόλεπτα

    std::cout << "CUDA Scaling completed successfully!\n";
    std::cout << "Total execution time (1st + 2nd Pass): " << elapsed_seconds << " secs\n";

    return 0;
}

// How to Compile and Run:
//  nvcc -O3 -std=c++17 src/scaler_cuda.cu -o scaler_cuda or make cuda
//  ./scaler_scuda data/test.bin out.bin 1000 8 standard
//  python3 scripts/validate.py
//  ./scaler_cuda data/test.bin out.bin 1000 8 minmax
//  python3 scripts/validate.py