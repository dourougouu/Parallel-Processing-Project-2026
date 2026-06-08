#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <omp.h>
#include <chrono>

int main(int argc, char *argv[])
{

    if (argc < 6)
    {
        std::cerr << "Usage-Help:\n";
        std::cerr << "./scaler_serial input.bin output.bin N D mode\n";
        std::cerr << "mode: standard or minmax\n";
        return 1;
    }

    std::string input_file = argv[1];  // Input binary file containing the dataset
    std::string output_file = argv[2]; // Output binary file to store the scaled dataset

    size_t N = std::stoul(argv[3]); // Number of samples in the dataset  N
    size_t D = std::stoul(argv[4]); // Number of features in the dataset D

    std::string mode = argv[5]; // Scaling mode: "standard" για Κανονικοποίηση
                                //              ή "minmax" για MinMaxScaler

    // Προαιρετικος αριθμος threads από γραμμή εντολών, και αν δεν δοθεί, τότε χρησιμοποιώ όλους τους διαθέσιμους πυρήνες του συστήματος
    if (argc >= 7)
    {
        omp_set_num_threads(std::stoi(argv[6]));
    }

    std::cout << "Using " << omp_get_max_threads() << " threads for OpenMP parallelization\n";

    const size_t block_rows = 10000;

    // Open the input binary file for reading
    std::ifstream fin(input_file, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Error opening input file: " << input_file << "!" << std::endl;
        return 1;
    }

    // Δηλώσεις-Αρχικοποιησεις για την Φάση Υπολογισμού των Στατιστικών (1st Pass)
    std::vector<double> sum(D, 0.0);            // Άθροισμα δεδομένων εισόδου
    std::vector<double> sum_squares(D, 0.0);    // Άθροισμα των τετραγώνων των δεδομένων εισόδου
    std::vector<double> min_value(D, DBL_MAX);  // Έλαχιστη τιμή
    std::vector<double> max_value(D, -DBL_MAX); // Μέγιστη τιμή
    std::vector<double> mean(D);                // Μέση τιμή
    std::vector<double> stddev(D);              // Τυπική απόκλιση
    std::vector<double> block(block_rows * D);  // Buffer to hold a block of data (Προσωρινός χώρος μνήμης-πίνακας για να διαβάζω το file block by block)
                                                // 3.4 Out of Core Επεξεργασία: block_size = B · D · sizeof(double)

    // Check για invalid mode από τον χρήστη
    if (mode != "standard" && mode != "minmax")
    {
        std::cerr << "Error! Invalid mode: " << mode << "!\n";
        std::cerr << "Use: standard or minmax , and try again :)\n";
        return 1;
    }

    //================================================================================
    //================================================================================
    // 3.1                1st PASS: ΥΠΟΛΟΓΙΣΜΟΣ ΣΤΑΤΙΣΤΙΚΩΝ
    //================================================================================
    //================================================================================

    auto start = std::chrono::high_resolution_clock::now(); // Ξεκινάω το χρονόμετρο

    size_t total_rows_processed = 0;

    while (total_rows_processed < N)
    {

        size_t rows_to_read = std::min(block_rows, N - total_rows_processed); // Determine how many rows to read in this block
        size_t elements_to_read = rows_to_read * D;

        fin.read(reinterpret_cast<char *>(block.data()), elements_to_read * sizeof(double)); // Read a block of data
        if (!fin)
        {
            std::cerr << "Error! Could not read the input file!\n";
            return 1;
        }

#pragma omp parallel
        {
            // Local στατιστικά ανα thread για αποφυγη race conditions
            std::vector<double> local_sum(D, 0.0);            // Άθροισμα δεδομένων εισόδου locally ανα thread
            std::vector<double> local_sum_squares(D, 0.0);    // Άθροισμα των τετραγώνων των δεδομένων εισόδου locally ανα thread
            std::vector<double> local_min_value(D, DBL_MAX);  // Έλαχιστη τιμή locally ανα thread
            std::vector<double> local_max_value(D, -DBL_MAX); // Μέγιστη τιμή locally ανα thread

#pragma omp for nowait
            for (size_t i = 0; i < rows_to_read; i++)
            {
                for (size_t j = 0; j < D; j++)
                {
                    double x = block[i * D + j];   // Access the (i,j) element of the block
                    local_sum[j] += x;             // Άθροισμα δεδομένων εισόδου (για τον υπολογισμό της μέσης τιμής)
                    local_sum_squares[j] += x * x; // Άθροισμα τετραγώνων δεδομένων εισόδου (για τον υπολογισμό της τυπικής απόκλισης/διακύμανσης)

                    if (x < local_min_value[j]) // Ενημέρωση της ελάχιστης τιμής
                        local_min_value[j] = x;

                    if (x > local_max_value[j]) // Ενημέρωση της μέγιστης τιμής
                        local_max_value[j] = x;
                }
            }

// Συγχώνευση των local στατιστικών των threads στα global στατιστικά με ασφάλεια (critical section)
#pragma omp critical
            {
                for (size_t j = 0; j < D; j++)
                {
                    sum[j] += local_sum[j];
                    sum_squares[j] += local_sum_squares[j];

                    if (local_min_value[j] < min_value[j])
                        min_value[j] = local_min_value[j];

                    if (local_max_value[j] > max_value[j])
                        max_value[j] = local_max_value[j];
                }
            }
        }

        total_rows_processed += rows_to_read; // Update the total number of rows processed
    }

    // Calculate mean and standard deviation for each feature (για Κανονικοποίηση)
    for (size_t j = 0; j < D; j++)
    {
        // Μέση τιμή
        mean[j] = sum[j] / N;

        // Διακύμανση , χρήση της εναλλακτικής σχέσης
        double variance = (sum_squares[j] / N) - (mean[j] * mean[j]);
        if (variance < 0.0)
            variance = 0.0; // Λόγω floating-point σφάλματος, η διακύμανση μπορεί να είναι αρνητική, οπότε την θέτω σε 0

        // Τυπική απόκλιση
        stddev[j] = std::sqrt(variance);
    }

    fin.close(); // Close the input file after the first pass

    std::vector<double> range(D); // Εύρος (για MinMaxScaler)
    for (size_t j = 0; j < D; j++)
    {
        range[j] = max_value[j] - min_value[j]; // Υπολογισμός εύρους για κάθε χαρακτηριστικό , πριν το 2nd Pass, για να αποφύγω επαναλαμβανόμενους υπολογισμούς μέσα στο loop του 2nd Pass
    }

    //================================================================================
    //================================================================================
    // 3.2                2nd PASS: ΜΕΤΑΣΧΗΜΑΤΙΣΜΟΣ ΔΕΔΟΜΕΝΩΝ
    //================================================================================
    //================================================================================
    fin.open(input_file, std::ios::binary);

    std::ofstream fout(output_file, std::ios::binary);
    if (!fout)
    {
        std::cerr << "Error opening output file: " << output_file << "!" << std::endl;
        return 1;
    }

    total_rows_processed = 0;

    while (total_rows_processed < N)
    {

        size_t rows_to_read = std::min(block_rows, N - total_rows_processed);
        size_t elements_to_read = rows_to_read * D;

        fin.read(reinterpret_cast<char *>(block.data()), elements_to_read * sizeof(double));
        if (!fin)
        {
            std::cerr << "Error! Could not read the input file!\n";
            return 1;
        }

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows_to_read; i++)
        {
            for (size_t j = 0; j < D; j++)
            {
                double x = block[i * D + j];

                if (mode == "standard")
                {
                    if (stddev[j] > 1e-12) // Χρησιμοποιώ ένα μικρό threshold αντί για 0 για να αποφύγω προβλήματα με πολύ μικρές τυπικές αποκλίσεις λόγω floating-point σφάλματος
                    {
                        x = (x - mean[j]) / stddev[j];
                    }
                    else
                    {
                        x = 0.0; // Αν η τυπική απόκλιση είναι μηδέν, τότε όλα τα δεδομένα είναι ίδια, οπότε το κανονικοποιημένο αποτέλεσμα είναι 0
                    }
                }

                else if (mode == "minmax")
                {
                    // note:  αν max_j = min_j, να αποφύγω διαίρεση με μηδέν
                    if (range[j] > 1e-12) // Χρησιμοποιώ ένα μικρό threshold αντί για 0 για να αποφύγω προβλήματα με πολύ μικρές τυπικές αποκλίσεις λόγω floating-point σφάλματος
                    {
                        x = (x - min_value[j]) / range[j];
                    }
                    else
                    {
                        x = 0.0; // Αν το εύρος είναι μηδέν, τότε όλα τα δεδομένα είναι ίδια, οπότε το κανονικοποιημένο αποτέλεσμα είναι 0
                    }
                }
                block[i * D + j] = x; // Update the block with the scaled value
            }
        }

        fout.write(reinterpret_cast<char *>(block.data()), elements_to_read * sizeof(double));

        total_rows_processed += rows_to_read;
    }

    fin.close();
    fout.close();

    auto end = std::chrono::high_resolution_clock::now(); // Σταματάω το χρονόμετρο
    double elapsed_seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "OpenMp parallelization completed successfully!\n";
    std::cout << "Total execution time (1st + 2nd Pass): " << elapsed_seconds << " secs\n";

    return 0;
}

// How to Compile and Run:
//  g++ -O3 -march=native -ffast-math -fopenmp -o src/scaler_OpenMP.cpp or make openmp
//  ./scaler_OpenMP data/test.bin out.bin 1000 8 standard
//  python3 scripts/validate.py
//  ./scaler_OpenMP data/test.bin out.bin 1000 8 minmax
//  python3 scripts/validate.py