#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <immintrin.h> // Header for SIMD intrinsics (AVX)
#include <algorithm>
#include <chrono>

// Function για το οριζόντιο άθροισμα των στοιχείων ενός __m256d vector (4 double τιμές) ,
// για να υπολογίσω το άθροισμα των στοιχείων του vector που περιέχει τα partial sums για κάθε feature
static inline double hsum256_pd(__m256d v)
{

    alignas(32) double temp[4];
    _mm256_store_pd(temp, v);

    return temp[0] + temp[1] + temp[2] + temp[3];
}

int main(int argc, char *argv[])
{

    if (argc < 6)
    {
        std::cerr << "Usage-Help:\n";
        std::cerr << "./scaler_simd input.bin output.bin N D mode\n";
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

        for (size_t i = 0; i < rows_to_read; i++)
        {
            size_t j = 0;

            // Χρήση SIMD για επεξεργασία 4 χαρακτηριστικών ταυτόχρονα (4 double τιμές σε κάθε __m256d vector)
            for (; j + 4 <= D; j += 4)
            {
                // Αθροισμα
                __m256d x = _mm256_loadu_pd(&block[i * D + j]); // Access the (i,j) to (i,j+3) elements of the block as a SIMD vector
                __m256d sum_vector = _mm256_loadu_pd(&sum[j]);
                sum_vector = _mm256_add_pd(sum_vector, x); // Προσθέτω τα στοιχεία του vector x στο αντίστοιχο vector sum_vector που περιέχει τα partial sums για κάθε feature
                _mm256_storeu_pd(&sum[j], sum_vector);     // Αποθηκεύω το ενημερωμένο vector sum_vector πίσω στον πίνακα sum

                // Άθροισμα τετραγώνων
                __m256d square = _mm256_mul_pd(x, x); // Πολλαπλασιάζω το vector x με τον εαυτό του για να πάρω τα τετράγωνα των στοιχείων
                __m256d sum_squares_vector = _mm256_loadu_pd(&sum_squares[j]);
                sum_squares_vector = _mm256_add_pd(sum_squares_vector, square); // Προσθέτω τα στοιχεία του vector square στο αντίστοιχο vector sum_squares_vector που περιέχει τα partial sums των τετραγώνων για κάθε feature
                _mm256_storeu_pd(&sum_squares[j], sum_squares_vector);          // Αποθηκεύω το ενημερωμένο vector sum_squares_vector πίσω στον πίνακα sum_squares

                // minimum
                __m256d min_vector = _mm256_loadu_pd(&min_value[j]);
                min_vector = _mm256_min_pd(min_vector, x);   // Παίρνω το στοιχείο-στοιχεία με τη μικρότερη τιμή μεταξύ του vector min_vector και του vector x
                _mm256_storeu_pd(&min_value[j], min_vector); // Αποθηκεύω το ενημερωμένο vector min_vector πίσω στον πίνακα min_value

                // maximum
                __m256d max_vector = _mm256_loadu_pd(&max_value[j]);
                max_vector = _mm256_max_pd(max_vector, x);   // Παίρνω το στοιχείο-στοιχεία με τη μεγαλύτερη τιμή μεταξύ του vector max_vector και του vector x
                _mm256_storeu_pd(&max_value[j], max_vector); // Αποθηκεύω το ενημερωμένο vector max_vector πίσω στον πίνακα max_value
            }

            for (; j < D; j++)
            {
                double x = block[i * D + j]; // Access the (i,j) element of the block

                sum[j] += x;             // Άθροισμα δεδομένων εισόδου
                sum_squares[j] += x * x; // Άθροισμα τετραγώνων δεδομένων εισόδου

                min_value[j] = std::min(min_value[j], x); // Ενημέρωση της ελάχιστης τιμής
                max_value[j] = std::max(max_value[j], x); // Ενημέρωση της μέγιστης τιμής
            }
        }

        total_rows_processed += rows_to_read; // Update the total number of rows processed
    }

    // Calculate mean and standard deviation for each feature (για Κανονικοποίηση)
    for (size_t j = 0; j < D; j++)
    {
        // Μέση τιμή
        mean[j] = sum[j] / static_cast<double>(N); // static_cast<double>(N) για να αποφύγω ακέραια διαίρεση

        // Διακύμανση , χρήση της εναλλακτικής σχέσης
        double variance = (sum_squares[j] / static_cast<double>(N)) - (mean[j] * mean[j]);
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

        for (size_t i = 0; i < rows_to_read; i++)
        {
            size_t j = 0;

            if (mode == "standard")
            {

                // Χρήση SIMD για επεξεργασία 4 χαρακτηριστικών ταυτόχρονα (4 double τιμές σε κάθε __m256d vector)
                for (; j + 4 <= D; j += 4)
                {
                    __m256d x = _mm256_loadu_pd(&block[i * D + j]);      // προσπελάση των (i,j) έως (i,j+3) στοιχείων του block ως SIMD vector
                    __m256d mean_vector = _mm256_loadu_pd(&mean[j]);     // Φορτώνω το vector mean_vector που περιέχει τις μέσες τιμές για τα 4 χαρακτηριστικά που επεξεργάζομαι αυτή τη στιγμή
                    __m256d stddev_vector = _mm256_loadu_pd(&stddev[j]); // Φορτώνω το vector stddev_vector που περιέχει τις τυπικές αποκλίσεις για τα 4 χαρακτηριστικά που επεξεργάζομαι αυτή τη στιγμή

                    x = _mm256_sub_pd(x, mean_vector);
                    x = _mm256_div_pd(x, stddev_vector);

                    _mm256_storeu_pd(&block[i * D + j], x); // Αποθηκεύω το κανονικοποιημένο vector x πίσω στο block
                }
            }
            else if (mode == "minmax")
            {
                // Χρήση SIMD για επεξεργασία 4 χαρακτηριστικών ταυτόχρονα (4 double τιμές σε κάθε __m256d vector)
                for (; j + 4 <= D; j += 4)
                {
                    __m256d x = _mm256_loadu_pd(&block[i * D + j]);               // προσπελάση των (i,j) έως (i,j+3) στοιχείων του block ως SIMD vector
                    __m256d min_vector = _mm256_loadu_pd(&min_value[j]);          // Φορτώνω το vector min_vector που περιέχει τις ελάχιστες τιμές για τα 4 χαρακτηριστικά που επεξεργάζομαι αυτή τη στιγμή
                    __m256d max_vector = _mm256_loadu_pd(&max_value[j]);          // Φορτώνω το vector max_vector που περιέχει τις μέγιστες τιμές για τα 4 χαρακτηριστικά που επεξεργάζομαι αυτή τη στιγμή
                    __m256d range_vector = _mm256_sub_pd(max_vector, min_vector); // Υπολογίζω το vector range_vector που περιέχει τα εύρη (max-min) για τα 4 χαρακτηριστικά που επεξεργάζομαι αυτή τη στιγμή

                    x = _mm256_sub_pd(x, min_vector);
                    x = _mm256_div_pd(x, range_vector);

                    _mm256_storeu_pd(&block[i * D + j], x); // Αποθηκεύω το κανονικοποιημένο vector x πίσω στο block
                }
            }

            for (; j < D; j++)
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
                    block[i * D + j] = x; // Ενημερώνω το στοιχείο (i,j) του block με την κανονικοποιημένη τιμή x
                }

                else if (mode == "minmax")
                {
                    // note:  αν max_j = min_j, να αποφύγω διαίρεση με μηδέν
                    double range = max_value[j] - min_value[j];
                    if (range > 1e-12) // Χρησιμοποιώ ένα μικρό threshold αντί για 0 για να αποφύγω προβλήματα με πολύ μικρά εύρη λόγω floating-point σφάλματος
                    {
                        x = (x - min_value[j]) / range;
                    }
                    else
                    {
                        x = 0.0; // Αν το εύρος είναι μηδέν, τότε όλα τα δεδομένα είναι ίδια, οπότε το κανονικοποιημένο αποτέλεσμα είναι 0
                    }

                    block[i * D + j] = x; // Ενημερώνω το στοιχείο (i,j) του block με την κανονικοποιημένη τιμή x
                }
            }
        }

        fout.write(reinterpret_cast<char *>(block.data()), elements_to_read * sizeof(double));

        total_rows_processed += rows_to_read;
    }

    fin.close();
    fout.close();

    auto end = std::chrono::high_resolution_clock::now(); // Σταματάω το χρονόμετρο

    double elapsed_seconds = std::chrono::duration<double>(end - start).count(); // Υπολογίζω τον χρόνο εκτέλεσης σε δευτερόλεπτα

    std::cout << "SIMD Scaling completed successfully!\n";
    std::cout << "Total execution time (1st + 2nd Pass): " << elapsed_seconds << " secs\n";

    return 0;
}

// How to Compile and Run:
//  g++ -O3 -march=native -mavx2 -ffast-math -std=c++17 src/scaler_simd.cpp -o scaler_simd or make simd
//  ./scaler_simd data/test.bin out.bin 1000 8 standard
//  python3 scripts/validate.py
//  ./scaler_simd data/test.bin out.bin 1000 8 minmax
//  python3 scripts/validate.py