#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_of_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);             // ποιος ειμαι εγώ (rank) μέσα στις διεργασίες που τρέχουν
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes); // πόσες διεργασίες έχω συνολικά

    if (argc < 6)
    {
        if (rank == 0)
        {
            std::cerr << "Usage-Help:\n";
            std::cerr << "./scaler_mpi input.bin output.bin N D mode\n";
            std::cerr << "mode: standard or minmax\n";
        }

        MPI_Finalize();
        return 1;
    }

    std::string input_file = argv[1];  // Input binary file containing the dataset
    std::string output_file = argv[2]; // Output binary file to store the scaled dataset

    size_t N = std::stoul(argv[3]); // Number of samples in the dataset  N
    size_t D = std::stoul(argv[4]); // Number of features in the dataset D

    std::string mode = argv[5]; // Scaling mode: "standard" για Κανονικοποίηση
                                //              ή "minmax" για MinMaxScaler

    if (mode != "standard" && mode != "minmax")
    {
        if (rank == 0)
        {
            std::cerr << "Error! Invalid mode: " << mode << "!\n";
            std::cerr << "Use: standard or minmax , and try again :)\n";
        }
        MPI_Finalize();
        return 1;
    }

    size_t rows_per_process = N / num_of_processes;                                           // Πόσες γραμμές θα επεξεργαστεί κάθε διεργασία
    size_t starting_row = rank * rows_per_process;                                            // Από ποια γραμμή ξεκινάει η κάθε διεργασία
    size_t ending_row = (rank == num_of_processes - 1) ? N : starting_row + rows_per_process; // Μέχρι ποια γραμμή θα επεξεργαστεί η κάθε διεργασία (η τελευταία μπορεί να πάρει παραπάνω αν N δεν διαιρείται ακριβώς με num_of_processes)
    size_t local_rows = ending_row - starting_row;                                            // Πόσες γραμμές θα επεξεργαστεί τελικά η κάθε διεργασία
    const size_t block_rows = 10000;

    // Δηλώσεις-Αρχικοποιησεις για την Φάση Υπολογισμού των Στατιστικών (1st Pass)
    std::vector<double> local_sum(D, 0.0);            // Τοπικό Άθροισμα δεδομένων εισόδου
    std::vector<double> local_sum_squares(D, 0.0);    // Τοπικό Άθροισμα των τετραγώνων των δεδομένων εισόδου
    std::vector<double> local_min_value(D, DBL_MAX);  // Τοπική Έλαχιστη τιμή
    std::vector<double> local_max_value(D, -DBL_MAX); // Τοπική Μέγιστη τιμή

    std::vector<double> global_sum(D, 0.0);            // Global Άθροισμα δεδομένων εισόδου
    std::vector<double> global_sum_squares(D, 0.0);    // Global Άθροισμα των τετραγώνων των δεδομένων εισόδου
    std::vector<double> global_min_value(D, DBL_MAX);  // Global Έλαχιστη τιμή
    std::vector<double> global_max_value(D, -DBL_MAX); // Global Μέγιστη τιμή

    std::vector<double> mean(D);               // Μέση τιμή
    std::vector<double> stddev(D);             // Τυπική απόκλιση
    std::vector<double> block(block_rows * D); // Buffer to hold a block of data (Προσωρινός χώρος μνήμης-πίνακας για να διαβάζω το file block by block)
                                               // 3.4 Out of Core Επεξεργασία: block_size = B · D · sizeof(double)
    std::vector<double> range(D);              // Εύρος (για MinMaxScaler)

    //================================================================================
    //================================================================================
    // 3.1                1st PASS: LOCAL ΥΠΟΛΟΓΙΣΜΟΣ ΣΤΑΤΙΣΤΙΚΩΝ
    // εδώ κάθε διεργασία διαβάζει και επεξεργάζεται μόνο το δικό της κομμάτι των δεδομένων
    // και υπολογίζει τοπικά στατιστικά
    //================================================================================
    //================================================================================
    // Open the input binary file for reading
    double elapsed = MPI_Wtime(); // ← ξεκινάω χρονόμετρο

    {
        std::ifstream fin(input_file, std::ios::binary);
        if (!fin)
        {
            std::cerr << "Process " << rank << ": Error opening input file\n";
            MPI_Finalize();
            return 1;
        }

        // Μεταφορά στην σωστή θέση του file για την εκάστοτε διεργασά
        fin.seekg(starting_row * D * sizeof(double));
        size_t rows_processed = 0;
        while (rows_processed < local_rows)
        {
            size_t rows_to_read = std::min(block_rows, local_rows - rows_processed);
            size_t elements_to_read = rows_to_read * D;

            fin.read(reinterpret_cast<char *>(block.data()), elements_to_read * sizeof(double));
            if (!fin)
            {
                std::cerr << "Process " << rank << ": Error reading input file!\n";
                MPI_Finalize();
                return 1;
            }
            for (size_t i = 0; i < rows_to_read; i++)
            {
                for (size_t j = 0; j < D; j++)
                {
                    double x = block[i * D + j];
                    local_sum[j] += x;
                    local_sum_squares[j] += x * x;
                    if (x < local_min_value[j])
                        local_min_value[j] = x;
                    if (x > local_max_value[j])
                        local_max_value[j] = x;
                }
            }

            rows_processed += rows_to_read;
        }

        fin.close();
    }

    //===============================================================================================
    // Συγχώνευση των τοπικών στατιστικών από όλες τις διεργασίες για να πάρουμε τα global στατιστικά
    //===============================================================================================
    MPI_Allreduce(local_sum.data(), global_sum.data(), D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_sum_squares.data(), global_sum_squares.data(), D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_min_value.data(), global_min_value.data(), D, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_max_value.data(), global_max_value.data(), D, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Υπολογισμός mean, stddev, range από τα global στατιστικά
    for (size_t j = 0; j < D; j++)
    {
        // Μέση τιμή
        mean[j] = global_sum[j] / N;

        // Διακύμανση, χρήση της εναλλακτικής σχέσης
        double variance = (global_sum_squares[j] / N) - (mean[j] * mean[j]);
        if (variance < 0.0)
            variance = 0.0; // Λόγω floating-point σφάλματος

        // Τυπική απόκλιση
        stddev[j] = std::sqrt(variance);

        // Εύρος (για MinMaxScaler)
        range[j] = global_max_value[j] - global_min_value[j];
    }

    //================================================================================
    //================================================================================
    // 3.2                2nd PASS: ΜΕΤΑΣΧΗΜΑΤΙΣΜΟΣ ΔΕΔΟΜΕΝΩΝ
    // εδώ κάθε διεργασία διαβάζει και επεξεργάζεται μόνο το δικό της κομμάτι των δεδομένων
    // εφαρμόζει scaling και γράφει το σωστό offset του output file
    //================================================================================
    //================================================================================
    {
        std::ifstream fin(input_file, std::ios::binary);
        if (!fin)
        {
            std::cerr << "Process " << rank << ": Error opening input file!\n";
            MPI_Finalize();
            return 1;
        }

        if (rank == 0)
        {
            std::ofstream create(output_file, std::ios::binary | std::ios::trunc);
            create.seekp(N * D * sizeof(double) - 1);
            char zero = 0;
            create.write(&zero, 1);
            create.close();
        }
        MPI_Barrier(MPI_COMM_WORLD); // Περιμένουμε να δημιουργηθεί το αρχείο

        std::fstream fout(output_file, std::ios::binary | std::ios::in | std::ios::out);
        if (!fout)
        {
            std::cerr << "Process " << rank << ": Error opening output file!\n";
            MPI_Finalize();
            return 1;
        }

        // Μεταφορά στη σωστή θέση για ανάγνωση και εγγραφή
        fin.seekg(starting_row * D * sizeof(double));
        fout.seekp(starting_row * D * sizeof(double));

        size_t rows_processed = 0;
        while (rows_processed < local_rows)
        {
            size_t rows_to_read = std::min(block_rows, local_rows - rows_processed);
            size_t elements_to_read = rows_to_read * D;

            fin.read(reinterpret_cast<char *>(block.data()), elements_to_read * sizeof(double));
            if (!fin)
            {
                std::cerr << "Process " << rank << ": Error reading input file!\n";
                MPI_Finalize();
                return 1;
            }

            // Εφαρμογή scaling — ίδια λογική με serial
            for (size_t i = 0; i < rows_to_read; i++)
            {
                for (size_t j = 0; j < D; j++)
                {
                    double x = block[i * D + j];

                    if (mode == "standard")
                    {
                        if (stddev[j] > 1e-12)
                        {
                            x = (x - mean[j]) / stddev[j];
                        }
                        else
                        {
                            x = 0.0;
                        }
                    }
                    else if (mode == "minmax")
                    {
                        if (range[j] > 1e-12) // Χρησιμοποιώ ένα μικρό threshold αντί για 0 για να αποφύγω προβλήματα με πολύ μικρές τυπικές αποκλίσεις λόγω floating-point σφάλματος
                        {
                            x = (x - global_min_value[j]) / range[j];
                        }
                        else
                        {
                            x = 0.0;
                        }
                    }

                    block[i * D + j] = x; // Update the block with the scaled value
                }
            }

            // Κάθε διεργασία γράφει στο σωστό offset — διατηρείται η σωστή σειρά
            fout.write(reinterpret_cast<char *>(block.data()), elements_to_read * sizeof(double));

            rows_processed += rows_to_read;
        }

        fin.close();
        fout.close();
    }

    // Barrier: περιμένουμε όλες τις διεργασίες να τελειώσουν πριν εκτυπώσουμε
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed = MPI_Wtime() - elapsed; // ← σταματάω χρονόμετρο

    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Μόνο η διεργασία 0 εκτυπώνει το αποτέλεσμα
    if (rank == 0)
    {
        std::cout << "MPI Scaling completed successfully!\n";
        std::cout << "Processes: " << num_of_processes << "\n";
        std::cout << "Total execution time (1st + 2nd Pass): " << max_time << " secs\n";
    }

    MPI_Finalize();
    return 0;
}

// How to Compile and Run:
//  mpicxx -O3 -march=native -ffast-math -std=c++17 src/scaler_MPI.cpp -o scaler_mpi or make mpi
//  mpirun -np 1 ./scaler_mpi data/test.bin out.bin 1000 8 standard
//  mpirun -np 2 ./scaler_mpi data/test.bin out.bin 1000 8 standard
//  mpirun -np 4 ./scaler_mpi data/test.bin out.bin 1000 8 standard
//  python3 scripts/validate.py
//  mpirun -np 1 ./scaler_mpi data/test.bin out.bin 1000 8 minmax
//  mpirun -np 2 ./scaler_mpi data/test.bin out.bin 1000 8 minmax
//  mpirun -np 4 ./scaler_mpi data/test.bin out.bin 1000 8 minmax
//  python3 scripts/validate.py