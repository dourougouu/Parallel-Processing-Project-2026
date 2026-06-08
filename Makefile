CXX = g++
CXXFLAGS = -O3 -march=native -ffast-math -std=c++17
MAVX_FLAGS = -mavx2
OMP_FLAGS = -fopenmp


all: serial simd openmp mpi

serial:
	$(CXX) $(CXXFLAGS) src/scaler_serial.cpp -o scaler_serial

simd:
	$(CXX) $(CXXFLAGS) $(MAVX_FLAGS) src/scaler_simd.cpp -o scaler_simd

openmp:
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) src/scaler_OpenMP.cpp -o scaler_OpenMP

mpi:
	mpicxx $(CXXFLAGS) src/scaler_MPI.cpp -o scaler_mpi


clean:
	rm -f scaler_serial scaler_simd scaler_OpenMP scaler_mpi