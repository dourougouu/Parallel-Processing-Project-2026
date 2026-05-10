CXX = g++
CXXFLAGS = -O3 -march=native -ffast-math -std=c++17

all: serial

serial:
	$(CXX) $(CXXFLAGS) src/scaler_serial.cpp -o scaler_serial

clean:
	rm -f scaler_serial