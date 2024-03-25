CC = g++-13
CC_FLAGS = -I include/

default: driver

network.o:
	$(CC) $(CC_FLAGS) -c src/cpp-nn/network.cpp -o build/$@

driver: src/driver.cpp network.o
	$(CC) $(CC_FLAGS) src/driver.cpp  build/network.o -o bin/driver
