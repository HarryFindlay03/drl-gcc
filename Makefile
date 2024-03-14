CC = g++-13
CC_FLAGS = -I include/

default: driver

utils.o:
	$(CC) $(CC_FLAGS) -c src/utils.cpp -o build/utils.o

iterative.o:
	$(CC) $(CC_FLAGS) -c src/iterative.cpp -o build/iterative.o

driver: src/driver.cpp utils.o iterative.o
	$(CC) $(CC_FLAGS) src/driver.cpp build/utils.o build/iterative.o -o bin/driver