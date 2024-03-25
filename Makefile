CC = g++-13
CC_FLAGS = -I include/

default: driver

network.o:
	$(CC) $(CC_FLAGS) -c src/cpp-nn/network.cpp -o build/$@

Agent.o:
	$(CC) $(CC_FLAGS) -c src/dqn/Agent.cpp -o build/$@

driver: src/driver.cpp network.o Agent.o
	$(CC) $(CC_FLAGS) src/driver.cpp  build/network.o build/Agent.o -o bin/$@
