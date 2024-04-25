CC = g++
CC_FLAGS = -I include/

network.o:
	$(CC) $(CC_FLAGS) -c src/mlp-cpp/network.cpp -o build/$@

funcs.o:
	$(CC) $(CC_FLAGS) -c src/mlp-cpp/funcs.cpp -o build/$@

Agent.o:
	$(CC) $(CC_FLAGS) -c src/dqn/Agent.cpp -o build/$@

utils.o:
	$(CC) $(CC_FLAGS) -c src/utils/utils.cpp -o build/$@

statetool:
	./plug.sh

driver_agent: network.o funcs.o Agent.o utils.o
	$(CC) $(CC_FLAGS) src/examples/driver_agent.cpp build/network.o build/funcs.o build/Agent.o build/utils.o -o bin/$@

example_mlp: network.o funcs.o
	$(CC) $(CC_FLAGS) src/examples/example_mlp.cpp build/network.o build/funcs.o -o bin/$@

example_agent: network.o funcs.o Agent.o utils.o
	$(CC) $(CC_FLAGS) src/examples/driver_agent.cpp build/network.o build/funcs.o build/Agent.o build/utils.o -o bin/$@
