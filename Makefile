SRC_DIR 	:= src/
TEST_DIR	:= tests/
TMP_DIR		:= tmp/
SCRIPT_DIR:= scripts/
BUILD_DIR	:= build/
BIN_DIR := bin/

C_FLAGS		:= -Wall
CC			:= gcc ${C_FLAGS}

.PHONY: libraries
libraries: bn combo helpers
	@:	
bn:
	$(CC) -c $(SRC_DIR)bn.c -o $(BUILD_DIR)bn.o
combo:
	$(CC) -c $(SRC_DIR)combo.c -o $(BUILD_DIR)combo.o
helpers:	# Used only for testing
	$(CC) -c $(SRC_DIR)helpers.c -o $(BUILD_DIR)helpers.o

tests: libraries
	${info Compiling combosMatch...}
	@$(CC) -c $(TEST_DIR)combosMatch.c -o $(BUILD_DIR)combosMatch.o
	@$(CC) $(BUILD_DIR)combosMatch.o $(BUILD_DIR)bn.o $(BUILD_DIR)combo.o $(BUILD_DIR)helpers.o -o $(BUILD_DIR)combosMatch

	${info Compiling combosCorrect...}
	@$(CC) -c $(TEST_DIR)combosCorrect.c -o $(BUILD_DIR)combosCorrect.o
	@$(CC) $(BUILD_DIR)combosCorrect.o $(BUILD_DIR)bn.o $(BUILD_DIR)combo.o $(BUILD_DIR)helpers.o -o $(BUILD_DIR)combosCorrect

run_tests: tests
	${info Testing if combos match...}
	@$(BUILD_DIR)combosMatch

	${info Testing if combos are correct...}
	@$(BUILD_DIR)combosCorrect $(TMP_DIR)sequentialCombinationsList.txt $(TMP_DIR)idCombinationsList.txt
	@python3.10 ${SCRIPT_DIR}makeAndCheckCombinationsList.py $(TMP_DIR) correctCombinationsList.txt sequentialCombinationsList.txt idCombinationsList.txt

cuda_library:
	# Compile library source files
	nvcc -c --x cu -rdc=true ${SRC_DIR}bn.c -o ${BUILD_DIR}bn.o
	nvcc -c --x cu -rdc=true ${SRC_DIR}combo.c -o ${BUILD_DIR}combo.o
	nvcc -c --x cu -rdc=true ${SRC_DIR}helpers.c -o ${BUILD_DIR}helpers.o

	# Create static library
	ar rcs ${BUILD_DIR}libtinycomb.a ${BUILD_DIR}combo.o ${BUILD_DIR}helpers.o ${BUILD_DIR}bn.o

cuda: cuda_library
	# Compile test files
	nvcc -c --x cu -rdc=true ${TEST_DIR}cudaBn.cu -o ${BUILD_DIR}cudaBn.o
	nvcc -c --x cu -rdc=true ${TEST_DIR}cudaCoherence.cu -o ${BUILD_DIR}cudaCoherence.o

	# Link everything into the executable
	nvcc ${BUILD_DIR}cudaBn.o -L${BUILD_DIR} -ltinycomb -o ${BUILD_DIR}cudaBnTest
	nvcc ${BUILD_DIR}cudaCoherence.o -L${BUILD_DIR} -ltinycomb -o main #${BUILD_DIR}cudaCoherenceTest

clean:
	rm build/*
	rm tmp/*