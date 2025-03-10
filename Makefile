SRC_DIR 	:= src/
TEST_DIR	:= tests/
TMP_DIR		:= tmp/
SCRIPT_DIR:= scripts/
BUILD_DIR	:= build/

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
