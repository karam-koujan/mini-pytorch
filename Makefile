Name = mini_pytorch
CC = cc
CFLAGS = -g -Wall -Wextra -Werror 
OBJ = main.o ./lib/print.o ./tensor_creation/helpers.o ./tensor_creation/data_creation.o ./tensor_creation/tensor_creation.o \
./tensor_operation/broadcasting.o ./tensor_creation/tensor_free.o ./tensor_operation/view.o ./tensor_operation/reshape.o \
./tensor_operation/transpose.o ./tensor_operation/permute.o ./tensor_operation/math_operations.o ./autograd/autograd.o \
./tensor_operation/tensor_matmul.o ./neural_network/linear.o ./neural_network/helper.o ./neural_network/nn_operation.o

HEADERS = ./headers/*

all : $(Name)

$(Name) : $(OBJ)
	$(CC) $(CFLAGS) -lm $(OBJ) -o $(Name)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean :
	rm -f $(OBJ)

fclean : clean 
	rm -f $(Name)

re: fclean all