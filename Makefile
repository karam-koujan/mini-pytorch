Name = mini_pytorch
CC = cc
CFLAGS = -g -Wall -Wextra -Werror
OBJ = tensor.o ./lib/print.o ./tensor_creation/helpers.o ./tensor_creation/data_creation.o ./tensor_creation/tensor_creation.o
HEADERS = ./headers/*

all : $(Name)

$(Name) : $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(Name)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean :
	rm -f $(OBJ)

fclean : clean 
	rm -f $(Name)

re: fclean all