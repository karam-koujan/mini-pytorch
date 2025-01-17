
OBJ = tensor.o tensor_operations.o autograd.o tensor_clean.o
NN_OBJ = neural_network.o

NAME = mini-pytorch.a
NN_EXEC = neural_network
all : $(NAME)

$(NAME) : $(OBJ)
	ar rc $(NAME) $^  $(NAME)

nn: $(NAME) $(NN_OBJ)
	$(CC) $(NN_OBJ) $(NAME) -o $(NN_EXEC)

%.o : %.c tensor.h
	$(CC) -c  $<  -o $@

clean :
	rm -f $(OBJ)
	rm -f $(NN_OBJ)
fclean : clean
	rm -f $(NAME)
	rm -f $(NN_EXEC)
re : fclean all