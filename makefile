
OBJ = tensor.o tensor_operations.o autograd.o tensor_clean.o neural_network.o plot_loss.o

NAME = mini-pytorch.a
NN_EXEC = 
all : $(NAME)

$(NAME) : $(OBJ)
	ar rc $(NAME) $^  $(NAME)

%.o : %.c tensor.h
	$(CC) -c  $<  -o $@

clean :
	rm -f $(OBJ)
fclean : clean
	rm -f $(NAME)

re : fclean all