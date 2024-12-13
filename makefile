
OBJ = tensor.o tensor_operations.o
CFLAGS = -Wall -Wextra -Werror
TESTOBJ = /tests/tensor.o /tests/tensor_operations.o
NAME = mini-pytorch.a
all : $(NAME)

$(NAME) : $(OBJ)
	ar rc $(NAME) $^

%.o : %.c tensor.h
	$(CC) -c  $< $(CFLAGS) -o $@

clean :
	rm -f $(OBJ)
fclean : clean
	rm -f $(NAME)

re : fclean all