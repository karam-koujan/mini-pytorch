
OBJ = tensor.o
CFLAGS = -Wall -Wextra -Werror

all: $(OBJ)

%.o : %.c tensor.h
	$(CC) -c $(CFLAGS) $< -o $@