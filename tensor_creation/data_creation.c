#include "../headers/tensor.h"
#include "../headers/print.h"

void    *create_zero_data(Dtype type, int size)
{
    int val_size = sizeof_type(type);
    if (val_size == -1)
        return (NULL);
    void *data = malloc(size * val_size);
    if (!data)
        return (error_msg("data creation failed!!"), NULL);
    for (int i = 0; i < size; i++)
    {
        if (type == FLOAT32)
            ((float *)data)[i] = 0.0F;
        else if (type == DOUBLE)
            ((double *)data)[i] = 0.0;
        else if (type == INT32)
            ((int *)data)[i] = 0;
        else if (type == INT64)
            ((int64_t *)data)[i] = 0L;
    }
    return (data);
}

void    *create_val_data(Dtype type, int size, void *val)
{
    int val_size = sizeof_type(type);
    if (val_size == -1)
        return (NULL);
    void *data = malloc(size * val_size);
    if (!data)
        return (error_msg("data creation failed!!"), NULL);
    for (int i = 0; i < size; i++)
    {
        if (type == FLOAT32)
            ((float *)data)[i] = *(float*)(val);
        else if (type == DOUBLE)
            ((double *)data)[i] = *(double*)(val);
        else if (type == INT32)
            ((int *)data)[i] = *(int*)(val);
        else if (type == INT64)
            ((int64_t *)data)[i] = *(int64_t*)(val);
    }
    return (data);
}

void    *create_one_data(Dtype type, int size)
{
    int val_size = sizeof_type(type);
    if (val_size == -1)
        return (NULL);
    void *data = malloc(size * val_size);
    if (!data)
        return (error_msg("data creation failed!!"), NULL);
    for (int i = 0; i < size; i++)
    {
        if (type == FLOAT32)
            ((float *)data)[i] = 1.0F;
        else if (type == DOUBLE)
            ((double *)data)[i] = 1.0;
        else if (type == INT32)
            ((int *)data)[i] = 1;
        else if (type == INT64)
            ((int64_t *)data)[i] = 1;
    }
    return (data);
}

void *copy_arr_data(void *arr, Dtype type, int64_t size)
{
    int element_size = sizeof_type(type);
    if (element_size == -1)
        return (NULL);
    void *data = malloc(size * element_size);
    if (!data)
        return (error_msg("data creation failed!!"), NULL);
    memcpy(data, arr, size * element_size);
    return (data);
}

void    *create_rand_data(Dtype type, int size)
{
    int val_size = sizeof_type(type);
    if (val_size == -1)
        return (NULL);
    void *data = malloc(size * val_size);
    if (!data)
        return (error_msg("data creation failed!!"), NULL);
    for (int i = 0; i < size; i++)
    {
        if (type == FLOAT32)

            ((float *)data)[i] = (float)generate_random();
        else if (type == DOUBLE)
            ((double *)data)[i] = (double)generate_random();
        else if (type == INT32)
            ((int *)data)[i] = (int)generate_random();
        else if (type == INT64)
            ((int64_t *)data)[i] = (int64_t)generate_random();
    }
    return (data);
}
