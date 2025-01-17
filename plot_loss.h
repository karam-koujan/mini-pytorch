#ifndef PLOT_LOSS_H
#define PLOT_LOSS_H

#include <stdio.h>
#include <string.h>

#define MAX_HISTORY 2000
#define GRAPH_WIDTH 60
#define GRAPH_HEIGHT 20

typedef struct {
    float values[MAX_HISTORY];
    int count;
    float min;
    float max;
} CostHistory;


void cost_history_init(CostHistory *history);
void cost_history_add(CostHistory *history, float value);
void plot_cost_ascii(CostHistory *history);
#endif