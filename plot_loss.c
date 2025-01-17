#include "plot_loss.h"

void cost_history_init(CostHistory *history) {
    history->count = 0;
    history->min = 1e9;
    history->max = -1e9;
}

void cost_history_add(CostHistory *history, float value) {
    if (history->count < MAX_HISTORY) {
        history->values[history->count++] = value;
        if (value < history->min) history->min = value;
        if (value > history->max) history->max = value;
    }
}

void plot_cost_ascii(CostHistory *history) {
    char graph[GRAPH_HEIGHT][GRAPH_WIDTH + 1];
    float range = history->max - history->min;
    
    // Initialize graph with spaces
    for (int i = 0; i < GRAPH_HEIGHT; i++) {
        memset(graph[i], ' ', GRAPH_WIDTH);
        graph[i][GRAPH_WIDTH] = '\0';
    }
    
    // Draw axis
    for (int i = 0; i < GRAPH_HEIGHT; i++) {
        graph[i][0] = '|';
    }
    for (int i = 0; i < GRAPH_WIDTH; i++) {
        graph[GRAPH_HEIGHT-1][i] = '-';
    }
    
    // Plot points with improved x-axis distribution
    int points_plotted[GRAPH_WIDTH] = {0};  // Track lowest y for each x position
    float min_y[GRAPH_WIDTH];  // Track minimum value for each x position
    for (int i = 0; i < GRAPH_WIDTH; i++) {
        min_y[i] = history->max;
    }
    
    // First pass: find minimum values for each x position
    for (int i = 0; i < history->count; i++) {
        int x = (int)((float)i / history->count * (GRAPH_WIDTH - 2)) + 1;
        if (x < GRAPH_WIDTH && history->values[i] < min_y[x]) {
            min_y[x] = history->values[i];
        }
    }
    
    // Second pass: plot the points
    for (int x = 1; x < GRAPH_WIDTH; x++) {
        if (min_y[x] != history->max) {
            float normalized = (min_y[x] - history->min) / range;
            int y = GRAPH_HEIGHT - 2 - (int)(normalized * (GRAPH_HEIGHT - 3));
            if (y >= 0 && y < GRAPH_HEIGHT) {
                graph[y][x] = '*';
            }
        }
    }
    
    // Connect adjacent points with lines
    for (int x = 1; x < GRAPH_WIDTH - 1; x++) {
        if (min_y[x] != history->max && min_y[x+1] != history->max) {
            float norm1 = (min_y[x] - history->min) / range;
            float norm2 = (min_y[x+1] - history->min) / range;
            int y1 = GRAPH_HEIGHT - 2 - (int)(norm1 * (GRAPH_HEIGHT - 3));
            int y2 = GRAPH_HEIGHT - 2 - (int)(norm2 * (GRAPH_HEIGHT - 3));
            
            // Draw connecting line
            int start_y = (y1 < y2) ? y1 : y2;
            int end_y = (y1 < y2) ? y2 : y1;
            for (int y = start_y + 1; y < end_y; y++) {
                if (y >= 0 && y < GRAPH_HEIGHT) {
                    graph[y][x] = '|';
                }
            }
        }
    }
    
    // Print graph with axis labels
    printf("\nCost Function Over Epochs\n");
    printf("%.4f ┐\n", history->max);
    for (int i = 0; i < GRAPH_HEIGHT; i++) {
        printf("%s\n", graph[i]);
    }
    printf("%.4f ┴", history->min);
    for (int i = 0; i < GRAPH_WIDTH-10; i++) printf("─");
    printf(" %d epochs\n", history->count);
    
    // Print epoch markers
    printf("        ");  // Align with graph
    for (int i = 0; i <= 4; i++) {
        int pos = (i * (GRAPH_WIDTH - 10) / 4);
        printf("%-12d", i * history->count / 4);
    }
    printf("\n");
    
    // Save to CSV for external plotting
    FILE *fp = fopen("cost_history.csv", "w");
    if (fp) {
        fprintf(fp, "epoch,cost\n");
        for (int i = 0; i < history->count; i++) {
            fprintf(fp, "%d,%.6f\n", i, history->values[i]);
        }
        fclose(fp);
        printf("\nCost history saved to 'cost_history.csv'\n");
    }
}
