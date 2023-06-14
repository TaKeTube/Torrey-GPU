// #ifndef MAT_MULT_H
// #define MAT_MULT_H
#pragma once

int mat_mult(float *hostA, 
             float *hostB, 
             float *hostC, 
             int numARows, 
             int numAColumns, 
             int numBRows,
             int numBColumns,
             int numCRows,   
             int numCColumns);

// #endif