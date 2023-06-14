#pragma once
// #ifndef MAT_MULT_VEC_H
// #define MAT_MULT_VEC_H

#include "vector.cuh"

int mat_mult_vec(Vector3f *hostA, 
                 Vector3f *hostB, 
                 float *hostC, 
                 int numARows, 
                 int numAColumns, 
                 int numBRows,
                 int numBColumns,
                 int numCRows,   
                 int numCColumns);

// #endif