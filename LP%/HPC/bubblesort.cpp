#include <iostream>
#include <omp.h>
#include <cstdlib>
using namespace std;

void sequentialBubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++){
        for (int j = 0; j < n - i - 1; j++){
            if (arr[j] > arr[j + 1]){
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
void parallelBubbleSort(int arr[], int n) {
    bool isSorted = false;
    for (int i = 0; i < n && !isSorted; i++) {
        isSorted = true;
        #pragma omp parallel for shared(arr, isSorted)
        for (int j = 1; j < n - 1; j += 2)
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                isSorted = false;
            }
        #pragma omp parallel for shared(arr, isSorted)
        for (int j = 0; j < n - 1; j += 2)
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                isSorted = false;
            }
    }
}
void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    cout << endl;
}

int main() {
    // int arr1[] = {64, 25, 12, 22, 11};
    // int n = 5;

    int n = 100000;

    int* arr1 = new int[n];
    int* arr2 = new int[n];

    for (int i = 0; i < n; i++) {
        arr1[i] = rand() % 1000000;
        arr2[i] = arr1[i]; // copy original values
    }

    printArray(arr1, 10);
    printArray(arr2, 10);

    double t1 = omp_get_wtime();
    sequentialBubbleSort(arr1, n);
    double t2 = omp_get_wtime();

    double t3 = omp_get_wtime();
    parallelBubbleSort(arr2, n);
    double t4 = omp_get_wtime();

    cout << "Sequential Sorted: ";
    // printArray(arr1, n);
    cout << "Time: " << t2 - t1 << " seconds" << endl;

    cout << "Parallel Sorted:   ";
    // printArray(arr2, n);
    cout << "Time: " << t4 - t3 << " seconds" << endl;

    return 0;
}
