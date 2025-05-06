#include <iostream>
#include <omp.h>
using namespace std;

// Merge two halves of the array
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = new int[n1];
    int* R = new int[n2];

    // Copy to temp arrays
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    // Merge the two halves
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    // Copy remaining elements
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

// Sequential Merge Sort
void sequentialMergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;
        sequentialMergeSort(arr, left, mid);
        sequentialMergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Parallel Merge Sort
void parallelMergeSort(int arr[], int left, int right, int depth = 0) {
    if (left < right) {
        int mid = (left + right) / 2;

        if (depth < 6) { // Limit depth to control parallelism
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMergeSort(arr, left, mid, depth + 1);

                #pragma omp section
                parallelMergeSort(arr, mid + 1, right, depth + 1);
            }
        } else {
            sequentialMergeSort(arr, left, mid);
            sequentialMergeSort(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
    }
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    cout << endl;
}

int main() {
    // int arr1[] = {38, 27, 43, 3, 9, 82, 10};
    // int arr2[] = {38, 27, 43, 3, 9, 82, 10};
    // int n = sizeof(arr1) / sizeof(arr1[0]);

    int n = 10000000;
    srand(time(0));

    int* arr1 = new int[n];
    int* arr2 = new int[n];

    for (int i = 0; i < n; i++) {
        arr1[i] = rand() % 1000000;
        arr2[i] = arr1[i]; // copy original values
    }

    cout << "Original Array: ";
    printArray(arr1, 10);

    double start1 = omp_get_wtime();
    sequentialMergeSort(arr1, 0, n - 1);
    double end1 = omp_get_wtime();

    double start2 = omp_get_wtime();
    parallelMergeSort(arr2, 0, n - 1);
    double end2 = omp_get_wtime();

    cout << "Sequential Sorted: ";
    printArray(arr1, 10);
    cout << "Time: " << end1 - start1 << " seconds\n";

    cout << "Parallel Sorted:   ";
    printArray(arr2, n);
    cout << "Time: " << end2 - start2 << " seconds\n";

    return 0;
}
