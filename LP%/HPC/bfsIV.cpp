#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

int main() {
    int V, E;
    cin >> V >> E;

    vector<vector<int>> graph(V);
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    vector<bool> visited(V, false);
    queue<int> q;
    int start;
    cin >> start;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int sz = q.size();
        vector<int> level;
        for (int i = 0; i < sz; i++) {
            int node = q.front(); q.pop();
            cout << node << " ";
            level.push_back(node);
        }

        vector<int> next;
        #pragma omp parallel for
        for (int i = 0; i < level.size(); i++) {
            int node = level[i];
            for (int nbr : graph[node]) {
                if (!visited[nbr]) {
                    #pragma omp critical
                    {
                        if (!visited[nbr]) {
                            visited[nbr] = true;
                            next.push_back(nbr);
                        }
                    }
                }
            }
        }

        for (int n : next) q.push(n);
    }
}