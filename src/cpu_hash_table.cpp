//
// Created by skunapan on 4/10/24.
//

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

/*
This has a simple hash table implementation, that maps onto a array based on FNV1-A hashing with linear probing
for collisions
*/

uint64_t fnv1a_hash(void *key, size_t size) {
    uint64_t hash = 0xcbf29ce484222325;
    auto *bytes = static_cast<uint8_t *>(key);
    for (size_t i = 0; i < size; i++) {
        hash = hash ^ bytes[i];
        hash = hash * 0x100000001b3;
    }
    return hash;
}

typedef struct MapElem {
    int key = 0;
    int data = 0;
} MapElem;
typedef struct Map {
    MapElem *array = nullptr;
    size_t max_size = 0;
    size_t size = 0;
    bool *init = nullptr;
} Map;

void insertKey(int key, int data, Map *map) {
    uint64_t hash = fnv1a_hash(static_cast<void *>(&key), sizeof(key));
    size_t table_size = map->max_size;
    uint64_t index = hash % table_size;
    uint64_t startIndex = index;
    while (map->init[index]) {
        if (map->init[index] == 2) // to handle deleted keys
            break;
        if (map->array[index].key == key) {
            map->array[index].data = data;
            return;
        }
        index = (index + 1) % map->max_size;
        if (index == startIndex)
            return;
    }
    if()
    map->array[index].key = key;
    map->array[index].data = data;
    map->init[index] = 1;
}

void deleteKey(int key, Map *map) {
    uint64_t hash = fnv1a_hash(static_cast<void *>(&key), sizeof(key));
    size_t table_size = map->max_size;
    uint64_t index = hash % table_size;
    uint64_t startIndex = index;
    while (map->init[index]) {
        if (map->array[index].key == key) {
            map->init[index] = 2;
            return;
        }
    }
}

MapElem *getVal(int key, Map *map) {
    uint64_t hash = fnv1a_hash(static_cast<void *>(&key), sizeof(key));
    size_t table_size = map->max_size;
    uint64_t index = hash % table_size;
    uint64_t startIndex = index;
    while (map->init[index]) {
        if (map->init[index] != 2 && map->array[index].key == key) {
            return &(map->array[index]);
        }
        index = (index + 1) % map->max_size;
        if (index == startIndex)
            return nullptr;
    }
    return nullptr;
}

void delMap(Map *map) {
    free(map->array);
    free(map->init);
    free(map);
}

int main() {
    size_t size = 10;
    Map *map = (Map *) malloc(sizeof(Map));
    map->max_size = size;
    map->array = (MapElem *) malloc(size * sizeof(MapElem));
    map->init = (bool *) malloc(size * sizeof(bool));
    memset(map->init, 0, size * sizeof(bool));
    for (int i = 0; i < 10; i++) {
        insertKey(i, i + 1, map);
    }
    for (int i = 0; i < 10; i++) {
        MapElem *curElem = getVal(i, map);
        if (curElem != nullptr)
            printf("key:%d, val:%d \n", curElem->key, curElem->data);
    }
    delMap(map);
}