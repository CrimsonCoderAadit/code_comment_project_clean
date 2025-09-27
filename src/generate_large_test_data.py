import csv
import random
import os

def generate_useful_comments():
    """Generate 50 useful comment-code pairs"""
    useful_samples = [
        # Algorithm explanations
        {
            "Comments": "// Binary search implementation - returns index of target or -1 if not found",
            "Surrounding Code Context": "int binary_search(int arr[], int n, int target) { int left = 0, right = n - 1; while (left <= right) { int mid = left + (right - left) / 2; if (arr[mid] == target) return mid; if (arr[mid] < target) left = mid + 1; else right = mid - 1; } return -1; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Quick sort partition function - places pivot in correct position and returns partition index",
            "Surrounding Code Context": "int partition(int arr[], int low, int high) { int pivot = arr[high]; int i = low - 1; for (int j = low; j < high; j++) { if (arr[j] < pivot) { i++; swap(&arr[i], &arr[j]); } } swap(&arr[i + 1], &arr[high]); return i + 1; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Merge sort merge function - combines two sorted subarrays into one sorted array",
            "Surrounding Code Context": "void merge(int arr[], int left, int mid, int right) { int n1 = mid - left + 1; int n2 = right - mid; int L[n1], R[n2]; for (int i = 0; i < n1; i++) L[i] = arr[left + i]; for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j]; int i = 0, j = 0, k = left; while (i < n1 && j < n2) { if (L[i] <= R[j]) arr[k++] = L[i++]; else arr[k++] = R[j++]; } while (i < n1) arr[k++] = L[i++]; while (j < n2) arr[k++] = R[j++]; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Floyd's cycle detection algorithm - detects loops in linked lists using two pointers",
            "Surrounding Code Context": "bool has_cycle(struct Node* head) { struct Node *slow = head, *fast = head; while (fast && fast->next) { slow = slow->next; fast = fast->next->next; if (slow == fast) return true; } return false; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Dijkstra's shortest path algorithm - finds minimum distance from source to all vertices",
            "Surrounding Code Context": "void dijkstra(int graph[V][V], int src) { int dist[V]; bool sptSet[V]; for (int i = 0; i < V; i++) { dist[i] = INT_MAX; sptSet[i] = false; } dist[src] = 0; for (int count = 0; count < V - 1; count++) { int u = minDistance(dist, sptSet); sptSet[u] = true; for (int v = 0; v < V; v++) { if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v]) dist[v] = dist[u] + graph[u][v]; } } }",
            "Class": "Useful"
        },
        
        # Memory management and safety
        {
            "Comments": "// Allocate memory with error checking - returns NULL if allocation fails",
            "Surrounding Code Context": "int* create_array(int size) { if (size <= 0) return NULL; int *arr = malloc(size * sizeof(int)); if (arr == NULL) { fprintf(stderr, \"Memory allocation failed\\n\"); exit(1); } return arr; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Safe string copy with bounds checking to prevent buffer overflow",
            "Surrounding Code Context": "void safe_strcpy(char *dest, const char *src, size_t dest_size) { if (dest == NULL || src == NULL || dest_size == 0) return; strncpy(dest, src, dest_size - 1); dest[dest_size - 1] = '\\0'; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Free linked list recursively to prevent memory leaks",
            "Surrounding Code Context": "void free_list(struct Node* head) { if (head == NULL) return; free_list(head->next); free(head); }",
            "Class": "Useful"
        },
        {
            "Comments": "// Allocate 2D array with proper cleanup on partial failure",
            "Surrounding Code Context": "int** allocate_2d_array(int rows, int cols) { int **arr = malloc(rows * sizeof(int*)); if (!arr) return NULL; for (int i = 0; i < rows; i++) { arr[i] = malloc(cols * sizeof(int)); if (!arr[i]) { for (int j = 0; j < i; j++) free(arr[j]); free(arr); return NULL; } } return arr; }",
            "Class": "Useful"
        },
        
        # Complex logic explanations
        {
            "Comments": "// Bit manipulation trick - swap two integers without temporary variable using XOR",
            "Surrounding Code Context": "void swap_xor(int *a, int *b) { if (a != b && *a != *b) { *a ^= *b; *b ^= *a; *a ^= *b; } }",
            "Class": "Useful"
        },
        {
            "Comments": "// Hash table with linear probing - handles collisions by checking next slot",
            "Surrounding Code Context": "int hash_insert(HashTable *table, int key, int value) { int index = hash_function(key) % table->size; while (table->entries[index].is_occupied) { if (table->entries[index].key == key) { table->entries[index].value = value; return 0; } index = (index + 1) % table->size; } table->entries[index].key = key; table->entries[index].value = value; table->entries[index].is_occupied = true; return 1; }",
            "Class": "Useful"
        },
        {
            "Comments": "// LRU cache implementation - maintains least recently used order with doubly linked list",
            "Surrounding Code Context": "void lru_put(LRUCache* cache, int key, int value) { Node* node = hash_get(cache->map, key); if (node) { node->value = value; move_to_head(cache, node); } else { Node* new_node = create_node(key, value); if (cache->size >= cache->capacity) { Node* tail = remove_tail(cache); hash_remove(cache->map, tail->key); free(tail); cache->size--; } add_to_head(cache, new_node); hash_put(cache->map, key, new_node); cache->size++; } }",
            "Class": "Useful"
        },
        
        # Mathematical algorithms
        {
            "Comments": "// Fast exponentiation using binary representation - O(log n) complexity instead of O(n)",
            "Surrounding Code Context": "long long fast_power(long long base, long long exp, long long mod) { long long result = 1; base = base % mod; while (exp > 0) { if (exp % 2 == 1) result = (result * base) % mod; exp = exp >> 1; base = (base * base) % mod; } return result; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Euclidean algorithm for GCD - repeatedly applies division until remainder is zero",
            "Surrounding Code Context": "int gcd(int a, int b) { while (b != 0) { int temp = b; b = a % b; a = temp; } return a; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Sieve of Eratosthenes - efficiently finds all primes up to n by marking multiples",
            "Surrounding Code Context": "void sieve_of_eratosthenes(int n) { bool prime[n + 1]; memset(prime, true, sizeof(prime)); for (int p = 2; p * p <= n; p++) { if (prime[p]) { for (int i = p * p; i <= n; i += p) prime[i] = false; } } for (int p = 2; p <= n; p++) { if (prime[p]) printf(\"%d \", p); } }",
            "Class": "Useful"
        },
        
        # Data structure operations
        {
            "Comments": "// AVL tree rotation to maintain balance - left rotation for right-heavy subtree",
            "Surrounding Code Context": "struct Node* rotate_left(struct Node* x) { struct Node* y = x->right; struct Node* T2 = y->left; y->left = x; x->right = T2; x->height = 1 + max(height(x->left), height(x->right)); y->height = 1 + max(height(y->left), height(y->right)); return y; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Trie insertion - builds prefix tree for efficient string searching and autocomplete",
            "Surrounding Code Context": "void insert_trie(struct TrieNode* root, char* word) { struct TrieNode* current = root; for (int i = 0; word[i]; i++) { int index = word[i] - 'a'; if (!current->children[index]) current->children[index] = create_trie_node(); current = current->children[index]; } current->is_end_of_word = true; }",
            "Class": "Useful"
        },
        
        # System programming concepts
        {
            "Comments": "// Thread-safe counter using atomic operations to prevent race conditions",
            "Surrounding Code Context": "void increment_counter(atomic_int* counter) { atomic_fetch_add(counter, 1); }",
            "Class": "Useful"
        },
        {
            "Comments": "// Signal handler setup - properly handles SIGINT for graceful shutdown",
            "Surrounding Code Context": "void setup_signal_handler() { struct sigaction sa; sa.sa_handler = sigint_handler; sigemptyset(&sa.sa_mask); sa.sa_flags = 0; if (sigaction(SIGINT, &sa, NULL) == -1) { perror(\"sigaction\"); exit(1); } }",
            "Class": "Useful"
        },
        
        # File I/O and parsing
        {
            "Comments": "// Robust file reading with error handling and proper resource cleanup",
            "Surrounding Code Context": "char* read_file(const char* filename) { FILE* file = fopen(filename, \"r\"); if (!file) { perror(\"fopen\"); return NULL; } fseek(file, 0, SEEK_END); long size = ftell(file); rewind(file); char* buffer = malloc(size + 1); if (!buffer) { fclose(file); return NULL; } fread(buffer, 1, size, file); buffer[size] = '\\0'; fclose(file); return buffer; }",
            "Class": "Useful"
        },
        {
            "Comments": "// CSV parser with quote handling - properly processes escaped commas and quotes",
            "Surrounding Code Context": "char** parse_csv_line(char* line, int* field_count) { char** fields = malloc(MAX_FIELDS * sizeof(char*)); int count = 0; bool in_quotes = false; char* start = line; for (char* p = line; *p; p++) { if (*p == '\"') in_quotes = !in_quotes; else if (*p == ',' && !in_quotes) { *p = '\\0'; fields[count++] = start; start = p + 1; } } fields[count++] = start; *field_count = count; return fields; }",
            "Class": "Useful"
        },
        
        # Network programming
        {
            "Comments": "// Socket setup with proper error handling and address reuse",
            "Surrounding Code Context": "int create_server_socket(int port) { int sockfd = socket(AF_INET, SOCK_STREAM, 0); if (sockfd < 0) return -1; int opt = 1; setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)); struct sockaddr_in addr; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port); if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { close(sockfd); return -1; } return sockfd; }",
            "Class": "Useful"
        },
        
        # String processing
        {
            "Comments": "// Boyer-Moore string search - skips characters using bad character heuristic for faster searching",
            "Surrounding Code Context": "int boyer_moore_search(char* text, char* pattern) { int m = strlen(pattern); int n = strlen(text); int bad_char[256]; fill_bad_char_table(pattern, m, bad_char); int s = 0; while (s <= (n - m)) { int j = m - 1; while (j >= 0 && pattern[j] == text[s + j]) j--; if (j < 0) return s; else s += max(1, j - bad_char[text[s + j]]); } return -1; }",
            "Class": "Useful"
        },
        
        # Add more samples to reach 50
        {
            "Comments": "// Circular buffer implementation - overwrites oldest data when full",
            "Surrounding Code Context": "void circular_buffer_put(CircularBuffer* cb, int data) { cb->buffer[cb->head] = data; cb->head = (cb->head + 1) % cb->size; if (cb->count < cb->size) cb->count++; else cb->tail = (cb->tail + 1) % cb->size; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Stack implementation with dynamic resizing to handle growth",
            "Surrounding Code Context": "void stack_push(Stack* stack, int value) { if (stack->top >= stack->capacity - 1) { stack->capacity *= 2; stack->data = realloc(stack->data, stack->capacity * sizeof(int)); } stack->data[++stack->top] = value; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Thread pool worker function - continuously processes tasks from queue",
            "Surrounding Code Context": "void* worker_thread(void* arg) { ThreadPool* pool = (ThreadPool*)arg; while (1) { pthread_mutex_lock(&pool->mutex); while (pool->queue_size == 0 && !pool->shutdown) { pthread_cond_wait(&pool->condition, &pool->mutex); } if (pool->shutdown) break; Task task = pool->queue[pool->queue_front]; pool->queue_front = (pool->queue_front + 1) % pool->queue_capacity; pool->queue_size--; pthread_mutex_unlock(&pool->mutex); task.function(task.arg); } return NULL; }",
            "Class": "Useful"
        }
    ]
    
    # Add more basic but useful comments to reach 50
    additional_useful = [
        {"Comments": "// Calculate factorial recursively with base case n <= 1", "Surrounding Code Context": "int factorial(int n) { if (n <= 1) return 1; return n * factorial(n - 1); }", "Class": "Useful"},
        {"Comments": "// Find maximum element in array by iterating through all elements", "Surrounding Code Context": "int find_max(int arr[], int n) { int max = arr[0]; for (int i = 1; i < n; i++) { if (arr[i] > max) max = arr[i]; } return max; }", "Class": "Useful"},
        {"Comments": "// Reverse string in-place using two pointers from both ends", "Surrounding Code Context": "void reverse_string(char* str) { int len = strlen(str); for (int i = 0; i < len / 2; i++) { char temp = str[i]; str[i] = str[len - 1 - i]; str[len - 1 - i] = temp; } }", "Class": "Useful"},
        {"Comments": "// Check if string is palindrome by comparing characters from both ends", "Surrounding Code Context": "bool is_palindrome(char* str) { int len = strlen(str); for (int i = 0; i < len / 2; i++) { if (tolower(str[i]) != tolower(str[len - 1 - i])) return false; } return true; }", "Class": "Useful"},
        {"Comments": "// Count occurrences of character in string using linear scan", "Surrounding Code Context": "int count_char(char* str, char c) { int count = 0; for (int i = 0; str[i]; i++) { if (str[i] == c) count++; } return count; }", "Class": "Useful"},
        {"Comments": "// Matrix multiplication with proper bounds checking", "Surrounding Code Context": "void matrix_multiply(int A[][MAX], int B[][MAX], int C[][MAX], int rows_A, int cols_A, int cols_B) { for (int i = 0; i < rows_A; i++) { for (int j = 0; j < cols_B; j++) { C[i][j] = 0; for (int k = 0; k < cols_A; k++) { C[i][j] += A[i][k] * B[k][j]; } } } }", "Class": "Useful"},
        {"Comments": "// Bubble sort with early termination when no swaps occur", "Surrounding Code Context": "void bubble_sort(int arr[], int n) { for (int i = 0; i < n - 1; i++) { bool swapped = false; for (int j = 0; j < n - i - 1; j++) { if (arr[j] > arr[j + 1]) { swap(&arr[j], &arr[j + 1]); swapped = true; } } if (!swapped) break; } }", "Class": "Useful"},
        {"Comments": "// Binary tree height calculation using recursive depth-first approach", "Surrounding Code Context": "int tree_height(struct Node* root) { if (root == NULL) return -1; int left_height = tree_height(root->left); int right_height = tree_height(root->right); return 1 + max(left_height, right_height); }", "Class": "Useful"},
        {"Comments": "// Queue implementation using circular array with full/empty detection", "Surrounding Code Context": "bool queue_enqueue(Queue* q, int data) { if ((q->rear + 1) % q->capacity == q->front) return false; q->data[q->rear] = data; q->rear = (q->rear + 1) % q->capacity; return true; }", "Class": "Useful"},
        {"Comments": "// Prime number check using trial division up to square root", "Surrounding Code Context": "bool is_prime(int n) { if (n <= 1) return false; if (n <= 3) return true; if (n % 2 == 0 || n % 3 == 0) return false; for (int i = 5; i * i <= n; i += 6) { if (n % i == 0 || n % (i + 2) == 0) return false; } return true; }", "Class": "Useful"},
        {"Comments": "// Fibonacci sequence using iterative approach to avoid recursion overhead", "Surrounding Code Context": "int fibonacci(int n) { if (n <= 1) return n; int a = 0, b = 1, result; for (int i = 2; i <= n; i++) { result = a + b; a = b; b = result; } return result; }", "Class": "Useful"},
        {"Comments": "// Array rotation by k positions using reversal algorithm", "Surrounding Code Context": "void rotate_array(int arr[], int n, int k) { k = k % n; reverse_array(arr, 0, n - 1); reverse_array(arr, 0, k - 1); reverse_array(arr, k, n - 1); }", "Class": "Useful"},
        {"Comments": "// Find intersection of two sorted arrays using two pointers", "Surrounding Code Context": "void find_intersection(int arr1[], int m, int arr2[], int n) { int i = 0, j = 0; while (i < m && j < n) { if (arr1[i] < arr2[j]) i++; else if (arr1[i] > arr2[j]) j++; else { printf(\"%d \", arr1[i]); i++; j++; } } }", "Class": "Useful"},
        {"Comments": "// Remove duplicates from sorted array by overwriting with unique elements", "Surrounding Code Context": "int remove_duplicates(int arr[], int n) { if (n == 0) return 0; int j = 0; for (int i = 1; i < n; i++) { if (arr[i] != arr[j]) { j++; arr[j] = arr[i]; } } return j + 1; }", "Class": "Useful"},
        {"Comments": "// Find second largest element in array with single pass", "Surrounding Code Context": "int second_largest(int arr[], int n) { int first = INT_MIN, second = INT_MIN; for (int i = 0; i < n; i++) { if (arr[i] > first) { second = first; first = arr[i]; } else if (arr[i] > second && arr[i] != first) { second = arr[i]; } } return second; }", "Class": "Useful"},
        {"Comments": "// Calculate power using iterative multiplication for positive exponents", "Surrounding Code Context": "double power(double base, int exp) { if (exp < 0) return 1.0 / power(base, -exp); double result = 1.0; for (int i = 0; i < exp; i++) { result *= base; } return result; }", "Class": "Useful"},
        {"Comments": "// Check if array is sorted in ascending order", "Surrounding Code Context": "bool is_sorted(int arr[], int n) { for (int i = 1; i < n; i++) { if (arr[i] < arr[i - 1]) return false; } return true; }", "Class": "Useful"},
        {"Comments": "// Find majority element using Boyer-Moore voting algorithm", "Surrounding Code Context": "int find_majority(int arr[], int n) { int candidate = 0, count = 0; for (int i = 0; i < n; i++) { if (count == 0) candidate = arr[i]; count += (arr[i] == candidate) ? 1 : -1; } return candidate; }", "Class": "Useful"},
        {"Comments": "// Convert string to integer with error handling for invalid input", "Surrounding Code Context": "int string_to_int(const char* str, bool* success) { if (!str || !*str) { *success = false; return 0; } int result = 0, sign = 1; if (*str == '-') { sign = -1; str++; } while (*str) { if (*str < '0' || *str > '9') { *success = false; return 0; } result = result * 10 + (*str - '0'); str++; } *success = true; return result * sign; }", "Class": "Useful"},
        {"Comments": "// Merge two sorted linked lists into single sorted list", "Surrounding Code Context": "struct Node* merge_sorted_lists(struct Node* l1, struct Node* l2) { if (!l1) return l2; if (!l2) return l1; if (l1->data <= l2->data) { l1->next = merge_sorted_lists(l1->next, l2); return l1; } else { l2->next = merge_sorted_lists(l1, l2->next); return l2; } }", "Class": "Useful"},
        {"Comments": "// Calculate square root using Newton's method for approximation", "Surrounding Code Context": "double sqrt_newton(double x, double precision) { if (x < 0) return -1; double guess = x / 2.0; while (fabs(guess * guess - x) > precision) { guess = (guess + x / guess) / 2.0; } return guess; }", "Class": "Useful"},
        {"Comments": "// Check if parentheses are balanced using stack approach", "Surrounding Code Context": "bool balanced_parentheses(char* expr) { Stack stack; init_stack(&stack); for (int i = 0; expr[i]; i++) { if (expr[i] == '(' || expr[i] == '[' || expr[i] == '{') { push(&stack, expr[i]); } else if (expr[i] == ')' || expr[i] == ']' || expr[i] == '}') { if (is_empty(&stack)) return false; char top = pop(&stack); if (!is_matching_pair(top, expr[i])) return false; } } return is_empty(&stack); }", "Class": "Useful"},
        {"Comments": "// Count number of set bits in integer using Brian Kernighan's algorithm", "Surrounding Code Context": "int count_set_bits(int n) { int count = 0; while (n) { n &= (n - 1); count++; } return count; }", "Class": "Useful"},
        {"Comments": "// Find least common multiple using GCD relationship", "Surrounding Code Context": "int lcm(int a, int b) { return (a * b) / gcd(a, b); }", "Class": "Useful"},
        {"Comments": "// Implement basic hash function using djb2 algorithm", "Surrounding Code Context": "unsigned long hash_djb2(unsigned char* str) { unsigned long hash = 5381; int c; while ((c = *str++)) { hash = ((hash << 5) + hash) + c; } return hash; }", "Class": "Useful"},
        {"Comments": "// Find peak element in array using binary search approach", "Surrounding Code Context": "int find_peak(int arr[], int n) { int left = 0, right = n - 1; while (left < right) { int mid = left + (right - left) / 2; if (arr[mid] < arr[mid + 1]) left = mid + 1; else right = mid; } return left; }", "Class": "Useful"},
        {"Comments": "// Convert binary string to decimal using positional notation", "Surrounding Code Context": "int binary_to_decimal(char* binary) { int decimal = 0; int base = 1; int len = strlen(binary); for (int i = len - 1; i >= 0; i--) { if (binary[i] == '1') decimal += base; base *= 2; } return decimal; }", "Class": "Useful"}
    ]
    
    return useful_samples + additional_useful[:50 - len(useful_samples)]

def generate_not_useful_comments():
    """Generate 50 not useful comment-code pairs"""
    not_useful_samples = [
        # Obvious statements
        {"Comments": "// Increment i", "Surrounding Code Context": "for (int i = 0; i < n; i++) { printf(\"%d \", arr[i]); }", "Class": "Not Useful"},
        {"Comments": "// Return the sum", "Surrounding Code Context": "int add(int a, int b) { return a + b; }", "Class": "Not Useful"},
        {"Comments": "// Print hello", "Surrounding Code Context": "void print_hello() { printf(\"Hello, World!\\n\"); }", "Class": "Not Useful"},
        {"Comments": "// Check if x equals 0", "Surrounding Code Context": "if (x == 0) { printf(\"Zero\\n\"); } else { printf(\"Non-zero\\n\"); }", "Class": "Not Useful"},
        {"Comments": "// Loop through array", "Surrounding Code Context": "for (int j = 0; j < length; j++) { sum += numbers[j]; }", "Class": "Not Useful"},
        {"Comments": "// Set x to 1", "Surrounding Code Context": "int x = 1;", "Class": "Not Useful"},
        {"Comments": "// Call function", "Surrounding Code Context": "result = calculate_total(values, count);", "Class": "Not Useful"},
        {"Comments": "// Close file", "Surrounding Code Context": "fclose(file);", "Class": "Not Useful"},
        {"Comments": "// Assign value", "Surrounding Code Context": "value = 42;", "Class": "Not Useful"},
        {"Comments": "// End of function", "Surrounding Code Context": "return 0; }", "Class": "Not Useful"},
        
        # Redundant comments
        {"Comments": "// Initialize variables", "Surrounding Code Context": "int a = 0, b = 0, c = 0;", "Class": "Not Useful"},
        {"Comments": "// Start of main function", "Surrounding Code Context": "int main() {", "Class": "Not Useful"},
        {"Comments": "// Declare integer variable", "Surrounding Code Context": "int counter;", "Class": "Not Useful"},
        {"Comments": "// Open file for reading", "Surrounding Code Context": "FILE* fp = fopen(\"data.txt\", \"r\");", "Class": "Not Useful"},
        {"Comments": "// Allocate memory", "Surrounding Code Context": "ptr = malloc(size);", "Class": "Not Useful"},
        {"Comments": "// Free memory", "Surrounding Code Context": "free(ptr);", "Class": "Not Useful"},
        {"Comments": "// Include header file", "Surrounding Code Context": "#include <stdio.h>", "Class": "Not Useful"},
        {"Comments": "// Define constant", "Surrounding Code Context": "#define MAX_SIZE 100", "Class": "Not Useful"},
        {"Comments": "// Create struct", "Surrounding Code Context": "struct Point { int x, y; };", "Class": "Not Useful"},
        {"Comments": "// Function declaration", "Surrounding Code Context": "int calculate(int a, int b);", "Class": "Not Useful"},
        
        # Stating the obvious
        {"Comments": "// If condition is true", "Surrounding Code Context": "if (condition == true) { execute_code(); }", "Class": "Not Useful"},
        {"Comments": "// While loop", "Surrounding Code Context": "while (i < 10) { process(i); i++; }", "Class": "Not Useful"},
        {"Comments": "// Switch statement", "Surrounding Code Context": "switch (option) { case 1: handle_option1(); break; }", "Class": "Not Useful"},
        {"Comments": "// Else clause", "Surrounding Code Context": "} else { handle_alternative(); }", "Class": "Not Useful"},
        {"Comments": "// Break from loop", "Surrounding Code Context": "if (found) break;", "Class": "Not Useful"},
        {"Comments": "// Continue loop", "Surrounding Code Context": "if (skip) continue;", "Class": "Not Useful"},
        {"Comments": "// Return statement", "Surrounding Code Context": "return result;", "Class": "Not Useful"},
        {"Comments": "// Exit program", "Surrounding Code Context": "exit(0);", "Class": "Not Useful"},
        {"Comments": "// Print number", "Surrounding Code Context": "printf(\"%d\", num);", "Class": "Not Useful"},
        {"Comments": "// Read input", "Surrounding Code Context": "scanf(\"%d\", &input);", "Class": "Not Useful"},
        
        # Meaningless or wrong comments
        {"Comments": "// This is code", "Surrounding Code Context": "int process_data(char* buffer) { return strlen(buffer); }", "Class": "Not Useful"},
        {"Comments": "// Do something", "Surrounding Code Context": "void process() { calculate_result(); display_output(); }", "Class": "Not Useful"},
        {"Comments": "// Important function", "Surrounding Code Context": "void update() { refresh_display(); save_state(); }", "Class": "Not Useful"},
        {"Comments": "// Main logic here", "Surrounding Code Context": "if (status == READY) { begin_processing(); } else { wait_for_ready(); }", "Class": "Not Useful"},
        {"Comments": "// Code goes here", "Surrounding Code Context": "void initialize() { setup_environment(); load_configuration(); }", "Class": "Not Useful"},
        {"Comments": "// Function body", "Surrounding Code Context": "int compute(int x, int y) { return x * y + 5; }", "Class": "Not Useful"},
        {"Comments": "// Some calculation", "Surrounding Code Context": "result = (a + b) * c - d;", "Class": "Not Useful"},
        {"Comments": "// Process input", "Surrounding Code Context": "validate_input(data); transform_data(data); store_result(data);", "Class": "Not Useful"},
        {"Comments": "// Handle case", "Surrounding Code Context": "case 2: execute_option2(); break;", "Class": "Not Useful"},
        {"Comments": "// Update value", "Surrounding Code Context": "counter += increment;", "Class": "Not Useful"},
        
        # Comments that add no information
        {"Comments": "// Variable declaration", "Surrounding Code Context": "double temperature, pressure;", "Class": "Not Useful"},
        {"Comments": "// Array initialization", "Surrounding Code Context": "int numbers[10] = {0};", "Class": "Not Useful"},
        {"Comments": "// Pointer assignment", "Surrounding Code Context": "char* ptr = buffer;", "Class": "Not Useful"},
        {"Comments": "// Comparison check", "Surrounding Code Context": "if (a > b) { max = a; } else { max = b; }", "Class": "Not Useful"},
        {"Comments": "// Loop iteration", "Surrounding Code Context": "for (int k = 0; k < size; k++) { array[k] = 0; }", "Class": "Not Useful"},
        {"Comments": "// Memory operation", "Surrounding Code Context": "memset(buffer, 0, sizeof(buffer));", "Class": "Not Useful"},
        {"Comments": "// String operation", "Surrounding Code Context": "strcpy(dest, source);", "Class": "Not Useful"},
        {"Comments": "// Mathematical operation", "Surrounding Code Context": "area = length * width;", "Class": "Not Useful"},
        {"Comments": "// Conditional execution", "Surrounding Code Context": "if (enabled) execute_task();", "Class": "Not Useful"},
        {"Comments": "// Data processing", "Surrounding Code Context": "output = transform(input);", "Class": "Not Useful"},
        
        # Placeholder or lazy comments
        {"Comments": "// TODO", "Surrounding Code Context": "void future_feature() { /* implementation pending */ }", "Class": "Not Useful"},
        {"Comments": "// FIXME", "Surrounding Code Context": "int buggy_function() { return -1; /* needs fixing */ }", "Class": "Not Useful"},
        {"Comments": "// Note", "Surrounding Code Context": "status = get_status(); /* check this later */", "Class": "Not Useful"},
        {"Comments": "// Temporary", "Surrounding Code Context": "int temp_solution() { return hardcoded_value; }", "Class": "Not Useful"},
        {"Comments": "// Debug", "Surrounding Code Context": "printf(\"Debug: %d\\n\", value);", "Class": "Not Useful"},
        {"Comments": "// Test", "Surrounding Code Context": "void test_function() { assert(1 == 1); }", "Class": "Not Useful"},
        {"Comments": "// Hack", "Surrounding Code Context": "workaround = dirty_fix(problem);", "Class": "Not Useful"},
        {"Comments": "// Magic number", "Surrounding Code Context": "size = 42;", "Class": "Not Useful"},
        {"Comments": "// Quick fix", "Surrounding Code Context": "if (error) return;", "Class": "Not Useful"},
        {"Comments": "// Placeholder", "Surrounding Code Context": "void stub() { /* not implemented */ }", "Class": "Not Useful"}
    ]
    
    return not_useful_samples

def save_large_synthetic_data():
    """Save large balanced synthetic dataset to CSV file"""
    # Generate balanced samples
    useful_samples = generate_useful_comments()
    not_useful_samples = generate_not_useful_comments()
    
    # Ensure we have exactly 50 of each
    useful_samples = useful_samples[:50]
    not_useful_samples = not_useful_samples[:50]
    
    # Combine and shuffle
    all_samples = useful_samples + not_useful_samples
    random.shuffle(all_samples)
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save to CSV
    output_file = os.path.join(data_dir, "large_synthetic_test_data.csv")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Comments', 'Surrounding Code Context', 'Class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for sample in all_samples:
            writer.writerow(sample)
    
    print(f"✅ Generated {len(all_samples)} synthetic samples")
    print(f"📁 Saved to: {output_file}")
    
    # Print distribution
    useful_count = sum(1 for s in all_samples if s['Class'] == 'Useful')
    not_useful_count = len(all_samples) - useful_count
    print(f"📊 Distribution: {useful_count} Useful, {not_useful_count} Not Useful")
    
    return output_file

if __name__ == "__main__":
    print("🤖 Generating LARGE balanced synthetic C code-comment pairs...")
    save_large_synthetic_data()
    print("✨ Ready for testing with 100 samples!")