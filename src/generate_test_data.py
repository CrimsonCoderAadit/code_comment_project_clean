import csv
import random
import os

# Sample synthetic C code-comment pairs
# In real implementation, you'd use an LLM API (OpenAI, Claude, etc.)
synthetic_data = [
    # Useful comments
    {
        "Comments": "// Binary search implementation - returns index of target or -1 if not found",
        "Surrounding Code Context": "int binary_search(int arr[], int n, int target) { int left = 0, right = n - 1; while (left <= right) { int mid = left + (right - left) / 2; if (arr[mid] == target) return mid; if (arr[mid] < target) left = mid + 1; else right = mid - 1; } return -1; }",
        "Class": "Useful"
    },
    {
        "Comments": "// Calculate factorial recursively with base case handling",
        "Surrounding Code Context": "int factorial(int n) { if (n <= 1) return 1; return n * factorial(n - 1); }",
        "Class": "Useful"
    },
    {
        "Comments": "// Swap two integers using XOR operation to avoid temporary variable",
        "Surrounding Code Context": "void swap_xor(int *a, int *b) { if (a != b) { *a ^= *b; *b ^= *a; *a ^= *b; } }",
        "Class": "Useful"
    },
    {
        "Comments": "// Dynamic memory allocation for integer array with error checking",
        "Surrounding Code Context": "int* create_array(int size) { int *arr = malloc(size * sizeof(int)); if (arr == NULL) { fprintf(stderr, \"Memory allocation failed\\n\"); exit(1); } return arr; }",
        "Class": "Useful"
    },
    {
        "Comments": "// Floyd's cycle detection algorithm for linked list loop detection",
        "Surrounding Code Context": "bool has_cycle(struct Node* head) { struct Node *slow = head, *fast = head; while (fast && fast->next) { slow = slow->next; fast = fast->next->next; if (slow == fast) return true; } return false; }",
        "Class": "Useful"
    },
    
    # Not Useful comments
    {
        "Comments": "// Increment i",
        "Surrounding Code Context": "for (int i = 0; i < n; i++) { printf(\"%d \", arr[i]); }",
        "Class": "Not Useful"
    },
    {
        "Comments": "// Return the sum",
        "Surrounding Code Context": "int add(int a, int b) { return a + b; }",
        "Class": "Not Useful"
    },
    {
        "Comments": "// This function prints hello",
        "Surrounding Code Context": "void print_hello() { printf(\"Hello, World!\\n\"); }",
        "Class": "Not Useful"
    },
    {
        "Comments": "// Check if x equals 0",
        "Surrounding Code Context": "if (x == 0) { printf(\"Zero\\n\"); } else { printf(\"Non-zero\\n\"); }",
        "Class": "Not Useful"
    },
    {
        "Comments": "// Loop through the array",
        "Surrounding Code Context": "for (int j = 0; j < length; j++) { sum += numbers[j]; }",
        "Class": "Not Useful"
    }
]

def generate_more_samples():
    """Generate additional synthetic samples by varying existing ones"""
    additional_samples = []
    
    # Add more useful comments
    useful_samples = [
        {
            "Comments": "// Quick sort partition function - places pivot in correct position",
            "Surrounding Code Context": "int partition(int arr[], int low, int high) { int pivot = arr[high]; int i = low - 1; for (int j = low; j < high; j++) { if (arr[j] < pivot) { i++; swap(&arr[i], &arr[j]); } } swap(&arr[i + 1], &arr[high]); return i + 1; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Thread-safe singleton implementation with double-checked locking",
            "Surrounding Code Context": "static Instance* getInstance() { if (instance == NULL) { pthread_mutex_lock(&mutex); if (instance == NULL) { instance = malloc(sizeof(Instance)); } pthread_mutex_unlock(&mutex); } return instance; }",
            "Class": "Useful"
        },
        {
            "Comments": "// Calculate CRC32 checksum for data integrity verification",
            "Surrounding Code Context": "uint32_t crc32(const uint8_t *data, size_t length) { uint32_t crc = 0xFFFFFFFF; for (size_t i = 0; i < length; i++) { crc ^= data[i]; for (int j = 0; j < 8; j++) { crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0); } } return ~crc; }",
            "Class": "Useful"
        }
    ]
    
    # Add more not useful comments
    not_useful_samples = [
        {
            "Comments": "// Set x to 1",
            "Surrounding Code Context": "int x = 1;",
            "Class": "Not Useful"
        },
        {
            "Comments": "// Call the function",
            "Surrounding Code Context": "result = calculate_total(values, count);",
            "Class": "Not Useful"
        },
        {
            "Comments": "// Close the file",
            "Surrounding Code Context": "fclose(file);",
            "Class": "Not Useful"
        }
    ]
    
    return useful_samples + not_useful_samples

def save_synthetic_data():
    """Save synthetic data to CSV file"""
    # Combine base samples with additional ones
    all_samples = synthetic_data + generate_more_samples()
    
    # Shuffle the data
    random.shuffle(all_samples)
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save to CSV
    output_file = os.path.join(data_dir, "synthetic_test_data.csv")
    
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
    print("🤖 Generating synthetic C code-comment pairs...")
    save_synthetic_data()
    print("✨ Ready for testing!")