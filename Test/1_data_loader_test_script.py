import os
import sys
import json

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.custom_dataloader import CustomDataLoader

def test_data_loader():
    data_path = "Data/funda_scrapped_amsterdam_sample.json"
    loader = CustomDataLoader(data_path)
    
    # First test: Can we load a single item?
    try:
        print("Testing load_all()...")
        all_data = loader.load_all()
        print(f"✓ Successfully loaded {len(all_data)} items")
        print(f"First item keys: {list(all_data[0].keys())}")
    except Exception as e:
        print(f"✗ Error in load_all(): {e}")
    
    # Second test: Can we stream data?
    try:
        print("\nTesting stream_data()...")
        batch_size = 5
        batch_count = 0
        total_items = 0
        
        for batch in loader.stream_data(batch_size):
            batch_count += 1
            total_items += len(batch)
            print(f"Batch {batch_count}: {len(batch)} items")
        
        print(f"✓ Successfully streamed {total_items} items in {batch_count} batches")
    except Exception as e:
        print(f"✗ Error in stream_data(): {e}")

if __name__ == "__main__":
    test_data_loader()