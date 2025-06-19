#!/usr/bin/env python3
"""
Extract image embeddings for all properties in the dataset.
Includes resume capability and progress tracking.
"""

import yaml
from pathlib import Path
from tqdm import tqdm
import sys
import time

# Path setup for nested project structure
script_dir = Path(__file__).parent
pipeline_root = script_dir.parent  # Real_Estate_Vision_Pipeline folder
project_root = pipeline_root.parent  # Main project folder
sys.path.append(str(project_root))

from Real_Estate_Vision_Pipeline.src.embeddings.extractor import EmbeddingExtractor
from Real_Estate_Vision_Pipeline.src.embeddings.manager import EmbeddingManager
from Real_Estate_Vision_Pipeline.src.utils.data_utils import get_all_properties, load_property_images

def main():
    # Load configuration using path relative to pipeline
    config_path = pipeline_root / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    print("Initializing BLIP model...")
    extractor = EmbeddingExtractor(device=config['model']['device'])
    
    # Create embeddings directory path relative to pipeline
    embeddings_dir = pipeline_root / config['embeddings']['image_embeddings_dir']
    manager = EmbeddingManager(str(embeddings_dir))
    
    # Get all properties from main project directory
    base_path = project_root / config['data']['base_path']
    all_properties = get_all_properties(base_path)
    
    # Get already completed properties
    completed = set(manager.list_completed_properties())
    remaining = [p for p in all_properties if p.name not in completed]
    
    print(f"\nDataset Status:")
    print(f"Total properties: {len(all_properties)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    
    if not remaining:
        print("\nAll properties already processed!")
        return
    
    # Process remaining properties
    print(f"\nProcessing {len(remaining)} properties...")
    start_time = time.time()
    processed_count = 0
    error_count = 0
    
    for prop_path in tqdm(remaining, desc="Extracting embeddings"):
        property_name = prop_path.name
        
        try:
            # Load property data
            metadata, image_paths = load_property_images(
                prop_path, 
                config['data']['resolution']
            )
            
            if not image_paths:
                print(f"\n⚠️  No images found for {property_name}, skipping...")
                continue
            
            # Extract embeddings
            embeddings = extractor.extract_image_embeddings(
                image_paths, 
                batch_size=config['data']['batch_size']
            )
            
            # Save embeddings
            save_path = manager.save_property_embeddings(
                property_name, 
                embeddings, 
                image_paths, 
                metadata
            )
            
            processed_count += 1
            
            # Print progress every 10 properties
            if processed_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                eta = (len(remaining) - processed_count) / rate
                print(f"\n✓ Processed {processed_count}/{len(remaining)} properties "
                      f"({rate:.1f} props/min, ETA: {eta/60:.1f} min)")
            
        except Exception as e:
            error_count += 1
            print(f"\n❌ Error processing {property_name}: {str(e)}")
            continue
    
    # Final statistics
    elapsed_total = time.time() - start_time
    final_stats = manager.get_embedding_stats()
    
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print(f"Time elapsed: {elapsed_total/60:.1f} minutes")
    print(f"Properties processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"\nFinal Statistics:")
    print(f"Total properties with embeddings: {final_stats['total_properties']}")
    print(f"Total image embeddings: {final_stats['total_embeddings']}")
    print(f"Embedding dimension: {final_stats['embedding_dimension']}")
    print(f"\nEmbeddings saved to: {embeddings_dir}")

if __name__ == "__main__":
    main()