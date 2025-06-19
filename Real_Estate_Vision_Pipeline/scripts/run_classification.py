#!/usr/bin/env python3
"""
Run full classification pipeline on all properties
Supports test mode and batch processing
"""

import yaml
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
import time
import argparse

# Path setup
script_dir = Path(__file__).parent
pipeline_root = script_dir.parent
project_root = pipeline_root.parent
sys.path.append(str(project_root))

from Real_Estate_Vision_Pipeline.src.embeddings.extractor import EmbeddingExtractor
from Real_Estate_Vision_Pipeline.src.embeddings.manager import EmbeddingManager
from Real_Estate_Vision_Pipeline.src.classification.classifier import RoomClassifier
from Real_Estate_Vision_Pipeline.src.utils.data_utils import get_all_properties

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run classification pipeline')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode with limited properties')
    parser.add_argument('--test-properties', type=int, default=5,
                       help='Number of properties to process in test mode')
    parser.add_argument('--specific-property', type=str,
                       help='Process only a specific property')
    args = parser.parse_args()
    
    # Load configurations
    config_path = pipeline_root / 'configs' / 'config.yaml'
    captions_path = pipeline_root / 'configs' / 'captions_classification.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(captions_path, 'r') as f:
        captions_config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.test_mode:
        config['classification']['test_mode'] = True
        config['classification']['test_properties'] = args.test_properties
    
    # Initialize components
    print("="*60)
    print("CLASSIFICATION PIPELINE")
    print("="*60)
    print(f"Model: {config['model']['type']}")
    print(f"Caption style: {config['classification']['caption_style']}")
    print(f"Aggregation: {config['classification']['aggregation']}")
    print(f"Test mode: {config['classification']['test_mode']}")
    print("="*60)
    
    print("\nInitializing components...")
    extractor = EmbeddingExtractor(device=config['model']['device'])
    
    # Image embeddings manager
    embeddings_dir = pipeline_root / config['embeddings']['image_embeddings_dir']
    manager = EmbeddingManager(str(embeddings_dir))
    
    # Initialize classifier
    classifier = RoomClassifier(config, captions_config)
    
    # Get text embeddings (computed once for this run)
    print("\nPreparing text embeddings...")
    text_embeddings = classifier.get_text_embeddings(extractor, manager)
    print(f"Text embeddings ready for {len(text_embeddings)} room types")
    
    # Get properties to process
    base_path = project_root / config['data']['base_path']
    
    if args.specific_property:
        # Process only specific property
        property_path = base_path / args.specific_property
        if not property_path.exists():
            print(f"Error: Property '{args.specific_property}' not found!")
            return
        all_properties = [property_path]
        print(f"\nProcessing specific property: {args.specific_property}")
    else:
        # Get all properties
        all_properties = get_all_properties(base_path)
        
        # Apply test mode filter
        if config['classification']['test_mode']:
            all_properties = all_properties[:config['classification']['test_properties']]
            print(f"\nTest mode: Processing first {len(all_properties)} properties")
        else:
            print(f"\nProcessing all {len(all_properties)} properties")
    
    # Process properties
    results = {}
    start_time = time.time()
    successful = 0
    failed = 0
    
    print("\nClassifying properties...")
    for prop_path in tqdm(all_properties, desc="Processing"):
        property_name = prop_path.name
        
        try:
            # Load embeddings
            property_data = manager.load_property_embeddings(property_name)
            if property_data is None:
                print(f"\n⚠️  No embeddings found for {property_name}, skipping...")
                failed += 1
                continue
            
            # Classify property
            classification_results = classifier.classify_property(
                property_data['embeddings'],
                text_embeddings,
                property_data['image_paths']
            )
            
            # Add metadata
            classification_results['property_name'] = property_name
            classification_results['timestamp'] = datetime.now().isoformat()
            classification_results['config'] = {
                'model_type': config['model']['type'],
                'caption_style': config['classification']['caption_style'],
                'aggregation': config['classification']['aggregation'],
                'num_room_types': len(classifier.room_types)
            }
            
            results[property_name] = classification_results
            successful += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {property_name}: {str(e)}")
            failed += 1
            continue
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    
    # Generate summary
    if results:
        summary = classifier.get_classification_summary(results)
        
        # Save results
        output_dir = pipeline_root / config['output']['classification_results']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        if args.specific_property:
            output_file = output_dir / f"classification_{args.specific_property}_{timestamp}.json"
        else:
            output_file = output_dir / f"classification_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary separately
        summary_file = output_dir / f"classification_summary_{timestamp}.json"
        summary['processing_time_seconds'] = elapsed_time
        summary['successful_properties'] = successful
        summary['failed_properties'] = failed
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print("\n" + "="*60)
        print("CLASSIFICATION COMPLETE")
        print("="*60)
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print(f"Properties processed: {successful}")
        print(f"Properties failed: {failed}")
        print(f"\nResults saved to:")
        print(f"  - {output_file.name}")
        print(f"  - {summary_file.name}")
        
        # Print summary statistics
        print("\n" + "-"*40)
        print("SUMMARY STATISTICS")
        print("-"*40)
        print(f"Total images classified: {summary['total_images']:,}")
        print(f"Average images per property: {summary['images_per_property']['mean']:.1f}")
        print(f"Overall confidence: {summary['confidence_statistics']['mean']:.3f} "
              f"(±{summary['confidence_statistics']['std']:.3f})")
        
        print("\nRoom distribution across all properties:")
        room_dist = summary['overall_room_distribution']
        total_images = summary['total_images']
        for room, count in sorted(room_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"  {room:15s}: {count:5d} images ({percentage:5.1f}%)")
    
    else:
        print("\n❌ No properties were successfully processed!")
    
    # Cleanup old text embeddings (keep last 5)
    print("\nCleaning up old text embeddings...")
    manager.cleanup_old_text_embeddings(keep_last_n=5)

if __name__ == "__main__":
    main()