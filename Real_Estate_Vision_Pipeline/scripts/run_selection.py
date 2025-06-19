#!/usr/bin/env python3
"""
Run selection pipeline on properties with classification results
Selects best 15 images per property using hierarchical algorithm
"""


import yaml
import json
from pathlib import Path
from typing import List
from datetime import datetime
from tqdm import tqdm
import sys
import time
import argparse
import numpy as np

# Path setup
script_dir = Path(__file__).parent
pipeline_root = script_dir.parent
project_root = pipeline_root.parent
sys.path.append(str(project_root))

from Real_Estate_Vision_Pipeline.src.embeddings.extractor import EmbeddingExtractor
from Real_Estate_Vision_Pipeline.src.embeddings.manager import EmbeddingManager
from Real_Estate_Vision_Pipeline.src.selection.selector import ImageSelector

def load_sample_properties(sample_file_path: Path) -> List[str]:
    """Load property names from sample dataset"""
    with open(sample_file_path, 'r') as f:
        sample_data = json.load(f)
    
    properties = []
    for bin_data in sample_data['bins'].values():
        for prop in bin_data['properties']:
            properties.append(prop['name'])
    
    return properties

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run selection pipeline')
    parser.add_argument('--classification-results', type=str, required=True,
                       help='Path to classification results JSON file')
    parser.add_argument('--sample-only', action='store_true',
                       help='Process only the 35 sample properties')
    parser.add_argument('--specific-property', type=str,
                       help='Process only a specific property')
    parser.add_argument('--max-properties', type=int,
                       help='Maximum number of properties to process')
    args = parser.parse_args()
    
    # Load configurations
    config_path = pipeline_root / 'configs' / 'config.yaml'
    selection_captions_path = pipeline_root / 'configs' / 'captions_selection.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(selection_captions_path, 'r') as f:
        selection_captions_config = yaml.safe_load(f)
    
    # Add selection parameters to config if not present
    if 'selection' not in config:
        config['selection'] = {
            'alpha': 0.3,
            'beta': 0.7,
            'aggregation': 'avg'
        }
    
    # Load classification results
    classification_file = Path(args.classification_results)
    if not classification_file.exists():
        print(f"Error: Classification results file not found: {classification_file}")
        return
    
    with open(classification_file, 'r') as f:
        all_classification_results = json.load(f)
    
    # Determine which properties to process
    if args.specific_property:
        if args.specific_property not in all_classification_results:
            print(f"Error: Property '{args.specific_property}' not found in classification results")
            return
        properties_to_process = [args.specific_property]
    elif args.sample_only:
        # Load sample properties
        sample_file = pipeline_root / 'data' / 'results' / 'classification' / 'sample_dataset_15_25_40_60.json'
        sample_properties = load_sample_properties(sample_file)
        properties_to_process = [p for p in sample_properties if p in all_classification_results]
        print(f"Processing {len(properties_to_process)} sample properties")
    else:
        properties_to_process = list(all_classification_results.keys())
        if args.max_properties:
            properties_to_process = properties_to_process[:args.max_properties]
    
    # Initialize components
    print("="*60)
    print("SELECTION PIPELINE")
    print("="*60)
    print(f"Classification results: {classification_file.name}")
    print(f"Properties to process: {len(properties_to_process)}")
    print(f"Selection weights: α={config['selection']['alpha']}, β={config['selection']['beta']}")
    print("="*60)
    
    print("\nInitializing components...")
    extractor = EmbeddingExtractor(device=config['model']['device'])
    
    # Image embeddings manager
    embeddings_dir = pipeline_root / config['embeddings']['image_embeddings_dir']
    manager = EmbeddingManager(str(embeddings_dir))
    
    # Initialize selector
    selector = ImageSelector(config, selection_captions_config['selection_captions'])
    
    # Extract text embeddings for selection captions
    print("\nExtracting selection text embeddings...")
    selection_text_embeddings = {}
    
    for room_type, captions in tqdm(selection_captions_config['selection_captions'].items(), 
                                   desc="Processing room types"):
        embeddings = extractor.extract_text_embeddings(captions)
        selection_text_embeddings[room_type] = embeddings
    
    print(f"Extracted embeddings for {len(selection_text_embeddings)} room types")
    
    # Process properties
    results = {}
    start_time = time.time()
    successful = 0
    failed = 0
    
    print("\nSelecting images for properties...")
    for property_name in tqdm(properties_to_process, desc="Processing"):
        try:
            # Load image embeddings
            embeddings_data = manager.load_property_embeddings(property_name)
            if embeddings_data is None:
                print(f"\n⚠️  No image embeddings found for {property_name}, skipping...")
                failed += 1
                continue
            
            # Get classification results
            classification_results = all_classification_results[property_name]
            
            # Select images
            selection_results = selector.select_images(
                property_name,
                embeddings_data['embeddings'],
                selection_text_embeddings,
                classification_results
            )
            
            results[property_name] = selection_results
            successful += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {property_name}: {str(e)}")
            failed += 1
            continue
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    
    # Generate summary
    if results:
        # Calculate summary statistics
        summary = {
            'total_properties': len(results),
            'processing_time_seconds': elapsed_time,
            'successful_properties': successful,
            'failed_properties': failed,
            'selection_statistics': {
                'images_selected_per_property': {
                    'mean': np.mean([r['num_selected'] for r in results.values()]),
                    'std': np.std([r['num_selected'] for r in results.values()]),
                    'min': min(r['num_selected'] for r in results.values()),
                    'max': max(r['num_selected'] for r in results.values())
                },
                'room_coverage': {}
            }
        }
        
        # Calculate room coverage statistics
        all_selected_rooms = []
        for result in results.values():
            all_selected_rooms.extend(result['selected_predictions'])
        
        from collections import Counter
        room_counts = Counter(all_selected_rooms)
        summary['selection_statistics']['room_coverage'] = dict(room_counts)
        
        # Save results
        output_dir = pipeline_root / 'data' / 'results' / 'selection'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine output filename
        if args.specific_property:
            output_file = output_dir / f"selection_{args.specific_property}_{timestamp}.json"
        elif args.sample_only:
            output_file = output_dir / f"selection_sample_{timestamp}.json"
        else:
            output_file = output_dir / f"selection_results_{timestamp}.json"
        
        # Save main results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_file = output_dir / f"selection_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print("\n" + "="*60)
        print("SELECTION COMPLETE")
        print("="*60)
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print(f"Properties processed: {successful}")
        print(f"Properties failed: {failed}")
        print(f"\nResults saved to:")
        print(f"  - {output_file.name}")
        print(f"  - {summary_file.name}")
        
        # Print summary statistics
        print("\n" + "-"*40)
        print("SELECTION STATISTICS")
        print("-"*40)
        print(f"Average images selected: {summary['selection_statistics']['images_selected_per_property']['mean']:.1f}")
        print(f"Selection range: {summary['selection_statistics']['images_selected_per_property']['min']}-"
              f"{summary['selection_statistics']['images_selected_per_property']['max']} images")
        
        print("\nRoom type distribution in selection:")
        total_selected = sum(room_counts.values())
        for room, count in sorted(room_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_selected * 100) if total_selected > 0 else 0
            print(f"  {room:15s}: {count:4d} images ({percentage:5.1f}%)")
    
    else:
        print("\n❌ No properties were successfully processed!")

if __name__ == "__main__":
    main()