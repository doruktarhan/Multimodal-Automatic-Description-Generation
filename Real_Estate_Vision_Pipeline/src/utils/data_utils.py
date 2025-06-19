"""
Data utilities for loading and processing property images and metadata
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_property_name(name: str) -> str:
    """
    Sanitize property name for filesystem usage
    
    Args:
        name: Original property name
        
    Returns:
        Sanitized name with spaces replaced by underscores
    """
    return name.replace(" ", "_")


def get_all_properties(base_path: Path) -> List[Path]:
    """
    Get all property directories that contain metadata.json
    
    Args:
        base_path: Base directory containing all properties
        
    Returns:
        List of Path objects for each valid property directory
    """
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        return []
    
    properties = []
    
    # Iterate through all directories in base path
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if metadata.json exists
            metadata_file = item / "metadata.json"
            if metadata_file.exists():
                properties.append(item)
            else:
                logger.debug(f"Skipping {item.name} - no metadata.json found")
    
    # Sort properties by name for consistent ordering
    properties.sort(key=lambda x: x.name)
    
    logger.info(f"Found {len(properties)} valid properties in {base_path}")
    return properties


def load_property_metadata(property_path: Path) -> Optional[Dict]:
    """
    Load metadata.json for a property
    
    Args:
        property_path: Path to property directory
        
    Returns:
        Dictionary containing metadata or None if not found
    """
    metadata_file = property_path / "metadata.json"
    
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return None
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing metadata.json for {property_path.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading metadata for {property_path.name}: {e}")
        return None


def load_property_images(property_path: Path, resolution: str = "360x240") -> Tuple[Dict, List[str]]:
    """
    Load property metadata and construct image paths for specified resolution
    
    Args:
        property_path: Path to property folder
        resolution: Image resolution to use (e.g., "360x240", "720x480")
        
    Returns:
        Tuple of (metadata dict, list of image paths)
    """
    # Load metadata
    metadata = load_property_metadata(property_path)
    if metadata is None:
        return {}, []
    
    # Get image directory for specified resolution
    image_dir = property_path / resolution
    image_paths = []
    
    if not image_dir.exists():
        logger.warning(f"Resolution directory not found: {image_dir}")
        return metadata, []
    
    # Find all jpg images
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg']])
    
    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return metadata, []
    
    # Convert to string paths
    image_paths = [str(f) for f in image_files]
    
    logger.debug(f"Found {len(image_paths)} images for {property_path.name} at {resolution}")
    
    return metadata, image_paths


def get_available_resolutions(property_path: Path) -> List[str]:
    """
    Get all available resolutions for a property
    
    Args:
        property_path: Path to property directory
        
    Returns:
        List of available resolutions (e.g., ["360x240", "720x480"])
    """
    resolutions = []
    
    # Check common resolution patterns
    for item in property_path.iterdir():
        if item.is_dir() and 'x' in item.name:
            # Check if it contains images
            has_images = any(f.suffix.lower() in ['.jpg', '.jpeg'] for f in item.iterdir())
            if has_images:
                resolutions.append(item.name)
    
    return sorted(resolutions)


def count_property_images(property_path: Path, resolution: Optional[str] = None) -> Dict[str, int]:
    """
    Count images in a property across resolutions
    
    Args:
        property_path: Path to property directory
        resolution: Specific resolution to count, or None for all
        
    Returns:
        Dictionary mapping resolution to image count
    """
    counts = {}
    
    if resolution:
        # Count for specific resolution
        _, images = load_property_images(property_path, resolution)
        counts[resolution] = len(images)
    else:
        # Count for all resolutions
        resolutions = get_available_resolutions(property_path)
        for res in resolutions:
            _, images = load_property_images(property_path, res)
            counts[res] = len(images)
    
    return counts


def validate_property_structure(property_path: Path) -> Dict[str, bool]:
    """
    Validate that a property has the expected structure
    
    Args:
        property_path: Path to property directory
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'exists': property_path.exists(),
        'is_directory': property_path.is_dir() if property_path.exists() else False,
        'has_metadata': False,
        'has_images': False,
        'valid_metadata': False,
        'resolutions_found': []
    }
    
    if not validation['exists'] or not validation['is_directory']:
        return validation
    
    # Check metadata
    metadata_file = property_path / "metadata.json"
    validation['has_metadata'] = metadata_file.exists()
    
    if validation['has_metadata']:
        metadata = load_property_metadata(property_path)
        validation['valid_metadata'] = metadata is not None
    
    # Check for images
    resolutions = get_available_resolutions(property_path)
    validation['resolutions_found'] = resolutions
    validation['has_images'] = len(resolutions) > 0
    
    return validation


def get_property_info(property_path: Path) -> Dict:
    """
    Get comprehensive information about a property
    
    Args:
        property_path: Path to property directory
        
    Returns:
        Dictionary containing property information
    """
    info = {
        'name': property_path.name,
        'path': str(property_path),
        'validation': validate_property_structure(property_path),
        'metadata': None,
        'image_counts': {},
        'total_images': 0
    }
    
    if info['validation']['valid_metadata']:
        info['metadata'] = load_property_metadata(property_path)
    
    if info['validation']['has_images']:
        info['image_counts'] = count_property_images(property_path)
        info['total_images'] = sum(info['image_counts'].values())
    
    return info


def batch_load_properties(base_path: Path, property_names: List[str], 
                         resolution: str = "360x240") -> Dict[str, Tuple[Dict, List[str]]]:
    """
    Load multiple properties at once
    
    Args:
        base_path: Base directory containing properties
        property_names: List of property names to load
        resolution: Image resolution to use
        
    Returns:
        Dictionary mapping property name to (metadata, image_paths) tuple
    """
    results = {}
    
    for name in property_names:
        sanitized_name = sanitize_property_name(name)
        property_path = base_path / sanitized_name
        
        if property_path.exists():
            metadata, image_paths = load_property_images(property_path, resolution)
            results[name] = (metadata, image_paths)
        else:
            logger.warning(f"Property not found: {name} (looked for {sanitized_name})")
            results[name] = ({}, [])
    
    return results


def filter_properties_by_image_count(properties: List[Path], 
                                   min_images: int = 1, 
                                   max_images: Optional[int] = None,
                                   resolution: str = "360x240") -> List[Path]:
    """
    Filter properties based on image count
    
    Args:
        properties: List of property paths
        min_images: Minimum number of images required
        max_images: Maximum number of images allowed (None for no limit)
        resolution: Resolution to check
        
    Returns:
        Filtered list of property paths
    """
    filtered = []
    
    for prop_path in properties:
        counts = count_property_images(prop_path, resolution)
        image_count = counts.get(resolution, 0)
        
        if image_count >= min_images:
            if max_images is None or image_count <= max_images:
                filtered.append(prop_path)
            else:
                logger.debug(f"Skipping {prop_path.name}: {image_count} images exceeds max {max_images}")
        else:
            logger.debug(f"Skipping {prop_path.name}: {image_count} images below min {min_images}")
    
    return filtered


def get_property_statistics(base_path: Path) -> Dict:
    """
    Get statistics about all properties in the dataset
    
    Args:
        base_path: Base directory containing properties
        
    Returns:
        Dictionary with dataset statistics
    """
    properties = get_all_properties(base_path)
    
    stats = {
        'total_properties': len(properties),
        'total_images': 0,
        'images_by_resolution': {},
        'properties_by_image_count': {
            '0-10': 0,
            '11-25': 0,
            '26-50': 0,
            '51-100': 0,
            '100+': 0
        },
        'missing_metadata': 0,
        'resolutions_available': set()
    }
    
    for prop_path in properties:
        info = get_property_info(prop_path)
        
        # Update totals
        stats['total_images'] += info['total_images']
        
        # Update resolution counts
        for res, count in info['image_counts'].items():
            stats['resolutions_available'].add(res)
            if res not in stats['images_by_resolution']:
                stats['images_by_resolution'][res] = 0
            stats['images_by_resolution'][res] += count
        
        # Categorize by image count
        total = info['total_images']
        if total == 0:
            stats['properties_by_image_count']['0-10'] += 1
        elif total <= 10:
            stats['properties_by_image_count']['0-10'] += 1
        elif total <= 25:
            stats['properties_by_image_count']['11-25'] += 1
        elif total <= 50:
            stats['properties_by_image_count']['26-50'] += 1
        elif total <= 100:
            stats['properties_by_image_count']['51-100'] += 1
        else:
            stats['properties_by_image_count']['100+'] += 1
        
        # Check metadata
        if not info['validation']['valid_metadata']:
            stats['missing_metadata'] += 1
    
    # Convert set to sorted list
    stats['resolutions_available'] = sorted(list(stats['resolutions_available']))
    
    return stats


# Example usage function for testing
def test_data_utils():
    """Test function to verify data utilities work correctly"""
    base_path = Path("funda_images")
    
    # Test getting all properties
    properties = get_all_properties(base_path)
    print(f"Found {len(properties)} properties")
    
    if properties:
        # Test loading a single property
        test_property = properties[0]
        print(f"\nTesting with property: {test_property.name}")
        
        # Get property info
        info = get_property_info(test_property)
        print(f"Property info: {json.dumps(info, indent=2, default=str)}")
        
        # Load images
        metadata, image_paths = load_property_images(test_property)
        print(f"\nLoaded {len(image_paths)} images")
        
        if image_paths:
            print(f"First image: {image_paths[0]}")
            print(f"Last image: {image_paths[-1]}")
    
    # Get dataset statistics
    print("\nDataset statistics:")
    stats = get_property_statistics(base_path)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    test_data_utils()