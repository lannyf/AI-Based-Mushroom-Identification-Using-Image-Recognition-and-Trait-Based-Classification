"""
Data utilities for loading, validating, and manipulating mushroom identification datasets.
"""

import pandas as pd
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def load_species_traits_xml(path: Path) -> pd.DataFrame:
    """
    Parse species_traits.xml and return a flat DataFrame with the same
    columns as the original species_traits.csv:
      species_id, trait_category, trait_name, trait_value, value_type, variability
    """
    tree = ET.parse(path)
    rows: List[Dict[str, str]] = []
    for species_el in tree.findall("species"):
        species_id = species_el.get("id", "")
        for grp_el in species_el.findall("trait_group"):
            category = grp_el.get("category", "")
            for trait_el in grp_el.findall("trait"):
                rows.append({
                    "species_id":     species_id,
                    "trait_category": category,
                    "trait_name":     trait_el.get("name", ""),
                    "trait_value":    trait_el.text or "",
                    "value_type":     trait_el.get("value_type", ""),
                    "variability":    trait_el.get("variability", ""),
                })
    return pd.DataFrame(rows)


class MushroomDataset:
    """Main dataset class for mushroom identification system."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the mushroom dataset.
        
        Args:
            data_dir: Path to the raw data directory
        """
        self.data_dir = Path(data_dir)
        self.species_df = None
        self.traits_df = None
        self.images_df = None
        self.lookalikes_df = None
        self.split_df = None
        
    def load_all(self) -> None:
        """Load all CSV files from the data directory."""
        self.species_df = pd.read_csv(self.data_dir / "species.csv")
        self.traits_df = load_species_traits_xml(self.data_dir / "species_traits.xml")
        self.images_df = pd.read_csv(self.data_dir / "species_images.csv")
        self.lookalikes_df = pd.read_csv(self.data_dir / "lookalikes.csv")
        self.split_df = pd.read_csv(self.data_dir / "dataset_split.csv")
        
    def get_species_info(self, species_id: str) -> Dict:
        """
        Get detailed information for a single species.
        
        Args:
            species_id: The species ID (e.g., 'CH001')
            
        Returns:
            Dictionary with species information
        """
        self.load_all()
        species = self.species_df[self.species_df['species_id'] == species_id]
        
        if species.empty:
            raise ValueError(f"Species {species_id} not found")
        
        species_info = species.iloc[0].to_dict()
        
        # Add traits
        traits = self.traits_df[self.traits_df['species_id'] == species_id]
        species_info['traits'] = traits.to_dict('records')
        
        # Add images
        images = self.images_df[self.images_df['species_id'] == species_id]
        species_info['images'] = images.to_dict('records')
        
        # Add lookalikes
        lookalikes = self.lookalikes_df[
            (self.lookalikes_df['edible_species_id'] == species_id) |
            (self.lookalikes_df['toxic_species_id'] == species_id)
        ]
        species_info['lookalikes'] = lookalikes.to_dict('records')
        
        return species_info
    
    def get_species_by_name(self, name: str, language: str = 'en') -> Optional[str]:
        """
        Find species ID by common name.
        
        Args:
            name: Common name of species
            language: 'en' for English, 'sv' for Swedish
            
        Returns:
            Species ID or None if not found
        """
        self.load_all()
        column = 'english_name' if language == 'en' else 'swedish_name'
        match = self.species_df[self.species_df[column].str.lower() == name.lower()]
        return match.iloc[0]['species_id'] if not match.empty else None
    
    def get_traits_for_species(self, species_id: str) -> pd.DataFrame:
        """Get all traits for a species."""
        self.load_all()
        return self.traits_df[self.traits_df['species_id'] == species_id]
    
    def get_images_for_species(self, species_id: str, 
                              suitable_only: bool = True) -> pd.DataFrame:
        """
        Get images for a species.
        
        Args:
            species_id: Species ID
            suitable_only: If True, only return suitable_for_training=TRUE
            
        Returns:
            DataFrame with image metadata
        """
        self.load_all()
        images = self.images_df[self.images_df['species_id'] == species_id]
        if suitable_only:
            images = images[images['suitable_for_training'] == True]
        return images
    
    def get_dangerous_lookalikes(self, species_id: str) -> List[Dict]:
        """
        Get dangerous lookalikes for a species.
        
        Args:
            species_id: Species ID
            
        Returns:
            List of lookalike dictionaries
        """
        self.load_all()
        lookalikes = self.lookalikes_df[
            self.lookalikes_df['edible_species_id'] == species_id
        ]
        return lookalikes.to_dict('records')
    
    def get_species_by_split(self, split_set: str) -> List[str]:
        """
        Get all species in a given split set (TRAIN, VALIDATION, TEST).
        
        Args:
            split_set: One of 'TRAIN', 'VALIDATION', 'TEST'
            
        Returns:
            List of unique species IDs
        """
        self.load_all()
        species = self.split_df[self.split_df['split_set'] == split_set]['species_id'].unique()
        return list(species)
    
    def get_images_by_split(self, split_set: str) -> pd.DataFrame:
        """Get all images in a given split set."""
        self.load_all()
        return self.split_df[self.split_df['split_set'] == split_set]
    
    def get_edible_species(self) -> pd.DataFrame:
        """Get all edible species."""
        self.load_all()
        return self.species_df[self.species_df['edible'] == True]
    
    def get_toxic_species(self) -> pd.DataFrame:
        """Get all toxic species."""
        self.load_all()
        return self.species_df[self.species_df['edible'] == False]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        self.load_all()
        
        stats = {
            'total_species': len(self.species_df),
            'edible_species': len(self.get_edible_species()),
            'toxic_species': len(self.get_toxic_species()),
            'total_images': len(self.images_df),
            'total_suitable_images': len(self.images_df[self.images_df['suitable_for_training'] == True]),
            'train_count': len(self.split_df[self.split_df['split_set'] == 'TRAIN']),
            'validation_count': len(self.split_df[self.split_df['split_set'] == 'VALIDATION']),
            'test_count': len(self.split_df[self.split_df['split_set'] == 'TEST']),
            'lookalike_pairs': len(self.lookalikes_df),
            'images_per_species': self.images_df.groupby('species_id').size().to_dict(),
        }
        
        return stats


class DataValidator:
    """Validate dataset integrity and quality."""
    
    def __init__(self, dataset: MushroomDataset):
        """Initialize validator with a dataset."""
        self.dataset = dataset
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks.
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        self._validate_species_completeness()
        self._validate_trait_coverage()
        self._validate_images_exist()
        self._validate_lookalikes()
        self._validate_splits()
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_species_completeness(self) -> None:
        """Check that all required species columns are present."""
        required_cols = ['species_id', 'scientific_name', 'swedish_name', 'english_name', 'edible', 'toxicity_level']
        missing = [col for col in required_cols if col not in self.dataset.species_df.columns]
        
        if missing:
            self.errors.append(f"Missing columns in species.csv: {missing}")
        
        # Check for duplicate species IDs
        if self.dataset.species_df['species_id'].duplicated().any():
            self.errors.append("Duplicate species IDs found in species.csv")
        
        # Check for null species IDs
        if self.dataset.species_df['species_id'].isna().any():
            self.errors.append("NULL species IDs found in species.csv")
    
    def _validate_trait_coverage(self) -> None:
        """Check that species have adequate trait coverage."""
        for species_id in self.dataset.species_df['species_id']:
            traits = self.dataset.traits_df[self.dataset.traits_df['species_id'] == species_id]
            if traits.empty:
                self.warnings.append(f"Species {species_id} has no trait data")
            elif len(traits) < 10:
                self.warnings.append(f"Species {species_id} has only {len(traits)} traits (target: 15+)")
    
    def _validate_images_exist(self) -> None:
        """Check that image files exist."""
        missing_files = []
        for idx, row in self.dataset.images_df.iterrows():
            image_path = self.dataset.data_dir / row['file_path']
            if not image_path.exists():
                missing_files.append(row['file_path'])
        
        if missing_files:
            self.warnings.append(f"Missing {len(missing_files)} image files (expected when dataset not fully populated)")
    
    def _validate_lookalikes(self) -> None:
        """Check that lookalike references point to valid species."""
        species_ids = set(self.dataset.species_df['species_id'])
        
        for idx, row in self.dataset.lookalikes_df.iterrows():
            if row['edible_species_id'] not in species_ids:
                self.errors.append(f"Lookalike references invalid species: {row['edible_species_id']}")
            if row['toxic_species_id'] not in species_ids:
                self.errors.append(f"Lookalike references invalid species: {row['toxic_species_id']}")
    
    def _validate_splits(self) -> None:
        """Check that splits are properly distributed."""
        total = len(self.dataset.split_df)
        train = len(self.dataset.split_df[self.dataset.split_df['split_set'] == 'TRAIN'])
        val = len(self.dataset.split_df[self.dataset.split_df['split_set'] == 'VALIDATION'])
        test = len(self.dataset.split_df[self.dataset.split_df['split_set'] == 'TEST'])
        
        if train + val + test != total:
            self.errors.append("Split sets do not account for all images")
        
        # Check split distribution (should be roughly 70/15/15)
        if total > 0:
            train_pct = train / total
            val_pct = val / total
            test_pct = test / total
            
            if abs(train_pct - 0.7) > 0.1:
                self.warnings.append(f"Train set is {train_pct:.1%} (target: 70%)")
            if abs(val_pct - 0.15) > 0.05:
                self.warnings.append(f"Validation set is {val_pct:.1%} (target: 15%)")
            if abs(test_pct - 0.15) > 0.05:
                self.warnings.append(f"Test set is {test_pct:.1%} (target: 15%)")


class DataExporter:
    """Export dataset to various formats for ML models."""
    
    def __init__(self, dataset: MushroomDataset):
        """Initialize exporter with a dataset."""
        self.dataset = dataset
        self.dataset.load_all()
    
    def export_to_json(self, output_path: str) -> None:
        """
        Export dataset to JSON format.
        
        Args:
            output_path: Path to output JSON file
        """
        data = {
            'species': self.dataset.species_df.to_dict('records'),
            'traits': self.dataset.traits_df.to_dict('records'),
            'images': self.dataset.images_df.to_dict('records'),
            'lookalikes': self.dataset.lookalikes_df.to_dict('records'),
            'splits': self.dataset.split_df.to_dict('records'),
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_trait_features(self, output_path: str) -> None:
        """
        Export trait data in ML-ready format.
        
        Args:
            output_path: Path to output CSV file
        """
        # Pivot traits to wide format
        trait_features = self.dataset.traits_df.pivot_table(
            index='species_id',
            columns=['trait_category', 'trait_name'],
            values='trait_value',
            aggfunc='first'
        )
        
        trait_features.to_csv(output_path)


if __name__ == "__main__":
    # Example usage
    dataset = MushroomDataset("data/raw")
    dataset.load_all()
    
    # Print statistics
    stats = dataset.get_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")
    
    # Validate dataset
    validator = DataValidator(dataset)
    is_valid, errors, warnings = validator.validate_all()
    
    if errors:
        print("\nValidation Errors:")
        for error in errors:
            print(f"  ❌ {error}")
    
    if warnings:
        print("\nValidation Warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    if is_valid:
        print("\n✅ Dataset validation passed!")
