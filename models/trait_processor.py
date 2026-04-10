"""
Trait Processing Pipeline for Mushroom Identification

This module handles encoding, preprocessing, and feature engineering of mushroom traits.
Converts categorical trait data into numerical feature vectors suitable for ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import pickle
import logging
from pathlib import Path

from data.dataset_utils import load_species_traits_xml

logger = logging.getLogger(__name__)


class TraitEncoder:
    """
    Encodes categorical trait values into numerical features.
    
    Supports:
    - One-hot encoding for categorical traits
    - Range encoding for numerical ranges (min/max extraction)
    - Binary encoding for yes/no traits
    - Ordinal encoding for ordered categorical traits
    """
    
    def __init__(self):
        self.encoders: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
    
    def fit(self, df: pd.DataFrame) -> 'TraitEncoder':
        """
        Learn encoding from trait data.
        
        Args:
            df: DataFrame with columns [species_id, trait_category, trait_name, 
                                        trait_value, value_type, variability]
        
        Returns:
            self for method chaining
        """
        # Create trait identifier (category + name)
        df['trait_id'] = df['trait_category'] + '.' + df['trait_name']
        
        # Group by trait_id and value_type
        for trait_id, group in df.groupby('trait_id'):
            value_type = group['value_type'].iloc[0]
            
            if value_type == 'categorical':
                # One-hot encoding: store unique values
                unique_vals = group['trait_value'].unique().tolist()
                self.encoders[trait_id] = {
                    'type': 'categorical',
                    'values': unique_vals,
                    'feature_names': [f"{trait_id}_{val}" for val in unique_vals]
                }
            
            elif value_type == 'range':
                # Range encoding: store min/max
                self.encoders[trait_id] = {
                    'type': 'range',
                    'feature_names': [f"{trait_id}_min", f"{trait_id}_max"]
                }
            
            elif value_type == 'ordinal':
                # Ordinal encoding: store order
                unique_vals = group['trait_value'].unique().tolist()
                self.encoders[trait_id] = {
                    'type': 'ordinal',
                    'values': unique_vals,
                    'feature_names': [f"{trait_id}"]
                }
        
        # Build feature names list
        for encoder in self.encoders.values():
            self.feature_names.extend(encoder['feature_names'])
        
        self.is_fitted = True
        logger.info(f"TraitEncoder fitted with {len(self.encoders)} traits -> {len(self.feature_names)} features")
        return self
    
    def transform(self, traits_dict: Dict[str, str]) -> np.ndarray:
        """
        Transform a single mushroom's traits to feature vector.
        
        Args:
            traits_dict: {trait_id: trait_value} mapping
                        e.g., {'CAP.color': 'white', 'CAP.shape': 'round', ...}
        
        Returns:
            Feature vector of shape (n_features,)
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        features = []
        
        for trait_id, encoder in self.encoders.items():
            value = traits_dict.get(trait_id, None)
            
            if encoder['type'] == 'categorical':
                # One-hot encode
                one_hot = [1.0 if val == value else 0.0 
                          for val in encoder['values']]
                features.extend(one_hot)
            
            elif encoder['type'] == 'range':
                # Parse range and extract min/max
                if value and isinstance(value, str):
                    try:
                        parts = value.split('-')
                        if len(parts) == 2:
                            min_val = float(parts[0].strip())
                            max_val = float(parts[1].strip())
                        else:
                            min_val = max_val = float(value)
                    except (ValueError, IndexError):
                        min_val = max_val = 0.0
                else:
                    min_val = max_val = 0.0
                
                features.extend([min_val, max_val])
            
            elif encoder['type'] == 'ordinal':
                # Map to ordinal index
                if value in encoder['values']:
                    ordinal_val = float(encoder['values'].index(value))
                else:
                    ordinal_val = 0.0
                
                features.append(ordinal_val)
        
        return np.array(features, dtype=np.float32)
    
    def fit_transform(self, df: pd.DataFrame) -> List[str]:
        """
        Fit the encoder on the provided DataFrame and return the learned feature names.

        Args:
            df: Input DataFrame containing trait columns to be encoded.

        Returns:
            List of feature names corresponding to the encoded traits.
        """
        self.fit(df)
        return self.feature_names
    
    def save(self, path: str) -> None:
        """Save encoder to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'encoders': self.encoders,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"TraitEncoder saved to {path}")
    
    def load(self, path: str) -> 'TraitEncoder':
        """Load encoder from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.encoders = data['encoders']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        logger.info(f"TraitEncoder loaded from {path}")
        return self


class TraitDataset:
    """
    Loads and preprocesses trait data from XML files.
    
    Creates feature vectors from species traits for training ML models.
    """
    
    def __init__(self, traits_xml: str, species_csv: str):
        """
        Initialize dataset.
        
        Args:
            traits_xml: Path to species_traits.xml
            species_csv: Path to species.csv (has species_id and edible flag)
        """
        self.traits_df = load_species_traits_xml(Path(traits_xml))
        self.species_df = pd.read_csv(species_csv)
        self.encoder = TraitEncoder()
        
        # Create species_id to label mapping
        self.species_to_id = {row['species_id']: idx 
                             for idx, row in self.species_df.iterrows()}
        self.id_to_species = {v: k for k, v in self.species_to_id.items()}
        
        logger.info(f"TraitDataset initialized: {len(self.species_df)} species")
    
    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and labels.
        
        Returns:
            X: Feature matrix of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            feature_names: List of feature names
        """
        # Fit encoder
        feature_names = self.encoder.fit_transform(self.traits_df)
        
        # Extract traits for each species
        X = []
        y = []
        
        for species_id, label_id in self.species_to_id.items():
            # Get all traits for this species
            species_traits = self.traits_df[
                self.traits_df['species_id'] == species_id
            ].copy()
            
            if len(species_traits) == 0:
                logger.warning(f"No traits found for species {species_id}")
                continue
            
            # Create trait dictionary for encoding
            traits_dict = {}
            for _, row in species_traits.iterrows():
                trait_id = f"{row['trait_category']}.{row['trait_name']}"
                traits_dict[trait_id] = row['trait_value']
            
            # Encode traits to feature vector
            features = self.encoder.transform(traits_dict)
            X.append(features)
            y.append(label_id)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        logger.info(f"Prepared dataset: X shape {X.shape}, y shape {y.shape}")
        return X, y, feature_names
    
    def get_species_name(self, label_id: int) -> str:
        """Get species name from label ID."""
        species_id = self.id_to_species.get(label_id, "Unknown")
        species_row = self.species_df[self.species_df['species_id'] == species_id]
        
        if len(species_row) > 0:
            # Return bilingual format
            swedish = species_row['swedish_name'].values[0]
            english = species_row['english_name'].values[0]
            return f"{swedish} ({english})"
        
        return species_id
    
    def save_encoder(self, path: str) -> None:
        """Save encoder for inference."""
        self.encoder.save(path)


class TraitObservation:
    """
    Encodes user-provided observations into trait vectors.
    
    Used during inference to convert user input (trait selections) to features.
    """
    
    def __init__(self, encoder: TraitEncoder):
        """
        Initialize with a fitted encoder.
        
        Args:
            encoder: Fitted TraitEncoder instance
        """
        self.encoder = encoder
    
    def from_dict(self, observations: Dict[str, str]) -> np.ndarray:
        """
        Convert observation dictionary to feature vector.
        
        Args:
            observations: {trait_id: value} dictionary
                         e.g., {'CAP.color': 'white', 'CAP.shape': 'round'}
        
        Returns:
            Feature vector of shape (n_features,)
        """
        return self.encoder.transform(observations)
    
    def from_list(self, trait_selections: List[Tuple[str, str]]) -> np.ndarray:
        """
        Convert list of trait selections to feature vector.
        
        Args:
            trait_selections: List of (trait_id, value) tuples
        
        Returns:
            Feature vector
        """
        observations = dict(trait_selections)
        return self.from_dict(observations)
