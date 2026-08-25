"""
Geocoding Module
Converts district/location names to GPS coordinates for map visualization
"""

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import pandas as pd
from typing import Tuple, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndiaGeocoder:
    """
    Geocode Indian administrative locations (districts, blocks) to GPS coordinates
    """
    
    def __init__(self, user_agent: str = "paimana-intelligence-platform"):
        """
        Initialize geocoder
        
        Args:
            user_agent: User agent string for geocoding service
        """
        self.geolocator = Nominatim(user_agent=user_agent)
        self.cache = {}  # Cache to avoid repeated lookups
    
    def geocode_location(self, district: str, state: str) -> Optional[Tuple[float, float]]:
        """
        Geocode a district to latitude/longitude
        
        Args:
            district: District name
            state: State name
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        # Check cache first
        cache_key = f"{district}, {state}, India"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Add delay to respect rate limits
            time.sleep(1)
            
            # Geocode
            location = self.geolocator.geocode(cache_key, timeout=10)
            
            if location:
                coords = (location.latitude, location.longitude)
                self.cache[cache_key] = coords
                logger.info(f"Geocoded {cache_key}: {coords}")
                return coords
            else:
                logger.warning(f"Could not geocode: {cache_key}")
                self.cache[cache_key] = None
                return None
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error(f"Geocoding error for {cache_key}: {e}")
            return None
    
    def add_coordinates_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add latitude and longitude columns to DataFrame using real geocoding
        
        Args:
            df: DataFrame with 'district' and 'state' columns
            
        Returns:
            DataFrame with added 'latitude' and 'longitude' columns
        """
        if not all(col in df.columns for col in ['district', 'state']):
            logger.warning("Missing 'district' or 'state' columns")
            return df
        
        logger.info(f"Geocoding {len(df)} projects...")
        
        # Fallback coordinates for common Maharashtra districts
        fallback_coords = {
            'Mumbai': (19.0760, 72.8777),
            'Pune': (18.5204, 73.8567),
            'Nagpur': (21.1458, 79.0882),
            'Nashik': (19.9975, 73.7898),
            'Aurangabad': (19.8762, 75.3433),
            'Solapur': (17.6599, 75.9064),
            'Kolhapur': (16.7050, 74.2433)
        }
        
        latitudes = []
        longitudes = []
        successful_geocodes = 0
        
        for _, row in df.iterrows():
            district = row['district']
            state = row['state']
            
            # Try real geocoding first (with error handling)
            try:
                coords = self.geocode_location(district, state)
                if coords:
                    latitudes.append(coords[0])
                    longitudes.append(coords[1])
                    successful_geocodes += 1
                else:
                    # Use fallback if available
                    fallback = fallback_coords.get(district, (19.5, 75.5))
                    latitudes.append(fallback[0])
                    longitudes.append(fallback[1])
            except Exception as e:
                # On any error, use fallback
                fallback = fallback_coords.get(district, (19.5, 75.5))
                latitudes.append(fallback[0])
                longitudes.append(fallback[1])
        
        df['latitude'] = latitudes
        df['longitude'] = longitudes
        
        logger.info(f"Geocoding complete. {successful_geocodes}/{len(df)} successful API lookups")
        
        return df


def main():
    """Demo usage"""
    # Sample data
    sample_df = pd.DataFrame({
        'project_id': ['P001', 'P002', 'P003'],
        'project_name': ['Project A', 'Project B', 'Project C'],
        'district': ['Mumbai', 'Pune', 'Nagpur'],
        'state': ['maharashtra', 'maharashtra', 'maharashtra']
    })
    
    print("Sample data:")
    print(sample_df)
    
    print("\nGeocoding districts...")
    geocoder = IndiaGeocoder()
    result_df = geocoder.add_coordinates_to_dataframe(sample_df)
    
    print("\nGeocoded data:")
    print(result_df[['project_id', 'district', 'latitude', 'longitude']])


if __name__ == '__main__':
    main()
