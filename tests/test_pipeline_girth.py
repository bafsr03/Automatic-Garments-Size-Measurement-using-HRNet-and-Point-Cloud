import sys
from unittest.mock import MagicMock

# Mock cv2 before importing pipeline
sys.modules["cv2"] = MagicMock()

import pytest
from api_app.pipeline import _build_size_scale

def test_pipeline_converts_half_width_to_girth():
    """
    Verify that _build_size_scale detects Half-Width measurements (e.g. Chest 50cm)
    and converts them to Girth (Chest 100cm) in the output scales.
    """
    # Input: Half-Width measurements (Chest 52cm)
    # Unit: CM
    measurements = {
        "chest": 52.0, # Flat
        "waist": 40.0, # Flat
        "length": 70.0 # Vertical (should NOT double)
    }
    
    scale_cm, scale_in = _build_size_scale(measurements, true_size="M", input_unit="cm")
    
    # Check Size M (Base)
    m_cm = scale_cm["M"]
    m_in = scale_in["M"]
    
    # Chest should be doubled: 52 * 2 = 104
    assert m_cm["chest"] == 104.0, f"Expected 104.0, got {m_cm['chest']}"
    
    # Waist should be doubled: 40 * 2 = 80
    assert m_cm["waist"] == 80.0, f"Expected 80.0, got {m_cm['waist']}"
    
    # Length should NOT be doubled: 70
    assert m_cm["length"] == 70.0, f"Expected 70.0, got {m_cm['length']}"
    
    # Check Inches
    # Chest 104cm -> ~40.94 inch
    assert abs(m_in["chest"] - 40.94) < 0.1
    
def test_pipeline_respects_girth_input():
    """
    Verify that if input is already Girth (Chest 104cm), it is NOT doubled.
    """
    measurements = {
        "chest": 104.0,
        "waist": 80.0,
        "length": 70.0
    }
    
    scale_cm, scale_in = _build_size_scale(measurements, true_size="M", input_unit="cm")
    
    m_cm = scale_cm["M"]
    
    # Should remain 104
    assert m_cm["chest"] == 104.0
