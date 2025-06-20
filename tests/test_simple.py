"""
test_simple.py

Simple, human-readable tests for the MLOps pipeline.

This file demonstrates basic testing patterns that are easy to understand
and maintain. It focuses on the most important functionality without
over-engineering.

Test Categories:
- Basic functionality tests
- Error handling tests
- Integration tests
"""

import pandas as pd
import pytest
from pathlib import Path

# Import the functions we want to test
from src.data_loader.data_loader import load_data_source, DataLoaderError


def test_load_csv_data():
    """Test loading CSV data - simple and straightforward."""
    # Create a simple test CSV file
    test_data = pd.DataFrame({
        'patient_id': [1, 2, 3],
        'age': [25, 45, 35],
        'diagnosis': ['positive', 'negative', 'positive']
    })
    
    # Save to temporary file
    csv_path = Path("tests/mock_data/test_simple.csv")
    test_data.to_csv(csv_path, index=False)
    
    try:
        # Test the function
        config = {
            'raw_path': str(csv_path),
            'type': 'csv',
            'header': 0,
            'encoding': 'utf-8'
        }
        
        result = load_data_source(config)
        
        # Simple assertions
        assert len(result) == 3
        assert 'patient_id' in result.columns
        assert 'age' in result.columns
        assert 'diagnosis' in result.columns
        
        print("✅ CSV loading test passed!")
        
    finally:
        # Clean up
        csv_path.unlink(missing_ok=True)


def test_load_excel_data():
    """Test loading Excel data - simple and straightforward."""
    # Create a simple test Excel file
    test_data = pd.DataFrame({
        'patient_id': [4, 5, 6],
        'age': [28, 52, 38],
        'diagnosis': ['negative', 'positive', 'negative']
    })
    
    # Save to temporary file
    excel_path = Path("tests/mock_data/test_simple.xlsx")
    test_data.to_excel(excel_path, index=False)
    
    try:
        # Test the function
        config = {
            'raw_path': str(excel_path),
            'type': 'excel',
            'header': 0,
            'sheet_name': 'Sheet1'
        }
        
        result = load_data_source(config)
        
        # Simple assertions
        assert len(result) == 3
        assert 'patient_id' in result.columns
        assert 'age' in result.columns
        assert 'diagnosis' in result.columns
        
        print("✅ Excel loading test passed!")
        
    finally:
        # Clean up
        excel_path.unlink(missing_ok=True)


def test_missing_file_error():
    """Test error handling when file is missing."""
    config = {
        'raw_path': 'nonexistent_file.csv',
        'type': 'csv',
        'header': 0,
        'encoding': 'utf-8'
    }
    
    # Test that the function raises the right error
    with pytest.raises(DataLoaderError) as error_info:
        load_data_source(config)
    
    # Check the error message
    assert "Data file not found" in str(error_info.value)
    print("✅ Missing file error test passed!")


def test_unsupported_file_type():
    """Test error handling for unsupported file types."""
    # Create a dummy file
    dummy_path = Path("tests/mock_data/dummy.txt")
    dummy_path.write_text("dummy data")
    
    try:
        config = {
            'raw_path': str(dummy_path),
            'type': 'txt',  # Unsupported type
            'header': 0,
            'encoding': 'utf-8'
        }
        
        # Test that the function raises the right error
        with pytest.raises(DataLoaderError) as error_info:
            load_data_source(config)
        
        # Check the error message
        assert "Unsupported data type" in str(error_info.value)
        print("✅ Unsupported file type test passed!")
        
    finally:
        # Clean up
        dummy_path.unlink(missing_ok=True)


def test_empty_csv_file():
    """Test handling of empty CSV file."""
    # Create an empty CSV file
    empty_path = Path("tests/mock_data/empty.csv")
    empty_path.write_text("")  # Empty file
    
    try:
        config = {
            'raw_path': str(empty_path),
            'type': 'csv',
            'header': 0,
            'encoding': 'utf-8'
        }
        
        with pytest.raises(DataLoaderError) as error_info:
            load_data_source(config)
        
        # Check the error message
        assert "No columns to parse from file" in str(error_info.value)
        print("✅ Empty file test passed!")
        
    finally:
        # Clean up
        empty_path.unlink(missing_ok=True)


def test_real_mock_data():
    """Test with the actual mock data files."""
    # Test CSV mock data
    csv_config = {
        'raw_path': 'tests/mock_data/mock_data.csv',
        'type': 'csv',
        'header': 0,
        'encoding': 'utf-8'
    }
    
    result_csv = load_data_source(csv_config)
    
    # Check basic properties
    assert len(result_csv) == 5  # 5 rows in our mock data
    assert 'patient_id' in result_csv.columns
    assert 'age' in result_csv.columns
    assert 'diagnosis' in result_csv.columns
    
    # Test Excel mock data
    excel_config = {
        'raw_path': 'tests/mock_data/mock_data.xlsx',
        'type': 'excel',
        'header': 0,
        'sheet_name': 'Sheet1'
    }
    
    result_excel = load_data_source(excel_config)
    
    # Check basic properties
    assert len(result_excel) == 5  # 5 rows in our mock data
    assert 'patient_id' in result_excel.columns
    assert 'age' in result_excel.columns
    assert 'diagnosis' in result_excel.columns
    
    print("✅ Real mock data test passed!")


def test_data_validation():
    """Test that loaded data has expected structure."""
    config = {
        'raw_path': 'tests/mock_data/mock_data.csv',
        'type': 'csv',
        'header': 0,
        'encoding': 'utf-8'
    }
    
    df = load_data_source(config)
    
    # Check data types
    assert df['patient_id'].dtype in ['int64', 'object']  # Could be int or object
    assert df['age'].dtype in ['int64', 'object']  # Could be int or object
    assert df['diagnosis'].dtype == 'object'  # Should be string
    
    # Check for missing values
    assert not df['patient_id'].isnull().any()
    assert not df['age'].isnull().any()
    assert not df['diagnosis'].isnull().any()
    
    # Check value ranges
    assert df['age'].min() >= 0
    assert df['age'].max() <= 100
    
    # Check diagnosis values
    valid_diagnoses = ['positive', 'negative']
    assert all(diagnosis in valid_diagnoses for diagnosis in df['diagnosis'])
    
    print("✅ Data validation test passed!")


# Integration test - testing multiple components together
def test_full_data_loading_workflow():
    """Test the complete data loading workflow."""
    # Test with different file types
    test_cases = [
        {
            'file': 'tests/mock_data/mock_data.csv',
            'type': 'csv',
            'expected_rows': 5
        },
        {
            'file': 'tests/mock_data/mock_data.xlsx',
            'type': 'excel',
            'expected_rows': 5
        }
    ]
    
    for case in test_cases:
        config = {
            'raw_path': case['file'],
            'type': case['type'],
            'header': 0,
            'encoding': 'utf-8' if case['type'] == 'csv' else None
        }
        
        if case['type'] == 'excel':
            config['sheet_name'] = 'Sheet1'
        
        # Load data
        df = load_data_source(config)
        
        # Verify results
        assert len(df) == case['expected_rows']
        assert not df.empty
        assert 'patient_id' in df.columns
        
        print(f"✅ {case['type'].upper()} workflow test passed!")
    
    print("✅ Full workflow test completed!")


if __name__ == "__main__":
    """Run tests manually for debugging."""
    print("🧪 Running simple tests...")
    
    try:
        test_load_csv_data()
        test_load_excel_data()
        test_missing_file_error()
        test_unsupported_file_type()
        test_empty_csv_file()
        test_real_mock_data()
        test_data_validation()
        test_full_data_loading_workflow()
        
        print("\n🎉 All simple tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise 