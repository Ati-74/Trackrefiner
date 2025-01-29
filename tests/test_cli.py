import subprocess
import os


def test_cli_help():
    """Test the --help option of the CLI."""
    result = subprocess.run(
        ["python", "-m", "Trackrefiner.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0  # Ensure the process exits successfully
    assert "analyzing CellProfiler output" in result.stdout  # Check help description


def test_cli_invalid_args():
    """Test the CLI with missing required arguments."""
    result = subprocess.run(
        ["python", "-m", "Trackrefiner.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0  # Should fail due to missing required arguments
    assert "usage: " in result.stderr  # Ensure the error mentions usage information


def test_cli_incorrect_data_types():
    """Test the CLI with incorrect data types for numeric arguments."""
    result = subprocess.run(
        [
            "python",
            "-m",
            "Trackrefiner.cli",
            "-i", "test_data/FilterObjects.csv",
            "-s", "test_data/objects/",
            "-n", "test_data/Object_relationships.csv",
            "-t", "not_a_number",  # Invalid interval_time
            "-d", "another_invalid",  # Invalid doubling_time
            "-o", "output",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0  # Should fail due to invalid argument types
    assert "Invalid value for interval_time:" in result.stderr or \
           "Invalid value for doubling_time:" in result.stderr, (
        "Expected error message for invalid data types was not found in stderr."
    )


def test_cli_missing_files():
    """Test the CLI with non-existent file paths."""
    result = subprocess.run(
        [
            "python",
            "-m",
            "Trackrefiner.cli",
            "-i", "non_existent_file.csv",  # Non-existent CP output file
            "-s", "non_existent_directory/",  # Non-existent segmentation folder
            "-n", "non_existent_neighbors.csv",  # Non-existent neighbor CSV file
            "-t", "3",
            "-d", "20",
            "-o", "output",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0  # Should fail due to missing files
    assert "The CP output CSV file does not exist" in result.stderr or \
           "The neighbor CSV file does not exist" in result.stderr, (
        "Expected error message for missing files was not found in stderr."
    )


def test_cli_disable_tracking_correction(tmp_path):
    """Test the CLI with valid arguments."""
    # Create a temporary output directory
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)

    # Simulate running the CLI with valid arguments
    result = subprocess.run(
        [
            "python",
            "-m",
            "Trackrefiner.cli",
            "-i", "test_data/FilterObjects.csv",
            "-n", "test_data/Object_relationships.csv",
            "-t", "3",
            "-d", "20",
            "--disable_tracking_correction",  # Tracking correction disabled
            "-o", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0  # Ensure the process exits successfully
    # Check for the final success log message in stdout
    assert "Trackrefiner Process completed at:" in result.stdout, (
        "Expected log message indicating successful completion was not found in stdout."
    )
    assert output_dir.exists()  # Check if the output directory was created


def test_cli_empty_cp_output(tmp_path):
    """Test the CLI with empty input files."""
    # Create empty test files
    # Create a temporary CP output CSV with only headers
    invalid_cp_output_csv = tmp_path / "invalid_cp_output.csv"
    # Only headers, no data
    invalid_cp_output_csv.write_text(
        "ImageNumber,ObjectNumber,AreaShape_MajorAxisLength,AreaShape_MinorAxisLength,AreaShape_Orientation\n")

    empty_neighbor_csv = tmp_path / "empty_neighbors.csv"
    empty_neighbor_csv.write_text("")  # Empty file

    # Create a temporary output directory
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)

    # Simulate running the CLI with empty input files
    result = subprocess.run(
        [
            "python",
            "-m",
            "Trackrefiner.cli",
            "-i", str(invalid_cp_output_csv),
            "-n", "test_data/Object_relationships.csv",
            "-t", "3",
            "-d", "20",
            "-o", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0  # Should fail due to empty input files
    assert f"The {invalid_cp_output_csv} is empty." in result.stderr, (
        "Expected error message for empty input files was not found in stderr."
    )


def test_cli_with_args(tmp_path):
    """Test the CLI with valid arguments."""
    # Create a temporary output directory
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)

    # Simulate running the CLI with valid arguments
    result = subprocess.run(
        [
            "python",
            "-m",
            "Trackrefiner.cli",
            "-i", "test_data/FilterObjects.csv",
            "-s", "test_data/objects/",
            "-n", "test_data/Object_relationships.csv",
            "-t", "3",
            "-d", "20",
            "-o", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0  # Ensure the process exits successfully
    # Check for the final success log message in stdout
    assert "Trackrefiner Process completed at:" in result.stdout, (
        "Expected log message indicating successful completion was not found in stdout."
    )
    assert output_dir.exists()  # Check if the output directory was created
