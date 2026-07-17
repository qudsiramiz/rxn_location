import pytest
from unittest.mock import patch

from rxn_location.batch_statistics_cli import parse_args, main

def test_parse_args():
    test_args = ["rxn_location.batch_statistics_cli.py", "-i", "dummy.txt", "--probe", "1", "--format", "pdf"]
    with patch("sys.argv", test_args):
        args = parse_args()
        assert args.input == "dummy.txt"
        assert args.probe == "1"
        assert args.format == "pdf"
        assert args.tsy_model == "T96"
        assert args.verbosity == 2

@patch("rxn_location.batch_statistics_cli.os.path.exists")
def test_main_file_not_found(mock_exists, capsys):
    mock_exists.return_value = False
    test_args = ["rxn_location.batch_statistics_cli.py", "-i", "nonexistent.txt"]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Input file 'nonexistent.txt' not found." in captured.out
