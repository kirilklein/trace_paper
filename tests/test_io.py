import textwrap
from pathlib import Path

import pytest

from trace.io import load_atc_dictionary


@pytest.fixture
def sample_atc_file(tmp_path: Path) -> Path:
    content = textwrap.dedent(
        """
        Code Text
        M Drug substance classification ATC
        MA01 remedies for diseases of the oral cavity
        MA01AA01 sodium fluoride
        MA01AA51 sodium fluoride, comb.
        """
    ).strip()
    file_path = tmp_path / "atc_sample.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_load_atc_dictionary_strips_prefix(sample_atc_file: Path) -> None:
    result = load_atc_dictionary(sample_atc_file)

    assert result == {
        "A01": "remedies for diseases of the oral cavity",
        "A01AA01": "sodium fluoride",
        "A01AA51": "sodium fluoride, comb.",
    }


def test_load_atc_dictionary_keeps_empty_when_requested(
    sample_atc_file: Path,
) -> None:
    result = load_atc_dictionary(sample_atc_file, keep_empty_codes=True)

    assert result[""] == "Drug substance classification ATC"
    assert result["A01AA01"] == "sodium fluoride"
