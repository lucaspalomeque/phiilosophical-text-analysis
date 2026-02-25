"""Tests for the CLI module."""

import pytest
from click.testing import CliRunner
from philosophical_analysis.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    """Tests for the command-line interface."""

    def test_cli_help(self, runner):
        """Test that the CLI help text displays correctly."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Philosophical Text Analysis' in result.output

    def test_cli_version(self, runner):
        """Test that the version flag works."""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0

    def test_info_command(self, runner):
        """Test the info command."""
        result = runner.invoke(cli, ['info'])
        assert result.exit_code == 0

    def test_analyze_missing_file(self, runner):
        """Test analyze command with a non-existent file."""
        result = runner.invoke(cli, ['analyze', '--text', '/nonexistent/file.txt'])
        assert result.exit_code != 0

    def test_analyze_basic_mode(self, runner, tmp_path):
        """Test basic analysis on a sample text file."""
        text_file = tmp_path / "test_text.txt"
        text_file.write_text(
            "Philosophy examines fundamental questions about reality and knowledge. "
            "These questions have been central to human inquiry for centuries. "
            "The systematic investigation of such problems defines philosophical methodology. "
            "Logical reasoning provides the foundation for philosophical argumentation. "
            "Ethics examines questions of right and wrong, good and evil. "
            "Metaphysics investigates the nature of reality and existence."
        )
        result = runner.invoke(cli, ['analyze', '--text', str(text_file), '--mode', 'basic'])
        assert result.exit_code == 0

    def test_analyze_with_json_output(self, runner, tmp_path):
        """Test that JSON output is written correctly."""
        text_file = tmp_path / "test_text.txt"
        text_file.write_text(
            "Philosophy examines fundamental questions about reality and knowledge. "
            "These questions have been central to human inquiry for centuries. "
            "The systematic investigation of such problems defines philosophical methodology. "
            "Logical reasoning provides the foundation for philosophical argumentation. "
            "Ethics examines questions of right and wrong, good and evil. "
            "Metaphysics investigates the nature of reality and existence."
        )
        output_file = tmp_path / "results.json"
        result = runner.invoke(cli, [
            'analyze', '--text', str(text_file),
            '--output', str(output_file), '--mode', 'basic'
        ])
        assert result.exit_code == 0
        assert output_file.exists()

    def test_test_command_basic(self, runner):
        """Test the built-in test command in basic mode."""
        result = runner.invoke(cli, ['test', '--mode', 'basic'])
        assert result.exit_code == 0
