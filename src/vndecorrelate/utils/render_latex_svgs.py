"""Render display-math blocks in Markdown as themed vector SVGs.

This developer utility requires a TeX distribution that provides ``latex`` and
``dvisvgm``. It is deliberately excluded from source and wheel distributions.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DISPLAY_MATH_PATTERN = re.compile(
    r'^\$\$\s*\n(?P<latex>.*?)^\$\$\s*$', re.MULTILINE | re.DOTALL
)
HEADING_PATTERN = re.compile(r'^#{1,6}\s+(?P<title>.*?)\s*#?\s*$', re.MULTILINE)
LIGHT_COLOR = '#1A1A1A'
DARK_COLOR = '#F5F5F7'


@dataclass(frozen=True, slots=True)
class Equation:
    """Describe a display-math block extracted from a Markdown document.

    Parameters
    ----------
    title : str
        Nearest preceding Markdown heading.
    latex : str
        LaTeX source between the display-math delimiters.
    stem : str
        Unique, filesystem-safe basename for generated SVG files.
    """

    title: str
    latex: str
    stem: str


def extract_equations(markdown: str, *, fallback_title: str = 'equation') -> list[Equation]:
    """Extract display-math blocks and assign stable SVG basenames.

    Parameters
    ----------
    markdown : str
        Markdown text containing ``$$``-delimited display-math blocks.
    fallback_title : str
        Title used when a display-math block has no preceding heading.

    Returns
    -------
    list[Equation]
        Extracted equations in document order. Equations with the same heading
        receive numbered basenames after the first occurrence.
    """
    headings = list(HEADING_PATTERN.finditer(markdown))
    heading_index = 0
    current_title = fallback_title
    occurrences: Counter[str] = Counter()
    equations: list[Equation] = []

    for math_match in DISPLAY_MATH_PATTERN.finditer(markdown):
        while (
            heading_index < len(headings)
            and headings[heading_index].start() < math_match.start()
        ):
            current_title = headings[heading_index]['title'].strip()
            heading_index += 1

        base_stem = slugify(current_title)
        occurrences[base_stem] += 1
        suffix = '' if occurrences[base_stem] == 1 else f'-{occurrences[base_stem]}'
        equations.append(
            Equation(
                title=current_title,
                latex=math_match['latex'].strip(),
                stem=f'{base_stem}{suffix}',
            )
        )

    return equations


def slugify(value: str) -> str:
    """Create a lowercase, filesystem-safe SVG basename.

    Parameters
    ----------
    value : str
        Human-readable heading to convert.

    Returns
    -------
    str
        Hyphen-separated basename containing only lowercase ASCII letters and
        digits.
    """
    slug = re.sub(r'[^a-z0-9]+', '-', value.casefold()).strip('-')
    return slug or 'equation'


def render_equations(
    source_path: Path,
    output_directory: Path,
    *,
    latex_command: str = 'latex',
    dvisvgm_command: str = 'dvisvgm',
) -> list[tuple[Equation, Path, Path]]:
    """Render every display-math block in a Markdown file as SVG pairs.

    Parameters
    ----------
    source_path : Path
        Markdown document containing display-math blocks.
    output_directory : Path
        Directory where the light and dark SVG files are written.
    latex_command : str
        Executable used to compile a temporary LaTeX document to DVI.
    dvisvgm_command : str
        Executable used to convert DVI output to vector SVG.

    Returns
    -------
    list[tuple[Equation, Path, Path]]
        Extracted equation together with its light and dark SVG paths.

    Raises
    ------
    RuntimeError
        If a required renderer is unavailable or either renderer reports an
        error.
    """
    equations = extract_equations(
        source_path.read_text(encoding='utf-8'), fallback_title=source_path.stem
    )
    if not equations:
        raise RuntimeError(f'No $$-delimited display-math blocks found in {source_path}.')

    output_directory.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[Equation, Path, Path]] = []
    for equation in equations:
        light_path = output_directory / f'{equation.stem}.svg'
        dark_path = output_directory / f'{equation.stem}-dark.svg'
        _render_svg(
            equation.latex,
            color=LIGHT_COLOR,
            output_path=light_path,
            latex_command=latex_command,
            dvisvgm_command=dvisvgm_command,
        )
        _render_svg(
            equation.latex,
            color=DARK_COLOR,
            output_path=dark_path,
            latex_command=latex_command,
            dvisvgm_command=dvisvgm_command,
        )
        rendered.append((equation, light_path, dark_path))

    return rendered


def picture_markup(
    equation: Equation, light_path: Path, dark_path: Path
) -> str:
    """Return GitHub-compatible responsive-image markup for an equation.

    Parameters
    ----------
    equation : Equation
        Equation described by the image markup.
    light_path : Path
        Repository-relative light-mode SVG path.
    dark_path : Path
        Repository-relative dark-mode SVG path.

    Returns
    -------
    str
        ``<picture>`` markup that selects a dark SVG when appropriate.
    """
    return (
        '<picture>\n'
        f'  <source media="(prefers-color-scheme: dark)" srcset="{dark_path.as_posix()}">\n'
        f'  <source media="(prefers-color-scheme: light)" srcset="{light_path.as_posix()}">\n'
        f'  <img alt="{equation.title}" src="{light_path.as_posix()}">\n'
        '</picture>'
    )


def _render_svg(
    latex: str,
    *,
    color: str,
    output_path: Path,
    latex_command: str,
    dvisvgm_command: str,
) -> None:
    """Compile one LaTeX expression and convert it to a vector SVG.

    Parameters
    ----------
    latex : str
        Display-math expression without ``$$`` delimiters.
    color : str
        Six-digit hexadecimal foreground color, prefixed with ``#``.
    output_path : Path
        Destination SVG path.
    latex_command : str
        LaTeX compiler executable.
    dvisvgm_command : str
        DVI-to-SVG converter executable.

    Returns
    -------
    None
    """
    color_hex = color.removeprefix('#')
    with tempfile.TemporaryDirectory(prefix='vndecorrelate-latex-') as temporary:
        temporary_path = Path(temporary)
        tex_path = temporary_path / 'equation.tex'
        tex_path.write_text(_latex_document(latex, color_hex), encoding='utf-8')
        _run(
            [
                latex_command,
                '-interaction=nonstopmode',
                '-halt-on-error',
                f'-output-directory={temporary_path}',
                tex_path.name,
            ],
            cwd=temporary_path,
        )
        _run(
            [
                dvisvgm_command,
                '--no-fonts',
                '--bbox=min',
                '--output',
                str(output_path),
                str(temporary_path / 'equation.dvi'),
            ],
            cwd=temporary_path,
        )


def _latex_document(latex: str, color_hex: str) -> str:
    """Wrap a display-math expression in a standalone LaTeX document.

    Parameters
    ----------
    latex : str
        Display-math expression without delimiters.
    color_hex : str
        Six-digit hexadecimal foreground color without a ``#`` prefix.

    Returns
    -------
    str
        Complete LaTeX document source.
    """
    return f'''\\documentclass[preview,border=2pt]{{standalone}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{xcolor}}
\\begin{{document}}
\\color[HTML]{{{color_hex}}}
\\[
{latex}
\\]
\\end{{document}}
'''


def _run(command: Sequence[str], *, cwd: Path) -> None:
    """Run one renderer command and surface a useful failure message.

    Parameters
    ----------
    command : Sequence[str]
        Renderer command and arguments.
    cwd : Path
        Temporary directory in which to invoke the command.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the executable is unavailable or returns a nonzero exit code.
    """
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        executable = command[0]
        raise RuntimeError(
            f'Unable to find {executable!r}. Install a TeX distribution that provides '
            "both 'latex' and 'dvisvgm', or pass the executable path with the CLI option."
        ) from error

    if completed.returncode:
        output = '\n'.join(part for part in (completed.stdout, completed.stderr) if part)
        raise RuntimeError(
            f"Renderer command failed ({' '.join(command)}):\n{output.strip()}"
        )


def main() -> None:
    """Render Markdown display math from the command line.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description='Render Markdown display math as light and dark vector SVGs.'
    )
    parser.add_argument('source', type=Path, help='Markdown file containing $$ blocks.')
    parser.add_argument(
        '--output-dir', type=Path, default=Path('img'), help='Destination directory.'
    )
    parser.add_argument(
        '--latex-command', default='latex', help='LaTeX executable or executable path.'
    )
    parser.add_argument(
        '--dvisvgm-command',
        default='dvisvgm',
        help='DVI-to-SVG executable or executable path.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List planned SVG names without invoking TeX.',
    )
    arguments = parser.parse_args()

    source_path = arguments.source.resolve()
    output_directory = arguments.output_dir.resolve()
    equations = extract_equations(
        source_path.read_text(encoding='utf-8'), fallback_title=source_path.stem
    )
    if arguments.dry_run:
        for equation in equations:
            print(arguments.output_dir / f'{equation.stem}.svg')
            print(arguments.output_dir / f'{equation.stem}-dark.svg')
        return

    rendered = render_equations(
        source_path,
        output_directory,
        latex_command=arguments.latex_command,
        dvisvgm_command=arguments.dvisvgm_command,
    )
    for equation, light_path, dark_path in rendered:
        print(
            picture_markup(
                equation,
                arguments.output_dir / light_path.name,
                arguments.output_dir / dark_path.name,
            )
        )


if __name__ == '__main__':
    main()
