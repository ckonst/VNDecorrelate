"""Render LaTeX math blocks in Markdown as themed vector SVGs.

This developer utility requires a TeX distribution that provides ``latex`` and
``dvisvgm``. The complete ``tools`` package is excluded from release artifacts.
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

MATH_PATTERN = re.compile(
    r'^\$\$\s*\n(?P<display_latex>.*?)^\$\$\s*$'
    r'|(?<![\\$])\$(?!\$)(?P<inline_latex>[^$\r\n]+?)(?<!\\)\$(?!\$)',
    re.MULTILINE | re.DOTALL,
)
HEADING_PATTERN = re.compile(r'^#{1,6}\s+(?P<title>.*?)\s*#?\s*$', re.MULTILINE)
LIGHT_COLOR = '#1A1A1A'
DARK_COLOR = '#F5F5F7'
DEFAULT_FONT_SIZE_PT = 14.0
LINE_HEIGHT_MULTIPLIER = 1.2


@dataclass(frozen=True, slots=True)
class Equation:
    """Describe a LaTeX math block extracted from a Markdown document.

    Parameters
    ----------
    title : str
        Nearest preceding Markdown heading.
    latex : str
        LaTeX source between the math delimiters.
    stem : str
        Unique, filesystem-safe basename for generated SVG files.
    """

    title: str
    latex: str
    stem: str


def extract_equations(
    markdown: str, *, fallback_title: str = 'equation'
) -> list[Equation]:
    """Extract display and inline math blocks and assign SVG basenames.

    Parameters
    ----------
    markdown : str
        Markdown text containing ``$$``-delimited display math and unescaped
        ``$``-delimited inline math.
    fallback_title : str
        Title used when a math block has no preceding heading.

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

    for math_match in MATH_PATTERN.finditer(markdown):
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
                latex=(
                    math_match['display_latex'] or math_match['inline_latex']
                ).strip(),
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
    font_size_pt: float = DEFAULT_FONT_SIZE_PT,
) -> list[tuple[Equation, Path, Path]]:
    """Render Markdown display and inline math as light and dark SVG pairs.

    Parameters
    ----------
    source_path : Path
        Markdown document containing LaTeX math blocks.
    output_directory : Path
        Directory where the light and dark SVG files are written.
    latex_command : str
        Executable used to compile temporary LaTeX documents to DVI.
    dvisvgm_command : str
        Executable used to convert DVI output to vector SVG.
    font_size_pt : float
        LaTeX font size for rendered math in points. The default is 12 pt,
        matching the scale of the repository's pre-existing equation SVGs.

    Returns
    -------
    list[tuple[Equation, Path, Path]]
        Extracted equation together with its light and dark SVG paths.

    Raises
    ------
    RuntimeError
        If no equations are found or a renderer is unavailable or fails.
    ValueError
        If ``font_size_pt`` is not positive.
    """
    if font_size_pt <= 0:
        raise ValueError('font_size_pt must be positive.')

    equations = extract_equations(
        source_path.read_text(encoding='utf-8'), fallback_title=source_path.stem
    )
    if not equations:
        raise RuntimeError(f'No LaTeX math blocks found in {source_path}.')

    output_directory.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[Equation, Path, Path]] = []
    for equation in equations:
        light_path = output_directory / f'{equation.stem}.svg'
        dark_path = output_directory / f'{equation.stem}-dark.svg'
        for color, output_path in (
            (LIGHT_COLOR, light_path),
            (DARK_COLOR, dark_path),
        ):
            _render_svg(
                equation.latex,
                color=color,
                output_path=output_path,
                latex_command=latex_command,
                dvisvgm_command=dvisvgm_command,
                font_size_pt=font_size_pt,
            )
        rendered.append((equation, light_path, dark_path))

    return rendered


def picture_markup(equation: Equation, light_path: Path, dark_path: Path) -> str:
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
    font_size_pt: float,
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
    font_size_pt : float
        LaTeX font size for the rendered expression in points.

    Returns
    -------
    None
    """
    with tempfile.TemporaryDirectory(prefix='vndecorrelate-latex-') as temporary:
        temporary_path = Path(temporary)
        tex_path = temporary_path / 'equation.tex'
        tex_path.write_text(
            _latex_document(
                latex,
                color.removeprefix('#'),
                font_size_pt=font_size_pt,
            ),
            encoding='utf-8',
        )
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
                f'--output={output_path}',
                str(temporary_path / 'equation.dvi'),
            ],
            cwd=temporary_path,
        )


def _latex_document(latex: str, color_hex: str, *, font_size_pt: float) -> str:
    """Wrap one expression in a standalone LaTeX document.

    Parameters
    ----------
    latex : str
        Display-math expression without delimiters.
    color_hex : str
        Six-digit hexadecimal foreground color without a ``#`` prefix.
    font_size_pt : float
        LaTeX font size for the expression in points.

    Returns
    -------
    str
        Complete LaTeX document source.

    Notes
    -----
    The preamble loads ``dsfont`` so expressions can use ``\\mathds{1}``
    for a clear indicator-function glyph.
    """
    font_size = f'{font_size_pt:g}'
    line_height = f'{font_size_pt * LINE_HEIGHT_MULTIPLIER:g}'
    return f"""\\documentclass[preview,border=2pt]{{standalone}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{dsfont}}
\\usepackage{{xcolor}}
\\begin{{document}}
\\color[HTML]{{{color_hex}}}
\\fontsize{{{font_size}pt}}{{{line_height}pt}}\\selectfont
\\[
{latex}
\\]
\\end{{document}}
"""


def _run(command: Sequence[str], *, cwd: Path) -> None:
    """Run a renderer command and surface useful failure diagnostics.

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
        raise RuntimeError(
            f'Unable to find {command[0]!r}. Install a TeX distribution that '
            "provides both 'latex' and 'dvisvgm', or pass executable paths with CLI options."
        ) from error

    if completed.returncode:
        output = '\n'.join(
            part for part in (completed.stdout, completed.stderr) if part
        )
        raise RuntimeError(
            f'Renderer command failed ({" ".join(command)}):\n{output.strip()}'
        )


def main() -> None:
    """Render Markdown LaTeX math from the command line.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description='Render Markdown LaTeX math as light and dark vector SVGs.'
    )
    parser.add_argument(
        'source', type=Path, help='Markdown file containing $ or $$ math blocks.'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('img'), help='Destination directory.'
    )
    parser.add_argument('--latex-command', default='latex', help='LaTeX executable.')
    parser.add_argument(
        '--dvisvgm-command', default='dvisvgm', help='DVI-to-SVG executable.'
    )
    parser.add_argument(
        '--font-size',
        type=float,
        default=DEFAULT_FONT_SIZE_PT,
        metavar='POINTS',
        help=(
            'LaTeX font size in points (default: 12, matching existing equation SVGs).'
        ),
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='List planned SVGs without TeX.'
    )
    arguments = parser.parse_args()

    source_path = arguments.source.resolve()
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
        arguments.output_dir.resolve(),
        latex_command=arguments.latex_command,
        dvisvgm_command=arguments.dvisvgm_command,
        font_size_pt=arguments.font_size,
    )
    for equation, light_path, dark_path in rendered:
        print(
            picture_markup(
                equation,
                arguments.output_dir / light_path.name,
                arguments.output_dir / dark_path.name,
            )
        )
