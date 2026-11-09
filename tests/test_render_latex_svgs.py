from pathlib import Path

import pytest

from tools.latex_svg_renderer import extract_equations, picture_markup
from tools.latex_svg_renderer.renderer import (
    DEFAULT_FONT_SIZE_PT,
    LINE_HEIGHT_MULTIPLIER,
    _latex_document,
    render_equations,
)


def test_extract_equations_uses_headings_and_unique_stems():
    equations = extract_equations(
        """# First Equation

$$
x = y
$$

## Second Equation

$$
a = b
$$

$$
c = d
$$
"""
    )

    assert [equation.title for equation in equations] == [
        'First Equation',
        'Second Equation',
        'Second Equation',
    ]
    assert [equation.stem for equation in equations] == [
        'first-equation',
        'second-equation',
        'second-equation-2',
    ]


def test_extract_equations_includes_inline_math_in_document_order():
    equations = extract_equations(
        r"""# Equation

The first expression is $x = y$.

$$
a = b
$$

The final expression is $c = d$; an escaped dollar sign \$ is ignored.
"""
    )

    assert [equation.latex for equation in equations] == ['x = y', 'a = b', 'c = d']
    assert [equation.stem for equation in equations] == [
        'equation',
        'equation-2',
        'equation-3',
    ]


def test_picture_markup_selects_the_dark_and_light_svg_paths():
    equation = extract_equations('# Equation\n\n$$\nx = y\n$$')[0]

    markup = picture_markup(
        equation,
        light_path=Path('img/equation.svg'),
        dark_path=Path('img/equation-dark.svg'),
    )

    assert 'prefers-color-scheme: dark' in markup
    assert 'img/equation-dark.svg' in markup
    assert 'img/equation.svg' in markup


def test_latex_document_uses_the_larger_default_font_size():
    document = _latex_document('x = y', '1A1A1A', font_size_pt=DEFAULT_FONT_SIZE_PT)

    font_size = f'{DEFAULT_FONT_SIZE_PT:g}'
    line_height = f'{DEFAULT_FONT_SIZE_PT * LINE_HEIGHT_MULTIPLIER:g}'
    assert f'\\fontsize{{{font_size}pt}}{{{line_height}pt}}\\selectfont' in document


def test_latex_document_loads_dsfont_for_indicator_functions():
    document = _latex_document(
        r'\mathds{1}_{x > 0}', '1A1A1A', font_size_pt=DEFAULT_FONT_SIZE_PT
    )

    assert r'\usepackage{dsfont}' in document
    assert r'\mathds{1}_{x > 0}' in document


def test_render_equations_rejects_nonpositive_font_sizes():
    with pytest.raises(ValueError, match='font_size_pt must be positive'):
        render_equations(
            Path('not-read-for-invalid-size.md'), Path('img'), font_size_pt=0
        )
