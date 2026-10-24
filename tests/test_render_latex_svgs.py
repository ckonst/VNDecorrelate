from pathlib import Path

from vndecorrelate.utils.render_latex_svgs import extract_equations, picture_markup


def test_extract_equations_uses_headings_and_unique_stems():
    equations = extract_equations(
        '''# First Equation

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
'''
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
