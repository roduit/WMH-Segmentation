# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-02 -*-
# -*- Last revision: 2025-12-02 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Helper functions for plots-*-

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


def set_style() -> None:
    """Set fontstyle of the matplotlib."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def darken(color: tuple, factor: float = 0.7) -> tuple[float, float, float]:
    """Darken a color by a given factor.

    Args:
        color (tuple): The color to darken.
        factor (float, optional): The factor by which to darken the color.
          Defaults to 0.7.

    Returns:
        tuple: A tuple representing the darkened RGB color.

    """
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)


MAPS_NAME_FOR_PLOTS = {
    "A": "PD*",
    "FA": "FA",
    "ICVF": "ICVF",
    "ISOVF": "ISOVF",
    "MD": "MD",
    "MT": "MTsat",
    "OD": "OD",
    "R1": "R1",
    "R2s_OLS": "R2*",
    "g_ratio": "g-ratio",
}
