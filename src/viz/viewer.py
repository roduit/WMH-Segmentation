# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-02 -*-
# -*- Last revision: 2025-12-02 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Interactive 3D viewer functions -*-

import numpy as np
import plotly.graph_objects as go

from viz.viz_utils import set_style

set_style()


def plot_3dview(prediction: np.ndarray, labels: np.ndarray, subject_id: str) -> None:
    """Plot 3D view of prediction and labels using Plotly.

    Args:
        prediction (np.ndarray): 3D array of predicted segmentation.
        labels (np.ndarray): 3D array of ground truth segmentation.
        subject_id (str): Identifier for the subject.

    Returns:
        fig: Plotly figure object.

    """
    overlap = (prediction == 1) & (labels == 1)
    only_pred = (prediction == 1) & (labels == 0)
    only_true = (prediction == 0) & (labels == 1)
    combined_mask = np.zeros_like(prediction)
    combined_mask[overlap] = 1  # Overlapping region
    combined_mask[only_pred] = 2  # Only predicted
    combined_mask[only_true] = 3  # Only true

    combined_mask = np.transpose(combined_mask, (0, 2, 1))

    x, y, z = np.where(combined_mask >= 0)  # Get coordinates of the mask
    values = combined_mask[combined_mask >= 0]

    fig = go.Figure(
        data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=values.flatten(),
            isomin=1,
            isomax=3,
            cmin=1,
            cmax=3,
            opacity=0.2,  # optional
            surface_count=3,  # optional
            colorscale="Viridis",
        )
    )

    # Set title and axis labels
    fig.update_layout(
        title=f"3D View for Subject {subject_id}"
        "(1: Overlap, 2: Only Predicted, 3: Only True)"
    )

    return fig
