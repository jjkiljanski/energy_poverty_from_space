from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch

PlotMode = Literal["continuous", "categorical"]
IslandLayer = tuple[str, gpd.GeoDataFrame]


def _set_map_extent(ax, gdf: gpd.GeoDataFrame, padding: float) -> None:
    if gdf.empty:
        ax.set_axis_off()
        return

    bounds = np.asarray(gdf.total_bounds, dtype=float)
    if not np.isfinite(bounds).all():
        ax.set_axis_off()
        return

    minx, miny, maxx, maxy = bounds
    padx = max((maxx - minx) * padding, 0.01)
    pady = max((maxy - miny) * padding, 0.01)
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)
    ax.set_xticks([])
    ax.set_yticks([])


def _boundary_for_layer(
    boundary: gpd.GeoDataFrame | None,
    layer: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame | None:
    if boundary is None or boundary.empty or layer.empty:
        return None
    minx, miny, maxx, maxy = layer.total_bounds
    return boundary.cx[minx:maxx, miny:maxy]


def _plot_boundary(ax, boundary: gpd.GeoDataFrame | None) -> None:
    if boundary is not None and not boundary.empty:
        boundary.boundary.plot(ax=ax, color="black", linewidth=1.8, zorder=20)


def _continuous_values(
    mainland: gpd.GeoDataFrame,
    islands: Sequence[IslandLayer],
    value_col: str,
) -> pd.Series:
    values = [pd.to_numeric(mainland[value_col], errors="coerce")]
    values.extend(pd.to_numeric(gdf[value_col], errors="coerce") for _, gdf in islands)
    return pd.concat(values, ignore_index=True)


def plot_portugal_index(
    mainland: gpd.GeoDataFrame,
    islands: Sequence[IslandLayer],
    value_col: str,
    *,
    mode: PlotMode,
    title: str,
    category_colors: Mapping[Any, str] | None = None,
    category_labels: Mapping[Any, str] | None = None,
    cmap: str = "viridis",
    legend_title: str | None = None,
    missing_color: str = "#EEEEEE",
    missing_label: str = "Missing",
    show_missing_count: bool = True,
    show_island_titles: bool = True,
    test_boundary: gpd.GeoDataFrame | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 200,
) -> Figure:
    """Plot a continuous or categorical Portugal index with island insets."""
    layers = [("mainland", mainland), *islands]
    missing_columns = [name for name, gdf in layers if value_col not in gdf.columns]
    if missing_columns:
        raise ValueError(
            f"Column '{value_col}' is missing from map layers: {missing_columns}"
        )
    if mode not in {"continuous", "categorical"}:
        raise ValueError("mode must be either 'continuous' or 'categorical'.")

    continuous_values: pd.Series | None = None
    norm: Normalize | None = None
    if mode == "continuous":
        continuous_values = _continuous_values(mainland, islands, value_col)
        available = continuous_values.dropna()
        if available.empty:
            raise ValueError(f"Column '{value_col}' has no numeric values to plot.")
        value_min = float(available.min())
        value_max = float(available.max())
        if value_min == value_max:
            value_min -= 0.5
            value_max += 0.5
        norm = Normalize(vmin=value_min, vmax=value_max)
    elif not category_colors:
        raise ValueError("category_colors is required for categorical plots.")
    else:
        observed = {
            value
            for _, gdf in layers
            for value in gdf[value_col].dropna().unique()
        }
        missing_colors = observed - set(category_colors)
        if missing_colors:
            raise ValueError(
                f"No category colors defined for '{value_col}' values: "
                f"{sorted(missing_colors, key=str)}"
            )

    fig = plt.figure(figsize=(10, 10))
    main_position = [0.33, 0.05, 0.64, 0.90] if islands else [0.08, 0.05, 0.84, 0.90]
    ax_main = fig.add_axes(main_position)

    def plot_layer(ax, gdf: gpd.GeoDataFrame, *, is_mainland: bool = False) -> None:
        if gdf.empty:
            ax.set_axis_off()
            return

        if mode == "continuous":
            values = pd.to_numeric(gdf[value_col], errors="coerce")
            colors = [
                plt.get_cmap(cmap)(norm(value)) if pd.notna(value) else missing_color
                for value in values
            ]
            gdf.plot(ax=ax, color=colors, linewidth=0)
        else:
            colors = gdf[value_col].map(category_colors).fillna(missing_color)
            gdf.plot(ax=ax, color=colors, edgecolor="white", linewidth=0.03)

        boundary = test_boundary if is_mainland else _boundary_for_layer(test_boundary, gdf)
        _plot_boundary(ax, boundary)
        _set_map_extent(ax, gdf, padding=0.0 if is_mainland else 0.05)

    plot_layer(ax_main, mainland, is_mainland=True)
    ax_main.set_axis_off()

    def layer_extent_area(layer: IslandLayer) -> float:
        bounds = np.asarray(layer[1].total_bounds, dtype=float)
        if not np.isfinite(bounds).all():
            return 0.0
        minx, miny, maxx, maxy = bounds
        return max(maxx - minx, 0.0) * max(maxy - miny, 0.0)

    islands_sorted = sorted(islands, key=layer_extent_area, reverse=True)
    if islands_sorted:
        top = 0.92
        bottom = 0.10
        left = 0.05
        width = 0.25
        gap = 0.02
        height = (top - bottom - gap * (len(islands_sorted) - 1)) / len(islands_sorted)

        for index, (name, island) in enumerate(islands_sorted):
            y0 = top - (index + 1) * height - index * gap
            ax_inset = fig.add_axes([left, y0, width, height], frameon=True)
            plot_layer(ax_inset, island)
            if show_island_titles:
                ax_inset.set_title(name, fontsize=10)
            ax_inset.set_facecolor("white")
            for spine in ax_inset.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.2)

    if mode == "continuous":
        missing_count = int(continuous_values.isna().sum())
    else:
        missing_count = sum(int(gdf[value_col].isna().sum()) for _, gdf in layers)
    displayed_title = title
    if show_missing_count:
        displayed_title += f"  |  missing: {missing_count}"
    ax_main.set_title(displayed_title, fontsize=14)

    if mode == "continuous":
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        fig.colorbar(
            scalar_map,
            ax=ax_main,
            label=legend_title or value_col,
            fraction=0.04,
            pad=0.02,
        )
    else:
        labels = category_labels or {}
        handles = [
            Patch(
                facecolor=color,
                edgecolor="none",
                label=labels.get(value, str(value)),
            )
            for value, color in category_colors.items()
        ]
        if missing_count:
            handles.append(
                Patch(facecolor=missing_color, edgecolor="none", label=missing_label)
            )
        ax_main.legend(
            handles=handles,
            title=legend_title,
            loc="lower left",
            frameon=True,
        )

    if out_path is not None:
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    if out_path is not None and not show:
        plt.close(fig)
    return fig
