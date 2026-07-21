from config import PRESSURECUTOFF, STRENGTH, INCREMENT, PARAMETERS
from data_io import lambdaFromFolder
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import os
import scipy
import scipy.ndimage
from matplotlib.colors import LogNorm


def finalPlots(allMultipoleData, folderList, allTimeSteps, allRadii):
    '''Plot the multipole moments and radii for each simulation folder. Generates multiple subplots for analysis.'''

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    TychoP2 = 3*10**(-7)
    TychoP3 = 1.4*10**(-7)
    TychoTime = 0.2

    # Distinct color per simulation (line color); time still shown via colorbar on axs[0]
    colorMap = plt.cm.tab10(np.linspace(0, 1, len(folderList)))

    # Global time range across all folders, for a shared color normalization on axs[0]
    allTimesFlat = np.concatenate([np.array(allTimeSteps[f]) for f in folderList])
    tNorm = LogNorm(vmin=allTimesFlat.min(), vmax=allTimesFlat.max())

    # Precompute R/Lambda label for each folder (R = final radius of that simulation)
    labelMap = {}
    lambdaMap = {}
    for folder in folderList:
        lam = lambdaFromFolder(folder)
        lambdaMap[folder] = lam
        R = allRadii[folder][-1]
        labelMap[folder] = f"{folder}: R/\u03bb = {R / lam:.3g}"

    sc = None  # keep a handle for the colorbar

    for folderIdx, folder in enumerate(folderList):
        transposedData = list(zip(*allMultipoleData[folder]))
        P2 = np.array(transposedData[0])
        P3 = np.array(transposedData[1])
        t = np.array(allTimeSteps[folder])
        radii = allRadii[folder]
        color = colorMap[folderIdx]
        label = labelMap[folder]

        # -------------------------------------------------------
        # 1. Phase plot with trajectory line + color
        # -------------------------------------------------------
        axs[0].plot(P3, P2, color=color, alpha=0.5, linewidth=1, label=label)
        axs[0].plot(TychoP3, TychoP2, marker='o', markersize=6, color='black', linestyle='none')
        axs[0].annotate(
            "Tycho",
            (TychoP3, TychoP2),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color='black'
        )
        sc = axs[0].scatter(
            P3, P2,
            c=t,
            cmap="viridis",
            norm=tNorm,
            s=40,
            edgecolor='k',
            linewidth=0.3
        )

        # Annotate a few time points along the trajectory for clarity
        def fmt_time(val):
            return f"{val:.2e}"

        idxs = np.linspace(0, len(t) - 1, 4, dtype=int)
        for i in idxs:
            axs[0].annotate(
                fmt_time(t[i]),
                (P3[i], P2[i]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8
            )

        # -------------------------------------------------------
        # 2. P2/P0 vs time
        # -------------------------------------------------------
        axs[1].plot(t, P2, marker='o', markersize=3, linewidth=1, color=color, label=label)
        axs[1].plot(TychoTime, TychoP2, marker='o', markersize=6, color='black', linestyle='none')
        axs[1].annotate(
            "Tycho",
            (TychoTime, TychoP2),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color='black'
        )
        # -------------------------------------------------------
        # 3. P3/P0 vs time
        # -------------------------------------------------------
        axs[2].plot(t, P3, marker='o', markersize=3, linewidth=1, color=color, label=label)
        axs[2].plot(TychoTime, TychoP3, marker='o', markersize=6, color='black', linestyle='none')
        axs[2].annotate(
            "Tycho",
            (TychoTime, TychoP3),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color='black'
        )

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel("P3/P0")
    axs[0].set_ylabel("P2/P0")
    axs[0].set_title("Multipole Time Evolutions")
    axs[0].legend(fontsize=8)

    cbar = fig.colorbar(sc, ax=axs[0])
    cbar.set_label("Time")

    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("P2/P0")
    axs[1].legend(fontsize=8)

    axs[2].set_yscale('log')
    axs[2].set_xscale('log')
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("P3/P0")
    axs[2].legend(fontsize=8)

    plt.show()

    # -------------------------------------------------------
    # 4. Radius
    # -------------------------------------------------------
    plt.figure(figsize=(6, 4))
    for folderIdx, folder in enumerate(folderList):
        t = np.array(allTimeSteps[folder])
        radii = allRadii[folder]
        plt.plot(t, radii, marker='o', markersize=3, color=colorMap[folderIdx], label=labelMap[folder])
    plt.xlabel("Time")
    plt.ylabel("Radius")
    plt.title("Radius evolution")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # 5. P2 and P3 vs Radius
    # -------------------------------------------------------
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for folderIdx, folder in enumerate(folderList):
        transposedData = list(zip(*allMultipoleData[folder]))
        P2 = np.array(transposedData[0])
        P3 = np.array(transposedData[1])
        radii = np.array(allRadii[folder])
        color = colorMap[folderIdx]
        label = labelMap[folder]

        axs2[0].plot(radii, P2, marker='o', markersize=3, linewidth=1, color=color, label=label)
        axs2[1].plot(radii, P3, marker='o', markersize=3, linewidth=1, color=color, label=label)

    axs2[0].set_yscale('log')
    axs2[0].set_xscale('log')
    axs2[0].set_xlabel("Radius")
    axs2[0].set_ylabel("P2/P0")
    axs2[0].set_title("P2/P0 vs Radius")
    axs2[0].legend(fontsize=8)

    axs2[1].set_yscale('log')
    axs2[1].set_xscale('log')
    axs2[1].set_xlabel("Radius")
    axs2[1].set_ylabel("P3/P0")
    axs2[1].set_title("P3/P0 vs Radius")
    axs2[1].legend(fontsize=8)

    plt.show()

    # -------------------------------------------------------
    # 6. P2 and P3 vs R/Lambda
    # -------------------------------------------------------
    fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for folderIdx, folder in enumerate(folderList):
        transposedData = list(zip(*allMultipoleData[folder]))
        P2 = np.array(transposedData[0])
        P3 = np.array(transposedData[1])
        t = np.array(allTimeSteps[folder])
        radii = np.array(allRadii[folder])
        color = colorMap[folderIdx]
        label = labelMap[folder]
        rOverLambda = radii / lambdaFromFolder(folder)

        # -------------------------------------------------------
        # 1. Phase plot with trajectory line + color + sparse labels
        # -------------------------------------------------------
        axs3[0].plot(rOverLambda, P2, color=color, alpha=0.5, linewidth=1, label=label)
        axs3[1].plot(rOverLambda, P3, color=color, alpha=0.5, linewidth=1, label=label)

    axs3[0].set_yscale('log')
    axs3[0].set_xscale('log')
    axs3[0].set_xlabel("R/Lam")
    axs3[0].set_ylabel("P2")
    axs3[0].set_title("P2 vs R/Lam")
    axs3[0].legend(fontsize=8)

    axs3[1].set_yscale('log')
    axs3[1].set_xscale('log')
    axs3[1].set_xlabel("R/Lam")
    axs3[1].set_ylabel("P3")
    axs3[1].set_title("P3 vs R/Lam")
    axs3[1].legend(fontsize=8)


def plotFiles(processedData, fileNumberList, max_cols=5):

    num_files = len(processedData)

    ncols = min(num_files, max_cols)
    nrows = math.ceil(num_files / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5*ncols, 4*nrows),
        constrained_layout=True
    )

    axes = np.array(axes).reshape(-1)

    for idx, data in enumerate(processedData):

        plotData = np.sum(data[0], axis=1)

        im = axes[idx].imshow(plotData, cmap='hot', aspect='auto')
        axes[idx].set_title(f"File {fileNumberList[idx]}")

        fig.colorbar(im, ax=axes[idx])

    # Hide unused axes
    for idx in range(num_files, len(axes)):
        axes[idx].axis('off')

    plt.show()