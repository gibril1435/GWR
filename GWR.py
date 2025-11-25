# 1. INSTALLATION & SETUP
# Run these in your terminal if you haven't already:
# pip install contextily splot mapclassify mgwr stargazer geopandas libpysal esda

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # Ensures pop-up windows work on Linux
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import libpysal as ps
from libpysal import weights
from libpysal.weights import Queen
import esda
from esda.moran import Moran, Moran_Local
import splot
from splot.esda import moran_scatterplot, plot_moran, lisa_cluster, plot_local_autocorrelation
from splot.libpysal import plot_spatial_weights
from giddy.directional import Rose
import statsmodels.api as sm
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer, LineLocation
from spreg import OLS, MoranRes, ML_Lag, ML_Error
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import ttk

# --- HELPER FUNCTION FOR POP-UP TABLES ---
def show_table(df, title="Table View"):
    """
    Opens a pop-up window displaying the DataFrame in a scrollable grid.
    Blocks script execution until the window is closed.
    """
    # Create the main window
    root = tk.Tk()
    root.title(title)
    root.geometry("1000x500")  # Set default size

    # Create a Frame to hold the table and scrollbars
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create Scrollbars
    tree_scroll_y = tk.Scrollbar(frame)
    tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    tree_scroll_x = tk.Scrollbar(frame, orient='horizontal')
    tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    # Create the Treeview (The Table)
    # We convert all columns to strings for display safety
    cols = list(df.columns)
    tree = ttk.Treeview(frame, columns=cols, show='headings', 
                        yscrollcommand=tree_scroll_y.set, 
                        xscrollcommand=tree_scroll_x.set)

    # Configure Scrollbars
    tree_scroll_y.config(command=tree.yview)
    tree_scroll_x.config(command=tree.xview)

    # Define Headings and Column widths
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="w") # Adjust width as needed

    # Add Data to the Treeview
    # We stick to the first 500 rows to keep it fast
    for index, row in df.head(500).iterrows():
        # Convert row values to a list of strings
        vals = [str(x) for x in row.tolist()]
        tree.insert("", "end", values=vals)

    tree.pack(fill=tk.BOTH, expand=True)

    # Start the loop (This pauses the script until you close the window)
    print(f"Opening popup: {title}...")
    root.mainloop()

# Configuration
pd.set_option('display.max_columns', None)
mpl.rcParams['figure.dpi'] = 72
sns.set_style("darkgrid")
sns.set_context(context="paper", font_scale=1.5)
sns.set(font="serif")
warnings.filterwarnings('ignore')

# 2. DATA LOADING
print("Loading Shapefile...")
try:
    gdf = gpd.read_file('spasial_jateng.zip') 
except Exception as e:
    print(f"Error loading shapefile: {e}. Please check the file path.")
    exit()

# Visual Check 1 (Map)
fig, ax = plt.subplots(figsize=(6, 6))
gdf.plot(color='white', edgecolor='black', ax=ax)
gdf.centroid.plot(ax=ax)
ax.set_title('Map of Jawa Tengah and centroids', fontsize=12)
ax.axis("off")
plt.show()

# Visual Check 2 (Map)
fig, ax = plt.subplots(figsize=(15, 15))
gdf.plot(column='Kabupaten', cmap='tab20', linewidth=0.01, legend=True, 
         legend_kwds={'bbox_to_anchor':(1.10, 0.96)}, ax=ax)
ax.set_title('Peta Jawa Tengah berdasarkan Kabupaten', fontsize=12)
ax.axis("off")
plt.show()

# 3. DATA PREPROCESSING
print("Filtering Data...")
kabupaten_to_remove = ['Hutan', 'Wadung Kedungombo']
gdf_filtered = gdf[~gdf['Kabupaten'].isin(kabupaten_to_remove)]

# --- POPUP 1: Filtered Data ---
show_table(gdf_filtered.drop(columns='geometry'), "Filtered Map Data")

print("Loading CSV Data...")
try:
    df = pd.read_csv("jateng_mmr.csv")
except Exception as e:
    print(f"Error loading CSV: {e}. Please check the file path.")
    exit()

# Merge Data
gdf_merged = gdf_filtered.merge(df, left_on='Kabupaten', right_on='Lokasi', how='left')

# --- POPUP 2: Merged Data ---
show_table(gdf_merged.drop(columns='geometry'), "Merged Data (Ready for GWR)")

# Prepare Variables
y = gdf_merged['MMR'].values.reshape((-1,1))
X = gdf_merged[['Jumlah_Dokter', 'Jumlah_Faskes', 'Pct_Persalinan_Ditolong_Nakes', 'Pct_Layanan_Ibu_Nifas']].values

# Prepare Coordinates (Projecting to UTM 49S for accurate meters)
gdf_utm = gdf_merged.to_crs(epsg=32749)
gdf_utm['rep_point'] = gdf_utm.geometry.representative_point()
u = gdf_utm['rep_point'].x
v = gdf_utm['rep_point'].y
coords = list(zip(u, v))

print(f"Shape of y: {y.shape}")
print(f"Shape of X: {X.shape}")

# 4. GWR ANALYSIS
print("Calculating Optimal Bandwidth...")
gwr_selector_gaussian = Sel_BW(coords, y, X, kernel='gaussian', fixed=False)
gwr_bw_gaussian = gwr_selector_gaussian.search(bw_max=len(coords)-1, bw_min=2)
print("Optimal adaptive bandwidth global:", gwr_bw_gaussian)

# Calculate Local Bandwidth Distances
nbrs = NearestNeighbors(n_neighbors=int(gwr_bw_gaussian) + 1, algorithm='ball_tree').fit(coords)
distances, indices = nbrs.kneighbors(coords)
gdf_merged['Bandwidth Lokal'] = distances[:, int(gwr_bw_gaussian)]

# --- POPUP 3: Local Bandwidths ---
show_table(gdf_merged[['Kabupaten', 'Bandwidth Lokal']], "Local Bandwidth Distances")

# Fit GWR Model
print("Fitting GWR Model...")
gwr_model_gaussian = GWR(coords, y, X, bw=gwr_bw_gaussian, kernel='gaussian', fixed=False)
gwr_results_gaussian = gwr_model_gaussian.fit()
print(gwr_results_gaussian.summary())

# 5. RESULTS & VISUALIZATION
gdf_merged['gwr_R2'] = gwr_results_gaussian.localR2

# Plot Local R2
fig, ax = plt.subplots(figsize=(6, 6))
gdf_merged.plot(column='gwr_R2', cmap='coolwarm', linewidth=0.01, scheme='FisherJenks', k=5, 
                legend=True, legend_kwds={'bbox_to_anchor':(1.10, 0.96)}, ax=ax)
ax.set_title('Local R2', fontsize=12)
ax.axis("off")
plt.show()

# Extract Parameters
params = gwr_results_gaussian.params
gdf_merged['intercept']           = params[:, 0]
gdf_merged['b1_Jumlah_Dokter']    = params[:, 1]
gdf_merged['b2_Faskes']           = params[:, 2]
gdf_merged['b3_Persalinan_Nakes'] = params[:, 3]
gdf_merged['b4_Layanan_Nifas']    = params[:, 4]
gdf_merged['y_pred_gwr']          = gwr_results_gaussian.predy

# --- POPUP 4: GWR Coefficients & Predictions ---
cols_to_view = ['Kabupaten', 'intercept', 'b1_Jumlah_Dokter', 'b2_Faskes', 'b3_Persalinan_Nakes', 'b4_Layanan_Nifas', 'y_pred_gwr', 'gwr_R2']
show_table(gdf_merged[cols_to_view], "GWR Coefficients and R2")

# Filter t-values
gwr_filtered_t = gwr_results_gaussian.filter_tvals(alpha=0.05)
gwr_filtered_tc = gwr_results_gaussian.filter_tvals()

# --- POPUP 5: T-Values ---
t_val_df = pd.DataFrame(gwr_filtered_t, columns=['Intercept_t', 'Dokter_t', 'Faskes_t', 'Persalinan_t', 'Nifas_t'])
# CORRECTED LABEL BELOW:
show_table(t_val_df, "Significant T-Values (Non-Zero = Significant, 0 = Not Significant)")

# Helper function to plot coefficients
def plot_gwr_coeffs(column_name, title_base, col_index):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
    
    # All coeffs
    gdf_merged.plot(column=column_name, cmap='coolwarm', linewidth=0.01, scheme='FisherJenks', k=5, 
                    legend=True, legend_kwds={'bbox_to_anchor':(1.10, 0.96)}, ax=axes[0])
    axes[0].set_title(f'(a) {title_base} (All)', fontsize=10)
    
    # Significant coeffs
    # NOTE: The logic here matches the PDF. We plot white where the t-value is 0 (Insignificant).
    gdf_merged.plot(column=column_name, cmap='coolwarm', linewidth=0.05, scheme='FisherJenks', k=5, 
                    legend=False, ax=axes[1])
    gdf_merged[gwr_filtered_t[:, col_index] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[1])
    axes[1].set_title(f'(b) {title_base} (Significant)', fontsize=10)
    
    # Corrected Significant coeffs
    gdf_merged.plot(column=column_name, cmap='coolwarm', linewidth=0.05, scheme='FisherJenks', k=5, 
                    legend=False, ax=axes[2])
    gdf_merged[gwr_filtered_tc[:, col_index] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[2])
    axes[2].set_title(f'(c) {title_base} (Corrected Sig)', fontsize=10)

    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.show()

# Plot variables
plot_gwr_coeffs('b1_Jumlah_Dokter', 'Jumlah Dokter', 1)
plot_gwr_coeffs('b2_Faskes', 'Jumlah Faskes', 2)
plot_gwr_coeffs('b3_Persalinan_Nakes', 'Persalinan Nakes', 3)
plot_gwr_coeffs('b4_Layanan_Nifas', 'Layanan Nifas', 4)

print("Analysis Complete.")