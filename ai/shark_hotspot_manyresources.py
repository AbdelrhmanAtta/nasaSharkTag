# combined_shark_hotspot_full_fixed.py
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import datetime

warnings.simplefilter("default")

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Optional mapping / gridding
try:
    from scipy import stats
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Optional cartopy for nicer maps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False

# ---------------- Basic settings ----------------
base_dir = os.path.abspath("/content/drive/MyDrive/nasa-data/monthly-data-folder")
if not os.path.exists(base_dir):
    base_dir = os.getcwd()  # fallback for testing

# Region subset (adjust if you want global)
# Use None for no subsetting
REGION = dict(lat=slice(-40, 40), lon=slice(-80, -20))  # Example Atlantic; set to None for global

# ---------- Utility helpers ----------
def detect_coord_name(ds):
    """Return latname, lonname heuristically from dataset coords/variables."""
    # Look in coords first
    coords = list(ds.coords)
    latname = None
    lonname = None
    for name in coords:
        n = name.lower()
        if 'lat' in n and latname is None:
            latname = name
        if 'lon' in n and lonname is None:
            lonname = name
    # If not found in coords, search variables
    if latname is None or lonname is None:
        for name in ds.variables:
            n = name.lower()
            if latname is None and 'lat' in n:
                latname = name
            if lonname is None and 'lon' in n:
                lonname = name
    # fallback to first two dims (dangerous but last option)
    if latname is None:
        latname = list(ds.dims)[0] if len(ds.dims) > 0 else None
    if lonname is None:
        lonname = list(ds.dims)[1] if len(ds.dims) > 1 else latname
    return latname, lonname

def find_data_var(ds, keywords):
    for v in ds.data_vars:
        vn = v.lower()
        for kw in keywords:
            if kw in vn:
                return v
    # fallback
    return list(ds.data_vars)[0]

def take_first_time_if_present(da):
    """Return the first time slice if 'time' is a dimension; else the DA itself."""
    if da is None:
        return da
    if 'time' in da.dims:
        # if time length 0 -> return squeezed da (will be empty)
        try:
            if da.sizes.get('time', 0) > 0:
                return da.isel(time=0)
            else:
                # empty time dimension; try dropping it
                return da.isel(time=0, drop=True)
        except Exception:
            # final fallback: squeeze
            return da.squeeze()
    else:
        return da.squeeze()

def safe_open(path):
    """Open dataset RAM-safe if possible."""
    try:
        # Use chunks to avoid loading all in memory (xarray will lazy load)
        return xr.open_dataset(path, decode_times=True, mask_and_scale=True, chunks={})
    except Exception:
        return xr.open_dataset(path)

def normalize_lon_array(lon):
    lon = np.array(lon)
    if lon.size == 0:
        return lon
    if lon.max() > 180:
        lon = ((lon + 180) % 360) - 180
    return lon

def robust_slice_lat(lat_vals, desired_slice):
    """Return an adjusted slice object that respects lat ordering."""
    if desired_slice is None:
        return None
    if lat_vals is None or len(lat_vals) == 0:
        return desired_slice
    # desired_slice is slice(start, stop)
    start = desired_slice.start
    stop = desired_slice.stop
    # ensure start/stop defined
    if start is None or stop is None:
        return desired_slice
    if lat_vals[0] > lat_vals[-1]:
        # lat decreasing; reverse desired slice
        return slice(stop, start)
    else:
        return slice(start, stop)

def subset_region(ds, region=REGION):
    """Subset ds to region (if provided) robustly handling lat ordering and lon normalization."""
    if region is None:
        return ds
    latname, lonname = detect_coord_name(ds)
    if latname not in ds or lonname not in ds:
        # nothing to subset
        return ds
    try:
        lat_vals = ds[latname].values
        lon_vals = ds[lonname].values
        # normalize lon coordinate if >180
        if np.isfinite(np.nanmax(lon_vals)) and np.nanmax(lon_vals) > 180:
            lon_norm = ((lon_vals + 180) % 360) - 180
            ds = ds.assign_coords({lonname: lon_norm})
            lon_vals = lon_norm
        # build slice for lat that respects ordering
        lat_slice = robust_slice_lat(lat_vals, region['lat'])
        ds2 = ds.sel({latname: lat_slice, lonname: region['lon']}, method=None, drop=True)
        # Some datasets may have transposed dims (lat/lon as 2D) - we try simple sel, fallback to no subset
        return ds2
    except Exception as e:
        print("subset_region: fallback, error:", e)
        return ds

# ---------- Input file paths ----------
file_paths = {
    # MODIS (baseline)
    "chl":  "/monthly-data-folder/AQUA_MODIS.20240601_20240630.L3m.MO.CHL.chlor_a.4km.nc",
    "flh":  "/monthly-data-folder/AQUA_MODIS.20240601_20240630.L3m.MO.FLH.ipar.4km.nc",
    "aph":  "/monthly-data-folder/AQUA_MODIS.20240601_20240630.L3m.MO.IOP.aph_443.4km.nc",
    "kd":   "/monthly-data-folder/AQUA_MODIS.20240601_20240630.L3m.MO.KD.Kd_490.4km.nc",
    "nsst": "/monthly-data-folder/AQUA_MODIS.20240601_20240630.L3m.MO.NSST.sst.4km.nc",
    "poc":  "/nasa-data/monthly-data-folder/AQUA_MODIS.20240601_20240630.L3m.MO.POC.poc.4km.nc",
    "aot":  "/content/drive/MyDrive/nasa-data/monthly-data-folder/AQUA_MODIS.20240601_20240630.L3m.MO.RRS.aot_869.4km.nc",
    "sst":  "/content/drive/MyDrive/nasa-data/monthly-data-folder/AQUA_MODIS.20240601_20240630.L3m.MO.SST.sst.4km.nc",
    "sst4": "/content/drive/MyDrive/nasa-data/monthly-data-folder/AQUA_MODIS.20240601_20240630.L3m.MO.SST4.sst4.4km.nc",

    # PACE products
    "pace_rrs_202507": "/content/drive/MyDrive/nasa-data/monthly-data-folder/PACE_OCI.20250701.L3m.DAY.RRS.V3_1.Rrs.0p1deg.NRT.nc",
    "pace_aer_202508": "/content/drive/MyDrive/nasa-data/monthly-data-folder/PACE_OCI.20250801.L3m.DAY.AER_UAA.V3_1.0p1deg.NRT.nc",
    "pace_aer_202509": "/content/drive/MyDrive/nasa-data/monthly-data-folder/PACE_OCI.20250915.L3m.DAY.AER_UAA.V3_1.1deg.NRT.nc",
    "pace_aer_202510": "/content/drive/MyDrive/nasa-data/monthly-data-folder/PACE_OCI.20251004.L3m.DAY.AER_UAA.V3_1.1deg.NRT.nc",

    # SM (Sea surface salinity or similar)
    "sm_1": "/content/drive/MyDrive/nasa-data/monthly-data-folder/SM_D2010152_Map_SATSSS_data_1day_1deg.nc",
    "sm_2": "/content/drive/MyDrive/nasa-data/monthly-data-folder/SM_D2010154_Map_SATSSS_data_3days.nc",
    "sm_3": "/content/drive/MyDrive/nasa-data/monthly-data-folder/SM_D2010155_Map_SATSSS_data_1day.nc",

    # SWOT (SSH)
    "swot_ssh": "/content/drive/MyDrive/nasa-data/monthly-data-folder/SWOT_L2_LR_SSH_Expert_632_018_20151230T234907_20151231T004012_DG10_01.nc",
}

# Check files presence
for k, p in file_paths.items():
    if not os.path.exists(p):
        print(f"Warning: file for key '{k}' not found at: {p}. Skipping.")

# ---------- Open datasets (RAM-safe) ----------
ds_objs = {}
for key, path in file_paths.items():
    if os.path.exists(path):
        try:
            ds = safe_open(path)
            ds = subset_region(ds, REGION)
            ds_objs[key] = ds
            print(f"Opened {key}: dims={ds.dims.keys()} vars={list(ds.data_vars)[:3]}")
        except Exception as e:
            print(f"Error opening {path}: {e}")

# ---------- Build reference (CHL) grid ----------
if "chl" not in ds_objs:
    raise RuntimeError("Chlorophyll reference file not found! Put a correct path to 'chl' in file_paths.")

chl_ds = ds_objs["chl"]
chl_lat_name, chl_lon_name = detect_coord_name(chl_ds)
chl_var = find_data_var(chl_ds, ['chlor', 'chl_a', 'chlor_a', 'chlorophyll'])
print("Reference (CHL):", chl_lat_name, chl_lon_name, chl_var)

chl_da = take_first_time_if_present(chl_ds[chl_var]).squeeze()
# If chl_da has unexpected dims (e.g., [time, lat, lon] with time length 0) this may be empty
if chl_da.size == 0:
    raise RuntimeError("Reference chlorophyll dataset contains no data after slicing. Check REGION or file contents.")

lat_vals = np.array(chl_da[chl_lat_name].values).ravel()
lon_vals = normalize_lon_array(np.array(chl_da[chl_lon_name].values).ravel())

if lat_vals.size == 0 or lon_vals.size == 0:
    raise RuntimeError("Reference grid lat/lon arrays are empty.")

# Use 1D lat/lon arrays as target
target_lat = xr.DataArray(lat_vals, dims=[chl_lat_name], coords={chl_lat_name: lat_vals})
target_lon = xr.DataArray(lon_vals, dims=[chl_lon_name], coords={chl_lon_name: lon_vals})

# ---------- Regrid function for L3-like datasets ----------
def extract_and_regrid(ds_obj, prefer_keys=None):
    latname, lonname = detect_coord_name(ds_obj)
    varname = find_data_var(ds_obj, prefer_keys if prefer_keys else [])
    da = ds_obj[varname]
    da = take_first_time_if_present(da)
    # If DA has 1D coords with different names, attempt to rename coords to latname/lonname
    if lonname in da.coords:
        src_lon = np.array(da[lonname].values).ravel()
        if src_lon.size > 0 and np.nanmax(src_lon) > 180:
            newlon = ((src_lon + 180) % 360) - 180
            da = da.assign_coords({lonname: newlon})
    # Try to interp to target grid (nearest)
    try:
        coords = {latname: target_lat, lonname: target_lon}
        da_on_target = da.interp(coords, method="nearest")
    except Exception:
        # fallback: attempt to transpose or rename dims
        try:
            da_on_target = da.rename({latname: chl_lat_name, lonname: chl_lon_name})
        except Exception:
            da_on_target = da
    # finally, ensure dims are named to chl grid dims
    try:
        da_on_target = da_on_target.rename({latname: chl_lat_name, lonname: chl_lon_name})
    except Exception:
        pass
    return da_on_target.squeeze()

# ---------- Convert SWOT L2 -> Level-3 (grid) ----------
def swot_to_level3_grid(swot_ds, targ_lats, targ_lons, ssh_var_candidates=('ssh','ssha','adt')):
    """Grids SWOT L2-like along-track points into the target lat/lon grid.
       Returns a numpy 2D array (nlat, nlon) with median SSH per cell (np.nan where empty).
    """
    # find SSH variable
    varname = None
    for v in swot_ds.data_vars:
        vn = v.lower()
        for cand in ssh_var_candidates:
            if cand in vn:
                varname = v
                break
        if varname:
            break
    if varname is None:
        varname = list(swot_ds.data_vars)[0]
        print("SWOT: couldn't locate ssh var, using", varname)

    # detect lat/lon names (could be different)
    latname, lonname = detect_coord_name(swot_ds)

    # Extract raw arrays: try coords first, else variables
    def get_array(ds, name_candidates):
        for n in name_candidates:
            if n in ds.coords:
                return np.array(ds.coords[n].values).ravel()
            if n in ds.variables:
                return np.array(ds[n].values).ravel()
        return None

    lon_pts = None
    lat_pts = None
    # try common names
    lat_candidates = [latname, 'latitude', 'lat', 'y']
    lon_candidates = [lonname, 'longitude', 'lon', 'x']
    lat_try = get_array(swot_ds, lat_candidates)
    lon_try = get_array(swot_ds, lon_candidates)
    ssh_try = np.array(swot_ds[varname].values).ravel()

    if lat_try is None or lon_try is None or ssh_try.size == 0:
        # brute force: search all variables for something like 'lat' or 'lon'
        for v in swot_ds.variables:
            nl = v.lower()
            if 'lat' in nl and lat_try is None:
                lat_try = np.array(swot_ds[v].values).ravel()
            if 'lon' in nl and lon_try is None:
                lon_try = np.array(swot_ds[v].values).ravel()
    lon_pts = lon_try
    lat_pts = lat_try
    ssh_pts = ssh_try

    if lon_pts is None or lat_pts is None or ssh_pts is None:
        raise RuntimeError("SWOT: couldn't extract lat/lon/ssh arrays for gridding.")

    # mask invalids
    valid = np.isfinite(lon_pts) & np.isfinite(lat_pts) & np.isfinite(ssh_pts)
    lon_pts = lon_pts[valid]
    lat_pts = lat_pts[valid]
    ssh_pts = ssh_pts[valid]
    if lon_pts.size == 0:
        raise RuntimeError("SWOT: no valid points after masking.")

    # normalize lon to -180..180
    lon_pts = ((lon_pts + 180) % 360) - 180

    # create meshgrid for target
    GLON, GLAT = np.meshgrid(targ_lons, targ_lats)
    grid_shape = GLAT.shape
    ssh_grid = np.full(grid_shape, np.nan, dtype=float)

    if SCIPY_AVAILABLE:
        try:
            # binned_statistic_2d expects x,y order; choose bins sizes equal to number of unique target coords
            xbins = len(targ_lons)
            ybins = len(targ_lats)
            # define range slightly padded
            x_range = [np.min(targ_lons)-1e-6, np.max(targ_lons)+1e-6]
            y_range = [np.min(targ_lats)-1e-6, np.max(targ_lats)+1e-6]
            stat, xe, ye, binnum = stats.binned_statistic_2d(
                lon_pts, lat_pts, ssh_pts, statistic='median', bins=[xbins, ybins], range=[x_range, y_range]
            )
            # stat shape is (xbins, ybins) -> transpose to (y, x)
            ssh_grid = stat.T
            return ssh_grid
        except Exception as e:
            print("SWOT binned_statistic_2d failed; falling back to griddata:", e)

    # fallback to griddata linear interpolation (may leave NaNs)
    try:
        pts = np.column_stack((lon_pts, lat_pts))
        grid_z = griddata(points=pts, values=ssh_pts, xi=(GLON, GLAT), method='linear')
        ssh_grid = grid_z
        return ssh_grid
    except Exception as e:
        raise RuntimeError("SWOT gridding failed (griddata fallback): " + str(e))

# ---------- Preferred keywords mapping ----------
pref_map = {
    "flh": ['flh', 'ipar'],
    "aph": ['aph', 'aph_443'],
    "kd": ['kd', 'kd_490'],
    "nsst": ['nsst', 'sst'],
    "poc": ['poc'],
    "aot": ['aot', 'aerosol'],
    "sst": ['sst', 'temperature'],
    "sst4": ['sst4'],
    "pace_rrs_202507": ['rrs', 'r_rs'],
    "pace_aer_202508": ['aer', 'aerosol'],
    "pace_aer_202509": ['aer', 'aerosol'],
    "pace_aer_202510": ['aer', 'aerosol'],
    "sm_1": ['sss', 'salinity', 'sss_mean'],
    "sm_2": ['sss', 'salinity'],
    "sm_3": ['sss', 'salinity'],
    "swot_ssh": ['ssh', 'ssha', 'adt'],
}

# ---------- Extract & regrid all non-SWOT features ----------
features = {}
# add chlorophyll reference
features["chlor_a"] = chl_da

for key in file_paths.keys():
    if key == "chl":
        continue
    if key not in ds_objs:
        continue
    ds_obj = ds_objs[key]
    try:
        if key == "swot_ssh":
            print("SWOT dataset detected; converting to level-3 grid.")
            swot_grid = swot_to_level3_grid(ds_obj, lat_vals, lon_vals)
            swot_da = xr.DataArray(swot_grid, coords={chl_lat_name: lat_vals, chl_lon_name: lon_vals}, dims=[chl_lat_name, chl_lon_name])
            features[key] = swot_da
            print(f"Converted SWOT to gridded SSH, shape: {swot_grid.shape}")
        else:
            da_reg = extract_and_regrid(ds_obj, prefer_keys=pref_map.get(key, None))
            # ensure we have lat/lon dims named as in reference
            try:
                da_reg = da_reg.rename({detect_coord_name(ds_obj)[0]: chl_lat_name, detect_coord_name(ds_obj)[1]: chl_lon_name})
            except Exception:
                pass
            features[key] = da_reg
            print(f"Loaded and regridded {key}, shape: {getattr(da_reg, 'shape', 'unknown')}")
    except Exception as e:
        print(f"Failed to process {key}: {e}")

# ---------- Convert features to numpy grids ----------
def da_to_grid(da):
    try:
        arr = np.array(da.values)
        return np.squeeze(arr)
    except Exception:
        return None

grid_arrays = {}
for k, v in features.items():
    arr = da_to_grid(v)
    if arr is None:
        print(f"Feature {k} could not be converted to grid (None).")
    else:
        grid_arrays[k] = arr

if 'chlor_a' not in grid_arrays:
    raise RuntimeError("Chlorophyll grid not available - aborting.")

chl_grid = grid_arrays['chlor_a']
grid_shape = chl_grid.shape
print("Grid shape:", grid_shape)

# Guard: no-data grid
if grid_shape[0] == 0 or grid_shape[1] == 0:
    raise RuntimeError("Reference grid has zero dimension (empty). Check the CHL file or REGION.")

# Flatten into feature vectors
flat_features = {k: arr.flatten() for k, arr in grid_arrays.items()}
n_pixels = flat_features['chlor_a'].size

# Choose feature order and which ones to use for model/rules
feature_keys = ['chlor_a', 'sst', 'sst4', 'nsst', 'poc', 'kd', 'aph', 'flh', 'aot', 'pace_rrs', 'pace_aer', 'swot_ssh']
X_cols = []
col_names = []
for k in feature_keys:
    if k in flat_features:
        X_cols.append(flat_features[k])
        col_names.append(k)
    else:
        print(f"Feature {k} not found - skipping.")

if len(X_cols) == 0:
    raise RuntimeError("No features available to build X. Check data loading.")

X = np.vstack(X_cols).T  # shape (n_pixels, n_features)

# ---------- Mask and fill ----------
mask_valid = np.isfinite(flat_features['chlor_a'])
# optionally require SST too if present
if any(name in flat_features for name in ['sst','nsst','sst4']):
    sst_any = None
    for n in ['sst','nsst','sst4']:
        if n in flat_features:
            sst_any = flat_features[n]
            break
    if sst_any is not None:
        mask_valid &= np.isfinite(sst_any)

# debug check
print(f"Valid pixels count: {np.sum(mask_valid)} / {mask_valid.size}")
if np.sum(mask_valid) == 0:
    raise RuntimeError("No valid pixels found after masking. Check REGION and dataset overlaps.")

# Fill NaNs per column using mean over valid mask
X_filled = X.copy()
for j in range(X_filled.shape[1]):
    col = X_filled[:, j]
    valid_for_col = mask_valid & np.isfinite(col)
    if np.any(valid_for_col):
        mean_val = np.nanmean(col[valid_for_col])
    else:
        mean_val = 0.0
    # replace NaNs
    nanidx = ~np.isfinite(col)
    col[nanidx] = mean_val
    X_filled[:, j] = col

scaler = StandardScaler()
# Fit only on valid pixels
X_valid_scaled = scaler.fit_transform(X_filled[mask_valid, :])

# ---------- Labels (simple heuristic pseudo-label) ----------
def find_index(name):
    try:
        return col_names.index(name)
    except ValueError:
        return None

idx_chl = find_index('chlor_a')
idx_sst = find_index('sst') or find_index('nsst') or find_index('sst4')

if idx_chl is None:
    raise RuntimeError("chlor_a feature missing from columns - cannot create labels.")

if idx_sst is None:
    # label using chlor only (top tercile)
    thr = np.nanpercentile(X_valid_scaled[:, idx_chl], 66)
    y_valid = (X_valid_scaled[:, idx_chl] > thr).astype(int)
else:
    # combined pseudo-label: high chlor & moderate SST preference
    y_valid = ((X_valid_scaled[:, idx_chl] > 0.5) & (X_valid_scaled[:, idx_sst] < 0.2)).astype(int)

# ---------- Train Random Forest ----------
n_max_samples = 20000
n_samples = min(n_max_samples, X_valid_scaled.shape[0])
rng = np.random.RandomState(42)
if n_samples < X_valid_scaled.shape[0]:
    idx_sample = rng.choice(np.arange(X_valid_scaled.shape[0]), size=n_samples, replace=False)
else:
    idx_sample = np.arange(X_valid_scaled.shape[0])

strat = y_valid[idx_sample] if len(np.unique(y_valid)) > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X_valid_scaled[idx_sample, :], y_valid[idx_sample],
    test_size=0.2, random_state=42, stratify=strat
)

clf = RandomForestClassifier(n_estimators=150, max_depth=12, n_jobs=-1, random_state=42, min_samples_leaf=8)
clf.fit(X_train, y_train)
print("Random Forest classification report (test set):")
print(classification_report(y_test, clf.predict(X_test), zero_division=0))

# ---------- Predict RF proba across grid ----------
if len(clf.classes_) == 1:
    probs_valid = clf.predict(X_valid_scaled).astype(float)
else:
    probs_valid = clf.predict_proba(X_valid_scaled)[:, 1]

prob_full = np.full(n_pixels, np.nan)
prob_full[mask_valid] = probs_valid
prob_map_rf = prob_full.reshape(grid_shape)

# ---------- Rule-based x,y,z,t,w,k score ----------
# Map letters to features (tune as you like)
# x -> chlor_a
# y -> sst
# z -> swot_ssh (frontiness via local std)
# t -> flh
# w -> poc
# k -> kd (lower kd preferred)

# convenience getter
get_full = lambda name: (flat_features[name] if name in flat_features else np.full(n_pixels, np.nan))

x_full = get_full('chlor_a')
y_full = get_full('sst') if 'sst' in flat_features else (get_full('nsst') if 'nsst' in flat_features else get_full('sst4'))
z_full = get_full('swot_ssh')
t_full = get_full('flh')
w_full = get_full('poc')
k_full = get_full('kd')

def percentile_rank(arr):
    arr = np.array(arr)
    nan_mask = ~np.isfinite(arr)
    ranks = np.full(arr.shape, np.nan)
    if np.all(nan_mask):
        return ranks
    valid = arr[~nan_mask]
    order = np.argsort(valid)
    ranks_valid = np.empty_like(order, dtype=float)
    ranks_valid[order] = np.linspace(0, 1, valid.size)
    ranks[~nan_mask] = ranks_valid
    return ranks

x_pct = percentile_rank(x_full)
t_pct = percentile_rank(t_full)
w_pct = percentile_rank(w_full)
k_pct = percentile_rank(k_full)  # lower kd better -> invert later

# SST preference: prefer around pref_temp
pref_temp = 22.0
y_arr = y_full.copy()
y_score = np.full(y_arr.shape, np.nan)
valid_y = np.isfinite(y_arr)
if np.any(valid_y):
    scale = 4.0
    delta = np.abs(y_arr[valid_y] - pref_temp)
    y_score_vals = np.exp(- (delta/scale)**2)
    y_score[valid_y] = y_score_vals

# z: compute local std of swot grid as proxy for frontiness
if np.isfinite(z_full).any():
    z_grid = z_full.reshape(grid_shape)
    pad = np.pad(z_grid, 1, mode='constant', constant_values=np.nan)
    z_local_std = np.full(grid_shape, np.nan)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            window = pad[i:i+3, j:j+3]
            z_local_std[i, j] = np.nanstd(window)
    z_pct = percentile_rank(z_local_std.flatten())
else:
    z_pct = np.full(n_pixels, np.nan)

k_pct_inv = 1.0 - k_pct

# weights (sum arbitrary; normalized in computation)
weights = dict(x=0.28, y=0.20, z=0.22, t=0.12, w=0.10, k=0.08)

comp_x = x_pct
comp_y = y_score
comp_z = z_pct
comp_t = t_pct
comp_w = w_pct
comp_k = k_pct_inv

rule_score = np.full(n_pixels, np.nan)
for idx in range(n_pixels):
    comps = []
    ws = []
    if np.isfinite(comp_x[idx]):
        comps.append(comp_x[idx]); ws.append(weights['x'])
    if np.isfinite(comp_y[idx]):
        comps.append(comp_y[idx]); ws.append(weights['y'])
    if np.isfinite(comp_z[idx]):
        comps.append(comp_z[idx]); ws.append(weights['z'])
    if np.isfinite(comp_t[idx]):
        comps.append(comp_t[idx]); ws.append(weights['t'])
    if np.isfinite(comp_w[idx]):
        comps.append(comp_w[idx]); ws.append(weights['w'])
    if np.isfinite(comp_k[idx]):
        comps.append(comp_k[idx]); ws.append(weights['k'])
    if len(comps) == 0:
        rule_score[idx] = np.nan
    else:
        ws = np.array(ws)
        comps = np.array(comps)
        rule_score[idx] = np.nansum(comps * ws) / np.sum(ws)

rule_map = rule_score.reshape(grid_shape)

# ---------- Combine RF + rule ----------
alpha = 0.7
final_prob_full = np.full(n_pixels, np.nan)
mask_idx = mask_valid
final_prob_full[mask_idx] = (alpha * prob_full[mask_idx]) + ((1 - alpha) * rule_score[mask_idx])
final_prob_map = final_prob_full.reshape(grid_shape)

# ---------- Visualization ----------
def plot_probability_map(lons, lats, prob_map, title="Predicted Shark Hotspots", cmap='Spectral_r'):
    if CARTOPY_AVAILABLE:
        fig = plt.figure(figsize=(14, 7))
        ax = plt.axes(projection=ccrs.Robinson())
        # pcolormesh expects 2D lon/lat or 1D (we'll pass 1D)
        im = ax.pcolormesh(lons, lats, prob_map, shading='auto', transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=0, vmax=1)
        ax.add_feature(cfeature.LAND.with_scale('110m'), zorder=2, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'), zorder=3)
        ax.set_global()
        ax.set_title(title)
        cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, fraction=0.05)
        cbar.set_label('Hotspot probability')
        plt.show()
    else:
        plt.figure(figsize=(14, 6))
        pcm = plt.pcolormesh(lons, lats, prob_map, shading='auto', cmap=cmap, vmin=0, vmax=1)
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.title(title + " (Cartopy not available)")
        plt.colorbar(pcm, label='Hotspot probability', orientation='vertical')
        plt.show()

# pcolormesh needs coordinates of length matching grid dims; our lon_vals and lat_vals are 1D arrays
plot_probability_map(lon_vals, lat_vals, final_prob_map, title="Predicted Shark Hotspots Probability (MODIS + PACE + SWOT)")

# Also show RF-only and Rule-only
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.title("RF probability")
plt.pcolormesh(lon_vals, lat_vals, prob_map_rf, shading='auto', vmin=0, vmax=1)
plt.colorbar(label='RF prob')
plt.xlabel("Longitude"); plt.ylabel("Latitude")

plt.subplot(1,2,2)
plt.title("Rule-based score")
plt.pcolormesh(lon_vals, lat_vals, rule_map, shading='auto', vmin=0, vmax=1)
plt.colorbar(label='rule score')
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.tight_layout()
plt.show()
###########################
# ---------- Export AI data (for publishing) ----------

timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
dataset_id = "AI_PREDICT_001"

# Example metadata placeholders (can be replaced dynamically if needed)
water_depth = "unknown"
temp_mean = np.nanmean(flat_features.get('sst', np.nan))
lat_mean = np.nanmean(lat_vals)
lon_mean = np.nanmean(lon_vals)

ai_data_line = f"{timestamp},{dataset_id},{water_depth},{temp_mean:.3f},{lat_mean:.3f},{lon_mean:.3f}"

out_txt = os.path.join(base_dir, "AI_published_data.txt")
with open(out_txt, "w") as f:
    f.write(ai_data_line)

print("Published AI data line:")
print(ai_data_line)

# ---------- Save outputs ----------
out_nc = os.path.join(base_dir, "combined_shark_hotspot_prob_map_fixed.nc")
ds_out = xr.Dataset(
    {"hotspot_prob": ((chl_lat_name, chl_lon_name), final_prob_map),
     "rf_prob": ((chl_lat_name, chl_lon_name), prob_map_rf),
     "rule_score": ((chl_lat_name, chl_lon_name), rule_map)},
    coords={chl_lat_name: lat_vals, chl_lon_name: lon_vals}
)
comp = {"zlib": True, "complevel": 4}
encoding = {k: comp for k in ds_out.data_vars}
ds_out.to_netcdf(out_nc, encoding=encoding)
print("Saved combined probability map to:", out_nc)

# ---------- End notes ----------
print("""
Notes:
- If you see many NaNs in the final map, your data coverage is sparse (SWOT sparse tracks or PACE product mismatch).
- Tweak REGION or set REGION=None to use the global reference grid.
- To improve SWOT L3: consider using more advanced objective mapping (optimal interpolation) or merging multiple SWOT passes.
- To tune ecological logic, change 'pref_temp', component weights, or mapping x,y,z,t,w,k to species-specific values.
""")