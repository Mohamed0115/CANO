# preprocess.py
import os, glob
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter

os.makedirs("data/processed", exist_ok=True)
files = sorted(glob.glob("data/raw/*.csv"))
print("Found", len(files), "raw files.")

for fpath in files:
    df = pd.read_csv(fpath)
    # expected columns: time (days), flux
    time = df['time'].values
    flux = df['flux'].values
    # remove NaNs
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]; flux = flux[mask]
    if len(time) < 10:
        continue
    # remove strong outliers (cosmic rays) using sigma clipping
    flux_clip = sigma_clip(flux, sigma=5).filled(np.nan)
    # interpolate missing values
    # resample to 1 hour cadence -> Kepler times are in days, so 1 hour = 1/24 day
    new_time = np.arange(time.min(), time.max(), 1.0/24.0)
    # linear interpolation over the original points
    interp_flux = np.interp(new_time, time, flux_clip, left=np.nan, right=np.nan)
    # simple detrend: subtract smoothed trend
    # choose window length odd and less than length
    w = min(len(interp_flux)-1, 101)
    if w % 2 == 0:
        w -= 1
    if w >= 5:
        trend = savgol_filter(np.nan_to_num(interp_flux, nan=np.nanmedian(interp_flux)), window_length=w, polyorder=3)
        flat_flux = interp_flux - trend + np.nanmedian(interp_flux)
    else:
        flat_flux = interp_flux
    # save processed
    base = os.path.basename(fpath).replace('.csv','')
    out_df = pd.DataFrame({'time': new_time, 'flux': flat_flux})
    out_df.to_csv(f"data/processed/{base}_proc.csv", index=False)
    print("Processed", base)
