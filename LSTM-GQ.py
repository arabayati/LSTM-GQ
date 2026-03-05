# -*- coding: utf-8 -*-
 
# ===============================================
# Beta-LSTM (Recession Windows) — Physics+Data Loss (Final Spec)
# ===============================================
 
print('AAhoyy')

# ---------- Standard Libs ----------
import os, math, time, random, warnings
warnings.filterwarnings("ignore")

# ---------- Third-Party ----------
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
 

# ===============================================
# HYPERPARAMETERS (all in one place)
# ===============================================

# ----- Device / SLURM -----
#if os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("MIG-"):
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#if "SLURM_ARRAY_TASK_ID" not in os.environ:
#    raise RuntimeError("This script must be run as a SLURM array job.")
#cluster = int(os.environ["SLURM_ARRAY_TASK_ID"])

cluster = 8
#cluster = 9  # gridcode = 2
#cluster = 8  # gridcode = 1465 # 0.13, SF = 0.4    # gridcode = 4  # 1.8, SF = 0
#cluster = 8 
Selected_GridCode = 4 #1465  # 1.8, SF = 0 
 
 
# ----- Paths / IO -----

folder = f"/lustre06/project/6047297/majidara/data/clustered_forcing_data/cluster_{cluster}/"
path_static = "/lustre06/project/6047297/majidara/data/attributes/merged_catchments_metadata.csv"

root_base = f"/lustre06/project/6047297/majidara/GG_results_postK_Nov_2025/cluster_{cluster}/weights_singlecode/"


# Two main folders: GENERAL (carry-forward) and FINETUNED (no carry-forward)
model_weights_path_general   = os.path.join(root_base, "general")
model_weights_path_finetuned = os.path.join(root_base, "finetuned")
os.makedirs(model_weights_path_general, exist_ok=True)
os.makedirs(model_weights_path_finetuned, exist_ok=True)
 
# Where to save CSVs and plots
#results_outdir = f"/lustre09/project/6047297/majidara/post_kirchner_GQ_results_Nov_2025/model_csv_and_plot_results/cluster_{cluster}/"

results_outdir = f"/lustre06/project/6047297/majidara/GG_results_postK_Nov_2025/cluster_{cluster}/results_singlecode/"
os.makedirs(results_outdir, exist_ok=True)


# ----- Training mode -----
# "General"   = carry-forward across gridcodes (transfer chain)
# "Finetuning"= init every gridcode from a fixed pretrained directory
Training_Mode = "General"   # or "Finetuning"

# Where to store/load checkpoints:
# - latest_dir is used only in "General" mode (rolling transfer)
# - pretrained_dir is used only in "Finetuning" mode (fixed init)
#pretrained_dir = "/lustre09/project/6047297/majidara/pretrained_fixed_by_TW/"  # set this to your pretrained weights root


# Font/plot parameters
font_path = "/home/majidara/Beta_LSTM_Sep_2024/Times New Roman.ttf"
DPI = 110
Font_Size_X_Y_Title = 14

# ----- Random Seeds -----
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----- Numeric / Modes -----
EPS  = 1e-6
#MODE = "log"   # log only (data loss in log-space; physics in physical space)

# ----- Data columns -----
X_normalized_dynamic_features = ["temperature_C", "precipitation_mmd", "pet_mmd"]  # normalized inputs

# Physical (not normalized, fed separately where needed)
X_physical_single_step = ["SM_%", "pet_mmd"]  # SM is unitless [0-1]; PET in mm/day

# Static (always used; statics-only LP/gamma head)
staticColumns = [
    "elevation_mean_m", "mean_slope_degree", "Median_DepthToBedrock_cm",
    "Prec_mm", "Temp_C", "PET_mm", "AET_mm", "P_AET_mm", "Aridity", "SF",
    "max_soil_moisture", "Porosity", "Seasonality_of_Moisture_Index",
    "low_high_ratio", "wet_days_ratio_1mm", "wet_days_ratio_5mm",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"
]

TargetLabel = "streamflow_mmd"

# ----- Trajectory & Windowing -----
INPUT_WINDOW   = 365
TARGET_WINDOW  = 4         # overridden per bootstrap element
TRAJECTORY_LEN = INPUT_WINDOW + TARGET_WINDOW
TrainRatio     = 0.7
BatchSize      = 1028
FineTuningEpochs = 1500  # 2000 


# ----- Recession Detection (physical space) -----
RECESSION_METHOD = "Simple"   # "Simple" or "HLE"   
QP_THRESHOLD     = 1.0  # 1.0
SEG_MIN_PROP     = 0.6
SEG_MAX_GAP      = 2
SEG_TRIM_ENDS    = True

# HLE parameters
HLE_MIN_LEN          = 4
HLE_DROP_FIRST       = 1 
HLE_DECREASING_RATE  = True
min_prominence_peaks = 0.0 #0.1  # for Simple  



# ----- Physics: m_t gate (Q/PET gate) -----
# m_t = 1 - sigmoid( GATE_K * ( (Q/PET) - GATE_TAU ) )
#GATE_K   = 2.0
#GATE_TAU = 7.0
GATE_K   = 2.0
GATE_TAU = 7.0   

USE_M_GATE = False    # set False to disable m_t 
USE_AET = True        

# ----- LP and gamma bounds -----
AET_parameters = "/lustre06/project/6047297/majidara/lp_gamma_fit_summary_with_recession.csv"
df_params = pd.read_csv(AET_parameters) 
df_params['gridcode'] = pd.to_numeric(df_params['gridcode'], errors='coerce')
#LP_MIN, LP_MAX       = 0.1, 1.0
#GAMMA_MIN, GAMMA_MAX = 0.1, 5.0
 
#LP_MIN, LP_MAX       = 1.0, 1.0
#GAMMA_MIN, GAMMA_MAX = 0.538, 0.538
#LP_MIN, LP_MAX       = 0.36, 0.44
#GAMMA_MIN, GAMMA_MAX = 0.5, 0.56  
 
# ----- Loss Weights (manual) -----
lambda_main = 2.2
lambda_g    = 2.0#1.0     # per-day g supervision
lambda_s    = 20.0    # smoothness on g 
lambda_c    = 1e-3    # L1 on context
lambda_cov  = 0#1e-3     # corr(g, AET)
lambda_Q0   = 10.0      # Q0 tether (log space) 
lambda_mono = 0  # 100.0   # monotonicity of g during recession (only penalize upward steps)    
# ----- Huber deltas -----
delta_data = 0.5
delta_ode  = 0.5
delta_g    = 0.2
delta_Q0   = 0.5
 
alpha_prior_val = 0.2
lambda_alpha = 0.01   # or something small like 0.1–0.5    
# ----- Optim ----- 
LearningRate = 1e-4 

# ----- Bootstrap target windows -----
#bootstrap_target_windows = [4, 5, 6, 7]
bootstrap_target_windows = [4]
recession_threshold  = 0.9    # fraction of days in TW that must be recession

# Burn-in and evaluation window control
#burn_in = 2
#n_last  = 2

burn_in = 0
n_last  = 4    
 
 
# ----- Network architecture hyperparameters (like old code) -----
HiddenSize        = 256    # or 128, whatever you were using before
DropoutRateLSTM   = 0.0
DropoutRateG      = 0.4
DropoutRateQ0Head = 0.4

# ===============================================
# DEVICE / FONTS / INFO
# ===============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No valid GPU device found")
print(f"Cluster: {cluster}")

if font_path and os.path.exists(font_path):
    font_properties = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_properties.get_name()
plt.rcParams['font.size'] = Font_Size_X_Y_Title

# ===============================================
# HELPERS (metrics, scalers, hubs)
# ===============================================
def _load_state_if_exists(module, path, device, *, ignore_keys=None):
    """Load a state_dict into `module` if file exists.

    - Skips any keys listed in ignore_keys.
    - Skips any keys whose shape doesn't match the current module.
    """
    if not os.path.exists(path):
        return False

    sd = torch.load(path, map_location=device)

    # Drop explicitly ignored keys
    if ignore_keys:
        sd = {
            k: v for k, v in sd.items()
            if not any(k == ik or k.endswith(ik) for ik in ignore_keys)
        }

    # Drop shape-mismatched keys (e.g. old 2-unit fc into new 3-unit fc)
    model_sd = module.state_dict()
    filtered_sd = {}
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered_sd[k] = v
        # else: silently skip (this is what we want for fc.weight/bias)

    module.load_state_dict(filtered_sd, strict=False)
    return True
 

class IdentityScaler:
    def fit_transform(self, X): return X
    def transform(self, X): return X
    def inverse_transform(self, X): return X

def NSE(obs, sim):
    obs, sim = np.asarray(obs), np.asarray(sim)
    if obs.size < 1:
        return np.nan
    denom = np.sum((obs - np.mean(obs))**2)
    return 1 - np.sum((obs - sim)**2) / denom if denom != 0 else np.nan

def KGE(obs, sim):
    obs, sim = np.asarray(obs), np.asarray(sim)
    if obs.size < 2:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    beta = np.mean(sim) / (np.mean(obs) + EPS)
    gamma = (np.std(sim)/(np.mean(sim)+EPS)) / (np.std(obs)/(np.mean(obs)+EPS) + EPS)
    return 1 - math.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)

def extract_first_number(s):
    num = ''.join([ch for ch in s if ch.isdigit()])
    return int(num) if num else None


def pearson_r_np(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if x.size < 2 or y.size < 2:
        return np.nan
    r = np.corrcoef(x, y)[0, 1]
    return r

def r2_from_r_np(r):
    if np.isnan(r):
        return np.nan
    return r * r

 

# Robust Huber in torch
def huber(x, delta):
    """
    Device/dtype-safe Huber loss.
    Same math as before, but avoids reallocations and cross-device issues.
    """
    d = x.new_tensor(delta)             # scalar on x's device & dtype
    absx = torch.abs(x)
    quad = torch.minimum(absx, d)       # piecewise: 0.5*absx^2 when absx<=d
    lin  = absx - quad                  # and d*(absx - 0.5*d) otherwise
    return 0.5 * quad**2 + d * lin

# ===============================================
# RECESSION DETECTION (physical space)
# ===============================================

def detect_recession_simple(Q):
    """
    Peak ? non-increasing after each peak; physical space.
    NaNs are treated as +inf so they cannot be flagged as recession.
    """
    Q = np.asarray(Q, dtype=float)
    Q_proc = Q.copy()
    Q_proc[np.isnan(Q_proc)] = np.inf  # NaNs break monotonic decrease

    mask = np.full_like(Q_proc, False, dtype=bool)
    peaks, _ = find_peaks(Q_proc, prominence=min_prominence_peaks)

    for peak in peaks:
        if peak >= len(Q_proc) - 1:
            continue
        i = peak
        while i < len(Q_proc) - 1 and Q_proc[i + 1] < Q_proc[i]:
            mask[i + 1] = True
            i += 1
    return mask


def detect_recession_paper(Q, min_len=5, drop_first=3, decreasing_rate=True):
    """
    HLE/paper-style; physical space; returns boolean mask.
    NaNs are treated as +inf so they cannot be flagged as recession.
    """
    Q = np.asarray(Q, dtype=float)
    Q_proc = Q.copy()
    Q_proc[np.isnan(Q_proc)] = np.inf  # ensure comparisons don't flag NaNs as recession

    N = len(Q_proc)
    mask = np.full(N, False, dtype=bool)
    i = 0
    while i < N - 1:
        if Q_proc[i + 1] < Q_proc[i]:
            seg = [i, i + 1]
            R_prev = -(Q_proc[i + 1] - Q_proc[i])
            j = i + 1
            while j < N - 1 and Q_proc[j + 1] < Q_proc[j]:
                R_cur = -(Q_proc[j + 1] - Q_proc[j])
                if (not decreasing_rate) or (R_cur < R_prev):
                    seg.append(j + 1)
                    R_prev = R_cur
                    j += 1
                else:
                    break
            if len(seg) >= min_len:
                for idx in seg[drop_first:]:
                    if 0 <= idx < N:
                        mask[idx] = True
            i = j
        else:
            i += 1
    return mask


def apply_qp_threshold_segmentwise(raw_mask, Q, P, threshold, min_prop=0.6, max_gap=2, trim_ends=True):
    """Segment-wise Q/P gate; P<=0 passes; all in physical space."""
    N = len(Q)
    final_mask = np.zeros(N, dtype=bool)

    ratio_now = np.full(N, np.nan, dtype=float)
    posP = P > 0
    ratio_now[posP] = Q[posP] / P[posP]
    pass_day = (P <= 0) | ((P > 0) & (ratio_now > threshold))

    idx = np.where(raw_mask)[0]
    if idx.size == 0:
        return final_mask, ratio_now

    splits = np.where(np.diff(idx) != 1)[0]
    segments = np.split(idx, splits + 1)

    for seg in segments:
        if seg.size == 0:
            continue
        seg_pass = pass_day[seg].copy()

        # fill small internal gaps
        if max_gap and max_gap > 0:
            z = seg_pass
            i = 0
            while i < len(z):
                if not z[i]:
                    j = i
                    while j < len(z) and not z[j]:
                        j += 1
                    gap_len = j - i
                    left_ok = (i - 1 >= 0) and z[i - 1]
                    right_ok = (j < len(z)) and (z[j] if j < len(z) else False)
                    if left_ok and right_ok and gap_len <= max_gap:
                        z[i:j] = True
                    i = j
                else:
                    i += 1
            seg_pass = z

        if seg_pass.mean() < min_prop:
            continue
        if trim_ends:
            pass_idx = np.where(seg_pass)[0]
            if pass_idx.size == 0:
                continue
            seg_keep = seg[pass_idx[0]: pass_idx[-1]+1]
            final_mask[seg_keep] = True
        else:
            final_mask[seg] = True

    return final_mask, ratio_now

# ===============================================
# DATA PREP
# ===============================================

def prepare_dataframe(raw_df):
    # Preserve original date string for export (exact as in file)
    df = raw_df.copy()
    assert 'date' in df.columns, "Input CSV must contain 'date' column"
    df['date_str_orig'] = df['date'].astype(str)

    # Parse/sort date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df['day_of_year'] = df['date'].dt.day_of_year

    # Clamp negatives for P, PET, Q in physical space
    for col in ['precipitation_mmd', 'pet_mmd', 'streamflow_mmd']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    # SM in [0,1] (column name is "SM_%" but it is unitless 0-1)
    if 'SM_%' in df.columns:
        df['SM_%'] = df['SM_%'].clip(lower=0.0, upper=1.0)

    return df

def compute_recession_mask(df_phys):
    Q = df_phys['streamflow_mmd'].values.astype(float)
    P = df_phys['precipitation_mmd'].values.astype(float)

    if RECESSION_METHOD == "HLE":
        raw = detect_recession_paper(Q, min_len=HLE_MIN_LEN, drop_first=HLE_DROP_FIRST,
                                     decreasing_rate=HLE_DECREASING_RATE)
        final_mask, _ = apply_qp_threshold_segmentwise(
            raw, Q, P, QP_THRESHOLD,
            min_prop=SEG_MIN_PROP, max_gap=SEG_MAX_GAP, trim_ends=SEG_TRIM_ENDS
        )
    else:
        raw = detect_recession_simple(Q)
        final_mask, _ = apply_qp_threshold_segmentwise(
            raw, Q, P, QP_THRESHOLD,
            min_prop=SEG_MIN_PROP, max_gap=SEG_MAX_GAP, trim_ends=SEG_TRIM_ENDS
        )
    return final_mask

def compute_hle_recession_mask_zero(df_phys):
    """Diagnostic HLE recession with QP threshold fixed at 0.0 (as requested)."""
    Q = df_phys['streamflow_mmd'].values.astype(float)
    P = df_phys['precipitation_mmd'].values.astype(float)
    raw = detect_recession_paper(Q, min_len=HLE_MIN_LEN, drop_first=HLE_DROP_FIRST,
                                 decreasing_rate=HLE_DECREASING_RATE)
    final_mask, _ = apply_qp_threshold_segmentwise(
        raw, Q, P, 0.0,
        min_prop=SEG_MIN_PROP, max_gap=SEG_MAX_GAP, trim_ends=SEG_TRIM_ENDS
    )
    return final_mask

def add_static_columns(df, static_cols, static_values):
    for i, col in enumerate(static_cols):
        df[col] = static_values[i]
    return df

def create_trajectories_sliding(df, trajectory_length):
    trajs = []
    total_points = len(df)
    for i in range(total_points - trajectory_length + 1):
        trajs.append(df.iloc[i: i+trajectory_length].copy())
    return trajs


def build_dataset(df_phys, df_norm, stat_vec, trajectory_length, input_window):
    """
    Returns on success:
      X_in   : (N, trajectory_length, feat_dim) normalized dyn + repeated statics
      Phys   : dict of per-sample physical series used in physics (all (N, trajectory_length))
               keys: 'Q', 'P', 'PET', 'SM', 'rec_mask', 'date'
      Y_log  : (N, target_window, 1) log targets (observed)
      Q0_obs_log : (N,1) log observed Q at t=input_window-1

    Returns (None, None, None, None) if no windows survive the NaN guard.
    """
    trajs_phys = create_trajectories_sliding(df_phys, trajectory_length)
    trajs_norm = create_trajectories_sliding(df_norm, trajectory_length)

    X_list, Y_log_list, Q0_obs_log_list = [], [], []
    Q_list, P_list, PET_list, SM_list, R_list, DATE_list = [], [], [], [], [], []

    for k in range(len(trajs_phys)):
        tp = trajs_phys[k].reset_index(drop=True)
        tn = trajs_norm[k].reset_index(drop=True)

        # ----- NaN guard: skip windows if required Q values contain NaN -----
        TW = trajectory_length - input_window
        q_needed = tp[TargetLabel].values[input_window - 1 : trajectory_length]  # includes Q0 and all targets
        if np.isnan(q_needed).any():
            continue
        # --------------------------------------------------------------------

        # Normalized dynamic features
        X_dyn = tn[X_normalized_dynamic_features].values.astype(float)
        # Repeat statics across time
        X_stat = np.repeat(stat_vec.reshape(1,-1), repeats=trajectory_length, axis=0)
        X_traj = np.concatenate([X_dyn, X_stat], axis=1)
        X_list.append(X_traj)

        # Physical series (per step)
        Q_phys   = tp["streamflow_mmd"].values.astype(float)
        P_phys   = tp["precipitation_mmd"].values.astype(float)
        PET_phys = tp["pet_mmd"].values.astype(float)
        SM_phys  = tp["SM_%"].values.astype(float)
        rec_mask = tp["recession_flag"].values.astype(int) if "recession_flag" in tp.columns else np.zeros(trajectory_length, dtype=int)
        dates    = tp["date"].values  # datetime64[ns]

        # Targets in log-space
        target_log = np.log(tp[TargetLabel].values[input_window:trajectory_length] + EPS).reshape(-1,1)
        Q0_obs_l   = np.log(tp[TargetLabel].values[input_window - 1] + EPS)

        Y_log_list.append(target_log)
        Q0_obs_log_list.append([Q0_obs_l])

        Q_list.append(Q_phys); P_list.append(P_phys); PET_list.append(PET_phys); SM_list.append(SM_phys)
        R_list.append(rec_mask); DATE_list.append(dates)

    # ---- handle the case where all windows were skipped ----
    if len(X_list) == 0:
        return None, None, None, None
    # -------------------------------------------------------

    X_in   = np.stack(X_list)                       # (N, T, D)
    Y_log  = np.stack(Y_log_list)                   # (N, TW, 1)
    Q0_log = np.array(Q0_obs_log_list)              # (N, 1)
    Phys   = {
        'Q':   np.stack(Q_list),    # (N,T)
        'P':   np.stack(P_list),
        'PET': np.stack(PET_list),
        'SM':  np.stack(SM_list),
        'rec_mask': np.stack(R_list),
        'date': np.stack(DATE_list)  # (N,T) datetime64
    }
    return X_in, Y_log, Q0_log, Phys
    

def filter_trajectories_by_recession(Phys, input_window, target_window, threshold):
    """
    Keep trajectories where (a) required Q values are all finite and
    (b) recession fraction = threshold over the target window.
    """
    R = Phys['rec_mask']  # (N, T)
    Q = Phys['Q']         # (N, T)
    tw_slice_Q = slice(input_window - 1, input_window + target_window)  # includes Q0 and all targets
    tw_slice_R = slice(input_window, input_window + target_window)      # target steps only

    keep = []
    for i in range(R.shape[0]):
        # Skip if any required Q is NaN
        if np.isnan(Q[i, tw_slice_Q]).any():
            continue
        # Recession fraction over target steps
        frac = np.mean(R[i, tw_slice_R] > 0)
        if frac >= threshold:
            keep.append(i)

    if len(keep) == 0:
        return None
    idx = np.array(keep, dtype=int)
    Phys_keep = {k: v[idx] for k, v in Phys.items()}
    return idx, Phys_keep


# ===============================================
# MODELS
# ===============================================

class ForcingEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.do = nn.Dropout(dropout)
    def forward(self, seq):  # seq: (T, B, D)
        out, _ = self.lstm(seq)
        return self.do(out)  # (T, B, H)

class ContextualGFunction(nn.Module):
    def __init__(self, hidden_size=50, dropout_rate=0.4):
        super().__init__()
        self.do = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Softplus()  # g >= 0
    def forward(self, h):  # h: (B,H)
        x = self.do(h)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x  # (B,1), positive

class Q0Head(nn.Module):
    """Predict log(Q0_hat) from context h_{t0}; return logQ0_hat and Q0_hat."""
    def __init__(self, hidden_size=128, dropout_rate=0.4):
        super().__init__()
        self.do = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, h0):  # (B,H)
        h0 = self.do(h0)
        logQ0_hat = self.fc(h0)
        Q0_hat = torch.clamp(torch.exp(logQ0_hat) - EPS, min=0.0)
        return logQ0_hat, Q0_hat

# >>> REPLACE: bounds are now attributes of the head (no globals needed)
class LPGammaHead(nn.Module):
    """
    Static head:
      - LP    in [lp_min, lp_max]
      - GAMMA in [gm_min, gm_max]
      - ALPHA in [alpha_min, alpha_max], scalar coeff for AET in ODE only
    """
    def __init__(self, in_dim, hidden=[32, 32], dropout_rate=0.3,
                 lp_bounds=(0.1, 1.0),
                 gamma_bounds=(0.1, 5.0),
                 alpha_bounds=(0.0, 1.0)):    # you can adjust this
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        self.mlp = nn.Sequential(*layers)
        self.do  = nn.Dropout(dropout_rate)

        # 3 outputs: LP, GAMMA, ALPHA
        self.fc  = nn.Linear(last, 3)
        self.sig = nn.Sigmoid()

        # bounds as buffers
        self.register_buffer("lp_min",     torch.tensor([lp_bounds[0]], dtype=torch.float32))
        self.register_buffer("lp_max",     torch.tensor([lp_bounds[1]], dtype=torch.float32))
        self.register_buffer("gm_min",     torch.tensor([gamma_bounds[0]], dtype=torch.float32))
        self.register_buffer("gm_max",     torch.tensor([gamma_bounds[1]], dtype=torch.float32))
        self.register_buffer("alpha_min",  torch.tensor([alpha_bounds[0]], dtype=torch.float32))
        self.register_buffer("alpha_max",  torch.tensor([alpha_bounds[1]], dtype=torch.float32))

    def forward(self, s):  # (B, S)
        x = self.mlp(s)
        x = self.do(x)
        sig = self.sig(self.fc(x))   # (B,3)

        LP    = self.lp_min    + (self.lp_max    - self.lp_min)    * sig[:, 0:1]
        GAMMA = self.gm_min    + (self.gm_max    - self.gm_min)    * sig[:, 1:2]
        ALPHA = self.alpha_min + (self.alpha_max - self.alpha_min) * sig[:, 2:3]

        return LP, GAMMA, ALPHA

 
# ===============================================
# TRAIN / EVAL UTILITIES
# ===============================================
 
def pet_sm_gate_and_aet(Q_phys, PET_phys, SM_phys, LP, GAMMA):  
    # --- FORCE ZERO AET OPTION ---
    if not USE_AET:
        m_t  = torch.ones_like(PET_phys)           # gate value (unused when AET=0)
        aett = torch.zeros_like(PET_phys)          # AET_t = 0 everywhere
        return m_t, aett
    # ------------------------------

    if LP.dim() == 2: LP = LP.unsqueeze(1)
    if GAMMA.dim() == 2: GAMMA = GAMMA.unsqueeze(1)

    ratio = Q_phys / (PET_phys + EPS)             # (B,T,1)
    if USE_M_GATE:
        m_t = 1.0 - torch.sigmoid(GATE_K * (ratio - GATE_TAU))
    else:
        m_t = torch.ones_like(ratio)

    sm_term = torch.clamp(SM_phys / (LP + EPS), min=EPS)
    aett    = m_t * PET_phys * torch.pow(sm_term, GAMMA)

    return m_t, aett




def rollout_closed_form(Q0_hat, g_seq, PET_phys, SM_phys, LP, GAMMA, ALPHA):
    """
    Q_{t+1} = (Q_t + alpha * AET_t) * exp(-g_t) - alpha * AET_t

    ALPHA: (B,1) or (B,1,1), static coefficient used ONLY in ODE.
    AET_seq is the unscaled AET_t from pet_sm_gate_and_aet.
    """
    B, TW, _ = g_seq.shape

    if ALPHA.dim() == 2:
        ALPHA = ALPHA.unsqueeze(1)   # (B,1,1)

    Q_hat = Q0_hat.unsqueeze(-1)     # (B,1,1)

    Q_list = [Q_hat]
    AET_list, m_list = [], []

    for t in range(TW):
        g_t   = g_seq[:, t:t+1, :]
        pet_t = PET_phys[:, t:t+1, :]
        sm_t  = SM_phys[:,  t:t+1, :]

        pet_t = torch.nan_to_num(pet_t, nan=0.0, posinf=0.0, neginf=0.0)
        sm_t  = torch.nan_to_num(sm_t,  nan=0.0, posinf=0.0, neginf=0.0)

        m_t, AET_t = pet_sm_gate_and_aet(Q_hat, pet_t, sm_t, LP, GAMMA)

        AET_eff = ALPHA * AET_t

        Q_next = (Q_hat + AET_eff) * torch.exp(-g_t) - AET_eff
        Q_next = torch.clamp(Q_next, min=0.0)

        Q_list.append(Q_next)
        AET_list.append(AET_t)   # store original AET
        m_list.append(m_t)
        Q_hat = Q_next

    Q_hat_all = torch.cat(Q_list, dim=1)   # (B, TW+1, 1)
    AET_seq   = torch.cat(AET_list, dim=1) # (B, TW,   1) unscaled
    m_seq     = torch.cat(m_list,  dim=1)
    return Q_hat_all, AET_seq, m_seq


def compute_midpoint_ode_residual(Q_hat_all, g_seq, AET_seq):
    B, TW, _ = g_seq.shape
    Q_t   = Q_hat_all[:, :-1, :]
    Q_tp1 = Q_hat_all[:,  1:, :]

    g_t   = g_seq
    g_tp1 = torch.cat([g_seq[:, 1:, :], g_seq[:, -1:, :]], dim=1)
    g_bar = 0.5 * (g_t + g_tp1)

    AET_t   = AET_seq
    AET_tp1 = torch.cat([AET_seq[:, 1:, :], AET_seq[:, -1:, :]], dim=1)
    AET_bar = 0.5 * (AET_t + AET_tp1)

    Q_bar = 0.5 * (Q_t + Q_tp1)

    r_t = (Q_tp1 - Q_t) + g_bar * (Q_bar + AET_bar)
    return r_t

def pearson_r2_torch(x, y):
    x = x.view(-1)
    y = y.view(-1)
    xm = x - torch.mean(x)
    ym = y - torch.mean(y)
    num = torch.sum(xm * ym)
    den = torch.sqrt(torch.sum(xm**2) * torch.sum(ym**2) + EPS)
    rho = num / (den + EPS)
    return rho * rho

# -------------- RE-ADDED: TRAIN STEP --------------
def train_model(dataset, encoder, g_net, q0_head, lp_gamma_head,
                batch_size, input_window, trajectory_length, target_window,
                s_Q, stat_dim, warm_up=False):
    """
    dataset = tuple of prepared (X_in, Y_log, Q0_log_obs, Phys, stat_vecs)
    """
    Number_of_epochs = 20 if warm_up else FineTuningEpochs
    print(f"\n{'Warm up' if warm_up else 'Main'} training for {Number_of_epochs} epochs\n")

    X_in, Y_log, Q0_log_obs, Phys, stat_mat = dataset
    # Build loader tensors
    ds = TensorDataset(
        torch.tensor(X_in, dtype=torch.float32),
        torch.tensor(Y_log, dtype=torch.float32),
        torch.tensor(Q0_log_obs, dtype=torch.float32),
        torch.tensor(Phys['Q'], dtype=torch.float32),
        torch.tensor(Phys['P'], dtype=torch.float32),
        torch.tensor(Phys['PET'], dtype=torch.float32),
        torch.tensor(Phys['SM'], dtype=torch.float32),
        torch.tensor(Phys['rec_mask'], dtype=torch.float32),
        torch.tensor(stat_mat, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    params = list(encoder.parameters()) + list(g_net.parameters()) \
           + list(q0_head.parameters()) + list(lp_gamma_head.parameters()) \
           + [s_Q]
    optimizer = optim.Adam(params, lr=LearningRate)

    loss_history = []
    for epoch in range(1, Number_of_epochs+1):
        epoch_losses = []

        for (batch_X, batch_Ylog, batch_Q0log_obs,
             Q_phys, P_phys, PET_phys, SM_phys, R_mask, stat_vecs) in loader:

            batch_X      = batch_X.to(device)       # (B,T,D)
            batch_Ylog   = batch_Ylog.to(device)    # (B,TW,1)
            batch_Q0log  = batch_Q0log_obs.to(device)  # (B,1)

            Q_phys  = Q_phys.to(device)             # (B,T)
            P_phys  = P_phys.to(device)
            PET_phys= PET_phys.to(device)
            SM_phys = SM_phys.to(device)
            R_mask  = R_mask.to(device)             # (B,T)
            stat_vecs = stat_vecs.to(device)        # (B,S)

            B, T, D = batch_X.shape
            TW = target_window
            t0 = input_window

            # LSTM encode full trajectory
            X_seq = batch_X.transpose(0,1)  # (T,B,D)
            H_seq = encoder(X_seq)          # (T,B,H)
            H_seq = H_seq.transpose(0,1)    # (B,T,H)
            H_t0  = H_seq[:, t0-1, :]       # (B,H)
            H_TW  = H_seq[:, t0:t0+TW, :]   # (B,TW,H)

            # g_t per step
            g_vals = []
            for t in range(TW):
                g_vals.append(g_net(H_TW[:, t, :]))    # (B,1)
            g_seq = torch.stack(g_vals, dim=1)         # (B,TW,1)

            # Q0_hat (learned) and physical rollout
            logQ0_hat, Q0_hat = q0_head(H_t0)          # (B,1), (B,1)

            # Physics inputs at target steps in physical space
            Q_phys_TW   = Q_phys[:, t0-1:t0+TW]        # (B, TW+1)
            Q_phys_TW   = Q_phys_TW.unsqueeze(-1)      # (B, TW+1, 1)
            PET_phys_TW = PET_phys[:, t0:t0+TW].unsqueeze(-1)  # (B, TW, 1)
            SM_phys_TW  = SM_phys[:,  t0:t0+TW].unsqueeze(-1)  # (B, TW, 1)

            # LP, gamma from statics (per-sample, constant across time)
            #LP, GAMMA = lp_gamma_head(stat_vecs)       # (B,1), (B,1)
            LP, GAMMA, ALPHA = lp_gamma_head(stat_vecs)   # (B,1) each
            
            
            alpha_prior_t = torch.tensor(alpha_prior_val, device=device, dtype=ALPHA.dtype)
            L_alpha = ((ALPHA - alpha_prior_t) ** 2).mean()

            
            LP    = LP.view(-1, 1, 1)      # (B,1,1) ############################################################################# just added 
            GAMMA = GAMMA.view(-1, 1, 1)   # (B,1,1)  ############################################################################# just added
            ALPHA = ALPHA.view(-1, 1, 1)

            # Rollout
            Q_hat_all, AET_seq, m_seq = rollout_closed_form(
                Q0_hat, g_seq, PET_phys_TW, SM_phys_TW, LP, GAMMA, ALPHA
            )
            
            # ---- SHAPE GUARD A: right after rollout_closed_form ----
            B_now = batch_X.shape[0]
            assert Q_hat_all.shape == (B_now, TW+1, 1), f"Q_hat_all bad shape: {Q_hat_all.shape}, expected {(B_now, TW+1, 1)}" ############################################################################# just added 
            assert AET_seq.shape    == (B_now, TW,   1), f"AET_seq bad shape: {AET_seq.shape}, expected {(B_now, TW, 1)}" ############################################################################# just added 
            assert g_seq.shape      == (B_now, TW,   1), f"g_seq bad shape: {g_seq.shape}, expected {(B_now, TW, 1)}" ############################################################################# just added 


            # ------------------------
            # LOSSES
            # ------------------------
            # Eval steps (respect burn-in and n_last)
            eval_start = max(burn_in, TW - n_last) if n_last < TW else burn_in
            eval_slice = slice(eval_start, TW)

            # 1) Data trend (log domain)
            #Qhat_log_eval = torch.log(Q_hat_all[:, 1:, :] + EPS)[:, eval_slice, :]  # (B, TW-eval_start, 1) ##############################commented
            Qhat_log_eval = torch.log(torch.clamp(Q_hat_all[:, 1:, :], min=EPS))[:, eval_slice, :] # (B, TW-eval_start, 1)

            Ylog_eval     = batch_Ylog[:, eval_slice, :]
            
            # ---- SHAPE GUARD B: after building eval tensors ---- ########################### just added
            expected_T = TW - eval_start ###########################
            assert Qhat_log_eval.shape == (B_now, expected_T, 1), f"Qhat_log_eval {Qhat_log_eval.shape} vs expected {(B_now, expected_T, 1)}" ############just added#
            assert Ylog_eval.shape     == (B_now, expected_T, 1), f"Ylog_eval {Ylog_eval.shape} vs expected {(B_now, expected_T, 1)}" #################just added

            
            
            L_data = torch.mean(huber(Qhat_log_eval - Ylog_eval, delta_data))

            # 2) (ODE residual term intentionally omitted from total per your spec)

            # 3) Per-day g supervision using observed Q (physical), recession-only, positive-only
            # g_obs_t = -ln((Q_{t+1}^{obs}+AET_t) / (Q_t^{obs}+AET_t))
            Qobs_t   = Q_phys_TW[:, :-1, :]  # (B,TW,1)
            Qobs_tp1 = Q_phys_TW[:, 1:,  :]  # (B,TW,1)
            AET_t    = AET_seq               # (B,TW,1)
            AET_eff  = ALPHA * AET_t                    # effective AET in ODE

            

            #g_obs = -torch.log( (Qobs_tp1 + AET_t + EPS) / (Qobs_t + AET_t + EPS) + EPS )  ############################################## commented
            den = torch.clamp(Qobs_t   + AET_eff, min=EPS)                                #################just added
            num = torch.clamp(Qobs_tp1 + AET_eff, min=EPS)                       #################just added
            g_obs = -torch.log(num / den)                                       #################just added
            

            # Mask: recession-only at target indices (use provided mask) and g_obs > 0
            R_TW = R_mask[:, t0:t0+TW].unsqueeze(-1)  # (B,TW,1)
            M = (R_TW > 0.5) & (g_obs > 0.0)
            if torch.any(M):
                L_g = torch.sum(huber(g_seq - g_obs, delta_g) * M) / (torch.sum(M).clamp_min(1.0))
            else:
                L_g = torch.tensor(0.0, device=device)

            # 4) Smoothness (2nd diff) within window
            if TW >= 3:
                g0 = g_seq[:, 0:-2, :]
                g1 = g_seq[:, 1:-1, :]
                g2 = g_seq[:, 2:,   :]
                second = g2 - 2*g1 + g0
                L_smooth = torch.mean(second**2)
            else:
                L_smooth = torch.tensor(0.0, device=device)

             
            
            # >>> ADD: 4b) Monotonicity-on-recession (hinge on first differences)
            # Penalize upward steps in g ONLY when both t and t+1 are recession days.
            if TW >= 2:
                dg = g_seq[:, 1:, :] - g_seq[:, :-1, :]                  # (B, TW-1, 1)
                R_pairs = (R_TW[:, 1:, :] > 0.5) & (R_TW[:, :-1, :] > 0.5)  # (B, TW-1, 1)
                mono_violation = torch.relu(dg) * R_pairs                # only increases (dg>0) and in recession
                L_mono = torch.sum(mono_violation) / (torch.sum(R_pairs).clamp_min(1.0))
            else:
                L_mono = torch.tensor(0.0, device=device)
            # <<< ADD END
            
            # 5) Context L1 on H_TW
            #L_ctx = torch.mean(torch.abs(H_TW))

            
            # 5) Context L1 on H_TW
            L_ctx = torch.mean(torch.abs(H_TW))

            # 6) Corr(g, AET) over flattened (B*TW)
            L_cov = pearson_r2_torch(g_seq, AET_seq)

            # 7) Q0 tether (log space), manual ?_Q0
            L_Q0 = torch.mean(huber(logQ0_hat - batch_Q0log, delta_Q0))

            # Learned uncertainty (data term only)
            #total = torch.exp(-s_Q) * L_data + s_Q \
                  #+ lambda_g    * L_g \
                  #+ lambda_s    * L_smooth \
                  #+ lambda_mono * L_mono \
                  #+ lambda_c    * L_ctx \
                  #+ lambda_cov  * L_cov \
                  #+ lambda_Q0   * L_Q0
            total = lambda_main * L_data\
                  + lambda_g    * L_g \
                  + lambda_s    * L_smooth \
                  + lambda_mono * L_mono \
                  + lambda_c    * L_ctx \
                  + lambda_cov  * L_cov \
                  + lambda_Q0   * L_Q0 \
                  + lambda_alpha * L_alpha   # <<< NEW


            

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            epoch_losses.append(total.item())

        loss_history.append(np.mean(epoch_losses))
        if epoch % 10 == 0:
            

            #print(f"Epoch {epoch:03d} | L_total(mean): {np.mean(epoch_losses):.6f} | "
             #     f"L_data(last): {L_data.item():.4f} | L_g(last): {L_g.item():.4f} | "
             #     f"L_s(last): {L_smooth.item():.4f} | L_mono(last): {L_mono.item():.4f} | L_ctx(last): {L_ctx.item():.4f} | "
             #     f"L_cov(last): {L_cov.item():.4f} | L_Q0(last): {L_Q0.item():.4f} | "
             #     f"s_Q: {s_Q.item():.3f} | exp(-s_Q): {torch.exp(-s_Q).item():.3f}")
                  
            print(f"Epoch {epoch:03d} | L_total(mean): {np.mean(epoch_losses):.6f} | "
                  f"L_data(last): {L_data.item():.4f} | L_g(last): {L_g.item():.4f} | "
                  f"L_s(last): {L_smooth.item():.4f} | L_mono(last): {L_mono.item():.4f} | "
                  f"L_ctx(last): {L_ctx.item():.4f} | L_cov(last): {L_cov.item():.4f} | "
                  f"L_Q0(last): {L_Q0.item():.4f} | L_alpha(last): {L_alpha.item():.4f} | "
                  f"s_Q: {s_Q.item():.3f} | exp(-s_Q): {torch.exp(-s_Q).item():.3f}")


            
            #print(f"Epoch {epoch:03d} | L_total(mean): {np.mean(epoch_losses):.6f} | "
            #      f"L_data(last): {L_data.item():.4f} | L_g(last): {L_g.item():.4f} | "
            #      f"L_s(last): {L_smooth.item():.4f} | L_ctx(last): {L_ctx.item():.4f} | "
            #      f"L_cov(last): {L_cov.item():.4f} | L_Q0(last): {L_Q0.item():.4f} | "
             #      f"s_Q: {s_Q.item():.3f} | exp(-s_Q): {torch.exp(-s_Q).item():.3f}")
    return loss_history
# -------------- END RE-ADDED TRAIN STEP --------------

# ===============================================
# EVALUATION
# ===============================================

@torch.no_grad()
def evaluate_model(dataset, encoder, g_net, q0_head, lp_gamma_head,
                   batch_size, input_window, trajectory_length, target_window):
    """
    Returns:
      pred_all  : (N, TW) predicted Q (physical)
      target_all: (N, TW) observed Q (physical)
      g_all     : (N, TW) predicted g
      aet_all   : (N, TW) predicted AET
      date_all  : (N, TW) datetime64 dates aligned with the above
    """
    encoder.eval()
    g_net.eval()
    q0_head.eval()
    lp_gamma_head.eval()

    X_in, Y_log, Q0_log_obs, Phys, stat_mat = dataset

    ds = TensorDataset(
        torch.tensor(X_in, dtype=torch.float32),
        torch.tensor(Y_log, dtype=torch.float32),
        torch.tensor(Q0_log_obs, dtype=torch.float32),
        torch.tensor(Phys['Q'], dtype=torch.float32),
        torch.tensor(Phys['P'], dtype=torch.float32),
        torch.tensor(Phys['PET'], dtype=torch.float32),
        torch.tensor(Phys['SM'], dtype=torch.float32),
        torch.tensor(Phys['rec_mask'], dtype=torch.float32),
        torch.tensor(stat_mat, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds_phys_list = []
    target_phys_list = []
    g_list = []
    aet_list = []
    date_list = []

    # Since shuffle=False, batch order follows dataset order; we can slice dates directly
    full_dates = Phys['date']  # (N, T) datetime64

    row_offset = 0
    for (batch_X, batch_Ylog, batch_Q0log_obs,
         Q_phys, P_phys, PET_phys, SM_phys, R_mask, stat_vecs) in loader:

        batch_size_now = batch_X.shape[0]
        batch_X = batch_X.to(device)
        Q_phys  = Q_phys.to(device)
        PET_phys= PET_phys.to(device)
        SM_phys = SM_phys.to(device)
        stat_vecs = stat_vecs.to(device)

        TW = target_window
        t0 = input_window

        X_seq = batch_X.transpose(0,1)
        H_seq = encoder(X_seq).transpose(0,1)
        H_t0  = H_seq[:, t0-1, :]
        H_TW  = H_seq[:, t0:t0+TW, :]

        g_vals = [g_net(H_TW[:, t, :]) for t in range(TW)]
        g_seq  = torch.stack(g_vals, dim=1)  # (B,TW,1)

        logQ0_hat, Q0_hat = q0_head(H_t0)

        Q_phys_TW   = Q_phys[:, t0-1:t0+TW].unsqueeze(-1)
        PET_phys_TW = PET_phys[:, t0:t0+TW].unsqueeze(-1)
        SM_phys_TW  = SM_phys[:,  t0:t0+TW].unsqueeze(-1)

        LP, GAMMA, ALPHA = lp_gamma_head(stat_vecs)
        ALPHA = ALPHA.view(-1, 1, 1)
        
        Q_hat_all, AET_seq, _ = rollout_closed_form(
            Q0_hat, g_seq, PET_phys_TW, SM_phys_TW, LP, GAMMA, ALPHA
        )

        target_phys = Q_phys[:, t0:t0+TW].unsqueeze(-1)

        preds_phys_list.append(Q_hat_all[:, 1:, :].cpu().numpy())  # (B,TW,1) -> Q
        target_phys_list.append(target_phys.cpu().numpy())          # (B,TW,1) -> Q obs
        g_list.append(g_seq.cpu().numpy())                          # (B,TW,1) -> g
        aet_list.append(AET_seq.cpu().numpy())                      # (B,TW,1) -> AET

        # Match date slices for these rows
        rows = np.arange(row_offset, row_offset + batch_size_now)
        dates_batch = full_dates[rows, t0:t0+TW]                    # (B,TW) datetime64
        date_list.append(dates_batch)

        row_offset += batch_size_now

    pred_all   = np.concatenate(preds_phys_list,  axis=0).squeeze(-1)   # (N,TW)
    target_all = np.concatenate(target_phys_list, axis=0).squeeze(-1)   # (N,TW)
    g_all      = np.concatenate(g_list,           axis=0).squeeze(-1)   # (N,TW)
    aet_all    = np.concatenate(aet_list,         axis=0).squeeze(-1)   # (N,TW)
    date_all   = np.concatenate(date_list,        axis=0)               # (N,TW) datetime64

    return pred_all, target_all, g_all, aet_all, date_all

# ===============================================
# MAIN
# ===============================================

def main():
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    # Load static table and standardize stat features
    static_df_full = pd.read_csv(path_static)
    assert 'gridcode' in static_df_full.columns, "Static attributes must contain 'gridcode'."

    scaler_static = StandardScaler()
    scaler_static.fit(static_df_full[staticColumns].values)
    static_df_full_std = static_df_full.copy()
    static_df_full_std[staticColumns] = scaler_static.transform(static_df_full[staticColumns].values)

    # Collect files
    all_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    random.shuffle(all_files)
    if len(all_files) == 0:
        print(f"No CSV files found in {folder}. Exiting.")
        return

    # Results accumulation
    #results_df = pd.DataFrame(columns=[
    #    'GridCode', 'train_NSE', 'train_KGE', 'test_NSE', 'test_KGE',
    #    'train_NSE_unique_target', 'train_KGE_unique_target',
    #    'test_NSE_unique_target', 'test_KGE_unique_target',
    #    'NSE_final_Q_train','KGE_final_Q_train','NSE_final_AET_train','KGE_final_AET_train',
    #    'NSE_final_Q_test','KGE_final_Q_test','NSE_final_AET_test','KGE_final_AET_test'
    #])
    #results_df = pd.DataFrame(columns=[
    #    'GridCode', 'train_NSE', 'train_KGE', 'test_NSE', 'test_KGE',
    #    'train_NSE_unique_target', 'train_KGE_unique_target',
    #    'test_NSE_unique_target', 'test_KGE_unique_target',
    #    'NSE_final_Q_train','KGE_final_Q_train','NSE_final_AET_train','KGE_final_AET_train',
    #    'R_final_AET_train','R2_final_AET_train',      #
    #    'NSE_final_Q_test','KGE_final_Q_test','NSE_final_AET_test','KGE_final_AET_test',
    #    'R_final_AET_test','R2_final_AET_test'         # 
    #])
    results_df = pd.DataFrame(columns=[
        'GridCode', 'train_NSE', 'train_KGE', 'test_NSE', 'test_KGE',
        'train_NSE_unique_target', 'train_KGE_unique_target',
        'test_NSE_unique_target', 'test_KGE_unique_target',
        'NSE_final_Q_train','KGE_final_Q_train','NSE_final_AET_train','KGE_final_AET_train',
        'R_final_AET_train','R2_final_AET_train',
        'NSE_final_Q_test','KGE_final_Q_test','NSE_final_AET_test','KGE_final_AET_test',
        'R_final_AET_test','R2_final_AET_test',
        'alpha'   # <<< NEW COLUMN
    ])



    start_time = time.time()

    for filename in all_files:
        csv_path = os.path.join(folder, filename)
        GridCode = extract_first_number(filename)
        if GridCode is None:
            print(f"Could not extract GridCode from {filename}; skipping.")
            continue
            
        if GridCode != Selected_GridCode: #2:  #1465:
         
            continue

        print(f"\nProcessing GridCode: {GridCode}")
        

        
        # locate row for this GridCode
        row_AET_parameters = df_params.loc[df_params['gridcode'] == GridCode]
        
        if not row_AET_parameters.empty:
            LP_MIN       = float(row_AET_parameters['Lp_lower_CI'].values[0])
            LP_MAX       = float(row_AET_parameters['Lp_higer_CI'].values[0])
            GAMMA_MIN    = float(row_AET_parameters['gamma_low'].values[0])
            GAMMA_MAX    = float(row_AET_parameters['gamma_high'].values[0])
            #LP_MIN, LP_MAX       = 0.1, 1.0
            #GAMMA_MIN, GAMMA_MAX = 0.1, 5.0
          
        else:
            print(f"GridCode {GridCode} not found in AET parameter file. Using defaults.") 
            LP_MIN, LP_MAX       = 0.1, 1.0
            GAMMA_MIN, GAMMA_MAX = 0.1, 5.0
        # <<< ADD END


        # Load dynamic daily CSV (physical)
        df_raw = pd.read_csv(csv_path)
        df_phys = prepare_dataframe(df_raw)

        # Diagnostic HLE mask with QP_THRESHOLD=0 (requested)
        df_phys['recession_HLE'] = compute_hle_recession_mask_zero(df_phys).astype(int)

        # Attach training-selection recession mask (default method)
        df_phys['recession_flag'] = compute_recession_mask(df_phys).astype(int)

        # Split train/test chronologically
        n_total = len(df_phys)
        n_train = int(n_total * TrainRatio)
        train_df_phys = df_phys.iloc[:n_train].reset_index(drop=True)
        test_df_phys  = df_phys.iloc[n_train:].reset_index(drop=True)

        # Statics for this gridcode
        row_stat = static_df_full_std.loc[static_df_full_std['gridcode'] == GridCode]
        if row_stat.empty:
            print(f"Statics not found for gridcode={GridCode}; skipping.")
            continue
        stat_vec = row_stat[staticColumns].values[0]  # (S,)

        # Build normalized dynamic features — fit scalers on train only
        train_df_norm = train_df_phys.copy()
        test_df_norm  = test_df_phys.copy()
        dyn_scalers = {}
        for col in X_normalized_dynamic_features:
            sc = StandardScaler()
            train_df_norm[col] = sc.fit_transform(train_df_norm[[col]])
            test_df_norm[col]  = sc.transform(test_df_norm[[col]])
            dyn_scalers[col]   = sc

        # carry flags + date
        for df_n, df_p in [(train_df_norm, train_df_phys), (test_df_norm, test_df_phys)]:
            df_n['recession_flag'] = df_p['recession_flag']
            df_n['date']           = df_p['date']

        # Metrics per TW across bootstrap
        bootstrap_metrics = []
        loss_curves_all = []

        # For date-level aggregation across all TWs (Train/Test separately)
        agg_train_frames = []
        agg_test_frames  = []

        for TW in bootstrap_target_windows:
            TRAJ_LEN = INPUT_WINDOW + TW
            print(f"  Bootstrapping TARGET_WINDOW={TW} (TRAJ_LEN={TRAJ_LEN})")

            # Build datasets (train)
            X_tr, Ylog_tr, Q0log_tr, Phys_tr = build_dataset(
                train_df_phys, train_df_norm, stat_vec, TRAJ_LEN, INPUT_WINDOW
            )
            
            # NEW: skip TW if nothing survived NaN filtering
            if X_tr is None:
                print("No train trajectories after NaN filtering; skip TW")
                continue
            
            filt_tr = filter_trajectories_by_recession(
                Phys_tr, INPUT_WINDOW, TW, recession_threshold
            )
            if filt_tr is None:
                print("   -> No train trajectories pass recession filter; skip TW.")
                continue
            idx_tr, Phys_tr = filt_tr
            X_tr       = X_tr[idx_tr]
            Ylog_tr    = Ylog_tr[idx_tr]
            Q0log_tr   = Q0log_tr[idx_tr]
            stat_tr    = np.repeat(stat_vec.reshape(1,-1), repeats=X_tr.shape[0], axis=0)

            # Build datasets (test)
            X_te, Ylog_te, Q0log_te, Phys_te = build_dataset(
                test_df_phys, test_df_norm, stat_vec, TRAJ_LEN, INPUT_WINDOW
            )
            
            # NEW: skip TW if nothing survived NaN filtering
            if X_te is None:
                print("No test trajectories after NaN filtering; skip TW.")
                continue
            
            filt_te = filter_trajectories_by_recession(
                Phys_te, INPUT_WINDOW, TW, recession_threshold
            )
            if filt_te is None:
                print("   -> No test trajectories pass recession filter; skip TW.")
                continue
            idx_te, Phys_te = filt_te
            X_te     = X_te[idx_te]
            Ylog_te  = Ylog_te[idx_te]
            Q0log_te = Q0log_te[idx_te]
            stat_te  = np.repeat(stat_vec.reshape(1,-1), repeats=X_te.shape[0], axis=0)

            # Init models fresh per TW
            #input_dim = X_tr.shape[2]
            #Hsize = 50

            #encoder = ForcingEncoder(input_size=input_dim, hidden_size=Hsize, num_layers=1, dropout=0.0).to(device)
            #g_net   = ContextualGFunction(hidden_size=Hsize, dropout_rate=0.4).to(device)
            #q0_head = Q0Head(hidden_size=Hsize).to(device)
            #lp_gamma_head = LPGammaHead(in_dim=len(staticColumns), hidden=[64,64]).to(device)
            # >>> REPLACE: pass per-gridcode bounds into the head
            input_dim = X_tr.shape[2]
            Hsize = HiddenSize
            
            encoder = ForcingEncoder(input_size=input_dim, hidden_size=Hsize, num_layers=1, dropout=0.0).to(device)
            g_net   = ContextualGFunction(hidden_size=Hsize, dropout_rate=0.4).to(device)
            q0_head = Q0Head(hidden_size=Hsize).to(device)
            lp_gamma_head = LPGammaHead(
                in_dim=len(staticColumns), hidden=[64, 64],
                lp_bounds=(LP_MIN, LP_MAX),
                gamma_bounds=(GAMMA_MIN, GAMMA_MAX)
            ).to(device)
            # <<< REPLACE END
            
            # ------------------ LOAD PREVIOUS WEIGHTS (carry-forward or pretrained) ------------------
            if Training_Mode == "General":
                # Load the most recent rolling-latest weights (carry-forward from previous grid)
                _load_state_if_exists(encoder, os.path.join(model_weights_path_general, f"encoder_tw{TW}.pth"), device)
                _load_state_if_exists(g_net,   os.path.join(model_weights_path_general, f"gnet_tw{TW}.pth"), device)
                _load_state_if_exists(q0_head, os.path.join(model_weights_path_general, f"q0_tw{TW}.pth"), device)
                _load_state_if_exists(
                    lp_gamma_head,
                    os.path.join(model_weights_path_general, f"lp_gamma_tw{TW}.pth"),
                    device,
                    ignore_keys={"lp_min", "lp_max", "gm_min", "gm_max"}
                )
            
            elif Training_Mode == "Finetuning":
                # Always start from the latest general model (no carry-forward chain)
                _load_state_if_exists(encoder, os.path.join(model_weights_path_general, f"encoder_tw{TW}.pth"), device)
                _load_state_if_exists(g_net,   os.path.join(model_weights_path_general, f"gnet_tw{TW}.pth"), device)
                _load_state_if_exists(q0_head, os.path.join(model_weights_path_general, f"q0_tw{TW}.pth"), device)
                _load_state_if_exists(
                    lp_gamma_head,
                    os.path.join(model_weights_path_general, f"lp_gamma_tw{TW}.pth"),
                    device,
                    ignore_keys={"lp_min", "lp_max", "gm_min", "gm_max"}
                )
            # -----------------------------------------------------------------------


 
            # Train
            dataset_tr = (X_tr, Ylog_tr, Q0log_tr, Phys_tr, stat_tr)
            s_Q   = torch.nn.Parameter(torch.tensor(0.0, device=device))
            loss_curve = train_model(dataset_tr, encoder, g_net, q0_head, lp_gamma_head,
                                     BatchSize, INPUT_WINDOW, TRAJ_LEN, TW,
                                     s_Q, len(staticColumns), warm_up=False)
            loss_curves_all.append(loss_curve)
            
            # --- Get alpha for this GridCode (static, from statics) ---
            with torch.no_grad():
                stat_vec_tensor = torch.tensor(stat_vec, dtype=torch.float32, device=device).unsqueeze(0)
                _, _, ALPHA_gc = lp_gamma_head(stat_vec_tensor)   # (1,1)
                alpha_gc = float(ALPHA_gc.squeeze().item())

            # Save per-TW weights
            #torch.save(encoder.state_dict(), os.path.join(model_weights_path, f"encoder_gc{GridCode}_tw{TW}.pth"))
            #torch.save(g_net.state_dict(),   os.path.join(model_weights_path, f"gnet_gc{GridCode}_tw{TW}.pth"))
            #torch.save(q0_head.state_dict(), os.path.join(model_weights_path, f"q0_gc{GridCode}_tw{TW}.pth"))
            #torch.save(lp_gamma_head.state_dict(), os.path.join(model_weights_path, f"lp_gamma_gc{GridCode}_tw{TW}.pth"))
            # ------------------ SAVE MODEL WEIGHTS (NEW SYSTEM) ------------------
            if Training_Mode == "General":
                # 1-Save per-grid (archival)
                torch.save(encoder.state_dict(), os.path.join(model_weights_path_general, f"encoder_gc{GridCode}_tw{TW}.pth"))
                torch.save(g_net.state_dict(),   os.path.join(model_weights_path_general, f"gnet_gc{GridCode}_tw{TW}.pth"))
                torch.save(q0_head.state_dict(), os.path.join(model_weights_path_general, f"q0_gc{GridCode}_tw{TW}.pth"))
                torch.save(lp_gamma_head.state_dict(), os.path.join(model_weights_path_general, f"lp_gamma_gc{GridCode}_tw{TW}.pth"))
            
                # 2-Update rolling-latest (carry-forward for next grid)
                torch.save(encoder.state_dict(), os.path.join(model_weights_path_general, f"encoder_tw{TW}.pth"))
                torch.save(g_net.state_dict(),   os.path.join(model_weights_path_general, f"gnet_tw{TW}.pth"))
                torch.save(q0_head.state_dict(), os.path.join(model_weights_path_general, f"q0_tw{TW}.pth"))
                torch.save(lp_gamma_head.state_dict(), os.path.join(model_weights_path_general, f"lp_gamma_tw{TW}.pth"))
            
            
            elif Training_Mode == "Finetuning":
                # Save only per-grid finetuned weights (no carry-forward)
                torch.save(encoder.state_dict(), os.path.join(model_weights_path_finetuned, f"encoder_gc{GridCode}_tw{TW}_Finetuned.pth"))
                torch.save(g_net.state_dict(),   os.path.join(model_weights_path_finetuned, f"gnet_gc{GridCode}_tw{TW}_Finetuned.pth"))
                torch.save(q0_head.state_dict(), os.path.join(model_weights_path_finetuned, f"q0_gc{GridCode}_tw{TW}_Finetuned.pth"))
                torch.save(lp_gamma_head.state_dict(), os.path.join(model_weights_path_finetuned, f"lp_gamma_gc{GridCode}_tw{TW}_Finetuned.pth"))
            # ---------------------------------------------------------------------


            # Evaluate (train split)
            pred_tr, target_tr, g_tr, aet_tr, date_tr = evaluate_model(
                (X_tr, Ylog_tr, Q0log_tr, Phys_tr, stat_tr),
                encoder, g_net, q0_head, lp_gamma_head,
                BatchSize, INPUT_WINDOW, TRAJ_LEN, TW
            )
            # Evaluate (test split)
            pred_te, target_te, g_te, aet_te, date_te = evaluate_model(
                (X_te, Ylog_te, Q0log_te, Phys_te, stat_te),
                encoder, g_net, q0_head, lp_gamma_head,
                BatchSize, INPUT_WINDOW, TRAJ_LEN, TW
            )

            # Overall metrics
            train_nse = NSE(target_tr.flatten(), pred_tr.flatten())
            train_kge = KGE(target_tr.flatten(), pred_tr.flatten())
            test_nse  = NSE(target_te.flatten(),  pred_te.flatten())
            test_kge  = KGE(target_te.flatten(),  pred_te.flatten())

            # Unique-Q metrics
            flat_t_tr = target_tr.flatten()
            flat_p_tr = pred_tr.flatten()
            flat_g_tr = g_tr.flatten()
            uniq_tr, inv_tr = np.unique(flat_t_tr, return_inverse=True)
            avg_pred_tr = np.bincount(inv_tr, weights=flat_p_tr) / np.maximum(np.bincount(inv_tr), 1)
            avg_g_tr    = np.bincount(inv_tr, weights=flat_g_tr) / np.maximum(np.bincount(inv_tr), 1)
            nse_value_train = NSE(uniq_tr, avg_pred_tr)
            kge_value_train = KGE(uniq_tr, avg_pred_tr)

            flat_t_te = target_te.flatten()
            flat_p_te = pred_te.flatten()
            flat_g_te = g_te.flatten()
            uniq_te, inv_te = np.unique(flat_t_te, return_inverse=True)
            avg_pred_te = np.bincount(inv_te, weights=flat_p_te) / np.maximum(np.bincount(inv_te), 1)
            avg_g_te    = np.bincount(inv_te, weights=flat_g_te) / np.maximum(np.bincount(inv_te), 1)
            nse_value_test = NSE(uniq_te, avg_pred_te)
            kge_value_test = KGE(uniq_te, avg_pred_te)

            # ---- Date-level aggregation for this TW (Train/Test) ----
            # Train
            df_tr_tw = pd.DataFrame({
                'date_dt': date_tr.reshape(-1),
                'Q_pred_step': pred_tr.reshape(-1),
                'GQ_pred_step': g_tr.reshape(-1),
                'AET_pred_step': aet_tr.reshape(-1)
            }).dropna(subset=['date_dt'])
            df_tr_tw = df_tr_tw.groupby('date_dt', as_index=False).mean()

            # Test
            df_te_tw = pd.DataFrame({
                'date_dt': date_te.reshape(-1),
                'Q_pred_step': pred_te.reshape(-1),
                'GQ_pred_step': g_te.reshape(-1),
                'AET_pred_step': aet_te.reshape(-1)
            }).dropna(subset=['date_dt'])
            df_te_tw = df_te_tw.groupby('date_dt', as_index=False).mean()

            agg_train_frames.append(df_tr_tw)
            agg_test_frames.append(df_te_tw)

            # Store metrics
            bootstrap_metrics.append({
                'TARGET_WINDOW': TW,
                'train_nse': train_nse, 'train_kge': train_kge,
                'test_nse':  test_nse,  'test_kge':  test_kge,
                'nse_value_train': nse_value_train, 'kge_value_train': kge_value_train,
                'nse_value_test':  nse_value_test,  'kge_value_test':  kge_value_test
            })

        if not bootstrap_metrics:
            print(f"No valid bootstrap runs for GridCode {GridCode}; skipping this grid.")
            continue

        # ---- Combine date-level aggregation across all TWs ----
        def combine_date_aggs(frames):
            if len(frames) == 0:
                return pd.DataFrame(columns=['date_dt','Q_pred','GQ_pred','AET_pred'])
            df = pd.concat(frames, ignore_index=True)
            df = df.groupby('date_dt', as_index=False).mean()
            df = df.rename(columns={
                'Q_pred_step':'Q_pred',
                'GQ_pred_step':'GQ_pred',
                'AET_pred_step':'AET_pred'
            })
            return df

        df_pred_train = combine_date_aggs(agg_train_frames)
        df_pred_test  = combine_date_aggs(agg_test_frames)

        # ---- Build final per-date CSV (both splits in one file) ----
        df_out = df_phys[['date','date_str_orig','precipitation_mmd','temperature_C','pet_mmd','aet_mm','SM_%','streamflow_mmd','recession_HLE']].copy()
        df_out['date_dt'] = df_out['date'].values
        df_out = df_out.drop(columns=['date'])
        df_out = df_out.rename(columns={
            'date_str_orig': 'date',   # export string date
            'precipitation_mmd':'P',
            'temperature_C':'T',
            'pet_mmd':'PET',
            'aet_mm':'AET',
            'SM_%':'SM',
            'streamflow_mmd':'Q',
            'recession_HLE':'is_this_date_recession_HLE'
        })

        # Left-join predictions
        df_out['Data_split_column'] = np.where(np.arange(len(df_out)) < n_train, 'Train', 'Test')

        # Merge train preds
        df_out = df_out.merge(df_pred_train, on='date_dt', how='left', suffixes=('',''))
        for col in ['Q_pred','GQ_pred','AET_pred']:
            df_out.loc[df_out['Data_split_column']!='Train', col] = np.nan

        # Merge test preds (fill only for Test rows)
        df_out = df_out.merge(df_pred_test, on='date_dt', how='left', suffixes=('','_te'))
        for col in ['Q_pred','GQ_pred','AET_pred']:
            te_col = f"{col}_te"
            df_out.loc[df_out['Data_split_column']=='Test', col] = df_out.loc[df_out['Data_split_column']=='Test', te_col]
            df_out.drop(columns=[te_col], inplace=True)

        # Reorder/export columns
        df_out['alpha'] = alpha_gc  # same for all rows in this catchment

        df_out = df_out[['date','P','T','PET','AET','SM','Q','Q_pred',
                         'GQ_pred','AET_pred','alpha',
                         'Data_split_column','is_this_date_recession_HLE']]

        
                # -------------------------------------------------------------
        # Scatter: Predicted g (GQ_pred) vs Q, Y on log scale
        # -------------------------------------------------------------
        plot_GE = True

        if plot_GE:
            # Prepare data (guard against NaN/inf and log-domain issues)
            df_scatter = df_out[['Q', 'GQ_pred', 'Data_split_column', 'is_this_date_recession_HLE']].copy()
            df_scatter = df_scatter.replace([np.inf, -np.inf], np.nan).dropna()
            df_scatter = df_scatter[df_scatter['GQ_pred'] > 0]   # log(Y) requires Y>0
            df_scatter = df_scatter[df_scatter['Q'] > 0]         # log(X) requires X>0
        
            # Keep only recession days (HLE)
            df_scatter = df_scatter[df_scatter['is_this_date_recession_HLE'] == 1]
        
            # Early exit if nothing to plot
            if not df_scatter.empty:
                fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=DPI)
        
                # Axes + grid
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)
                ax.minorticks_on()
                ax.tick_params(axis='both', which='both', direction='out', length=4, width=0.8)
        
                # Clean decimal tick labels (e.g., 0.128)
                fmt = FuncFormatter(lambda x, pos: f"{x:.3f}".rstrip('0').rstrip('.'))
                ax.xaxis.set_major_formatter(fmt)
                ax.yaxis.set_major_formatter(fmt)
        
                # Split by Train/Test for clarity
                split_styles = {
                    'Train': dict(s=8,  marker='o', alpha=0.45, linewidths=0, rasterized=True),
                    'Test' : dict(s=10, marker='o', alpha=0.85, linewidths=0, rasterized=True),
                }
                for split, style in split_styles.items():
                    sub = df_scatter[df_scatter['Data_split_column'] == split]
                    if not sub.empty:
                        ax.scatter(sub['Q'].values, sub['GQ_pred'].values, label=split, **style)
        
                # Labels
                ax.set_xlabel(r'$Q$ (mm day$^{-1}$)', labelpad=6)
                ax.set_ylabel(r'Predicted $g$ (day$^{-1}$)', labelpad=6)
        
                # Sensible axis limits for log–log (avoid tails/zeros)
                x = df_scatter['Q'].values
                y = df_scatter['GQ_pred'].values
                x_pos = x[np.isfinite(x) & (x > 0)]
                y_pos = y[np.isfinite(y) & (y > 0)]
        
                if x_pos.size:
                    x_left  = np.nanpercentile(x_pos, 0.5) * 0.9
                    x_right = np.nanpercentile(x_pos, 99.5)
                    if np.isfinite(x_left) and np.isfinite(x_right) and x_right > x_left and x_left > 0:
                        ax.set_xlim(left=x_left, right=x_right)
        
                if y_pos.size:
                    y_bottom = np.nanpercentile(y_pos, 0.5) * 0.9
                    y_top    = np.nanpercentile(y_pos, 99.5)
                    if np.isfinite(y_bottom) and np.isfinite(y_top) and y_top > y_bottom and y_bottom > 0:
                        ax.set_ylim(bottom=y_bottom, top=y_top)
        
                # Legend (no box)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(frameon=False, ncols=2, fontsize=max(Font_Size_X_Y_Title-2, 10))
        
                # Title
                ax.set_title(f'GridCode {GridCode}: $g$ vs $Q$', fontsize=Font_Size_X_Y_Title)
        
                fig.tight_layout()
                out_png = os.path.join(results_outdir, f'GQ_vs_Q_gc{GridCode}.png')
                plt.savefig(out_png, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
            else:
                print("Plot skipped: no valid (Q, GQ_pred) points after filtering for log-scale.")
          
 
        # Write the per-gridcode CSV
        per_gc_csv = os.path.join(results_outdir, f'date_level_predictions_gc{GridCode}.csv')
        df_out.to_csv(per_gc_csv, index=False)

        # ---- Compute final NSE/KGE on Q and AET from the exported CSV ----
        def safe_metrics(obs, pred):
            m_nse = NSE(obs, pred)
            m_kge = KGE(obs, pred)
            return m_nse, m_kge

        # Train metrics
        dft = df_out[df_out['Data_split_column']=='Train'].copy()
        dft_q = dft[['Q','Q_pred']].dropna()
        dft_a = dft[['AET','AET_pred']].dropna()
        NSE_final_Q_train,  KGE_final_Q_train  = (np.nan, np.nan) if dft_q.empty else safe_metrics(dft_q['Q'].values,  dft_q['Q_pred'].values)
        NSE_final_AET_train,KGE_final_AET_train= (np.nan, np.nan) if dft_a.empty else safe_metrics(dft_a['AET'].values, dft_a['AET_pred'].values)

        # Test metrics
        dfe = df_out[df_out['Data_split_column']=='Test'].copy()
        dfe_q = dfe[['Q','Q_pred']].dropna()
        dfe_a = dfe[['AET','AET_pred']].dropna()
        NSE_final_Q_test,   KGE_final_Q_test   = (np.nan, np.nan) if dfe_q.empty else safe_metrics(dfe_q['Q'].values,  dfe_q['Q_pred'].values)
        NSE_final_AET_test, KGE_final_AET_test = (np.nan, np.nan) if dfe_a.empty else safe_metrics(dfe_a['AET'].values, dfe_a['AET_pred'].values)
        
        # >>> ADD: Pearson R and R^2 for AET (Train/Test)
        if dft_a.empty:
            R_final_AET_train  = np.nan
            R2_final_AET_train = np.nan
        else:
            R_final_AET_train  = pearson_r_np(dft_a['AET'].values, dft_a['AET_pred'].values)
            R2_final_AET_train = r2_from_r_np(R_final_AET_train)
        
        if dfe_a.empty:
            R_final_AET_test  = np.nan
            R2_final_AET_test = np.nan
        else:
            R_final_AET_test  = pearson_r_np(dfe_a['AET'].values, dfe_a['AET_pred'].values)
            R2_final_AET_test = r2_from_r_np(R_final_AET_test)
        # <<< ADD END


        print(f"Final (date-level) metrics for GridCode {GridCode}:")
        print(f"  Train: NSE_Q={NSE_final_Q_train:.4f}, KGE_Q={KGE_final_Q_train:.4f}, R_AET={R_final_AET_train:.4f}, R2_AET={R2_final_AET_train:.4f}")
        print(f"  Test : NSE_Q={NSE_final_Q_test:.4f},  KGE_Q={KGE_final_Q_test:.4f},  R_AET={R_final_AET_test:.4f},  R2_AET={R2_final_AET_test:.4f}")


        # ---- Aggregate per-TW (existing behavior) ----
        avg_metrics = {
            'train_nse': np.mean([m['train_nse'] for m in bootstrap_metrics]),
            'train_kge': np.mean([m['train_kge'] for m in bootstrap_metrics]),
            'test_nse':  np.mean([m['test_nse']  for m in bootstrap_metrics]),
            'test_kge':  np.mean([m['test_kge']  for m in bootstrap_metrics]),
            'nse_value_train': np.mean([m['nse_value_train'] for m in bootstrap_metrics]),
            'kge_value_train': np.mean([m['kge_value_train'] for m in bootstrap_metrics]),
            'nse_value_test':  np.mean([m['nse_value_test']  for m in bootstrap_metrics]),
            'kge_value_test':  np.mean([m['kge_value_test']  for m in bootstrap_metrics])
        }

        #row = {
        #    'GridCode': GridCode,
        #    'train_NSE': avg_metrics['train_nse'],
        #    'train_KGE': avg_metrics['train_kge'],
        #    'test_NSE':  avg_metrics['test_nse'],
        #    'test_KGE':  avg_metrics['test_kge'],
        #    'train_NSE_unique_target': avg_metrics['nse_value_train'],
        #    'train_KGE_unique_target': avg_metrics['kge_value_train'],
        #    'test_NSE_unique_target':  avg_metrics['nse_value_test'],
        #    'test_KGE_unique_target':  avg_metrics['kge_value_test'],
        #    'NSE_final_Q_train':  NSE_final_Q_train,
        #    'KGE_final_Q_train':  KGE_final_Q_train,
        #    'NSE_final_AET_train':NSE_final_AET_train,
        #    'KGE_final_AET_train':KGE_final_AET_train, 
        #    'NSE_final_Q_test':   NSE_final_Q_test,
        #    'KGE_final_Q_test':   KGE_final_Q_test,
        #    'NSE_final_AET_test': NSE_final_AET_test,
        #    'KGE_final_AET_test': KGE_final_AET_test
        #}
        row = {
            'GridCode': GridCode,
            'train_NSE': avg_metrics['train_nse'],
            'train_KGE': avg_metrics['train_kge'],
            'test_NSE':  avg_metrics['test_nse'],
            'test_KGE':  avg_metrics['test_kge'],
            'train_NSE_unique_target': avg_metrics['nse_value_train'],
            'train_KGE_unique_target': avg_metrics['kge_value_train'],
            'test_NSE_unique_target':  avg_metrics['nse_value_test'],
            'test_KGE_unique_target':  avg_metrics['kge_value_test'],
            'NSE_final_Q_train':  NSE_final_Q_train,
            'KGE_final_Q_train':  KGE_final_Q_train,
            'NSE_final_AET_train':NSE_final_AET_train,
            'KGE_final_AET_train':KGE_final_AET_train,
            'R_final_AET_train':  R_final_AET_train,     # >>> ADD
            'R2_final_AET_train': R2_final_AET_train,    # >>> ADD
            'NSE_final_Q_test':   NSE_final_Q_test,
            'KGE_final_Q_test':   KGE_final_Q_test,
            'NSE_final_AET_test': NSE_final_AET_test,
            'KGE_final_AET_test': KGE_final_AET_test,
            'R_final_AET_test':   R_final_AET_test,      # >>> ADD
            'R2_final_AET_test':  R2_final_AET_test,      # >>> ADD
            'alpha':              alpha_gc      # <<< NEW FIELD
        }


        for m in bootstrap_metrics:
            tw = m['TARGET_WINDOW']
            row[f"train_nse_tw{tw}"] = m['train_nse']
            row[f"train_kge_tw{tw}"] = m['train_kge']
            row[f"test_nse_tw{tw}"]  = m['test_nse']
            row[f"test_kge_tw{tw}"]  = m['test_kge']
            row[f"nse_value_train_tw{tw}"] = m['nse_value_train']
            row[f"kge_value_train_tw{tw}"] = m['kge_value_train']
            row[f"nse_value_test_tw{tw}"]  = m['nse_value_test']
            row[f"kge_value_test_tw{tw}"]  = m['kge_value_test']

        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        results_df.to_csv(os.path.join(results_outdir, f"FineTuning_AccuracyResults_Cluster_{cluster}.csv"), index=False)

    end_time = time.time()
    print("Training and evaluation completed. Metrics saved.")
    print("Elapsed time (s):", (end_time - start_time))
    print("Elapsed time (m):", (end_time - start_time)/60)
    print("Elapsed time (h):", (end_time - start_time)/3600)

# ===============================================
# RUN
# ===============================================
if __name__ == "__main__":
    main()
