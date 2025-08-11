# Toy Monte Carlo run: generate sample events, plots, and a short search-note PDF.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

np.random.seed(42)

# Parameters (match Appendix example choices plus variety)
N_EVENTS = 3000
tau0 = 1e-9       # 1 ns base
alpha = 0.9
K_mean = 1.2
E_top_perK = 1000.0
detector_time_res = 200e-12  # 200 ps

def sample_complexity(mean=K_mean):
    k = np.random.poisson(mean)
    return max(0, k)

def tau_of_K(K):
    return tau0 * np.exp(alpha * K)

def n_daughters_of_K(K):
    lam = 1 + K
    return np.random.poisson(lam) + 1

def emission_times_for_event(K, tau):
    n = n_daughters_of_K(K)
    times = []
    while len(times) < n:
        t = np.random.exponential(scale=tau)
        if t <= 5 * tau:
            times.append(t)
    return np.array(times)

def energies_for_event(K, n):
    total_E = E_top_perK * max(1, K)
    parts = np.random.exponential(scale=1.0, size=n)
    parts = parts / parts.sum()
    return parts * total_E

rows = []
all_emission_times = []
for ev in range(N_EVENTS):
    K = sample_complexity()
    tau = tau_of_K(K)
    times = emission_times_for_event(K, tau)
    n = len(times)
    energies = energies_for_event(K, n)
    measured_times = times + np.random.normal(scale=detector_time_res, size=n)
    measured_times = np.clip(measured_times, 0, None)
    for i in range(n):
        rows.append({
            "event_id": ev,
            "K": int(K),
            "tau_s": tau,
            "daughter_index": int(i),
            "true_t_s": times[i],
            "measured_t_s": measured_times[i],
            "energy": energies[i]
        })
    all_emission_times.extend(times.tolist())

df = pd.DataFrame(rows)
event_summary = df.groupby('event_id').agg({
    'K': 'first',
    'tau_s': 'first',
    'daughter_index': 'count',
    'energy': 'sum',
    'true_t_s': ['min','max']
}).reset_index()
event_summary.columns = ['event_id', 'K', 'tau_s', 'n_daughters', 'total_energy', 't_min_true', 't_max_true']

# Save CSV sample
out_csv = "mnt/data/toy_topology_events_full.csv"
df.to_csv(out_csv, index=False)

# Plot 1: Histogram of true emission times (ns) with log y
plt.figure(figsize=(8,4))
plt.hist(np.array(all_emission_times)*1e9, bins=200, color='tab:blue', alpha=0.8)
plt.xlabel("Emission time (ns)")
plt.ylabel("Counts")
plt.title("Histogram of true emission times (toy topology MC)")
plt.grid(alpha=0.3)
plt.tight_layout()
plot1 = "mnt/data/emit_time_hist.png"
plt.savefig(plot1, dpi=200)
plt.close()

# Plot 2: Time-spread (t_max - t_min) distribution for events, in ns
event_summary['delta_t_ns'] = (event_summary['t_max_true'] - event_summary['t_min_true']) * 1e9
plt.figure(figsize=(8,4))
plt.hist(event_summary['delta_t_ns'], bins=120, color='tab:orange', alpha=0.9)
plt.xlabel("Event time-spread Δt (ns)")
plt.ylabel("Number of events")
plt.title("Distribution of per-event time-spread Δt (toy MC)")
plt.grid(alpha=0.3)
plt.tight_layout()
plot2 = "mnt/data/delta_t_hist.png"
plt.savefig(plot2, dpi=200)
plt.close()

# Plot 3: Example event timeline for a representative K=3 event (or nearest)
candidates = event_summary[event_summary['K']>=3]
if len(candidates)==0:
    example_ev = event_summary.sample(1, random_state=7).iloc[0]['event_id']
else:
    example_ev = candidates.sample(1, random_state=7).iloc[0]['event_id']

ev_rows = df[df['event_id'] == example_ev].sort_values('true_t_s')
plt.figure(figsize=(9,2.5))
y = np.arange(len(ev_rows))
plt.scatter(ev_rows['true_t_s']*1e9, y, marker='o', label='true emission times')
plt.scatter(ev_rows['measured_t_s']*1e9, y, marker='x', label='measured times (smeared)')
for _, r in ev_rows.iterrows():
    plt.plot([r['true_t_s']*1e9, r['measured_t_s']*1e9], [r['daughter_index'], r['daughter_index']], linestyle='--', linewidth=0.8)
plt.xlabel("Time (ns)")
plt.yticks(y, [f"d{i}" for i in ev_rows['daughter_index']])
plt.legend(loc='upper right', fontsize=8)
plt.title(f"Example event timeline (event_id={int(example_ev)}, K={int(ev_rows['K'].iloc[0])}, tau={ev_rows['tau_s'].iloc[0]:.2e} s)")
plt.tight_layout()
plot3 = "mnt/data/example_event_timeline.png"
plt.savefig(plot3, dpi=200)
plt.close()

# Create a short search-note PDF with recommended analysis steps and plots included
pdf_path = "mnt/data/search_note_topology_tuning.pdf"
with PdfPages(pdf_path) as pdf:
    # Page 1: Title
    fig = plt.figure(figsize=(8.5,11))
    fig.text(0.1,0.85, "Search Note: Reanalysis for Topology-Delayed (Temporal Caching) Signals", fontsize=16, weight='bold')
    fig.text(0.1,0.80, "Author: Kristian Magda", fontsize=10)
    fig.text(0.1,0.76, "Date: August 2025", fontsize=10)
    fig.text(0.1,0.68, "Summary:", fontsize=12, weight='bold')
    text = ("This short note describes a practical reanalysis plan and recommended selections to search for\n"
            "temporally delayed decays arising from metastable topological excitations (``temporal caching'').\n\n"
            "Key signatures:\n"
            "- Temporally smeared decay vertices with time-spread Δt ~ τ(K) (ps–ms range depending on parameters).\n"
            "- Apparent long-lived particle events where lifetime arises from topology unraveling.\n"
            "- Correlated timing excess across subdetectors and nonstandard daughter ordering.\n\n"
            "The companion toy Monte Carlo (toy_topology_mc.py) generates templates for f(t;τ), Δt, and energy partitions; example plots are included in this PDF.\n")
    fig.text(0.1,0.64, text, fontsize=9)
    fig.text(0.1,0.24, "Recommended immediate actions for experimental groups:", fontsize=12, weight='bold')
    actions = ("1) Use zero-bias and calibration streams to build unbiased timing background models.\n"
               "2) Reconstruct per-daughter times using precision timing layers (HGTD, MIP timing) and calorimeter timing.\n"
               "3) Compute Δt_event = t_max - t_min and compare to prompt templates; use likelihood ratio tests.\n"
               "4) Require cross-detector timing consistency to reduce noise/artifact backgrounds.\n"
               "5) Inject toy MC templates into full detector sim (Delphes or experiment-specific) to estimate sensitivity.\n")
    fig.text(0.1,0.20, actions, fontsize=9)
    pdf.savefig(fig); plt.close()

    # Page 2: include histogram plot1
    fig = plt.figure(figsize=(8.5,11))
    img = plt.imread(plot1)
    plt.imshow(img); plt.axis('off')
    plt.title("Histogram of true emission times (toy MC)")
    pdf.savefig(fig); plt.close()

    # Page 3: include delta_t histogram
    fig = plt.figure(figsize=(8.5,11))
    img = plt.imread(plot2)
    plt.imshow(img); plt.axis('off')
    plt.title("Distribution of per-event time-spread Δt (toy MC)")
    pdf.savefig(fig); plt.close()

    # Page 4: example timeline
    fig = plt.figure(figsize=(8.5,11))
    img = plt.imread(plot3)
    plt.imshow(img); plt.axis('off')
    plt.title("Example event timeline (true vs measured times)")
    pdf.savefig(fig); plt.close()

# List produced files for user
produced = [out_csv, plot1, plot2, plot3, pdf_path]
for p in produced:
    print("Produced:", p)

# Display small summary dataframe (first 8 rows)
event_summary_sample = event_summary.head(8).copy()
event_summary_sample['tau_ns'] = event_summary_sample['tau_s']*1e9
event_summary_sample['delta_t_ns'] = event_summary_sample['delta_t_ns']
event_summary_sample = event_summary_sample[['event_id','K','tau_ns','n_daughters','total_energy','delta_t_ns']]
print("\nSample event summary (first 8 rows):")
print(event_summary_sample.to_string(index=False))

# Return paths to user (they will be shown in the notebook output)
produced_files = produced
produced_files

