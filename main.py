import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi gaya visualisasi grafik
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
sns.set_palette("viridis")

def generate_sample_dataset(filepath="air_quality_dataset.csv"):
    """
    Membuat dataset sampel kualitas udara jika file CSV kosong atau belum tersedia.
    """
    np.random.seed(42)
    days = 60
    dates = pd.date_range(start="2025-01-01", periods=days, freq="D")
    
    pm25 = np.random.lognormal(mean=4.1, sigma=0.35, size=days).round(2)
    pm10 = (pm25 * 1.35 + np.random.normal(0, 5, size=days)).round(2)
    so2 = np.random.normal(loc=22, scale=4.5, size=days).clip(5, 60).round(2)
    co = np.random.normal(loc=16, scale=4.0, size=days).clip(3, 50).round(2)
    o3 = np.random.normal(loc=48, scale=12.0, size=days).clip(10, 120).round(2)
    no2 = np.random.normal(loc=30, scale=7.0, size=days).clip(5, 80).round(2)
    ispu = (pm25 * 1.25).round(0).astype(int)

    df = pd.DataFrame({
        "Tanggal": dates,
        "Stasiun": ["DKI1 Bundaran HI"] * days,
        "PM2.5": pm25,
        "PM10": pm10,
        "SO2": so2,
        "CO": co,
        "O3": o3,
        "NO2": no2,
        "ISPU": ispu
    })
    df.to_csv(filepath, index=False)
    print(f"[INFO] Dataset sampel berhasil dibuat di: {filepath}")
    return df

def build_monte_carlo_table(series, num_bins=6):
    """
    Membentuk tabel distribusi frekuensi, probabilitas, kumulatif (CDF),
    dan interval angka acak (00 - 99) untuk Metode Monte Carlo.
    """
    counts, bin_edges = np.histogram(series.dropna(), bins=num_bins)
    total = counts.sum()
    
    table_rows = []
    cumulative = 0.0
    prev_end = -1
    
    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        midpoint = (low + high) / 2.0
        freq = counts[i]
        prob = freq / total
        cumulative += prob
        cdf = 1.0 if i == num_bins - 1 else cumulative
        
        rand_start = prev_end + 1
        rand_end = 99 if i == num_bins - 1 else int(round(cdf * 100)) - 1
        if freq > 0 and rand_end >= rand_start:
            prev_end = rand_end
            interval_str = f"{rand_start:02d} - {rand_end:02d}"
        else:
            interval_str = "-"
            
        table_rows.append({
            "Kelas_Bawah": round(low, 2),
            "Kelas_Atas": round(high, 2),
            "Nilai_Tengah": round(midpoint, 2),
            "Frekuensi": freq,
            "Probabilitas": round(prob, 4),
            "Prob_Kumulatif": round(cdf, 4),
            "Interval_Angka_Acak": interval_str,
            "Start_Int": rand_start,
            "End_Int": rand_end
        })
        
    return pd.DataFrame(table_rows)

def run_monte_carlo_simulation(df_table, n_simulations=1000, threshold=100.0, seed=101):
    """
    Menjalankan simulasi Monte Carlo sebanyak n_simulations iterasi.
    """
    np.random.seed(seed)
    random_u = np.random.uniform(0, 1, size=n_simulations)
    random_int = (random_u * 100).astype(int)
    
    simulated_values = []
    categories = []
    
    for u in random_u:
        # Mencari baris di mana u <= Prob_Kumulatif
        matched = df_table[df_table["Prob_Kumulatif"] >= u].iloc[0]
        # Interpolasi seragam di dalam kelas interval terpilih
        val = np.random.uniform(matched["Kelas_Bawah"], matched["Kelas_Atas"])
        simulated_values.append(round(val, 2))
        
        if val <= 50:
            categories.append("Baik")
        elif val <= 100:
            categories.append("Sedang")
        elif val <= 150:
            categories.append("Tidak Sehat")
        elif val <= 200:
            categories.append("Sangat Tidak Sehat")
        else:
            categories.append("Berbahaya")
            
    sim_df = pd.DataFrame({
        "Iterasi": range(1, n_simulations + 1),
        "Bilangan_Acak_U": random_u.round(4),
        "Angka_Acak_0_99": random_int,
        "Hasil_Simulasi": simulated_values,
        "Kategori": categories
    })
    
    return sim_df

def visualize_monte_carlo(df_hist, df_table, sim_df, target_col="PM2.5", threshold=100.0, horizon=30, save_path="visualisasi_monte_carlo.png"):
    """
    Membuat dashboard visualisasi 4 panel untuk analisis Simulasi Monte Carlo:
    1. Histogram Distribusi Hasil Simulasi & Garis Ambang Batas
    2. Kurva Distribusi Probabilitas Kumulatif (CDF) & Persentil 95%
    3. Simulasi Jalur Stokastik (Monte Carlo Trajectory Paths) H+30
    4. Proporsi Kategori Risiko Kualitas Udara (ISPU)
    """
    sim_values = sim_df["Hasil_Simulasi"].values
    hist_mean = df_hist[target_col].mean()
    sim_mean = np.mean(sim_values)
    p95 = np.percentile(sim_values, 95)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f"Dashboard Visualisasi Simulasi Monte Carlo - Kualitas Udara ({target_col})\n"
        f"Total Iterasi: {len(sim_df):,} | Rata-rata Simulasi: {sim_mean:.2f} | Risiko >= {threshold}: {(sim_values >= threshold).mean()*100:.1f}%",
        fontsize=15, fontweight="bold", y=0.98
    )

    # ---------------------------------------------------------
    # Panel 1: Histogram & Kurva KDE Hasil Simulasi
    # ---------------------------------------------------------
    ax1 = axes[0, 0]
    sns.histplot(sim_values, bins=20, kde=True, ax=ax1, color="#10b981", edgecolor="black", alpha=0.6)
    ax1.axvline(hist_mean, color="#3b82f6", linestyle="--", linewidth=2, label=f"Rata-rata Historis ({hist_mean:.1f})")
    ax1.axvline(sim_mean, color="#059669", linestyle="-", linewidth=2, label=f"Rata-rata Simulasi ({sim_mean:.1f})")
    ax1.axvline(threshold, color="#ef4444", linestyle="-.", linewidth=2, label=f"Ambang Bahaya ({threshold:.0f})")
    ax1.axvline(p95, color="#f59e0b", linestyle=":", linewidth=2, label=f"VaR / Persentil 95% ({p95:.1f})")
    ax1.set_title(f"1. Distribusi Hasil Simulasi Monte Carlo ({target_col})", fontsize=12, fontweight="bold")
    ax1.set_xlabel(f"Konsentrasi {target_col}")
    ax1.set_ylabel("Frekuensi")
    ax1.legend(loc="upper right", frameon=True)

    # ---------------------------------------------------------
    # Panel 2: Kurva Probabilitas Kumulatif (CDF)
    # ---------------------------------------------------------
    ax2 = axes[0, 1]
    sorted_vals = np.sort(sim_values)
    cdf_vals = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax2.plot(sorted_vals, cdf_vals, color="#0ea5e9", linewidth=2.5, label="CDF Simulasi")
    
    # Plot titik-titik kelas interval dari tabel Monte Carlo
    ax2.scatter(df_table["Kelas_Atas"], df_table["Prob_Kumulatif"], color="#ef4444", zorder=5, s=50, label="Batas Kelas Empiris")
    ax2.axhline(0.95, color="#f59e0b", linestyle=":", linewidth=1.5, label="Probabilitas 95%")
    ax2.axvline(p95, color="#f59e0b", linestyle=":", linewidth=1.5)
    ax2.set_title("2. Kurva Distribusi Probabilitas Kumulatif (CDF)", fontsize=12, fontweight="bold")
    ax2.set_xlabel(f"Konsentrasi {target_col}")
    ax2.set_ylabel("Probabilitas Kumulatif F(x)")
    ax2.legend(loc="lower right", frameon=True)

    # ---------------------------------------------------------
    # Panel 3: Jalur Stokastik (Monte Carlo Paths) 30 Hari
    # ---------------------------------------------------------
    ax3 = axes[1, 0]
    days = np.arange(1, horizon + 1)
    num_paths = 8
    np.random.seed(2025)
    
    all_paths = []
    for p in range(num_paths):
        u_path = np.random.uniform(0, 1, size=horizon)
        path_vals = []
        for u in u_path:
            matched = df_table[df_table["Prob_Kumulatif"] >= u].iloc[0]
            path_vals.append(np.random.uniform(matched["Kelas_Bawah"], matched["Kelas_Atas"]))
        all_paths.append(path_vals)
        ax3.plot(days, path_vals, alpha=0.45, linewidth=1.2, label=f"Skenario #{p+1}" if p < 3 else None)

    mean_path = np.mean(all_paths, axis=0)
    ax3.plot(days, mean_path, color="#1e293b", linewidth=2.5, label="Rata-rata Skenario")
    ax3.axhline(threshold, color="#ef4444", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.0f})")
    ax3.set_title(f"3. Simulasi Jalur Stokastik ({horizon} Hari ke Depan)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Hari Prediksi (H+)")
    ax3.set_ylabel(f"Estimasi {target_col}")
    ax3.legend(loc="upper right", fontsize=9, frameon=True)

    # ---------------------------------------------------------
    # Panel 4: Proporsi Kategori Kualitas Udara (ISPU)
    # ---------------------------------------------------------
    ax4 = axes[1, 1]
    kategori_order = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]
    color_map = {
        "Baik": "#10b981",
        "Sedang": "#0ea5e9",
        "Tidak Sehat": "#f59e0b",
        "Sangat Tidak Sehat": "#ef4444",
        "Berbahaya": "#7e22ce"
    }
    
    cat_counts = sim_df["Kategori"].value_counts().reindex(kategori_order, fill_value=0)
    cat_pct = (cat_counts / len(sim_df)) * 100
    bars = ax4.bar(
        cat_pct.index,
        cat_pct.values,
        color=[color_map[k] for k in cat_pct.index],
        edgecolor="black",
        alpha=0.85
    )

    for bar in bars:
        height = bar.get_height()
        ax4.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax4.set_title("4. Distribusi Probabilitas Kategori ISPU", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Kategori Kualitas Udara")
    ax4.set_ylabel("Persentase Kejadian (%)")
    ax4.set_ylim(0, max(cat_pct.values) * 1.2 + 5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300)
    print(f"\n[INFO] Grafik visualisasi berhasil disimpan ke file: {save_path}")
    plt.show()

if __name__ == "__main__":
    csv_path = "air_quality_dataset.csv"
    
    # Cek apakah file ada dan tidak kosong
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("[PERINGATAN] File air_quality_dataset.csv tidak ditemukan atau kosong.")
        df = generate_sample_dataset(csv_path)
    else:
        df = pd.read_csv(csv_path)
        
    target_col = "PM2.5" if "PM2.5" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    print(f"\n=== SIMULASI METODE MONTE CARLO: PARAMETER {target_col} ===")
    
    # 1. Buat Tabel Probabilitas & Interval Angka Acak
    mc_table = build_monte_carlo_table(df[target_col], num_bins=6)
    print("\n1. Tabel Distribusi Probabilitas & Interval Angka Acak:")
    print(mc_table[["Kelas_Bawah", "Kelas_Atas", "Nilai_Tengah", "Frekuensi", "Probabilitas", "Prob_Kumulatif", "Interval_Angka_Acak"]].to_string(index=False))
    
    # 2. Jalankan Simulasi
    sim_results = run_monte_carlo_simulation(mc_table, n_simulations=1000, threshold=100.0)
    print("\n2. 10 Sampel Pertama Hasil Simulasi:")
    print(sim_results.head(10).to_string(index=False))
    
    # 3. Ringkasan Evaluasi
    hist_mean = df[target_col].mean()
    sim_mean = sim_results["Hasil_Simulasi"].mean()
    prob_unhealthy = (sim_results["Hasil_Simulasi"] >= 100.0).mean() * 100
    
    print("\n3. Ringkasan Statistik:")
    print(f"   - Rata-rata Historis         : {hist_mean:.2f}")
    print(f"   - Rata-rata Simulasi (N=1000): {sim_mean:.2f}")
    print(f"   - Akurasi Konvergensi        : {100 - abs(sim_mean - hist_mean) / hist_mean * 100:.2f}%")
    print(f"   - Risiko Udara Tidak Sehat   : {prob_unhealthy:.2f}% (Nilai >= 100)")

    # 4. Tampilkan & Simpan Visualisasi Grafik
    visualize_monte_carlo(
        df_hist=df,
        df_table=mc_table,
        sim_df=sim_results,
        target_col=target_col,
        threshold=100.0,
        horizon=30,
        save_path="visualisasi_monte_carlo.png"
    )
