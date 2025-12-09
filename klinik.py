import streamlit as st
import json
import pandas as pd
from typing import Dict, List
import numpy as np
import os

# Base dir of the app (useful when deployed where CWD may differ)
BASE_DIR = os.path.dirname(__file__)


@st.cache_data
def _read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class Kasus:
    def __init__(self, id_kasus: str, nama_hewan: str, ras: str, umur: float, berat: float,
                 riwayat_medis: str, gejala: List[str], solusi: str):
        self.id_kasus = id_kasus
        self.nama_hewan = nama_hewan
        self.ras = ras
        self.umur = umur
        self.berat = berat
        self.riwayat_medis = riwayat_medis
        self.gejala = gejala
        self.solusi = solusi

    def ke_dict(self):
        return {
            "id_kasus": self.id_kasus,
            "nama_hewan": self.nama_hewan,
            "ras": self.ras,
            "umur": self.umur,
            "berat": self.berat,
            "riwayat_medis": self.riwayat_medis,
            "gejala": self.gejala,
            "solusi": self.solusi
        }

    @staticmethod
    def dari_dict(data: Dict):
        return Kasus(
            id_kasus=data["id_kasus"],
            nama_hewan=data["nama_hewan"],
            ras=data["ras"],
            umur=data["umur"],
            berat=data["berat"],
            riwayat_medis=data["riwayat_medis"],
            gejala=data["gejala"],
            solusi=data["solusi"]
        )

class BasisKasus:
    def __init__(self, nama_file="basis_kasus_hewan.json"):
        # store JSON file path relative to app directory for deploy compatibility
        self.nama_file = os.path.join(BASE_DIR, nama_file)
        self.kasus_list: List[Kasus] = []
        self.muatan_kasus()

    def muatan_dari_csv(self, jalur_csv: str):
        # resolve path relative to app directory
        csv_path = os.path.join(BASE_DIR, jalur_csv)
        if not os.path.exists(csv_path):
            st.error(f"File CSV tidak ditemukan: {csv_path}")
            return

        try:
            df = _read_csv_cached(csv_path)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            return
        self.kasus_list = []

        peta_diagnosa = {
            'rabies': ['Hydrophobia', 'Shyness or aggression', 'drooping ears'],
            'kennel cough': ['Coughing', 'Sneezing', 'Lethargy'],
            'parvovirus': ['Vomiting', 'Diarrhea', 'Lethargy', 'Weight loss'],
            'feline leukemia': ['Weight loss', 'Anorexia', 'Fever'],
            'skin infection': ['Itchiness', 'Hair loss', 'Redness'],
            'parasites': ['Weight loss', 'Diarrhea', 'Scratching']
        }

        # Normalisasi peta diagnosa ke lowercase agar cocok dengan gejala yang
        # sudah dinormalisasi (lowercase) saat dimuat dari CSV / input user.
        peta_diagnosa = {k: [s.lower() for s in v] for k, v in peta_diagnosa.items()}

        for idx, row in df.iterrows():
            gejala = []
            for kol in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5']:
                if kol in row and pd.notna(row[kol]) and str(row[kol]).strip():
                    gejala.append(str(row[kol]).strip().lower())

            diagnosa = "Diagnosa tidak diketahui"
            for penyakit, lista_gejala in peta_diagnosa.items():
                if any(sym in gejala for sym in lista_gejala):
                    diagnosa = f"{penyakit.capitalize()} - Antibiotik + Perawatan pendukung"
                    break

            kasus = Kasus(
                id_kasus=f"K{idx+1}",
                nama_hewan=str(row['AnimalName']),
                ras=str(row['Breed']),
                umur=float(row['Age']),
                berat=float(row['Weight_kg']),
                riwayat_medis=str(row['MedicalHistory']),
                gejala=gejala,
                solusi=diagnosa
            )
            self.kasus_list.append(kasus)

        self.simpan_kasus()
        st.success(f"✅ Memuat {len(self.kasus_list)} kasus dari CSV")

    def tambah_kasus(self, kasus: Kasus):
        self.kasus_list.append(kasus)
        self.simpan_kasus()

    def dapatkan_semua_kasus(self):
        return self.kasus_list

    def muatan_kasus(self):
        try:
            if os.path.exists(self.nama_file):
                with open(self.nama_file, "r") as file:
                    try:
                        data = json.load(file)
                        self.kasus_list = [Kasus.dari_dict(item) for item in data]
                    except json.JSONDecodeError:
                        # file may be empty or invalid on deploy; start with empty list
                        self.kasus_list = []
        except Exception as e:
            st.warning(f"File JSON tidak valid: {e}")
            self.kasus_list = []

        # Normalisasi gejala ke lowercase untuk konsistensi matching
        for kasus in self.kasus_list:
            kasus.gejala = [g.lower() for g in kasus.gejala]

    def simpan_kasus(self):
        with open(self.nama_file, "w") as file:
            json.dump([kasus.ke_dict() for kasus in self.kasus_list], file, indent=4)

class Similaritas:
    @staticmethod
    def jaccard_similarity(gejala1: List[str], gejala2: List[str]) -> float:
        set1 = set(gejala1)
        set2 = set(gejala2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    @staticmethod
    def attribute_similarity(kasus1: Kasus, kasus2: Kasus) -> float:
        umur_sim = 1 - abs(kasus1.umur - kasus2.umur) / max(kasus1.umur, kasus2.umur, 1)
        berat_sim = 1 - abs(kasus1.berat - kasus2.berat) / max(kasus1.berat, kasus2.berat, 1)
        animal_sim = 1.0 if kasus1.nama_hewan.lower() == kasus2.nama_hewan.lower() else 0.5
        riwayat_sim = 1.0 if kasus1.riwayat_medis.lower() == kasus2.riwayat_medis.lower() else 0.3
        return (0.4 * Similaritas.jaccard_similarity(kasus1.gejala, kasus2.gejala) +
                0.2 * umur_sim + 0.2 * berat_sim +
                0.1 * animal_sim + 0.1 * riwayat_sim)

class CBR:
    def __init__(self, basis_kasus: BasisKasus):
        self.basis_kasus = basis_kasus

    def ambil_kasus_terdekat(self, kasus_baru: Kasus, top_k: int = 3) -> List[Kasus]:
        if not self.basis_kasus.dapatkan_semua_kasus():
            return []
        
        semua_kasus = self.basis_kasus.dapatkan_semua_kasus()
        similarity_scores = [(kasus, Similaritas.attribute_similarity(kasus_baru, kasus)) 
                           for kasus in semua_kasus]
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        return [kasus for kasus, skor in similarity_scores[:top_k]]

    def gunakan_solusi(self, kasus_terambil: List[Kasus]) -> str:
        if not kasus_terambil:
            return "Tidak ditemukan kasus serupa"
        return kasus_terambil[0].solusi

    def simpan_kasus_baru(self, kasus_baru: Kasus, solusi: str):
        kasus_baru.solusi = solusi
        kasus_baru.id_kasus = f"K{len(self.basis_kasus.dapatkan_semua_kasus()) + 1}"
        self.basis_kasus.tambah_kasus(kasus_baru)

# Streamlit App
def main():
    st.markdown('<h1 class="main-header">Sistem CBR Pada Klinik Hewan Anjing Dan Kucing</h1>', unsafe_allow_html=True)
    
    # Inisialisasi session state
    if 'basis_kasus' not in st.session_state:
        st.session_state.basis_kasus = BasisKasus()
        st.session_state.cbr = CBR(st.session_state.basis_kasus)
        st.session_state.csv_loaded = False
    
    tab1, tab2 = st.tabs(["📋 Diagnosa Baru", "📊 Lihat Kasus"])
    
    with tab1:
        st.header("Input Kasus Baru")
        
        col1, col2 = st.columns(2)
        with col1:
            nama_hewan = st.text_input("Jenis Hewan", placeholder="Anjing/Kucing")
            ras = st.text_input("Ras")
            umur = st.number_input("Umur (tahun)", min_value=0.0, value=2.0)
        with col2:
            berat = st.number_input("Berat (kg)", min_value=0.0, value=5.0)
            riwayat_medis = st.text_area("Riwayat Medis", height=80, placeholder="Sehat/Normal")
        
        gejala_input = st.text_area("Gejala (pisahkan dengan koma)", 
                      placeholder="Vomiting, Diarrhea, Lethargy", height=100)
        
        if st.button("Lakukan Diagnosa", type="primary"):
            if nama_hewan and gejala_input:
                gejala_list = [g.strip().lower() for g in gejala_input.split(",") if g.strip()]
                
                kasus_baru = Kasus("BARU", nama_hewan, ras or "Tidak diketahui", 
                                 umur, berat, riwayat_medis or "Tidak ada", gejala_list, "")
                
                with st.spinner("Mencari kasus serupa..."):
                    kasus_terdekat = st.session_state.cbr.ambil_kasus_terdekat(kasus_baru)
                    
                    if not st.session_state.basis_kasus.dapatkan_semua_kasus():
                        st.warning("Basis kasus kosong — muat data dari CSV di tab '📊 Lihat Kasus' terlebih dahulu.")
                        kasus_terdekat = []
                    if kasus_terdekat:
                        solusi = st.session_state.cbr.gunakan_solusi(kasus_terdekat)
                        
                        st.success("**DIAGNOSA SELESAI**")
                        col_a, col_b, col_c = st.columns([1, 2, 1])
                        
                        with col_b:
                            st.metric("Rekomendasi Solusi", solusi)
                        
                        # Tampilkan kasus terdekat
                        st.subheader("3 Kasus Paling Serupa")
                        for i, kasus in enumerate(kasus_terdekat[:3], 1):
                            skor = Similaritas.attribute_similarity(kasus_baru, kasus)
                            with st.expander(f"{i}. {kasus.nama_hewan} {kasus.ras} ({skor:.1%})"):
                                st.write(f"**ID:** {kasus.id_kasus}")
                                st.write(f"**Umur:** {kasus.umur} tahun, **Berat:** {kasus.berat} kg")
                                st.write(f"**Gejala:** {', '.join(kasus.gejala)}")
                                st.write(f"**Solusi:** {kasus.solusi}")
                        
                        # Simpan kasus
                        if st.button("Simpan Kasus Baru ke Database"):
                            st.session_state.cbr.simpan_kasus_baru(kasus_baru, solusi)
                            st.success("Kasus baru berhasil disimpan!")
                    else:
                        st.warning("❌ Tidak ada kasus serupa ditemukan")
            else:
                st.error("⚠️ Lengkapi data hewan dan gejala")
    
    with tab2:
        st.header("Statistik Kasus")
        if st.button("Muat Kasus dari CSV (veterinary_clinical_data.csv)"):
            st.session_state.basis_kasus.muatan_dari_csv("veterinary_clinical_data.csv")
            st.session_state.csv_loaded = True
        
        total_kasus = len(st.session_state.basis_kasus.dapatkan_semua_kasus())
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Kasus", total_kasus)
        
        if total_kasus > 0:
            kasus_sample = st.session_state.basis_kasus.dapatkan_semua_kasus()[:10]
            df_display = pd.DataFrame([
                {
                    "ID": kasus.id_kasus,
                    "Hewan": f"{kasus.nama_hewan} {kasus.ras}",
                    "Gejala": ", ".join(kasus.gejala[:3]),
                    "Solusi": kasus.solusi
                }
                for kasus in kasus_sample
            ])
            st.dataframe(df_display, use_container_width=True)

if __name__ == "__main__":
    main()