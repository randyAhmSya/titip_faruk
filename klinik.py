import streamlit as st
import json
import pandas as pd
from typing import List, Dict
import os


# ============================
# PATH FILE
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "basis_kasus_hewan.json")
CSV_PATH = os.path.join(BASE_DIR, "veterinary_clinical_data.csv")


# ============================
# MODEL KASUS
# ============================
class Kasus:
    def __init__(self, id_kasus, nama_hewan, ras, umur, berat, riwayat, gejala, solusi):
        self.id_kasus = id_kasus
        self.nama_hewan = nama_hewan
        self.ras = ras
        self.umur = umur
        self.berat = berat
        self.riwayat = riwayat
        self.gejala = gejala
        self.solusi = solusi

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(data: Dict):
        return Kasus(**data)


# ============================
# BASIS KASUS (JSON + CSV)
# ============================
class BasisKasus:
    def __init__(self):
        self.kasus_list: List[Kasus] = []
        self.load_json()

    def load_json(self):
        """Load data JSON jika file tersedia, jika kosong → tidak error."""
        if not os.path.exists(JSON_PATH):
            self.kasus_list = []
            return

        try:
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
                self.kasus_list = [Kasus.from_dict(item) for item in data]
        except:
            self.kasus_list = []  # JSON rusak → tetap aman

    def save_json(self):
        """Simpan JSON dengan aman (ignore error karena server read-only)."""
        try:
            with open(JSON_PATH, "w") as f:
                json.dump([k.to_dict() for k in self.kasus_list], f, indent=4)
        except:
            pass  # Jangan crash saat deploy

    def load_from_csv(self):
        """Menambahkan kasus dari CSV ke basis kasus."""
        if not os.path.exists(CSV_PATH):
            st.error("CSV tidak ditemukan!")
            return

        df = pd.read_csv(CSV_PATH)

        for idx, row in df.iterrows():
            gejala = [
                str(row[c]).strip().lower()
                for c in df.columns
                if "Symptom" in c and pd.notna(row[c])
            ]

            kasus = Kasus(
                id_kasus=f"CSV{idx+1}",
                nama_hewan=row["AnimalName"],
                ras=row["Breed"],
                umur=float(row["Age"]),
                berat=float(row["Weight_kg"]),
                riwayat=row["MedicalHistory"],
                gejala=gejala,
                solusi="Solusi dari data CSV"
            )

            self.kasus_list.append(kasus)

        self.save_json()
        st.success("CSV berhasil dimasukkan ke basis kasus!")

    def add_case(self, kasus: Kasus):
        self.kasus_list.append(kasus)
        self.save_json()

    def all_cases(self):
        return self.kasus_list


# ============================
# SIMILARITAS
# ============================
class Similaritas:
    @staticmethod
    def jaccard(g1, g2):
        s1, s2 = set(g1), set(g2)
        if not (s1 or s2):
            return 0
        return len(s1 & s2) / len(s1 | s2)

    @staticmethod
    def attribute(k1, k2):
        umur_sim = 1 - abs(k1.umur - k2.umur) / max(k1.umur, k2.umur, 1)
        berat_sim = 1 - abs(k1.berat - k2.berat) / max(k1.berat, k2.berat, 1)
        gejala_sim = Similaritas.jaccard(k1.gejala, k2.gejala)

        return 0.5 * gejala_sim + 0.25 * umur_sim + 0.25 * berat_sim


# ============================
# CBR ENGINE
# ============================
class CBR:
    def __init__(self, basis: BasisKasus):
        self.basis = basis

    def cari_terdekat(self, kasus_baru, k=3):
        if not self.basis.all_cases():
            return []

        skor = [
            (kasus, Similaritas.attribute(kasus_baru, kasus))
            for kasus in self.basis.all_cases()
        ]

        skor.sort(key=lambda x: x[1], reverse=True)
        return skor[:k]


# ============================
# STREAMLIT UI
# ============================
def main():
    st.title("Sistem CBR Klinik Hewan (JSON + CSV)")

    # Load database JSON
    if "basis" not in st.session_state:
        st.session_state.basis = BasisKasus()
        st.session_state.cbr = CBR(st.session_state.basis)

    tab1, tab2 = st.tabs(["Diagnosa Baru", "Basis Kasus"])

    # ---------------------
    # TAB 1 – Diagnosa Baru
    # ---------------------
    with tab1:
        nama_hewan = st.text_input("Jenis Hewan")
        ras = st.text_input("Ras")
        umur = st.number_input("Umur (tahun)", 0.0, 30.0, 2.0)
        berat = st.number_input("Berat (kg)", 0.0, 120.0, 5.0)
        riwayat = st.text_area("Riwayat Medis", "Normal")
        gejala_input = st.text_area("Gejala (pisahkan koma)")

        if st.button("Diagnosa"):
            gejala = [g.strip().lower() for g in gejala_input.split(",") if g.strip()]

            kasus_baru = Kasus(
                id_kasus="BARU",
                nama_hewan=nama_hewan,
                ras=ras,
                umur=umur,
                berat=berat,
                riwayat=riwayat,
                gejala=gejala,
                solusi=""
            )

            hasil = st.session_state.cbr.cari_terdekat(kasus_baru)

            if not hasil:
                st.warning("Basis kasus kosong! Muat CSV atau isi JSON terlebih dahulu.")
            else:
                st.success("Kasus paling mirip:")
                for i, (k, skor) in enumerate(hasil, 1):
                    st.write(f"**{i}. {k.nama_hewan} - {k.ras}**")
                    st.write(f"Kemiripan: {skor*100:.2f}%")
                    st.write(f"Gejala: {', '.join(k.gejala)}")
                    st.write("---")

    # ---------------------
    # TAB 2 – Basis Kasus
    # ---------------------
    with tab2:
        if st.button("Load CSV ke Basis Kasus"):
            st.session_state.basis.load_from_csv()

        st.subheader("Data Basis Kasus (JSON + CSV)")
        data = [
            {
                "ID": k.id_kasus,
                "Hewan": k.nama_hewan,
                "Ras": k.ras,
                "Gejala": ", ".join(k.gejala)
            }
            for k in st.session_state.basis.all_cases()
        ]
        st.dataframe(pd.DataFrame(data))


if __name__ == "__main__":
    main()
