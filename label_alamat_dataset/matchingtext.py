import difflib

kecamatan_list = [
    "Binakal", "Bondowoso", "Botolinggo", "Cermee", "Curahdami",
    "Grujugan", "Jambesari Darus Sholah", "Klabang", "Maesan",
    "Pakem", "Prajekan", "Pujer", "Sempol", "Sukosari",
    "Sumber Wringin", "Taman Krocok", "Tamanan", "Tapen",
    "Tegalampel", "Tenggarang", "Tlogosari", "Wonosari", "Wringin"
]

input_user = input("Masukkan nama kecamatan: ")

# cari yang mirip (suggestions)
hasil = difflib.get_close_matches(input_user, kecamatan_list, n=3, cutoff=0.5)

if hasil:
    print("Mungkin maksud Anda:")
    for h in hasil:
        print("-", h)
else:
    print("Tidak ada yang mirip ditemukan")
