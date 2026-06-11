"""
AKADEMIKAI - MINI RAG DEMO
UAS Kecerdasan Buatan
Kode Final - Siap Jalan di VS Code
"""

import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("🎓 AKADEMIKAI - MINI RAG DEMO")
print("=" * 60)

# =========================================================
# 1. BACA FILE PDF
# =========================================================

def baca_pdf(path):
    """Baca teks dari file PDF"""
    teks = ""
    try:
        with pdfplumber.open(path) as pdf:
            for halaman in pdf.pages:
                t = halaman.extract_text()
                if t:
                    teks += t + "\n"
        return teks
    except Exception as e:
        print(f"  Error: {e}")
        return ""

# Daftar file PDF (pastikan nama file SAMA PERSIS)
file_pdf = [
    "buku-pedoman-KTI-KBK-2016-7072019-final-cetak.pdf",
    "Buku-Pedoman-TA-MBKM-2021.pdf",
    "44601acb5bf273e6ec2e26cde84a8bc2.pdf"
]

print("\n📄 MEMBACA FILE PDF...")
semua_dokumen = []

for file in file_pdf:
    if os.path.exists(file):
        teks = baca_pdf(file)
        if teks and len(teks) > 500:
            semua_dokumen.append({
                "nama": file,
                "teks": teks
            })
            print(f"  ✅ {file} ({len(teks)} karakter)")
        else:
            print(f"  ⚠️ {file} - teks terlalu pendek ({len(teks)} karakter)")
    else:
        print(f"  ❌ File tidak ditemukan: {file}")

if len(semua_dokumen) == 0:
    print("\n❌ Tidak ada file PDF yang berhasil dibaca!")
    print("   Pastikan 3 file PDF ada di folder yang sama")
    exit()

print(f"\n✅ Total {len(semua_dokumen)} dokumen berhasil dibaca")

# =========================================================
# 2. POTONG TEKS MENJADI CHUNK
# =========================================================

def potong_teks(teks, ukuran_chunk=1000, overlap=100):
    """Potong teks menjadi chunk dengan overlap"""
    kata = teks.split()
    chunks = []
    for i in range(0, len(kata), ukuran_chunk - overlap):
        chunk = " ".join(kata[i:i + ukuran_chunk])
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

print("\n✂️ MEMOTONG TEKS MENJADI CHUNK...")

semua_chunk = []
for doc in semua_dokumen:
    chunks = potong_teks(doc["teks"])
    for i, chunk in enumerate(chunks):
        semua_chunk.append({
            "teks": chunk,
            "source": doc["nama"],
            "chunk_id": i
        })
    print(f"  📄 {doc['nama'][:30]}... -> {len(chunks)} chunk")

print(f"\n✅ Total {len(semua_chunk)} chunk")

# =========================================================
# 3. BUAT EMBEDDING
# =========================================================

print("\n🔢 MEMBUAT EMBEDDING...")
print("   (Loading model, tunggu sebentar...)")

# Load model embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Buat embedding untuk semua chunk
teks_chunk = [c["teks"] for c in semua_chunk]
embeddings = model.encode(teks_chunk, show_progress_bar=True)

print(f"✅ Embedding selesai! {len(embeddings)} vektor dibuat")

# =========================================================
# 4. SIMPAN KE CHROMADB
# =========================================================

print("\n💾 MENYIMPAN KE CHROMADB...")

# Hapus folder lama jika ada
import shutil
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# Buat ChromaDB client
client = chromadb.PersistentClient(path="./chromatika_db")

# Buat collection baru
collection = client.create_collection(
    name="dokumen_akademik",
    metadata={"hnsw:space": "cosine"}
)

# Masukkan data ke ChromaDB
for i, chunk in enumerate(semua_chunk):
    collection.add(
        ids=[f"chunk_{i}"],
        documents=[chunk["teks"]],
        metadatas=[{
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }],
        embeddings=[embeddings[i].tolist()]
    )

print(f"✅ Data tersimpan di ChromaDB ({len(semua_chunk)} chunk)")

# =========================================================
# 5. FUNGSI PENCARIAN
# =========================================================

def cari_dokumen(pertanyaan, k=3, threshold=0.4):
    """Mencari chunk yang paling relevan dengan pertanyaan"""
    query_embedding = model.encode([pertanyaan])[0]
    
    hasil = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    
    if not hasil["documents"][0]:
        return None
    
    relevan = []
    for i in range(len(hasil["documents"][0])):
        distance = hasil["distances"][0][i] if hasil["distances"] else 1.0
        similarity = 1 - distance
        
        if similarity >= threshold:
            relevan.append({
                "teks": hasil["documents"][0][i],
                "source": hasil["metadatas"][0][i]["source"],
                "similarity": similarity
            })
    
    return relevan if relevan else None

# =========================================================
# 6. FUNGSI MENJAWAB
# =========================================================

def tanya_akademikai(pertanyaan):
    """Fungsi utama untuk bertanya ke AkademikAI"""
    print("\n" + "=" * 60)
    print(f"📝 PERTANYAAN: {pertanyaan}")
    print("=" * 60)
    
    hasil = cari_dokumen(pertanyaan)
    
    if not hasil:
        print("\n❌ INFORMASI TIDAK DITEMUKAN")
        print("   Jawaban tidak tersedia dalam dokumen yang diunggah.")
        return
    
    print("\n📚 SUMBER YANG DITEMUKAN:")
    for h in hasil:
        print(f"   - {h['source']} (kemiripan: {h['similarity']:.2%})")
    
    chunk_terbaik = hasil[0]["teks"]
    sumber_terbaik = hasil[0]["source"]
    
    print(f"\n📄 KONTEKS DARI {sumber_terbaik}:")
    print("-" * 50)
    print(chunk_terbaik[:500] + "..." if len(chunk_terbaik) > 500 else chunk_terbaik)
    print("-" * 50)
    
    print(f"\n🤖 JAWABAN RINGKAS:")
    print(f"   Berdasarkan {sumber_terbaik}, ditemukan:")
    print(f"   {chunk_terbaik[:300]}...")
    
    print("\n" + "-" * 60)
    print("⚠️ DISCLAIMER: Tulis ulang dengan kata-kata sendiri sebelum")
    print("   dikumpulkan ke dosen. Verifikasi ulang ke dokumen asli.")
    print("-" * 60)

# =========================================================
# 7. DEMO DENGAN 5 PERTANYAAN
# =========================================================

print("\n" + "=" * 60)
print("🎓 MEMULAI DEMO AKADEMIKAI")
print("=" * 60)

daftar_pertanyaan = [
    "Bagaimana struktur laporan tugas akhir yang benar?",
    "Apa saja kriteria penilaian dalam tugas akhir?",
    "Bagaimana cara menulis kutipan yang benar?",
    "Apa yang dimaksud dengan metodologi penelitian?",
    "Bagaimana format penulisan tabel dan gambar?"
]

for q in daftar_pertanyaan:
    tanya_akademikai(q)
    print("\n" + "🟦" * 30)

print("\n" + "=" * 60)
print("✅ DEMO SELESAI!")
print("=" * 60)

print("\n📊 RINGKASAN:")
print(f"   - Dokumen yang diproses: {len(semua_dokumen)} file")
print(f"   - Total chunk: {len(semua_chunk)}")
print(f"   - Pertanyaan yang diuji: {len(daftar_pertanyaan)}")
