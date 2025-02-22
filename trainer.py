import os
import pandas as pd
import threading
import asyncio
from tkinter import Tk, filedialog, Label, Button, StringVar
from textblob import TextBlob  # Basit AI tabanlı metin analizi
from langdetect import detect  # Dil tespiti için langdetect kütüphanesi
import subprocess
import sys

def install_requirements():
    required_packages = ['pandas', 'tkinter', 'textblob', 'langdetect']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def prepare_data(data_dir):
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                data.append(file.read())
    return data

def detect_task(data):
    # Görev tespiti için basit bir uygulama
    if all(isinstance(d, str) for d in data):
        return "text_classification"
    return "unknown"

def clean_data(data):
    cleaned_data = []
    for text in data:
        # Temel temizlik: ekstra boşlukları, yeni satırları vb. kaldırma
        cleaned_text = ' '.join(text.split())
        cleaned_data.append(cleaned_text)
    return cleaned_data

def filter_nonsensical_data(data):
    filtered_data = []
    for text in data:
        # langdetect kullanarak dil tespiti ve TextBlob kullanarak anlamsız verileri filtreleme
        if detect(text) == 'tr':
            blob = TextBlob(text)
            if blob.sentiment.polarity != 0:
                filtered_data.append(text)
    return filtered_data

def ensure_consistent_labeling(data):
    # Tutarlı etiketleme için basit bir uygulama
    labeled_data = [{"text": text, "label": "label"} for text in data]
    return labeled_data

def format_data(data):
    # DataFrame'e dönüştürme
    df = pd.DataFrame(data)
    return df

def save_to_csv(data, output_file):
    data.to_csv(output_file, index=False, encoding='utf-8')

def select_directory():
    root = Tk()
    root.withdraw()  # Ana pencereyi gizle
    data_dir = filedialog.askdirectory(title="Metin Dosyalarını İçeren Dizini Seçin")
    return data_dir

def split_text_into_chunks(text, max_tokens=128):
    # Metni 128 tokenlık parçalara bölme
    tokens = text.split()
    chunks = [' '.join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks

async def main():
    global status
    # Metin dosyalarını içeren dizini seç
    data_dir = select_directory()
    
    # Adım 1: Veriyi hazırla
    data = await asyncio.to_thread(prepare_data, data_dir)
    status.set("Veri hazırlandı")
    
    # Adım 2: Görev türünü tespit et
    task_type = detect_task(data)
    status.set(f"Tespit edilen görev türü: {task_type}")
    
    # Adım 3: Veriyi temizle
    cleaned_data = await asyncio.to_thread(clean_data, data)
    status.set("Veri temizlendi")
    
    # Adım 4: Anlamsız verileri filtrele
    filtered_data = await asyncio.to_thread(filter_nonsensical_data, cleaned_data)
    status.set("Anlamsız veriler filtrelendi")
    
    # Adım 5: Metinleri 128 tokenlık parçalara böl
    chunked_data = []
    for text in filtered_data:
        chunks = split_text_into_chunks(text)
        chunked_data.extend(chunks)
    
    # Adım 6: Tutarlı etiketlemeyi sağla
    labeled_data = await asyncio.to_thread(ensure_consistent_labeling, chunked_data)
    status.set("Tutarlı etiketleme sağlandı")
    
    # Adım 7: Model eğitimi için veriyi formatla
    formatted_data = await asyncio.to_thread(format_data, labeled_data)
    status.set("Veri formatlandı")
    
    # Adım 8: CSV'ye kaydet
    output_file = 'output/dataset.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    await asyncio.to_thread(save_to_csv, formatted_data, output_file)
    status.set(f"Veri {output_file} dosyasına kaydedildi")

def start_process():
    asyncio.run(main())

if __name__ == "__main__":
    install_requirements()

    # GUI oluşturma
    root = Tk()
    root.title("Veri Hazırlama Aracı")

    status = StringVar()
    status.set("Başlamak için 'Başlat' düğmesine tıklayın")

    label = Label(root, textvariable=status)
    label.pack(pady=20)

    start_button = Button(root, text="Başlat", command=start_process)
    start_button.pack(pady=20)

    root.mainloop()