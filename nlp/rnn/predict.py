import numpy as np
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence

# Model parametreleri
max_features = 10_000
max_len = 200

# Kelime indeksini yükle
word_index = imdb.get_word_index()

# Reverse word index (sayıdan kelimeye çevirmek için)
reverse_word_index = {value: key for key, value in word_index.items()}

def encode_text(text):
    """
    Metni IMDB formatına uygun şekilde encode eder
    """
    # Metni küçük harfe çevir ve kelimelere ayır
    words = text.lower().split()
    
    # Her kelimeyi indeksine çevir
    # IMDB veri setinde indeksler 3'ten başlar (0,1,2 rezerve)
    # max_features'dan büyük indeksleri ve bilinmeyen kelimeleri filtrele
    encoded = []
    for word in words:
        if word in word_index:
            idx = word_index[word]
            if idx < max_features:
                encoded.append(idx + 3)  # IMDB offset
        # Bilinmeyen kelimeler için 2 kullanılır
        else:
            encoded.append(2)
    
    # Padding uygula
    encoded = sequence.pad_sequences([encoded], maxlen=max_len, padding='post')
    return encoded

def predict_sentiment(text, model, model_name):
    """
    Verilen metni model ile tahmin eder
    """
    # Metni encode et
    encoded_text = encode_text(text)
    
    # Tahmin yap
    prediction = model.predict(encoded_text, verbose=0)[0][0]
    
    # Sonucu yorumla
    sentiment = "POZİTİF" if prediction > 0.5 else "NEGATİF"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Tahmin: {sentiment}")
    print(f"Güven Skoru: {prediction:.4f} ({confidence*100:.2f}% güven)")
    print(f"{'='*60}")
    
    return prediction, sentiment

def main():
    print("Modeller yükleniyor...")
    
    # Modelleri yükle
    try:
        simple_rnn = load_model("simple_rnn_model.keras")
        print("✓ SimpleRNN modeli yüklendi")
    except Exception as e:
        print(f"✗ SimpleRNN modeli yüklenemedi: {e}")
        simple_rnn = None
    
    try:
        lstm = load_model("lstm_model.keras")
        print("✓ LSTM modeli yüklendi")
    except Exception as e:
        print(f"✗ LSTM modeli yüklenemedi: {e}")
        lstm = None
    
    try:
        gru = load_model("gru_model.keras")
        print("✓ GRU modeli yüklendi")
    except Exception as e:
        print(f"✗ GRU modeli yüklenemedi: {e}")
        gru = None
    
    # Test cümlesi
    print("\n" + "="*60)
    print("SENTİMENT ANALİZİ - ÜÇ MODEL KARŞILAŞTIRMASI")
    print("="*60)
    
    # Kullanıcıdan cümle al veya örnek cümle kullan
    test_sentence = input("\nTest etmek istediğiniz cümleyi girin (veya Enter'a basarak örnek cümleyi kullanın): ").strip()
    
    if not test_sentence:
        test_sentence = "This movie was absolutely wonderful! I loved every minute of it."
        print(f"\nÖrnek cümle kullanılıyor: '{test_sentence}'")
    
    print(f"\nTest Cümlesi: {test_sentence}")
    
    results = {}
    
    # Her model ile tahmin yap
    if simple_rnn:
        pred, sent = predict_sentiment(test_sentence, simple_rnn, "SimpleRNN")
        results['SimpleRNN'] = (pred, sent)
    
    if lstm:
        pred, sent = predict_sentiment(test_sentence, lstm, "LSTM")
        results['LSTM'] = (pred, sent)
    
    if gru:
        pred, sent = predict_sentiment(test_sentence, gru, "GRU")
        results['GRU'] = (pred, sent)
    
    # Özet
    if results:
        print("\n" + "="*60)
        print("ÖZET")
        print("="*60)
        for model_name, (pred, sent) in results.items():
            print(f"{model_name:12s}: {sent:8s} (skor: {pred:.4f})")
        print("="*60)
    
    # Başka cümle test etmek ister mi?
    while True:
        print("\n")
        choice = input("Başka bir cümle test etmek ister misiniz? (e/h): ").strip().lower()
        if choice == 'e':
            test_sentence = input("\nTest etmek istediğiniz cümleyi girin: ").strip()
            if test_sentence:
                print(f"\nTest Cümlesi: {test_sentence}")
                
                results = {}
                if simple_rnn:
                    pred, sent = predict_sentiment(test_sentence, simple_rnn, "SimpleRNN")
                    results['SimpleRNN'] = (pred, sent)
                
                if lstm:
                    pred, sent = predict_sentiment(test_sentence, lstm, "LSTM")
                    results['LSTM'] = (pred, sent)
                
                if gru:
                    pred, sent = predict_sentiment(test_sentence, gru, "GRU")
                    results['GRU'] = (pred, sent)
                
                # Özet
                if results:
                    print("\n" + "="*60)
                    print("ÖZET")
                    print("="*60)
                    for model_name, (pred, sent) in results.items():
                        print(f"{model_name:12s}: {sent:8s} (skor: {pred:.4f})")
                    print("="*60)
        else:
            print("\nProgram sonlandırılıyor...")
            break

if __name__ == "__main__":
    main()

