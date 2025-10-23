import numpy as np

# ============================================================================
# SEQ2SEQ + ATTENTION MEKANIZMASI SİMÜLASYONU
# ============================================================================
# Bu dosya, gerçek bir eğitim yapmadan seq2seq + attention mekanizmasını
# adım adım göstermek için tasarlanmıştır.
# ============================================================================

def softmax(x):
    """
    Softmax fonksiyonu: girdi vektörünü olasılık dağılımına çevirir
    exp(x_i) / sum(exp(x))
    """
    # Numerik stabilite için max değeri çıkarıyoruz
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def create_word_embeddings(words, embedding_dim=4):
    """
    Her kelime için rastgele embedding vektörü oluşturur
    Gerçek uygulamada bu vektörler öğrenilir, burada rastgele atıyoruz
    """
    embeddings = {}
    for word in words:
        # Her kelime için embedding_dim boyutunda rastgele vektör
        embeddings[word] = np.random.randn(embedding_dim)
    return embeddings

def encoder_step(word_embedding, prev_hidden):
    """
    Encoder'ın bir adımını simüle eder
    Gerçek RNN/LSTM'de karmaşık hesaplamalar var, burada basitleştirilmiş
    """
    # Basit linear transformation simülasyonu
    # Gerçekte: h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
    hidden_size = prev_hidden.shape[0]
    
    # Rastgele ağırlık matrisleri (normalde eğitilir)
    W_h = np.random.randn(hidden_size, hidden_size) * 0.1
    W_x = np.random.randn(hidden_size, word_embedding.shape[0]) * 0.1
    
    # Yeni hidden state hesaplama
    new_hidden = np.tanh(W_h @ prev_hidden + W_x @ word_embedding)
    
    return new_hidden

def calculate_attention(decoder_hidden, encoder_hiddens, encoder_words):
    """
    Attention mekanizmasını hesaplar
    1. Her encoder hidden state ile decoder hidden state arasında dot product
    2. Softmax ile attention ağırlıklarını hesapla
    3. Context vector oluştur
    """
    print("\n    --- Attention Hesaplama ---")
    
    # 1. Attention skorlarını hesapla (dot product)
    attention_scores = []
    for i, enc_hidden in enumerate(encoder_hiddens):
        # Dot product: decoder_hidden · encoder_hidden
        score = np.dot(decoder_hidden, enc_hidden)
        attention_scores.append(score)
        print(f"    '{encoder_words[i]}' kelimesi için attention skoru: {score:.4f}")
    
    attention_scores = np.array(attention_scores)
    
    # 2. Softmax ile normalize et (olasılık dağılımı)
    attention_weights = softmax(attention_scores)
    
    print("\n    --- Attention Ağırlıkları (Softmax sonrası) ---")
    for i, word in enumerate(encoder_words):
        print(f"    '{word}': {attention_weights[i]:.4f} ({attention_weights[i]*100:.2f}%)")
    
    # 3. Context vector oluştur (weighted sum)
    # Context = sum(attention_weight_i * encoder_hidden_i)
    context_vector = np.zeros_like(encoder_hiddens[0])
    for i, enc_hidden in enumerate(encoder_hiddens):
        context_vector += attention_weights[i] * enc_hidden
    
    print(f"\n    Context vector oluşturuldu: shape={context_vector.shape}")
    
    return context_vector, attention_weights

def decoder_step(prev_word_embedding, prev_hidden, context_vector):
    """
    Decoder'ın bir adımını simüle eder
    Context vector ile decoder hidden state birleştirilerek yeni output üretilir
    """
    hidden_size = prev_hidden.shape[0]
    
    # Rastgele ağırlık matrisleri (normalde eğitilir)
    W_h = np.random.randn(hidden_size, hidden_size) * 0.1
    W_x = np.random.randn(hidden_size, prev_word_embedding.shape[0]) * 0.1
    W_c = np.random.randn(hidden_size, context_vector.shape[0]) * 0.1
    
    # Yeni hidden state: context vector da dahil edilir
    # h_t = tanh(W_h * h_{t-1} + W_x * x_t + W_c * context + b)
    new_hidden = np.tanh(
        W_h @ prev_hidden + 
        W_x @ prev_word_embedding + 
        W_c @ context_vector
    )
    
    return new_hidden

def select_output_word(decoder_hidden, vocab, vocab_embeddings):
    """
    Decoder hidden state'e göre en uygun kelimeyi seçer
    Gerçekte: softmax(W * hidden + b) ile olasılık dağılımı hesaplanır
    Burada basitçe: hidden state ile vocab embeddings arasında similarity
    """
    # Decoder hidden state boyutunu embedding boyutuna project et
    # Gerçek uygulamada bu bir öğrenilen linear layer olurdu
    hidden_size = decoder_hidden.shape[0]
    
    # İlk kelime embedding'inin boyutunu al
    embedding_dim = vocab_embeddings[vocab[0]].shape[0]
    
    # Projection matrisi (hidden_size -> embedding_dim)
    W_projection = np.random.randn(embedding_dim, hidden_size) * 0.1
    
    # Decoder hidden'ı embedding boyutuna project et
    projected_hidden = W_projection @ decoder_hidden
    
    scores = {}
    for word in vocab:
        # Her kelime embedding'i ile projected hidden state arasında benzerlik (dot product)
        score = np.dot(projected_hidden, vocab_embeddings[word])
        scores[word] = score
    
    # En yüksek skora sahip kelimeyi seç
    selected_word = max(scores, key=scores.get)
    return selected_word, scores

# ============================================================================
# ANA SİMÜLASYON
# ============================================================================

def run_simulation():
    """
    Seq2seq + Attention mekanizmasının tam simülasyonu
    """
    print("=" * 80)
    print("SEQ2SEQ + ATTENTION MEKANIZMASI SİMÜLASYONU")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. GİRDİ CÜMLESI VE HEDEF CÜMLE
    # -------------------------------------------------------------------------
    input_sentence = "Ben okula gidiyorum"
    input_words = input_sentence.split()
    
    # Hedef cümle (decoder'ın üretmesi gereken)
    target_sentence = "I am going to school"
    target_words = target_sentence.split()
    
    print(f"\n📝 Girdi Cümlesi (Türkçe): {input_sentence}")
    print(f"🎯 Hedef Cümle (İngilizce): {target_sentence}")
    print(f"\nEncoder'a giren kelimeler: {input_words}")
    
    # -------------------------------------------------------------------------
    # 2. KELİME EMBEDDİNGLERİ OLUŞTUR
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("ADIM 1: KELİME EMBEDDİNGLERİ OLUŞTURMA")
    print("-" * 80)
    
    embedding_dim = 4  # Her kelimenin vektör boyutu
    hidden_size = 6    # Hidden state boyutu
    
    # Tüm kelimelerin listesi (girdi + hedef + özel tokenlar)
    all_words = input_words + target_words + ["<START>", "<END>"]
    
    # Her kelime için embedding vektörü
    word_embeddings = create_word_embeddings(all_words, embedding_dim)
    
    print(f"\nToplam {len(all_words)} kelime için {embedding_dim} boyutlu embedding oluşturuldu")
    for word in input_words:
        print(f"  '{word}': {word_embeddings[word]}")
    
    # -------------------------------------------------------------------------
    # 3. ENCODER İŞLEMİ
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("ADIM 2: ENCODER İŞLEMİ")
    print("-" * 80)
    print("Encoder, girdi cümlesindeki her kelimeyi sırayla işler")
    print("ve her adımda bir hidden state üretir.\n")
    
    # Encoder'ın başlangıç hidden state'i (sıfırlarla başla)
    encoder_hidden = np.zeros(hidden_size)
    encoder_hiddens = []  # Her adımdaki hidden state'leri sakla
    
    for i, word in enumerate(input_words):
        print(f"Encoder Adım {i+1}: '{word}' kelimesi işleniyor...")
        
        # Kelimenin embedding'ini al
        word_embedding = word_embeddings[word]
        print(f"  Embedding: {word_embedding}")
        
        # Encoder step
        encoder_hidden = encoder_step(word_embedding, encoder_hidden)
        encoder_hiddens.append(encoder_hidden)
        
        print(f"  Yeni hidden state: {encoder_hidden}")
        print()
    
    print(f"✓ Encoder tamamlandı. {len(encoder_hiddens)} adet hidden state oluşturuldu.")
    
    # -------------------------------------------------------------------------
    # 4. DECODER İŞLEMİ (ATTENTION İLE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADIM 3: DECODER İŞLEMİ (ATTENTION MEKANIZMASI İLE)")
    print("=" * 80)
    print("Decoder, her adımda:")
    print("  1. Attention mekanizması ile encoder çıktılarına odaklanır")
    print("  2. Context vector oluşturur")
    print("  3. Yeni bir kelime üretir")
    print("=" * 80)
    
    # Decoder'ın başlangıç hidden state'i (encoder'ın son hidden state'i)
    decoder_hidden = encoder_hiddens[-1]
    
    # Başlangıç tokeni
    prev_word = "<START>"
    
    # Çıktı vocabulary (decoder'ın üretebileceği kelimeler)
    output_vocab = target_words + ["<END>"]
    
    # Üretilen kelimeler
    generated_words = []
    
    # Maksimum decoder adımı (sonsuz döngüyü önlemek için)
    max_decoder_steps = 10
    
    for step in range(max_decoder_steps):
        print(f"\n{'='*80}")
        print(f"DECODER ADIM {step+1}")
        print(f"{'='*80}")
        print(f"Önceki kelime: '{prev_word}'")
        
        # Önceki kelimenin embedding'i
        prev_word_embedding = word_embeddings[prev_word]
        
        # -----------------------------------------------------------------------
        # ATTENTION MEKANIZMASI
        # -----------------------------------------------------------------------
        print("\n🔍 Attention mekanizması devreye giriyor...")
        context_vector, attention_weights = calculate_attention(
            decoder_hidden, 
            encoder_hiddens, 
            input_words
        )
        
        # -----------------------------------------------------------------------
        # DECODER STEP
        # -----------------------------------------------------------------------
        print("\n⚙️  Decoder hidden state güncelleniyor...")
        decoder_hidden = decoder_step(prev_word_embedding, decoder_hidden, context_vector)
        print(f"Yeni decoder hidden state: {decoder_hidden}")
        
        # -----------------------------------------------------------------------
        # ÇIKTI KELİMESİ SEÇME
        # -----------------------------------------------------------------------
        print("\n🎲 Çıktı kelimesi seçiliyor...")
        selected_word, scores = select_output_word(
            decoder_hidden, 
            output_vocab, 
            word_embeddings
        )
        
        print("Tüm kelimeler için skorlar:")
        for word, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            marker = " ← SEÇİLDİ" if word == selected_word else ""
            print(f"  '{word}': {score:.4f}{marker}")
        
        print(f"\n✨ Üretilen kelime: '{selected_word}'")
        
        # Üretilen kelimeyi kaydet
        generated_words.append(selected_word)
        
        # Eğer <END> tokeni üretildiyse dur
        if selected_word == "<END>":
            print("\n🏁 <END> tokeni üretildi. Decoder durduruluyor.")
            break
        
        # Sonraki adım için bu kelimeyi kullan
        prev_word = selected_word
    
    # -------------------------------------------------------------------------
    # 5. SONUÇLARI GÖSTER
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SİMÜLASYON SONUÇLARI")
    print("=" * 80)
    print(f"\n📥 Girdi Cümlesi    : {input_sentence}")
    print(f"📤 Üretilen Cümle   : {' '.join(generated_words)}")
    print(f"🎯 Hedef Cümle      : {target_sentence}")
    
    print("\n" + "=" * 80)
    print("AÇIKLAMA")
    print("=" * 80)
    print("""
Bu simülasyonda:
- Encoder, Türkçe cümleyi kelime kelime işledi ve her kelime için bir 
  hidden state üretti.
  
- Decoder, her adımda:
  * Attention mekanizması ile encoder'ın hangi kelimesine odaklanacağına karar verdi
  * Dot product ile attention skorları hesaplandı
  * Softmax ile bu skorlar olasılık dağılımına çevrildi
  * Weighted sum ile context vector oluşturuldu
  * Bu context vector kullanılarak yeni bir kelime üretildi

- Attention ağırlıkları, decoder'ın her adımda encoder'ın hangi kelimesine
  ne kadar önem verdiğini gösterir.

NOT: Bu bir simülasyondur. Gerçek bir seq2seq modelinde ağırlıklar
     eğitim sırasında öğrenilir ve anlamlı çeviriler üretir.
     Burada ağırlıklar rastgele olduğu için çıktı da rastgeledir.
    """)
    print("=" * 80)

# ============================================================================
# PROGRAMI ÇALIŞTIR
# ============================================================================

if __name__ == "__main__":
    # Random seed (her çalıştırmada aynı sonucu görmek için)
    np.random.seed(42)
    
    # Simülasyonu başlat
    run_simulation()

