import streamlit as st
import pandas as pd
import plotly.express as px

# Osnovna nastavitev strani
st.set_page_config(page_title="Brand Reputation Monitor", layout="wide")

# Funkcija za nalaganje AI modela (Optimizirana za 512MB RAM)
@st.cache_resource
def load_sentiment_model():
    # Uvozimo znotraj funkcije, da ne porabimo RAM-a takoj ob zagonu
    from transformers import pipeline
    # Uporaba majhnega, a učinkovitega modela
    return pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")

# Nalaganje podatkov s predpomnjenjem
@st.cache_data
def load_data():
    df = pd.read_csv('podatki_2023.csv')
    df['Datum'] = pd.to_datetime(df['Datum'])
    return df

# Naložimo podatke
df = load_data()

# --- NAVIGACIJA (Zahteva 2.1) ---
st.sidebar.title("Navigacija")
choice = st.sidebar.radio("Izberite razdelek:", ["Products", "Testimonials", "Reviews"])

if choice in ["Products", "Testimonials"]:
    st.title(f"Prikaz razdelka: {choice}")
    st.dataframe(df, use_container_width=True)

elif choice == "Reviews":
    st.title("🔎 Analiza mnenj (Reviews)")
    
    # Slider za mesec (Zahteva 2.2.b)
    month_num = st.select_slider(
        "Izberite mesec v letu 2023:",
        options=list(range(1, 13)),
        value=6,
        format_func=lambda x: pd.to_datetime(f"2023-{x}-01").strftime('%B')
    )

    # Filtriranje podatkov
    df_filtered = df[df['Datum'].dt.month == month_num].copy()

    if not df_filtered.empty:
        # AI Analiza (Zahteva 3)
        with st.spinner('AI analizira mnenja (prosim počakajte)...'):
            try:
                sentiment_pipeline = load_sentiment_model()
                results = sentiment_pipeline(df_filtered['Komentar'].tolist())
                df_filtered['Sentiment'] = [res['label'] for res in results]
                df_filtered['Confidence'] = [res['score'] for res in results]
                
                # Prikaz tabele z rezultati
                st.dataframe(df_filtered, use_container_width=True)

                # Vizualizacija (Zahteva 4)
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Grafikon sentimenta
                    sentiment_counts = df_filtered['Sentiment'].value_counts()
                    fig = px.bar(
                        sentiment_counts, 
                        title="Porazdelitev sentimenta",
                        labels={'value': 'Število', 'index': 'Kategorija'},
                        color=sentiment_counts.index,
                        color_discrete_map={'POS':'green', 'NEU':'gray', 'NEG':'red'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Povprečno zaupanje AI", f"{df_filtered['Confidence'].mean():.2%}")

                # --- BONUS: WORD CLOUD ---
                st.subheader("☁️ Besedni oblak mnenj")
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                
                text = " ".join(review for review in df_filtered['Komentar'])
                if text.strip():
                    wc = WordCloud(background_color="white", width=800, height=400).generate(text)
                    fig_wc, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_wc)
                
            except Exception as e:
                st.error(f"Napaka pri AI analizi: {e}")
                st.info("To se lahko zgodi zaradi omejitve RAM-a na strežniku.")
    else:
        st.warning("Ni podatkov za izbrani mesec.")
