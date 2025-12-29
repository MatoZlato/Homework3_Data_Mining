import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Brand Reputation Monitor", layout="wide")

# Nujno za stabilnost na Renderju (512MB RAM)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Nalaganje podatkov
@st.cache_data
def load_data():
    df = pd.read_csv('podatki_2023.csv')
    df['Datum'] = pd.to_datetime(df['Datum'])
    return df

df = load_data()

# Navigacija (Zahteva 2.1)
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
        format_func=lambda x: pd.to_datetime(f"2023-{x}-01").month_name()
    )

    df_filtered = df[df['Datum'].dt.month == month_num].copy()

    if not df_filtered.empty:
        # AI Analiza (Zahteva 3)
        with st.spinner('AI analizira...'):
            sentiment_pipeline = load_sentiment_model()
            results = sentiment_pipeline(df_filtered['Komentar'].tolist())
            df_filtered['Sentiment'] = [res['label'] for res in results]
            df_filtered['Confidence'] = [res['score'] for res in results]
        
        st.dataframe(df_filtered, use_container_width=True)

        # Vizualizacija (Zahteva 4)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df_filtered['Sentiment'].value_counts(), 
                         title="Sentiment Distribution",
                         color=df_filtered['Sentiment'].value_counts().index,
                         color_discrete_map={'POSITIVE':'green', 'NEGATIVE':'red'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Avg Confidence", f"{df_filtered['Confidence'].mean():.2%}")

        # BONUS: Word Cloud
        st.subheader("☁️ Word Cloud")
        text = " ".join(review for review in df_filtered['Komentar'])
        wc = WordCloud(background_color="white", width=800, height=400).generate(text)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.warning("Ni podatkov za ta mesec.")
