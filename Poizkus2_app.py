import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px

st.set_page_config(page_title="Brand Reputation Monitor", layout="wide")

# --- 3. SENTIMENT ANALYSIS SETUP ---
@st.cache_resource
def load_sentiment_model():
    # Uporaba specifičnega modela iz navodil
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# --- NALOGA 2.1: SIDEBAR NAVIGATION ---
st.sidebar.title("Navigacija")
choice = st.sidebar.radio("Izberite razdelek:", ["Products", "Testimonials", "Reviews"])

@st.cache_data
def load_data():
    df = pd.read_csv('podatki_2023.csv')
    df['Datum'] = pd.to_datetime(df['Datum'])
    return df

df = load_data()

# --- NALOGA 2.2: SECTION BEHAVIOR ---
if choice == "Products" or choice == "Testimonials":
    st.title(f"Prikaz razdelka: {choice}")
    st.dataframe(df, use_container_width=True)

elif choice == "Reviews":
    st.title("🔎 Analiza mnenj (Reviews)")
    
    # 2.2.b.i: Month Selection
    month_num = st.select_slider(
        "Izberite mesec v letu 2023:",
        options=list(range(1, 13)),
        value=6, # Privzeto Junij, ker imamo tam podatke
        format_func=lambda x: pd.to_datetime(f"2023-{x}-01").month_name()
    )

    # 2.2.b.ii: Filtriranje
    df_filtered = df[df['Datum'].dt.month == month_num].copy()

    if df_filtered.empty:
        st.warning(f"Ni najdenih mnenj za izbrani mesec.")
    else:
        # --- NALOGA 3: SENTIMENT ANALYSIS ---
        with st.spinner('AI analizira sentiment...'):
            results = sentiment_pipeline(df_filtered['Komentar'].tolist())
            df_filtered['Sentiment'] = [res['label'] for res in results]
            df_filtered['Confidence'] = [res['score'] for res in results]
        
        st.subheader(f"Rezultati za {pd.to_datetime(f'2023-{month_num}-01').month_name()}")
        st.dataframe(df_filtered, use_container_width=True)

        # --- NALOGA 4: VISUALIZATION ---
        st.markdown("---")
        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar Chart: Count of Positive vs Negative
            sentiment_counts = df_filtered['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            fig = px.bar(sentiment_counts, x='Sentiment', y='Count', 
                         color='Sentiment', title="Število pozitivnih vs. negativnih mnenj",
                         color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Average Confidence Score
            avg_confidence = df_filtered['Confidence'].mean()
            st.metric("Povprečno zaupanje modela (Confidence)", f"{avg_confidence:.2%}")
            
            st.info("Model klasificira vsako mnenje na podlagi verjetnosti (0-100%).")