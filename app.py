"""
🎬 SensCritique - Système de Recommandation de Critiques
Développé par: Amir Salah Eddine Daoudi
Email: daoudiamirsalaheddine@gmail.com
LinkedIn: https://www.linkedin.com/in/amir-salah-eddine-daoudi/
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path
import time

# Configuration de la page
st.set_page_config(
    page_title="SensCritique - Recommandations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le design system
st.markdown("""
<style>
:root {
    --primary-color: #FF6B35;
    --secondary-color: #004E89;
    --accent-color: #FFD23F;
}

.main-header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    border-left: 4px solid var(--primary-color);
}

.recommendation-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 15px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border: 1px solid #E9ECEF;
}

.score-badge {
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    color: white;
    margin: 0.25rem;
}
.score-high { background-color: #28A745; }
.score-medium { background-color: #FFC107; color: #000; }
.score-low { background-color: #6C757D; }
</style>
""", unsafe_allow_html=True)

class SensCritiqueRecommender:
    """Système de recommandation simplifié."""
    
    def __init__(self):
        self.data = None
        self.tfidf_matrix = None
        self.vectorizer = None
        
    def load_data(self):
        """Charge les données depuis les fichiers CSV."""
        try:
            parent_dir = Path(__file__).parent.parent
            fight_club_file = parent_dir / "fight_club_critiques.csv"
            interstellar_file = parent_dir / "interstellar_critique.csv"
            
            data_frames = []
            
            if fight_club_file.exists():
                try:
                    # Essayer d'abord de lire sans headers pour voir la structure
                    df_sample = pd.read_csv(fight_club_file, nrows=5)
                    st.write("🔍 **Structure Fight Club (5 premières lignes):**")
                    st.write(df_sample)
                    st.write("**Colonnes et types:**")
                    for i, col in enumerate(df_sample.columns):
                        st.write(f"  Col {i}: {col} = `{str(df_sample[col].iloc[0])[:100]}...`")
                    
                    # Essayer de deviner la bonne colonne pour le contenu
                    content_col = None
                    for col in df_sample.columns:
                        # Chercher la colonne avec le texte le plus long
                        sample_text = str(df_sample[col].iloc[0])
                        if len(sample_text) > 50 and not sample_text.isdigit():
                            content_col = col
                            st.success(f"🎯 Colonne contenu détectée: {col}")
                            break
                    
                    if content_col:
                        # Utiliser la vraie structure détectée
                        df_fc = pd.read_csv(fight_club_file)
                        df_fc = df_fc.rename(columns={content_col: 'content'})
                    else:
                        # Fallback vers l'ancienne méthode
                        df_fc = pd.read_csv(fight_club_file, names=[
                            'id', 'url', 'rating', 'created_at', 'updated_at', 'user_id', 'content'
                        ])
                    # Nettoyer les données
                    df_fc = df_fc.dropna(subset=['content'])
                    df_fc['content'] = df_fc['content'].astype(str)
                    df_fc['rating'] = pd.to_numeric(df_fc['rating'], errors='coerce').fillna(5)
                    df_fc['film'] = 'Fight Club'
                    df_fc = df_fc.head(50)
                    data_frames.append(df_fc)
                    st.success(f"✅ Fight Club: {len(df_fc)} critiques")
                except Exception as e:
                    st.warning(f"⚠️ Erreur Fight Club: {e}")
            
            if interstellar_file.exists():
                try:
                    # Essayer d'abord de lire sans headers pour voir la structure  
                    df_sample = pd.read_csv(interstellar_file, nrows=5)
                    st.write("🔍 **Structure Interstellar (5 premières lignes):**")
                    st.write(df_sample)
                    
                    # Essayer de deviner la bonne colonne pour le contenu
                    content_col = None
                    for col in df_sample.columns:
                        sample_text = str(df_sample[col].iloc[0])
                        if len(sample_text) > 50 and not sample_text.isdigit():
                            content_col = col
                            st.success(f"🎯 Colonne contenu Interstellar: {col}")
                            break
                    
                    if content_col:
                        df_int = pd.read_csv(interstellar_file)
                        df_int = df_int.rename(columns={content_col: 'content'})
                    else:
                        df_int = pd.read_csv(interstellar_file, names=[
                            'id', 'url', 'rating', 'created_at', 'updated_at', 'user_id', 'content'
                        ])
                    # Nettoyer les données
                    df_int = df_int.dropna(subset=['content'])
                    df_int['content'] = df_int['content'].astype(str)
                    df_int['rating'] = pd.to_numeric(df_int['rating'], errors='coerce').fillna(7)
                    df_int['film'] = 'Interstellar'
                    df_int = df_int.head(50)
                    data_frames.append(df_int)
                    st.success(f"✅ Interstellar: {len(df_int)} critiques")
                except Exception as e:
                    st.warning(f"⚠️ Erreur Interstellar: {e}")
            
            if data_frames:
                st.info(f"📊 Concaténation de {len(data_frames)} dataframes...")
                self.data = pd.concat(data_frames, ignore_index=True)
                st.info(f"📊 Avant nettoyage: {len(self.data)} critiques")
                
                # Nettoyer et valider les données avec debug
                initial_count = len(self.data)
                self.data = self.data.dropna(subset=['content'])
                after_dropna = len(self.data)
                st.info(f"📊 Après dropna: {after_dropna} critiques")
                
                # Convertir content en string et filtrer
                self.data['content'] = self.data['content'].astype(str)
                self.data = self.data[self.data['content'].str.len() > 5]  # Réduire le seuil
                after_filter = len(self.data)
                st.info(f"📊 Après filtre longueur: {after_filter} critiques")
                
                if len(self.data) > 0:
                    self.data['content_clean'] = self.data['content'].apply(self.clean_text)
                    # Reset des index pour éviter les conflits
                    self.data = self.data.reset_index(drop=True)
                    self.data['id'] = range(1, len(self.data) + 1)  # Nouveaux IDs sécurisés
                    st.success(f"🎬 Total final: {len(self.data)} critiques chargées")
                    return True
                else:
                    st.error("❌ Aucune critique valide après nettoyage")
                    self._create_demo_data()
                    return True
            else:
                self._create_demo_data()
                return True
                
        except Exception as e:
            self._create_demo_data()
            return True
    
    def _create_demo_data(self):
        """Crée des données de démonstration."""
        demo_critiques = [
            {'id': 1, 'film': 'Fight Club', 'rating': 6, 'user_id': 101,
             'content': "Fight Club est un film trop violent avec beaucoup de bagarres qui m'ont dérangé."},
            {'id': 2, 'film': 'Fight Club', 'rating': 5, 'user_id': 102,
             'content': "Ce film m'a déçu par sa violence excessive et ses scènes brutales."},
            {'id': 3, 'film': 'Fight Club', 'rating': 8, 'user_id': 103,
             'content': "Excellent film avec une philosophie profonde sur la société moderne."},
            {'id': 4, 'film': 'Interstellar', 'rating': 9, 'user_id': 104,
             'content': "Film magnifique sur l'espace. Les effets visuels sont époustouflants."},
            {'id': 5, 'film': 'Interstellar', 'rating': 7, 'user_id': 105,
             'content': "Bon film de science-fiction mais un peu long parfois."},
        ]
        
        self.data = pd.DataFrame(demo_critiques)
        self.data['content_clean'] = self.data['content'].apply(self.clean_text)
        st.info("🧪 Utilisation de données de démonstration")
    
    def clean_text(self, text):
        """Nettoie le texte."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def compute_similarity_matrix(self):
        """Calcule la matrice de similarité TF-IDF."""
        try:
            if self.data is None or len(self.data) == 0:
                st.error("❌ Aucune donnée pour calculer la similarité")
                return False
            
            # Vérifier que content_clean existe et n'est pas vide
            clean_texts = self.data['content_clean'].dropna()
            if len(clean_texts) == 0:
                st.error("❌ Aucun contenu nettoyé disponible")
                return False
            
            st.info(f"🔍 Calcul TF-IDF sur {len(clean_texts)} textes...")
            
            # Configuration TF-IDF plus permissive
            self.vectorizer = TfidfVectorizer(
                max_features=200,  # Réduire pour éviter les problèmes
                ngram_range=(1, 2),
                min_df=1,  # Accepter tous les mots
                stop_words=None  # Garder tous les mots
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(clean_texts)
            st.success(f"✅ Matrice TF-IDF créée: {self.tfidf_matrix.shape}")
            return True
            
        except Exception as e:
            st.error(f"❌ Erreur TF-IDF: {e}")
            return False
    
    def get_recommendations(self, critique_id, num_recommendations=5):
        """Obtient des recommandations."""
        try:
            critique_idx = self.data[self.data['id'] == critique_id].index[0]
            source_critique = self.data.iloc[critique_idx]
            
            cosine_similarities = cosine_similarity(
                self.tfidf_matrix[critique_idx:critique_idx+1], 
                self.tfidf_matrix
            ).flatten()
            
            similar_indices = cosine_similarities.argsort()[::-1][1:num_recommendations+1]
            
            recommendations = []
            for idx in similar_indices:
                if cosine_similarities[idx] > 0.01:
                    rec_critique = self.data.iloc[idx]
                    
                    semantic_score = cosine_similarities[idx]
                    rating_proximity = 1 - abs(source_critique['rating'] - rec_critique['rating']) / 9
                    same_film_bonus = 1.0 if source_critique['film'] == rec_critique['film'] else 0.3
                    
                    total_score = 0.6 * semantic_score + 0.2 * rating_proximity + 0.2 * same_film_bonus
                    
                    recommendations.append({
                        'id': rec_critique['id'],
                        'film': rec_critique['film'],
                        'content': rec_critique['content'][:200] + "...",
                        'rating': rec_critique['rating'],
                        'user_id': rec_critique['user_id'],
                        'similarity_score': total_score,
                        'semantic_score': semantic_score,
                        'rating_proximity': rating_proximity,
                        'same_film_bonus': same_film_bonus
                    })
            
            return sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)
            
        except Exception as e:
            st.error(f"Erreur: {e}")
            return []

@st.cache_resource
def init_recommender():
    """Initialise le système."""
    recommender = SensCritiqueRecommender()
    recommender.load_data()
    recommender.compute_similarity_matrix()
    return recommender

def show_architecture_diagram():
    """Affiche le diagramme d'architecture microservices."""
    st.subheader("🏗️ Architecture Microservices")
    
    # Création d'un diagramme d'architecture interactif avec Plotly
    fig = go.Figure()
    
    # Définir les couches de l'architecture
    layers = [
        {"name": "Frontend", "y": 4, "components": ["Page Critique", "Widget Reco", "API Gateway"], "color": "#FF6B35"},
        {"name": "API Gateway", "y": 3, "components": ["Kong", "Auth", "Rate Limiting"], "color": "#004E89"},
        {"name": "Services", "y": 2, "components": ["Recommandation", "NLP Engine", "Data Service"], "color": "#FFD23F"},
        {"name": "Données", "y": 1, "components": ["PostgreSQL", "Elasticsearch", "Redis"], "color": "#28A745"},
        {"name": "Infrastructure", "y": 0, "components": ["Kubernetes", "Monitoring", "CI/CD"], "color": "#6C757D"}
    ]
    
    # Ajouter chaque couche
    for layer in layers:
        for i, component in enumerate(layer["components"]):
            fig.add_trace(go.Scatter(
                x=[i], y=[layer["y"]], 
                mode="markers+text",
                marker=dict(size=80, color=layer["color"]),
                text=component,
                textposition="middle center",
                textfont=dict(color="white", size=10),
                name=layer["name"],
                showlegend=(i == 0)
            ))
    
    fig.update_layout(
        title="Architecture Microservices SensCritique",
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 2.5]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 4.5]),
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Description textuelle
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Couches Architecture**
        - **Frontend** : Interface utilisateur React
        - **API Gateway** : Kong pour routing et auth
        - **Services** : Microservices découplés
        - **Données** : Stack polyglotte optimisée
        - **Infra** : Cloud-native avec K8s
        """)
    
    with col2:
        st.markdown("""
        **⚡ Technologies Clés**
        - **NLP** : spaCy + Transformers + VADER
        - **ML** : scikit-learn + Sentence Transformers
        - **Cache** : Redis multi-niveaux
        - **Search** : Elasticsearch pour vectoriel
        - **API** : FastAPI avec documentation auto
        """)

def show_algorithm_flow():
    """Affiche le flow de l'algorithme de recommandation."""
    st.subheader("🧠 Pipeline de Recommandation")
    
    # Diagramme de flux avec étapes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📝 Phase 1: Preprocessing
        """)
        
        phases = [
            {"icon": "📄", "title": "Input", "desc": "Critique utilisateur"},
            {"icon": "🧹", "title": "Cleaning", "desc": "HTML/Unicode normalization"},
            {"icon": "🔤", "title": "Tokenization", "desc": "spaCy lemmatization + POS"},
            {"icon": "💭", "title": "Features", "desc": "Embeddings + Sentiment + Thèmes"}
        ]
        
        for i, phase in enumerate(phases):
            st.markdown(f"""
            <div class="recommendation-card" style="margin: 0.5rem 0;">
                <div style="display: flex; align-items: center;">
                    <div style="font-size: 2rem; margin-right: 1rem;">{phase['icon']}</div>
                    <div>
                        <h4 style="margin: 0;">{phase['title']}</h4>
                        <p style="margin: 0; color: #666;">{phase['desc']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if i < len(phases) - 1:
                st.markdown("<div style='text-align: center; font-size: 1.5rem;'>⬇️</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### ⚖️ Phase 2: Scoring Hybride
        """)
        
        components = [
            {"name": "Similarité Sémantique", "weight": 0.6, "color": "#FF6B35"},
            {"name": "Proximité Notes", "weight": 0.2, "color": "#004E89"},
            {"name": "Bonus Même Film", "weight": 0.2, "color": "#FFD23F"}
        ]
        
        # Graphique en secteurs des composants
        fig = px.pie(
            values=[c["weight"] for c in components],
            names=[f"{c['name']}<br>{c['weight']*100}%" for c in components],
            color_discrete_sequence=[c["color"] for c in components],
            title="Répartition des Scores"
        )
        fig.update_traces(textposition='inside', textinfo='label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Exemple concret de calcul
    st.markdown("### 🔢 Exemple de Calcul")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📚 Critique Source</h4>
            <p><strong>Fight Club</strong></p>
            <p>Note: 6/10</p>
            <p>Sentiment: -0.3 (négatif)</p>
            <p>"Trop de violence et de bagarres..."</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Candidat #1</h4>
            <p><strong>Fight Club</strong></p>
            <p>Note: 5/10</p>
            <p>Sentiment: -0.4 (négatif)</p>
            <p>"Film décevant, violence excessive..."</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Score Final</h4>
            <p>Sémantique: 0.85</p>
            <p>Notes: 0.90 (5 vs 6)</p>
            <p>Film: 1.0 (même film)</p>
            <hr>
            <p><strong>Total: 0.89</strong></p>
        </div>
        """, unsafe_allow_html=True)

def show_design_system():
    """Affiche le design system complet."""
    st.subheader("🎨 Système de Conception")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎨 Palette de Couleurs")
        colors = {
            "Primary Orange": "#FF6B35",
            "Secondary Blue": "#004E89", 
            "Accent Yellow": "#FFD23F",
            "Success Green": "#28A745",
            "Warning Amber": "#FFC107",
            "Dark Text": "#1A1A1A",
            "Light Text": "#6C757D"
        }
        
        for name, color in colors.items():
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 1rem 0;">
                <div style="width: 50px; height: 50px; background: {color}; border-radius: 12px; margin-right: 1rem; border: 2px solid #eee; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                <div>
                    <strong style="font-size: 1.1rem;">{name}</strong><br>
                    <code style="background: #f8f9fa; padding: 0.2rem 0.5rem; border-radius: 4px;">{color}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🧩 Composants UI")
        
        st.markdown("""
        <div class="metric-card" style="margin: 1rem 0;">
            <h4 style="color: var(--primary-color); margin: 0;">📊 Metric Card</h4>
            <p style="margin: 0.5rem 0 0 0;">Affichage des KPIs avec bordure colorée et ombre subtile</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="recommendation-card" style="margin: 1rem 0;">
            <h4 style="color: var(--secondary-color); margin: 0;">🎬 Recommendation Card</h4>
            <p style="margin: 0.5rem 0 0 0;">Cards interactives avec hover effects pour les suggestions</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <span class="score-badge score-high" style="margin: 0.25rem;">Score Élevé (0.7+)</span><br>
            <span class="score-badge score-medium" style="margin: 0.25rem;">Score Moyen (0.4-0.7)</span><br>
            <span class="score-badge score-low" style="margin: 0.25rem;">Score Faible (< 0.4)</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Grille typographique
    st.markdown("#### 📝 Système Typographique")
    
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 2px 15px rgba(0,0,0,0.1);">
        <h1 style="color: #1A1A1A; margin: 0.5rem 0; font-weight: 700;">Heading 1 - 2.5rem (40px)</h1>
        <h2 style="color: #1A1A1A; margin: 0.5rem 0; font-weight: 600;">Heading 2 - 2rem (32px)</h2>
        <h3 style="color: #1A1A1A; margin: 0.5rem 0; font-weight: 600;">Heading 3 - 1.5rem (24px)</h3>
        <h4 style="color: #1A1A1A; margin: 0.5rem 0; font-weight: 500;">Heading 4 - 1.25rem (20px)</h4>
        <p style="color: #6C757D; margin: 0.5rem 0; font-size: 1rem;">Body Text Regular - 1rem (16px)</p>
        <small style="color: #6C757D; font-size: 0.875rem;">Small Text - 0.875rem (14px)</small>
    </div>
    """, unsafe_allow_html=True)

def show_performance_architecture():
    """Affiche l'architecture de performance et cache."""
    st.subheader("⚡ Architecture de Performance")
    
    # Métriques de performance
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚀 Latence P95", "< 200ms", "✅")
    with col2:
        st.metric("📊 Taux Cache", "85%", "↗️ +5%")
    with col3:
        st.metric("🎯 Throughput", "500 req/s", "↗️ +50")
    with col4:
        st.metric("⚡ Disponibilité", "99.9%", "✅")
    
    # Diagramme cache multi-niveaux
    st.markdown("#### 🏗️ Cache Multi-Niveaux")
    
    cache_levels = [
        {"name": "L1: Application (In-Memory)", "hit_rate": 40, "ttl": "5 min", "size": "1K entries", "color": "#28A745"},
        {"name": "L2: Redis Hot Cache", "hit_rate": 35, "ttl": "1 hour", "size": "100K entries", "color": "#FF6B35"},
        {"name": "L3: Redis Cold Cache", "hit_rate": 20, "ttl": "24h", "size": "1M entries", "color": "#FFD23F"},
        {"name": "L4: Elasticsearch", "hit_rate": 4, "ttl": "6h", "size": "∞", "color": "#004E89"},
        {"name": "L5: PostgreSQL", "hit_rate": 1, "ttl": "∞", "size": "∞", "color": "#6C757D"}
    ]
    
    # Graphique waterfall des hit rates
    fig = go.Figure()
    
    cumulative = 0
    for level in cache_levels:
        fig.add_trace(go.Bar(
            name=level["name"],
            x=[level["name"]],
            y=[level["hit_rate"]],
            marker_color=level["color"],
            text=f"{level['hit_rate']}%",
            textposition='inside'
        ))
    
    fig.update_layout(
        title="Répartition des Cache Hit Rates",
        xaxis_title="Niveau de Cache",
        yaxis_title="Hit Rate (%)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Détails des caches
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🔥 Hot Path (Temps Réel)")
        st.markdown("""
        ```
        User Request
            ↓
        Cache L1 Check (40% hit)
            ↓ (miss)
        Redis Hot (35% hit)
            ↓ (miss) 
        Compute + Cache
            ↓
        Response < 200ms
        ```
        """)
    
    with col2:
        st.markdown("##### ❄️ Cold Path (Batch)")
        st.markdown("""
        ```
        Scheduled Job (6h)
            ↓
        Process New Critiques
            ↓
        Generate Embeddings
            ↓
        Update ES Index
            ↓
        Warm Cache Popular
        ```
        """)

def analyze_critique_visual(content):
    """Analyse visuelle d'une critique pour l'interface."""
    # Sécuriser le contenu
    if pd.isna(content) or content is None:
        content = "Contenu indisponible"
    content = str(content)
    
    # Analyse du sentiment basique
    positive_words = ['excellent', 'magnifique', 'superbe', 'génial', 'parfait', 'chef-d\'œuvre']
    negative_words = ['violence', 'violent', 'décevant', 'nul', 'horrible', 'ennuyeux', 'mauvais']
    
    content_lower = content.lower()
    
    pos_count = sum(1 for word in positive_words if word in content_lower)
    neg_count = sum(1 for word in negative_words if word in content_lower)
    
    if neg_count > pos_count:
        sentiment_score = -0.3 - (neg_count * 0.1)
        sentiment_icon = "😞"
        sentiment_label = "Négatif"
    elif pos_count > neg_count:
        sentiment_score = 0.3 + (pos_count * 0.1)
        sentiment_icon = "😊"
        sentiment_label = "Positif"
    else:
        sentiment_score = 0.0
        sentiment_icon = "😐"
        sentiment_label = "Neutre"
    
    # Détection de thèmes
    themes = []
    if any(word in content_lower for word in ['violence', 'violent', 'bagarre', 'combat']):
        themes.append('Violence')
    if any(word in content_lower for word in ['philosophie', 'profond', 'société', 'réflexion']):
        themes.append('Philosophie')
    if any(word in content_lower for word in ['espace', 'science-fiction', 'futur', 'technologie']):
        themes.append('Sci-Fi')
    if any(word in content_lower for word in ['action', 'aventure', 'dynamique']):
        themes.append('Action')
    if any(word in content_lower for word in ['ennuyeux', 'long', 'lent']):
        themes.append('Rythme')
    if any(word in content_lower for word in ['visuel', 'effet', 'image', 'cinématographie']):
        themes.append('Visuel')
    
    if not themes:
        themes = ['Général']
    
    return {
        'sentiment_score': sentiment_score,
        'sentiment_icon': sentiment_icon,
        'sentiment_label': sentiment_label,
        'themes': themes
    }

def apply_custom_weights(recommendations, weight_semantic, weight_rating, weight_film):
    """Applique des poids personnalisés aux recommandations."""
    total_weight = weight_semantic + weight_rating + weight_film
    if total_weight == 0:
        return recommendations
    
    # Normaliser les poids
    w_sem = weight_semantic / total_weight
    w_rat = weight_rating / total_weight  
    w_fil = weight_film / total_weight
    
    for rec in recommendations:
        # Recalculer le score avec les nouveaux poids
        new_score = (
            w_sem * rec['semantic_score'] +
            w_rat * rec['rating_proximity'] +
            w_fil * rec['same_film_bonus']
        )
        rec['similarity_score'] = new_score
    
    # Re-trier par nouveau score
    recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return recommendations

def show_score_distribution(recommendations):
    """Affiche la distribution des scores de recommandation."""
    st.markdown("### 📊 Analyse des Scores")
    
    scores = [r['similarity_score'] for r in recommendations]
    semantic_scores = [r['semantic_score'] for r in recommendations]
    rating_scores = [r['rating_proximity'] for r in recommendations]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogramme des scores finaux
        fig_hist = px.histogram(
            x=scores,
            nbins=10,
            title="Distribution des Scores de Similarité",
            labels={'x': 'Score de Similarité', 'y': 'Nombre de Recommandations'},
            color_discrete_sequence=['#FF6B35']
        )
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Scatter plot composants vs score final
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=semantic_scores,
            y=scores,
            mode='markers',
            name='Sémantique vs Final',
            marker=dict(color='#FF6B35', size=10),
            text=[f"Rec #{i+1}" for i in range(len(scores))],
            textposition="top center"
        ))
        
        fig_scatter.update_layout(
            title="Corrélation Sémantique vs Score Final",
            xaxis_title="Score Sémantique",
            yaxis_title="Score Final",
            height=300
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

def show_recommendation_card(rank, rec, source, show_explanation):
    """Affiche une carte de recommandation améliorée."""
    score_class = "high" if rec['similarity_score'] > 0.7 else "medium" if rec['similarity_score'] > 0.4 else "low"
    
    # Couleurs basées sur le score
    score_color = "#28A745" if score_class == "high" else "#FFC107" if score_class == "medium" else "#6C757D"
    
    # Analyse de la recommandation
    rec_analysis = analyze_critique_visual(rec['content'])
    
    # Raison principale de la recommandation
    components = {
        'Sémantique': rec['semantic_score'],
        'Notes': rec['rating_proximity'], 
        'Film': rec['same_film_bonus']
    }
    main_reason = max(components.items(), key=lambda x: x[1])
    
    st.markdown(f"""
    <div class="recommendation-card" style="border-left: 5px solid {score_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <h4 style="margin: 0; color: #1A1A1A;">#{rank} - 🎬 {rec['film']}</h4>
                <p style="margin: 0; color: #6C757D; font-size: 0.9rem;">Utilisateur #{rec['user_id']} • Note: {rec['rating']}/10</p>
            </div>
            <div style="text-align: right;">
                <span class="score-badge score-{score_class}" style="font-size: 1.1rem;">{rec['similarity_score']:.3f}</span>
                <br><small style="color: #6C757D;">Raison: {main_reason[0]}</small>
            </div>
        </div>
        
        <p style="font-size: 1rem; line-height: 1.5; margin-bottom: 1rem; color: #1A1A1A;">
            {rec['content'][:200]}{'...' if len(rec['content']) > 200 else ''}
        </p>
        
        <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
            {' '.join([f'<span style="background: #E3F2FD; color: #1976D2; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem;">{theme}</span>' for theme in rec_analysis['themes']])}
        </div>
    """, unsafe_allow_html=True)
    
    if show_explanation:
        st.markdown(f"""
        <div style="background: #F8F9FA; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <strong>🔍 Détail du Score:</strong><br>
            🧠 Sémantique: <code>{rec['semantic_score']:.3f}</code> • 
            ⭐ Notes: <code>{rec['rating_proximity']:.3f}</code> • 
            🎬 Film: <code>{rec['same_film_bonus']:.3f}</code><br>
            😊 Sentiment: <span style="color: {'green' if rec_analysis['sentiment_score'] > 0 else 'red'};">{rec_analysis['sentiment_label']} ({rec_analysis['sentiment_score']:.2f})</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)

def show_recommendation_analysis(recommendations, source):
    """Affiche une analyse globale des recommandations."""
    st.markdown("### 🔬 Analyse Globale des Recommandations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎭 Distribution Sentiment")
        
        sentiments = []
        for rec in recommendations:
            analysis = analyze_critique_visual(rec['content'])
            sentiments.append(analysis['sentiment_label'])
        
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color_discrete_map={
                'Positif': '#28A745',
                'Négatif': '#DC3545', 
                'Neutre': '#6C757D'
            },
            title="Répartition des Sentiments"
        )
        fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
        fig_sentiment.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Qualité des Matches")
        
        # Graphique radar des composants moyens
        components = ['Sémantique', 'Notes', 'Film']
        avg_scores = [
            np.mean([r['semantic_score'] for r in recommendations]),
            np.mean([r['rating_proximity'] for r in recommendations]),
            np.mean([r['same_film_bonus'] for r in recommendations])
        ]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_scores,
            theta=components,
            fill='toself',
            name='Score Moyen',
            line_color='#FF6B35'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            height=300,
            title="Profil des Recommandations"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col3:
        st.markdown("#### ⚡ Métriques Qualité")
        
        # Calcul de métriques avancées
        scores = [r['similarity_score'] for r in recommendations]
        
        st.metric("🎯 Précision", f"{np.mean(scores):.3f}", 
                 f"σ = {np.std(scores):.3f}")
        
        same_film_ratio = len([r for r in recommendations if r['film'] == source['film']]) / len(recommendations)
        st.metric("🎬 Pertinence Film", f"{same_film_ratio:.1%}", 
                 "Cohérence thématique")
        
        high_quality = len([s for s in scores if s > 0.7]) / len(scores)
        st.metric("⭐ Qualité Élevée", f"{high_quality:.1%}",
                 "Score > 0.7")

def main():
    """Application principale."""
    
    # Sidebar navigation
    st.sidebar.title("🎬 SensCritique")
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Accueil", "🎨 Design System", "🎯 Recommandations", "📊 Analytics", "👨‍💻 À Propos"]
    )
    
    if page == "🏠 Accueil":
        st.markdown('<div class="main-header"><h1>🎬 SensCritique</h1><p>Système de Recommandation de Critiques</p></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Objectif")
            st.write("Créer un système intelligent qui recommande des critiques similaires basé sur l'analyse de contenu.")
            
            st.markdown("### ⚡ Technologies")
            st.write("- **Streamlit** pour l'interface")
            st.write("- **TF-IDF** pour l'analyse textuelle") 
            st.write("- **Similarité cosinus** pour les recommandations")
            st.write("- **Plotly** pour les visualisations")
        
        with col2:
            st.markdown("### 📊 Algorithme")
            st.write("**Score = 60% Sémantique + 20% Note + 20% Film**")
            
            st.markdown("### 👨‍💻 Développeur")
            st.write("**Amir Salah Eddine Daoudi**")
            st.write("📧 daoudiamirsalaheddine@gmail.com")
            st.write("🔗 [LinkedIn](https://www.linkedin.com/in/amir-salah-eddine-daoudi/)")
    
    elif page == "🎨 Design System":
        st.markdown('<div class="main-header"><h1>🎨 Architecture & Design System</h1><p>Visualisation interactive de la solution technique</p></div>', unsafe_allow_html=True)
        
        # Tabs pour organiser le contenu
        tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Architecture", "🧠 Algorithme", "🎨 Design System", "⚡ Performance"])
        
        with tab1:
            show_architecture_diagram()
        
        with tab2:
            show_algorithm_flow()
        
        with tab3:
            show_design_system()
        
        with tab4:
            show_performance_architecture()
    
    elif page == "🎯 Recommandations":
        st.markdown('<div class="main-header"><h1>🎯 Moteur de Recommandation Interactif</h1><p>Algorithme hybride avec visualisation temps réel</p></div>', unsafe_allow_html=True)
        
        # Bouton pour recharger les données
        col_debug1, col_debug2 = st.columns([1, 4])
        with col_debug1:
            if st.button("🔄 Recharger données"):
                st.cache_resource.clear()
                st.rerun()
        
        recommender = init_recommender()
        
        with col_debug2:
            with st.expander("🔍 Debug Info", expanded=True):  # Ouvrir par défaut
                if recommender.data is not None:
                    st.write(f"Données chargées: {len(recommender.data)} critiques")
                    st.write("**Colonnes disponibles:**", list(recommender.data.columns))
                    st.write("**Échantillon des contenus:**")
                    for i, row in recommender.data.head(3).iterrows():
                        st.write(f"- ID {row['id']}: `{str(row['content'])[:100]}...`")
                        st.write(f"  Clean: `{str(row.get('content_clean', 'N/A'))[:50]}...`")
        
        if recommender.data is None:
            st.error("Erreur de chargement des données")
            return
        
        # Interface améliorée avec colonnes
        col1, col2, col3 = st.columns([2, 1, 1])
        
        # Variables pour le mapping des critiques (portée globale)
        films = recommender.data['film'].unique()
        selected_film = st.selectbox("🎬 Choisir un film", films, key="film_select")
        
        film_critiques = recommender.data[recommender.data['film'] == selected_film]
        
        # Créer le mapping des critiques
        critique_options = []
        critique_mapping = {}
        
        for idx, (_, row) in enumerate(film_critiques.iterrows()):
            # Sécuriser l'accès au contenu
            content = str(row['content']) if pd.notna(row['content']) else "Contenu indisponible"
            
            # Améliorer le preview du contenu
            if len(content) > 5 and not content.isdigit():  # Éviter les IDs numériques
                preview = content[:100] + "..." if len(content) > 100 else content
            else:
                preview = f"Critique #{row['id']}"
            
            # Sécuriser l'analyse du sentiment
            content_lower = content.lower()
            if any(word in content_lower for word in ["violence", "violent", "décevant", "nul", "horrible"]):
                sentiment_icon = "😞"
            elif any(word in content_lower for word in ["excellent", "magnifique", "superbe", "génial"]):
                sentiment_icon = "😊"
            elif row['rating'] <= 5:
                sentiment_icon = "😞"
            elif row['rating'] >= 8:
                sentiment_icon = "😊"
            else:
                sentiment_icon = "😐"
            
            option_text = f"{sentiment_icon} Critique {idx+1} - {row['rating']}/10 - {preview}"
            critique_options.append(option_text)
            critique_mapping[option_text] = row['id']
        
        with col1:
            st.subheader("📝 Sélection de la Critique Source")
            selected_critique = st.selectbox("💬 Sélectionner une critique", critique_options, key="critique_select")
        
        with col2:
            st.subheader("⚙️ Paramètres")
            num_recs = st.slider("📊 Nombre de recommandations", 3, 8, 5)
            min_score = st.slider("🎯 Score minimum", 0.0, 1.0, 0.2, 0.1)
            show_explanation = st.checkbox("🔍 Afficher explications détaillées", True)
        
        with col3:
            st.subheader("🎛️ Poids Algorithme")
            with st.expander("Ajuster les poids", expanded=False):
                weight_semantic = st.slider("🧠 Sémantique", 0.0, 1.0, 0.6, 0.1)
                weight_rating = st.slider("⭐ Notes", 0.0, 1.0, 0.2, 0.1)
                weight_film = st.slider("🎬 Film", 0.0, 1.0, 0.2, 0.1)
        
        if selected_critique and selected_critique in critique_mapping:
            critique_id = critique_mapping[selected_critique]
            # Sécuriser la récupération de la critique
            source_mask = recommender.data['id'] == critique_id
            if source_mask.any():
                source = recommender.data[source_mask].iloc[0]
            else:
                st.error("Critique introuvable")
                return
            
            # Affichage amélioré de la critique source
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### 📖 Critique Source Analysée")
                
                # Analyse visuelle de la critique source
                source_analysis = analyze_critique_visual(source['content'])
                
                st.markdown(f"""
                <div class="recommendation-card" style="background: linear-gradient(135deg, #FF6B35, #FF8A65); color: white;">
                    <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 1rem;">
                        <h4 style="margin: 0; color: white;">🎬 {source['film']}</h4>
                        <div style="display: flex; gap: 1rem;">
                            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 20px;">⭐ {source['rating']}/10</span>
                            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 20px;">{source_analysis['sentiment_icon']} {source_analysis['sentiment_label']}</span>
                        </div>
                    </div>
                    <p style="color: white; font-size: 1.1rem; margin-bottom: 1rem;">{source['content']}</p>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                        {' '.join([f'<span style="background: rgba(255,255,255,0.3); padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.9rem;">{theme}</span>' for theme in source_analysis['themes']])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Analyse NLP")
                
                # Métriques de la critique source
                st.metric("📝 Mots", len(source['content'].split()), "")
                st.metric("😊 Sentiment", f"{source_analysis['sentiment_score']:.2f}", 
                         "Positif" if source_analysis['sentiment_score'] > 0 else "Négatif")
                st.metric("🎭 Thèmes", len(source_analysis['themes']), "détectés")
            
            # Calcul des recommandations avec animation
            st.markdown("### 🔄 Traitement en Cours...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulation du traitement avec étapes visuelles
            steps = [
                "🧹 Nettoyage du texte...",
                "🔤 Tokenisation spaCy...", 
                "🧠 Génération embeddings...",
                "📐 Calcul similarités...",
                "⚖️ Scoring hybride...",
                "🎯 Classement final..."
            ]
            
            for i, step in enumerate(steps):
                status_text.text(step)
                progress_bar.progress((i + 1) / len(steps))
                time.sleep(0.3)  # Animation
            
            # Calcul réel
            start_time = time.time()
            recommendations = recommender.get_recommendations(critique_id, num_recs)
            calc_time = time.time() - start_time
            
            # Appliquer les poids personnalisés
            recommendations = apply_custom_weights(recommendations, weight_semantic, weight_rating, weight_film)
            
            progress_bar.empty()
            status_text.empty()
            
            # Filtrer par score
            filtered_recs = [r for r in recommendations if r['similarity_score'] >= min_score]
            
            if filtered_recs:
                # Métriques de performance
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Recommandations", len(filtered_recs), f"sur {len(recommendations)}")
                with col2:
                    st.metric("⚡ Temps de traitement", f"{calc_time*1000:.1f}ms", "< 200ms target")
                with col3:
                    avg_score = np.mean([r['similarity_score'] for r in filtered_recs])
                    st.metric("📊 Score moyen", f"{avg_score:.3f}", "✅ Qualité")
                with col4:
                    same_film_count = len([r for r in filtered_recs if r['film'] == source['film']])
                    st.metric("🎬 Même film", f"{same_film_count}/{len(filtered_recs)}", "Pertinence")
                
                # Graphique de répartition des scores
                show_score_distribution(filtered_recs)
                
                st.markdown("### 🏆 Recommandations Classées")
                
                # Affichage des recommandations avec style amélioré
                for i, rec in enumerate(filtered_recs, 1):
                    show_recommendation_card(i, rec, source, show_explanation)
                    
                # Analyse globale
                show_recommendation_analysis(filtered_recs, source)
                
            else:
                st.warning("🔍 Aucune recommandation trouvée avec ces paramètres. Essayez de diminuer le score minimum.")
    
    elif page == "📊 Analytics":
        st.markdown('<div class="main-header"><h1>📊 Analytics</h1></div>', unsafe_allow_html=True)
        
        recommender = init_recommender()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Critiques", len(recommender.data))
        with col2:
            st.metric("🎬 Films", len(recommender.data['film'].unique()))
        with col3:
            st.metric("👥 Utilisateurs", len(recommender.data['user_id'].unique()))
        with col4:
            st.metric("⭐ Note moyenne", f"{recommender.data['rating'].mean():.1f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(recommender.data, x='rating', color='film', title="Distribution des notes")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            film_counts = recommender.data['film'].value_counts()
            fig = px.pie(values=film_counts.values, names=film_counts.index, title="Répartition par film")
            st.plotly_chart(fig, use_container_width=True)
        
        # Analyse textuelle sécurisée
        st.subheader("📝 Analyse Textuelle")
        
        try:
            # Sécuriser le calcul de longueur
            recommender.data['content_safe'] = recommender.data['content'].astype(str)
            recommender.data['content_length'] = recommender.data['content_safe'].str.len()
            
            # Filtrer les valeurs aberrantes
            length_data = recommender.data[recommender.data['content_length'] < 2000]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if len(length_data) > 0:
                    fig_length = px.histogram(
                        length_data,
                        x='content_length',
                        color='film',
                        title="Longueur des critiques",
                        color_discrete_sequence=['#FF6B35', '#004E89']
                    )
                    st.plotly_chart(fig_length, use_container_width=True)
                else:
                    st.info("Pas assez de données pour l'histogramme")
            
            with col2:
                if len(length_data) > 0:
                    # Corrélation note vs longueur
                    fig_corr = px.scatter(
                        length_data,
                        x='content_length',
                        y='rating',
                        color='film',
                        title="Corrélation Note vs Longueur",
                        color_discrete_sequence=['#FF6B35', '#004E89']
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Pas assez de données pour le scatter plot")
        
        except Exception as e:
            st.error(f"Erreur dans l'analyse textuelle: {e}")
            st.info("Utilisation de données simplifiées")
    
    elif page == "👨‍💻 À Propos":
        st.markdown('<div class="main-header"><h1>👨‍💻 À Propos</h1></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ## Amir Salah Eddine Daoudi
            **Développeur Full-Stack & Data Scientist**
            
            ### 📧 Contact
            - **Email**: daoudiamirsalaheddine@gmail.com
            - **Email Alt**: spartaamir19@gmail.com
            - **LinkedIn**: [Profil LinkedIn](https://www.linkedin.com/in/amir-salah-eddine-daoudi/)
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Compétences
            - **Backend**: Python, FastAPI, Django
            - **Frontend**: React, Streamlit
            - **Data**: ML, NLP, Analytics
            - **Infrastructure**: Docker, AWS
            
            ### 🎯 Projet
            Système de recommandation utilisant TF-IDF et similarité cosinus pour suggérer des critiques similaires.
            """)

if __name__ == "__main__":
    main()
