import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import io
from PIL import Image
 

# Set page config
st.set_page_config(
    page_title="Application de Segmentation Client Selon les Usages OM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stDownloadButton>button {
        background-color: #2196F3;
        color: white;
        border-radius: 5px;
    }
    .header-img {
        max-width: 100%;
        height: auto;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Header image
header_image = Image.open("seg.jpg") if "seg.jpg" else None
if header_image:
    st.image(header_image, use_column_width=True, caption="Segmentation des clients OM selon leurs usages")

# App title
st.title("📊 Segmentation des clients OM selon leurs usages")
st.markdown("""
Cette application effectue une segmentation client en utilisant l'ACP et le clustering K-Means sur des données de transactions.
""")

# Sidebar
st.sidebar.header("Importation des Données")
uploaded_file = st.sidebar.file_uploader("Choisir un fichier CSV", type="csv")

# Define the expected columns
selected_columns = [
    'num_ligne',
    'nombre_p2p_in',
    'montant_p2p_in',
    'nombre_p2p_out',
    'montant_p2p_out',
    'nombre_cashin',
    'montant_cashin',
    'nombre_cashout',
    'montant_cashout',
    'nombre_merchpay',
    'montant_merchpay',
    'nombre_facture',
    'montant_facture',
    'nombre_produit_telco',
    'montant_produit_telco',
    'nombre_b2w_in',
    'nombre_b2w_out',
    'montant_b2w_in',
    'montant_b2w_out',
    'nombre_irt_in',
    'nombre_irt_out',
    'montant_irt_in',
    'montant_irt_out',
    'nombre_webpay',
    'montant_webpay',
    'anciennete',
    'commune'
]

def process_data(df):
    """Process the uploaded data"""
    # Remove duplicates
    data_duplicated_values = df['num_ligne'][df['num_ligne'].duplicated(keep=False)]
    df = df[~df['num_ligne'].isin(data_duplicated_values)]

    # Select relevant columns
    df_quant = df[selected_columns].copy()

    # Monthly normalization based on 'anciennete'
    conditions = [
        df_quant['anciennete'] >= 6,
        (df_quant['anciennete'] < 6) & (df_quant['anciennete'] >= 5),
        (df_quant['anciennete'] < 5) & (df_quant['anciennete'] >= 4),
        (df_quant['anciennete'] < 4) & (df_quant['anciennete'] >= 3),
        (df_quant['anciennete'] < 3) & (df_quant['anciennete'] >= 2),
        (df_quant['anciennete'] < 2) & (df_quant['anciennete'] >= 1),
    ]
    divisors = [6, 5, 4, 3, 2, 1]
    df_quant['divisor'] = np.select(conditions, divisors, default=np.nan)

    # Divide columns
    columns_to_divide = [col for col in df_quant.columns if
                         col not in ['num_ligne', 'anciennete', 'commune', 'divisor']]
    df_quant[columns_to_divide] = df_quant[columns_to_divide].astype(float).div(df_quant['divisor'], axis=0)

    # Keep only customers with 6+ months activity
    df_quant = df_quant[df_quant['divisor'] == 6]

    # Round values
    columns = [col for col in df_quant.columns if col not in ['num_ligne', 'anciennete', 'commune']]
    df_quant[columns] = df_quant[columns].applymap(np.ceil)

    # Create diversity index
    columns_nombre = [col for col in df_quant.columns if 'nombre' in col]
    df_quant['Indice_diversite'] = df_quant[columns_nombre].gt(0).sum(axis=1)

    # Final quantitative data
    data_quanti_mensuel = df_quant.drop(["divisor", "commune"], axis=1)

    return data_quanti_mensuel

def perform_pca(data):
    """Perform PCA analysis"""
    # Normalize data
    scaler = MinMaxScaler()
    columns_to_scale = data.columns.drop('num_ligne')
    df_scaled = scaler.fit_transform(data[columns_to_scale])

    # Perform PCA
    pca = PCA()
    acp = pca.fit_transform(df_scaled)

    return pca, acp, df_scaled

def perform_clustering(acp, n_clusters=5):
    """Perform K-Means clustering"""
    composantes = acp[:, :2]
    clusters = KMeans(n_clusters=n_clusters, n_init=25, random_state=42).fit(composantes)
    return clusters, composantes

def main():
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            # Process data
            with st.spinner('Traitement des données en cours...'):
                data_processed = process_data(data)

            st.success("Données traitées avec succès!")

            # Show processed data
            st.subheader("Aperçu des Données Traitées")
            st.dataframe(data_processed.head())

            # Descriptive statistics
            st.subheader("Statistiques Descriptives")
            st.dataframe(data_processed.describe().round(1).T)

            # Correlation matrix
            st.subheader("Matrice de Corrélation")
            corr = data_processed.drop('num_ligne', axis=1).corr(method="pearson").round(1)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                        vmin=-1, vmax=+1, center=0, cmap='coolwarm',
                        linewidths=0.5, annot=True, ax=ax)
            plt.title('Matrice de Corrélation', fontsize=10)
            st.pyplot(fig)

            # PCA Analysis
            st.subheader("Analyse en Composantes Principales (ACP)")
            pca, acp, df_scaled = perform_pca(data_processed)

            # Explained variance
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                   pca.explained_variance_ratio_, alpha=0.5, align='center',
                   label='Variance expliquée individuelle')
            ax.step(range(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1),
                    np.cumsum(pca.explained_variance_ratio_), where='mid',
                    label='Variance expliquée cumulative')
            ax.set_ylabel('Ratio de variance expliquée')
            ax.set_xlabel('Index des composantes principales')
            ax.set_title('Variance Expliquée par les Composantes Principales')
            ax.legend(loc='best')
            st.pyplot(fig)

            # PCA Scatter plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(acp[:, 0], acp[:, 1])
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('Nuage de Points des Composantes ACP')
            st.pyplot(fig)

            # Clustering
            st.subheader("Segmentation Client")
            n_clusters = st.slider("Sélectionner le nombre de segments", 2, 10, 5)

            with st.spinner('Segmentation en cours...'):
                clusters, composantes = perform_clustering(acp, n_clusters)

                # Assign cluster names
                cluster_names = {
                    0: "Dormants",
                    1: "Occasionnels",
                    2: "Nouveaux",
                    3: "Réguliers",
                    4: "Intensifs"
                }

                # Add clusters to data
                data_processed['Cluster'] = clusters.labels_
                data_processed['Nom_Cluster'] = data_processed['Cluster'].map(cluster_names)

                # Custom palette
                custom_palette = {
                    "Dormants": '#FF5733',
                    "Occasionnels": '#8E44AD',
                    "Nouveaux": '#ffd609',
                    "Réguliers": '#1ABC9C',
                    "Intensifs": '#E67E22'
                }

                # Plot clusters
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.scatterplot(
                    x=composantes[:, 0],
                    y=composantes[:, 1],
                    hue=data_processed['Nom_Cluster'],
                    palette=custom_palette,
                    ax=ax
                )
                ax.set_title('Visualisation des Segments Clients')
                ax.set_xlabel('Composante 1')
                ax.set_ylabel('Composante 2')
                ax.legend(title='Segments')
                st.pyplot(fig)

                # Show cluster distribution
                st.subheader("Distribution des Segments")
                cluster_counts = data_processed['Nom_Cluster'].value_counts()
                st.bar_chart(cluster_counts)

                # Show segment characteristics
                st.subheader("Caractéristiques des Segments")
                moyenne_par_segment = data_processed.groupby('Nom_Cluster')[
                    data_processed.columns.drop(['num_ligne', 'Cluster', 'Nom_Cluster'])].mean().round(2)
                st.dataframe(moyenne_par_segment.T.round(1))

                # Download buttons
                st.subheader("Télécharger les Résultats")

                # Download segmented data
                csv = data_processed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les Données Segmentées (CSV)",
                    data=csv,
                    file_name='clients_segmentes.csv',
                    mime='text/csv'
                )

                # Download segment characteristics
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    moyenne_par_segment.to_excel(writer, sheet_name='Caractéristiques Segments')
                    data_processed.describe().round(1).T.to_excel(writer, sheet_name='Stats Descriptives')
                excel_data = output.getvalue()
                st.download_button(
                    label="Télécharger le Rapport Complet (Excel)",
                    data=excel_data,
                    file_name='rapport_segmentation_clients.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

        except Exception as e:
            st.error(f"Une erreur est survenue: {str(e)}")
    else:
        st.info("Veuillez importer un fichier CSV pour commencer")

if __name__ == "__main__":
    main()