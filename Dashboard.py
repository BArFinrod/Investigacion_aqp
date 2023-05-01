import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import base64
from PIL import Image
from pathlib import Path


image = Image.open(Path(__file__).parent / 'Logo_blanco.jpeg')
st.image(image, width=150)

st.title("Potencial de vinculación entre los Centros de Investigación y las industrias de la región de Arequipa, Perú")

p1 = Path(__file__).parent / "data_aqp_research_industries_embedding.pickle"
p2 = Path(__file__).parent / "Infografía investigación AQP centros.pdf"

with open(p1,'rb') as f:
    dfciiu_aqp1 = pickle.load(f)
    dfcen = pickle.load(f)
    vis_dims = pickle.load( f)

x = np.array([x for x,y in vis_dims])
y = np.array([y for x,y in vis_dims])
nterms = dfcen.shape[0]
ntables = dfciiu_aqp1.shape[0]


fig1 = go.Figure([go.Scatter(x=x[nterms:], y=y[nterms:], mode='markers', hovertext=[f"<b>{dfciiu_aqp1.loc[row, 'real_name']} </b> <br>Total de ventas en millones de soles (2015): {dfciiu_aqp1.loc[row, 'size_sales_1m']} <br>Número de empresas (2015): {dfciiu_aqp1.loc[row, 'n_empresas']}" for row in dfciiu_aqp1.index], marker=dict(size=dfciiu_aqp1['size_sales_1m']/3,color='steelblue'),
                            hoverinfo='text',name='Sectores industriales en Arequipa'),
                  go.Scatter(x=x[:nterms], y=y[:nterms], mode='markers', text=dfcen['name'], marker={'color':'red','size':20}, hovertemplate ='<b>%{text}</b>', name='Centros de investigación en Arequipa <br> autorizados por CONCYTEC (Ley 30309)')])
fig1.update_layout(
    plot_bgcolor='white',
    width=2000,
    height=1000)

st.plotly_chart(fig1, use_container_width=True)
st.caption('Nota: La cercanía entre industrias y los centros de investigación se calculó utilizando herramientas de inteligencia artificial y análisis semántico (Embedding de OpenAI). Se mide la distancia semántica entre los títulos de los proyectos de investigación de los centros y la descripción completa de las actividades económicas de las empresas. La representación visual se realiza con el algoritmo t-SNE.Fuentes: Se utiliza información actualizada a 2023 de los centros de investigación autorizados bajo la Ley 30309 y los datos de 2015 de la estructura empresarial en Arequipa bajo la clasificación CIIU rev. 3.' , unsafe_allow_html=True)

with open(p2,"rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')

contenido = f"data:application/pdf;base64,{base64_pdf}"
pdf_display = f'<iframe src={contenido} width="800" height="800" type="application/pdf"></iframe>'
st.markdown(pdf_display, unsafe_allow_html=True)
