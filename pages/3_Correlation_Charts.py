import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Retail Sales Correlation", layout="wide")

st.markdown("<h1 style='text-align: center;'>Correlation Charts</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size: 24;">

    **Welcome to the Retail Sales Correlation Analysis page.**

    Choose a Product Group on the left to Visualize.

    </div>
    """,
    unsafe_allow_html=True
)


@st.cache_data()
def load_dataset():
    df = pd.read_csv('./pages/clean_retail_data.csv')
    return df


df = load_dataset()

heatmap_cols = df.select_dtypes(include=np.number).columns.tolist()

corr_matrix = df[heatmap_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
corr_lower_triangle = corr_matrix.mask(mask)

if not corr_matrix.empty:
    heatmap = px.imshow(
        corr_lower_triangle,
        text_auto='.3f',
        aspect="equal",
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    ).update_layout(
        height=1000,
        width=1000,
        coloraxis_colorbar=dict(title="Correlation")
    ).update_xaxes(
        tickangle=45
    ).update_layout(
        title={
            'text': "Correlation Heatmap of Numerical Features",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    st.plotly_chart(heatmap, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    * Values close to **+1** (dark blue) indicate a strong positive correlation.
    * Values close to **-1** (dark red) indicate a strong negative correlation.
    * Values close to **0** (white/light color) indicate a weak or no linear correlation.
    """)
else:
    st.warning("Could not compute correlation matrix. Ensure your DataFrame has numerical columns.")

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
numeric_columns.sort()

all_columns = df.columns.tolist()
all_columns.sort()

st.sidebar.header("Scatter Plot Configuration")

x_axis = st.sidebar.selectbox(
    'Select column for X-axis:',
    options=numeric_columns,
    index=0
)

default_y_index = 1 if len(numeric_columns) > 1 else 0
y_axis = st.sidebar.selectbox(
    'Select column for Y-axis:',
    options=numeric_columns,
    index=default_y_index
)

color_option = st.sidebar.selectbox(
    'Optional: Select column for Color:',
    options=[None] + all_columns,
    index=0
)

size_option = st.sidebar.selectbox(
    'Optional: Select column for Size (must be numeric):',
    options=[None] + numeric_columns,
    index=0
)

hover_options = st.sidebar.multiselect(
    'Optional: Select additional columns for hover info:',
    options=all_columns,
    default=[]
)

st.subheader(f'Scatter Plot: {y_axis} vs. {x_axis}')

hover_data_list = hover_options
if df.index.name and df.index.name not in hover_data_list:
    hover_data_list.insert(0, df.index.name)  # Add index name to the start

try:
    fig = px.scatter(
        data_frame=df,
        x=x_axis,
        y=y_axis,
        color=color_option,
        size=size_option,
        hover_name=df.index.name if df.index.name and df.index.name not in hover_options else None,
        hover_data=hover_options,
        title=f'{y_axis} vs. {x_axis}',
        labels={x_axis: f'{x_axis} Label', y_axis: f'{y_axis} Label'}
    ).update_layout(
        xaxis_title=f'{x_axis}',  # Ensure axis titles are set
        yaxis_title=f'{y_axis}',
        legend_title_text=str(color_option) if color_option else ''
    )

    if size_option:
        fig.update_traces(marker=dict(sizemin=3))

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred while generating the plot: {e}")
    st.error("Please check column selections and data types.")

# --- Optional: Show Data Table ---
if st.checkbox("Show selected data table"):
    st.subheader("Data Used in Plot")

    cols_to_show = [x_axis, y_axis]
    if color_option:
        cols_to_show.append(color_option)
    if size_option:
        cols_to_show.append(size_option)
    if hover_options:
        cols_to_show.extend(hover_options)
    cols_to_show = list(dict.fromkeys(cols_to_show))
    st.dataframe(df[cols_to_show])
