import streamlit as st
st.set_page_config(page_title="Façade Design AHP Ranking", layout="wide")

import pandas as pd
import numpy as np
import os
import plotly.express as px
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# --- Run Optimization ---
def run_optimization():
    df = pd.read_csv("JoyceStreet_data.csv")
    objectives = ['N_EUI', 'N_illuminace']
    df = df.dropna(subset=objectives)
    F = df[objectives].copy()
    F['N_illuminace'] = 1 - F['N_illuminace']
    F_matrix = F.to_numpy()

    class EvaluatedProblem(Problem):
        def __init__(self, F):
            super().__init__(n_var=1, n_obj=F.shape[1], n_constr=0, xl=0, xu=1)
            self.F = F

        def _evaluate(self, x, out, *args, **kwargs):
            idx = np.clip((x[:, 0] * (self.F.shape[0] - 1)).  astype(int), 0, self.F.shape[0]-1)
            out["F"] = self.F[idx]

    problem = EvaluatedProblem(F_matrix)
    algorithm = NSGA2(pop_size=100)
    termination = get_termination("n_gen", 100)

    res = minimize(problem, algorithm, termination, seed=1, save_history=False, verbose=False)
    indices = [int(ind.X[0] * (F_matrix.shape[0] - 1)) for ind in res.pop]
    pareto_df = df.iloc[indices].reset_index(drop=True)
    pareto_df.to_csv("pareto_front.csv", index=False)

    nds = NonDominatedSorting()
    fronts = nds.do(F_matrix, only_non_dominated_front=False)
    for i, front in enumerate(fronts[1:6], start=2):
        front_df = df.iloc[front].drop_duplicates().reset_index(drop=True)
        front_df.to_csv(f"elitist_front_{i}.csv", index=False)

# --- Streamlit UI ---
st.title("📊 Façade Design Ranking Using AHP")
st.write("Upload or use optimization results to rank design alternatives using adjustable AHP weights.")

@st.cache_data
def load_data():
    pareto_df = pd.read_csv("pareto_front.csv")
    pareto_df['Front'] = 1
    all_fronts = [pareto_df]
    for i in range(2, 7):
        path = f"elitist_front_{i}.csv"
        if os.path.exists(path):
            df_i = pd.read_csv(path)
            df_i['Front'] = i
            all_fronts.append(df_i)
    full_df = pd.concat(all_fronts, ignore_index=True)
    return full_df

# Load dataset
if st.session_state.get('just_optimized'):
    del st.session_state['just_optimized']
df = load_data()
normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
df['norm_EUI'] = normalize(df['N_EUI'])
df['norm_Illuminance'] = normalize(df['N_illuminace'])

# Sidebar sliders for AHP weights
st.sidebar.header("🔧 Set AHP Weights")
w_ill = st.sidebar.slider("Illuminance Weight", 0.0, 1.0, 0.5, 0.01)
w_eui = st.sidebar.slider("EUI Weight", 0.0, 1.0, 0.5, 0.01)
total_weight = w_ill + w_eui
if total_weight > 0:
    w_ill /= total_weight
    w_eui /= total_weight

# Value function for sensitivity and ranking
df['Value_Function'] = w_eui * df['norm_EUI'] + w_ill * (1 - df['norm_Illuminance'])

# Sensitivity analysis
input_params = [
    'in:WWR_N', 'in:WWR_W', 'in:WWR_S', 'in:WWR_E',
    'in:Wall_U_Value', 'in:Chi_Value',
    'in:Shading_N', 'in:Shading_W', 'in:Shading_S', 'in:Shading_E',
    'in:Window_U_Value', 'in:SHGC'
]

# Filter to only existing columns
input_params = [col for col in input_params if col in df.columns]

# Drop rows with NaNs in any required columns
required_cols = input_params + ['Value_Function']
sa_df = df.dropna(subset=required_cols)

# Drop constant columns
sa_df = sa_df.loc[:, sa_df.nunique() > 1]

# Check that Value_Function is still valid
if 'Value_Function' not in sa_df.columns or sa_df['Value_Function'].nunique() <= 1:
    st.warning("⚠️ 'Value_Function' is constant or missing. Cannot compute sensitivity.")
    sensitivity = pd.Series(dtype=float)
else:
    # Safe correlation
    valid_inputs = [col for col in input_params if col in sa_df.columns and sa_df[col].nunique() > 1]
    sensitivity = sa_df[valid_inputs].apply(lambda col: col.corr(sa_df['Value_Function'])).abs().sort_values(ascending=False)
    top5_params = sensitivity.head(5)


# AHP scoring and ranking
df['AHP_Score'] = df['Value_Function']

# Sidebar: Select fronts + sensitivity
st.sidebar.header("📂 Select Fronts to Include")
selected_fronts = []
for i in range(1, 7):
    if st.sidebar.checkbox(f"Include Front {i}", value=(i == 1)):
        selected_fronts.append(i)

st.sidebar.header("📈 Top 5 important Design Parameters")
sorted_top5_df = pd.DataFrame({
    'Parameter': top5_params.index.str.replace('in:', '', regex=False),
    'Sensitivity': top5_params.values
}).sort_values(by='Sensitivity', ascending=True)

fig_sens = px.bar(
    sorted_top5_df,
    x='Sensitivity',
    y='Parameter',
    orientation='h',
    text='Sensitivity',
    labels={'Sensitivity': 'Correlation with Value Function', 'Parameter': 'Input Parameter'},
    title=None
)
fig_sens.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_sens.update_layout(
    height=300,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(range=[0, sorted_top5_df['Sensitivity'].max() + 0.05])
)
st.sidebar.plotly_chart(fig_sens, use_container_width=True)

df_ranked = df[df['Front'].isin(selected_fronts)].sort_values("AHP_Score").reset_index(drop=True)
df_ranked['Rank'] = df_ranked.index + 1

# Results table
st.subheader("🏅 Ranked Design Alternatives")
st.dataframe(df_ranked[['Rank', 'in:Run', 'N_EUI', 'N_illuminace', 'AHP_Score', 'Front']].rename(columns={'in:Run': 'Run'}))

# Image grid
st.subheader("🖼️ Design Images")
image_dir = "DSC_CaseStudy2_Output"
image_column = 'img'

cols = st.columns(3)
for i, row in df_ranked.iterrows():
    img_name = row.get(image_column)
    if isinstance(img_name, str):
        img_path = os.path.join(image_dir, img_name)
        with cols[i % 3]:
            st.markdown(f"**Rank {row['Rank']} – {row['in:Run']}**")
            if os.path.exists(img_path):
                try:
                    st.image(img_path, caption=f"Score: {row['AHP_Score']:.3f}", use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Could not display image: {img_name} — {e}")
            else:
                st.warning(f"Image not found: {img_name}")

# Plot NSGA-II Pareto Front and Elitist Fronts
st.subheader("📌 Optimization Fronts (NSGA-II + Elitist Fronts)")
if 'N_EUI' in df.columns and 'N_illuminace' in df.columns:
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    plt.figure(figsize=(12, 8), dpi=150)
    original_df = pd.read_csv("JoyceStreet_data.csv")
    original_df = original_df.dropna(subset=['N_EUI', 'N_illuminace'])
    plt.scatter(original_df['N_illuminace'], original_df['N_EUI'], color='lightgray', label='All Designs', alpha=0.5, s=20, zorder=1)
    for i, color in zip(range(1, 7), colors):
        front_df = df[df['Front'] == i].sort_values('N_illuminace')
        if not front_df.empty:
            if i in selected_fronts:
                plt.plot(front_df['N_illuminace'], front_df['N_EUI'], label=f'Front {i}', color=color, linewidth=2.5)
                plt.scatter(front_df['N_illuminace'], front_df['N_EUI'], edgecolors='black', color=color, marker='s')
    if not df_ranked.empty:
        top_two = df_ranked.iloc[:2]
        star_colors = ['gold', 'darkorange']
        for idx, (i, row) in enumerate(top_two.iterrows()):
            plt.scatter(row['N_illuminace'], row['N_EUI'], color=star_colors[idx], edgecolors='black', marker='*', s=300, linewidths=1.2, label=f'Top {idx+1}')
    plt.xlabel('Normalized Illuminance')
    plt.ylabel('Normalized EUI')
    plt.title('Multiple Elitist Fronts vs NSGA-II Pareto Front')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Parallel Coordinates Plot
    pcp_columns = [
        'in:WWR_N', 'in:WWR_W', 'in:WWR_S', 'in:WWR_E',
        'in:Wall_U_Value', 'in:Chi_Value',
        'in:Shading_N', 'in:Shading_W', 'in:Shading_S', 'in:Shading_E',
        'in:Window_U_Value', 'in:SHGC',
        'out:Illuminance %', 'out:EUI kWh/m2',
        'N_illuminace', 'N_EUI', 'I_EUI'
    ]
    available_pcp_cols = [col for col in pcp_columns if col in df.columns]
    if len(available_pcp_cols) >= 2:
        st.subheader("📈 Parallel Coordinates Plot")
        highlighted = df_ranked.nsmallest(1, 'AHP_Score').copy()
        others = df_ranked[~df_ranked.index.isin(highlighted.index)].copy()
        highlighted['color_val'] = 1
        others['color_val'] = 0
        combined = pd.concat([highlighted, others])
        fig = px.parallel_coordinates(
            combined,
            dimensions=available_pcp_cols,
            color='color_val',
            color_continuous_scale=[[0, 'black'], [1, 'red']],
            range_color=[0, 1],
            labels={col: col for col in available_pcp_cols}
        )
        fig.update_layout(width=2000, height=600, font=dict(color='black'))
        st.plotly_chart(fig, use_container_width=False)
    else:
        st.info("Not enough input/output columns to draw a parallel coordinates plot.")
