import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path

# Must be the first Streamlit command
st.set_page_config(page_title='Alloy Calculator', page_icon=':wrench:')
st.title("🔧 Alloy Calculator")

# Load your data (replace this with actual data loading)
@st.cache_data
def load_data():
   # df = pd.read_csv('alloy.csv', index_col=0)  # Assume index is alloy names
    
    DATA_FILENAME = Path(__file__).parent/'alloys.csv'
    df = pd.read_csv(DATA_FILENAME, index_col=0)
    return df, list(df.columns)

df, parameter_names = load_data()

# --- Streamlit UI ---
#.set_page_config(page_title='Alloy Optimizer', page_icon=':wrench:')

target_alloy = st.selectbox("Select Target Alloy", df.index)
fixed_alloys = st.multiselect("Select Fixed Alloys", [a for a in df.index if a != target_alloy])
n_additional = st.number_input("Number of Additional Contributing Alloys", min_value=1, max_value=10, value=2, step=1)

if st.button("Run Optimization"):

    target_vector = df.loc[target_alloy].values
    candidate_alloys = [a for a in df.index if a not in fixed_alloys and a != target_alloy]

    # Step 1: Global optimization
    X_fixed = df.loc[fixed_alloys].values.T if fixed_alloys else np.empty((df.shape[1], 0))
    X_candidates = df.loc[candidate_alloys].values.T
    X_full = np.hstack([X_fixed, X_candidates])

    num_fixed = len(fixed_alloys)
    num_candidates = len(candidate_alloys)

    x0 = np.ones(num_fixed + num_candidates) / (num_fixed + num_candidates)
    bounds = [(0, 1)] * (num_fixed + num_candidates)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    def loss(w):
        return np.linalg.norm(X_full @ w - target_vector)

    result = minimize(loss, x0, bounds=bounds, constraints=constraints)

    if not result.success:
        st.error("Initial optimization failed.")
    else:
        full_weights = result.x
        candidate_weights = full_weights[num_fixed:]
        sorted_indices = np.argsort(candidate_weights)[::-1]
        sorted_candidates = [candidate_alloys[i] for i in sorted_indices]

        # Step 2: Iterative selection
        selected_alloys = []
        i = 0
        while len(selected_alloys) < n_additional and i < len(sorted_candidates):
            trial_alloys = fixed_alloys + selected_alloys + [sorted_candidates[i]]
            X_trial = df.loc[trial_alloys].values.T
            x0_trial = np.ones(len(trial_alloys)) / len(trial_alloys)
            bounds_trial = [(0, 1)] * len(trial_alloys)
            constraints_trial = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

            def loss_trial(w):
                return np.linalg.norm(X_trial @ w - target_vector)

            result_trial = minimize(loss_trial, x0_trial, bounds=bounds_trial, constraints=constraints_trial)

            if result_trial.success and result_trial.x[-1] > 1e-4:
                selected_alloys.append(sorted_candidates[i])
            i += 1

        # Final optimization
        final_alloys = fixed_alloys + selected_alloys
        X_final = df.loc[final_alloys].values.T
        x0_final = np.ones(len(final_alloys)) / len(final_alloys)
        bounds_final = [(0, 1)] * len(final_alloys)
        constraints_final = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        def loss_final(w):
            return np.linalg.norm(X_final @ w - target_vector)

        result_final = minimize(loss_final, x0_final, bounds=bounds_final, constraints=constraints_final)

        if result_final.success:
            final_weights = result_final.x
            combined_vector = X_final @ final_weights

            st.success("Optimization successful!")
            st.subheader(f"Final Alloy Composition for {target_alloy}")
            for a, w in zip(final_alloys, final_weights):
                st.write(f"**{a}**: {w:.4f}")

            # Plot
            actual = target_vector
            calculated = combined_vector
            delta = np.abs(actual - calculated)

            exclude_param = "Al"
            indices_keep = [i for i, p in enumerate(parameter_names) if p != exclude_param]
            params_filtered = [parameter_names[i] for i in indices_keep]
            actual_filtered = [actual[i] for i in indices_keep]
            calculated_filtered = [calculated[i] for i in indices_keep]
            delta_filtered = [delta[i] for i in indices_keep]

            fig, ax = plt.subplots(figsize=(12, 6))
            bar_width = 0.35
            indices = np.arange(len(params_filtered))

            ax.bar(indices, actual_filtered, bar_width, label='Actual', color='skyblue')
            ax.bar(indices + bar_width, calculated_filtered, bar_width, label='Calculated', color='lightcoral')
            max_val = max(max(actual_filtered), max(calculated_filtered))
            ax.set_ylim(0, max_val * 1.15)

            for i in range(len(params_filtered)):
                y_pos = max(actual_filtered[i], calculated_filtered[i]) + max_val * 0.03
                ax.text(indices[i] + bar_width / 2, y_pos, f"Δ={delta_filtered[i]:.3f}",
                        ha='center', va='bottom', fontsize=8, rotation=20)

            ax.set_xticks(indices + bar_width / 2)
            ax.set_xticklabels(params_filtered, rotation=45)
            ax.set_ylabel('Parameter Value')
            ax.set_title(f'Actual vs Calculated for {target_alloy} using {", ".join(final_alloys)}')
            ax.legend()
            st.pyplot(fig)

        else:
            st.error("Final optimization failed.")
