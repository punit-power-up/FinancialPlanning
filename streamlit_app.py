import streamlit as st #type:ignore
import pandas as pd
from datetime import datetime
import main as logic
import io
import zipfile

def main():

    st.set_page_config(page_title="Financial Planning", layout="wide")

    st.title("🎯 Financial Planning")

    # Initialize session state for goals and effects
    if 'goals' not in st.session_state:
        st.session_state.goals = []
    if 'effects' not in st.session_state:
        st.session_state.effects = []

    # Section 1: Basic Information
    st.header("📊 Basic Information")
    col1, col2 = st.columns(2)

    with col1:
        current_date = st.date_input("Current Date", value=datetime(2025, 12, 23))
        current_corpus = st.number_input("Current Corpus (₹)", value=10000000, step=100000)
        yearly_sip_step_up = st.number_input("Yearly SIP Step-up (%)", value=10.0, step=0.1)

    with col2:
        current_age = st.number_input("Current Age", value=30, step=1)
        current_sip = st.number_input("Current SIP (₹/month)", value=100000, step=1000)

    st.divider()

    # Section 2: Financial Goals
    st.header("🎯 Financial Goals")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Add Goal", use_container_width=True):
            st.session_state.goals.append({
                'id': len(st.session_state.goals),
                'name': '',
                'type': 'Non-Negotiable',
                'maturity_date': datetime(2040, 3, 5),
                'downpayment_present_value': 5000000,
                'rate_for_future_value': 6.0
            })
            st.rerun()

    if len(st.session_state.goals) == 0:
        st.info("No goals added yet. Click 'Add Goal' to get started.")
    else:
        for i, goal in enumerate(st.session_state.goals):
            with st.expander(f"Goal {i+1}: {goal['name'] if goal['name'] else 'Unnamed'}", expanded=True):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    goal['name'] = st.text_input(
                        "Goal Name", 
                        value=goal['name'], 
                        key=f"goal_name_{i}",
                        placeholder="e.g., Home, Child Education"
                    )
                    goal['downpayment_present_value'] = st.number_input(
                        "Present Value (₹)", 
                        value=goal['downpayment_present_value'], 
                        step=100000,
                        key=f"goal_pv_{i}"
                    )
                
                with col2:
                    goal['type'] = st.selectbox(
                        "Goal Type", 
                        options=['Non-Negotiable', 'Negotiable', 'Semi-Negotiable'],
                        index=['Non-Negotiable', 'Negotiable', 'Semi-Negotiable'].index(goal['type']),
                        key=f"goal_type_{i}"
                    )
                    goal['rate_for_future_value'] = st.number_input(
                        "Inflation Rate (%)", 
                        value=goal['rate_for_future_value'], 
                        step=0.1,
                        key=f"goal_rate_{i}"
                    )
                
                with col3:
                    goal['maturity_date'] = st.date_input(
                        "Maturity Date", 
                        value=goal['maturity_date'],
                        min_value=pd.Timestamp(1900, 1, 1),
                        max_value=pd.Timestamp(2200, 12, 31),
                        key=f"goal_date_{i}"
                    )
                    if st.button("🗑️ Remove", key=f"remove_goal_{i}", use_container_width=True):
                        st.session_state.goals.pop(i)
                        st.rerun()

    st.divider()

    # Section 3: Effects on Cashflows
    st.header("💰 Effects on Cashflows")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Add Cashflow Effect", use_container_width=True):
            st.session_state.effects.append({
                'id': len(st.session_state.effects),
                'start_date': datetime(2030, 1, 1),
                'end_date': datetime(2050, 1, 1),
                'monthly_amount': -30000
            })
            st.rerun()

    if len(st.session_state.effects) == 0:
        st.info("No cashflow effects added yet. Click 'Add Cashflow Effect' to get started.")
    else:
        for i, effect in enumerate(st.session_state.effects):
            with st.expander(f"Cashflow Effect {i+1}", expanded=True):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    effect['start_date'] = st.date_input(
                        "Start Date", 
                        value=effect['start_date'],
                        min_value=pd.Timestamp(1900, 1, 1),
                        max_value=pd.Timestamp(2200, 12, 31),
                        key=f"effect_start_{i}"
                    )
                
                with col2:
                    effect['end_date'] = st.date_input(
                        "End Date", 
                        value=effect['end_date'],
                        min_value=pd.Timestamp(1900, 1, 1),
                        max_value=pd.Timestamp(2200, 12, 31),
                        key=f"effect_end_{i}"
                    )
                
                with col3:
                    effect['monthly_amount'] = st.number_input(
                        "Monthly Amount (₹)", 
                        value=effect['monthly_amount'],
                        step=1000,
                        key=f"effect_amount_{i}",
                        help="Use negative values for outflow"
                    )
                
                with col4:
                    st.write("")  # Spacing
                    if st.button("🗑️ Remove", key=f"remove_effect_{i}", use_container_width=True):
                        st.session_state.effects.pop(i)
                        st.rerun()

    st.divider()

    # Section 4: Instrument Parameters
    st.header("⚙️ Instrument Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hybrid")
        hybrid_return = st.number_input("Return (%)", value=10.0, step=0.1, key="hybrid_return")
        hybrid_tax = st.number_input("Tax (%)", value=12.5, step=0.1, key="hybrid_tax")
        
        st.subheader("Debt")
        debt_return = st.number_input("Return (%)", value=6.0, step=0.1, key="debt_return")
        debt_tax = st.number_input("Tax (%)", value=30.0, step=0.1, key="debt_tax")

    with col2:
        st.subheader("Goal")
        goal_return = st.number_input("Return (%)", value=0.0, step=0.1, key="goal_return")
        goal_tax = st.number_input("Tax (%)", value=0.0, step=0.1, key="goal_tax")
        
        st.subheader("Core Corpus")
        core_return = st.number_input("Return (%)", value=15.0, step=0.1, key="core_return")
        core_tax = st.number_input("Tax (%)", value=12.5, step=0.1, key="core_tax")

    st.divider()

    # Generate Output Button
    if st.button("🚀 Show Simulation Results", type="primary", use_container_width=True):
        
        # Create input_variables dictionary
        input_variables = {
            'current_date': pd.Timestamp(current_date),
            'current_age': int(current_age),
            'current_corpus': int(current_corpus),
            'current_sip': int(current_sip),
            'yearly_sip_step_up_%': float(yearly_sip_step_up),
            'goals': [
                {
                    'name': goal['name'],
                    'type': goal['type'],
                    'maturity_date': pd.Timestamp(goal['maturity_date']),
                    'downpayment_present_value': int(goal['downpayment_present_value']),
                    'rate_for_future_value%': float(goal['rate_for_future_value'])
                }
                for goal in st.session_state.goals
            ],
            'effects_on_cashflows': [
                {
                    'start_date': pd.Timestamp(effect['start_date']),
                    'end_date': pd.Timestamp(effect['end_date']),
                    'monthly_amount': int(effect['monthly_amount'])
                }
                for effect in st.session_state.effects
            ]
        }
        
        # Create instrument_params dictionary (convert percentages to decimals)
        instrument_params = {
            'hybrid': {'return': hybrid_return / 100, 'tax': hybrid_tax / 100},
            'debt': {'return': debt_return / 100, 'tax': debt_tax / 100},
            'goal': {'return': goal_return / 100, 'tax': goal_tax / 100},
            'core_corpus': {'return': core_return / 100, 'tax': core_tax / 100}
        }
        
        # Run simulation
        result = logic.run_simulation(input_variables, instrument_params)
        
        if result:
            if result['status'] == 'error':
                st.error(result['message'])
                if 'sip_df' in result['data']:
                    st.subheader('Please look at the adjustments in cashflows')
                    st.dataframe(result['data']['sip_df'])
            
            elif result['status'] in ['success', 'failure']:
                data = result['data']
                success_metrics = data['success_metrics']
                daily_corpus_value_df = data['daily_corpus_value_df']
                final_trans_df = data['final_trans_df']
                goal_dfs = data['goal_dfs']
                last_goal_date = data['last_goal_date']
                
                if result['status'] == 'success':
                    st.success('All goals met Successfully')
                    
                    st.header(
                        f"Corpus on {last_goal_date.strftime('%d-%b-%Y')} "
                        f"will be {logic.format_inr(daily_corpus_value_df['current_value'].iloc[-1])}"
                    )
                    
                    st.subheader("Daily Corpus Value")
                    st.line_chart(daily_corpus_value_df, x='Date', y='current_value')

                    st.divider()

                    # Create zip file in memory
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zf:
                        # Helper to add df to zip
                        def add_df_to_zip(df, name):
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            zf.writestr(f"{name}.csv", csv_data)

                        add_df_to_zip(daily_corpus_value_df, "daily_corpus_value")
                        add_df_to_zip(final_trans_df, "final_trans_df")
                        add_df_to_zip(data.get('nav_df', pd.DataFrame()), "nav_df")
                        add_df_to_zip(data.get('sip_df', pd.DataFrame()), "sip_df")
                        add_df_to_zip(data.get('sip_trans_df', pd.DataFrame()), "sip_trans_df")
                        add_df_to_zip(data.get('withdrawls_df', pd.DataFrame()), "withdrawls_df")
                        
                        for goal_name, goal_df in goal_dfs.items():
                             add_df_to_zip(goal_df, f"goal_{goal_name}")
                    
                    st.download_button(
                        label="Download All Data (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="simulation_results.zip",
                        mime="application/zip"
                    )

                else:
                    st.error('Goals could not be met')
                    st.header(f"Cashflows started going negative on {success_metrics['simulation_broke_date'].strftime('%d-%b-%Y')}")

                st.subheader('Cashflows in core corpus:')
                st.dataframe(final_trans_df)

                st.subheader('Goal wise cashflows:')
                for goal_name, goal_df in goal_dfs.items():
                    st.write(goal_name)
                    st.dataframe(goal_df)

if __name__ == "__main__":
    main()
