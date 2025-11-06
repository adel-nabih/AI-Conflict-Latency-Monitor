"""
analyze_intervention_v36.py

Analyzes the "v36_nudge" experiment.

This script has two critical fixes:
1.  It loads the 'clm_data_v36_nudge.csv' file.
2.  It uses the CORRECT 'epsilon_decay = 0.005' to match
    the experiment, finally fixing the Epsilon plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def get_project_root():
    try: current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src': return os.path.dirname(current_dir)
    return current_dir

def load_data(filename, name):
    """Load the v36 experiment's CSV file."""
    root_dir = get_project_root()
    filepath = os.path.join(root_dir, 'experiments', filename)
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    print(f"Loaded {filename}: {len(df)} records for {name}")
    return df

def analyze_intervention_patterns(df):
    """
    Analyzes when and how the "Nudge" SAI intervened.
    """
    print("\n" + "="*60)
    print("ANALYSIS: 'v3.6 Nudge' SAI Intervention Patterns")
    print("="*60)
    
    total_interventions = df['is_conflict'].sum()
    print(f"Total SAI Interventions (Nudges): {total_interventions}")
    
    # --- *** CRITICAL FIX *** ---
    # Calculate Epsilon using the CORRECT decay rate
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.005 # <-- This now matches the v36 experiment
    # --- *** END FIX *** ---
    
    df['epsilon'] = epsilon_end + (epsilon_start - epsilon_end) * \
                        np.exp(-epsilon_decay * df['episode'])

    # --- Analysis 1: When are interventions happening? ---
    interventions_df = df[df['is_conflict'] == 1]
    
    if len(interventions_df) == 0:
        print("No interventions found.")
    else:
        print(f"\nAverage Epsilon during an Intervention: {interventions_df['epsilon'].mean():.3f}")
        
        # Check if interventions happened during "exploitation" (low epsilon)
        # This is what we EXPECT to see now.
        interventions_during_exploration = interventions_df[interventions_df['epsilon'] > 0.1]
        interventions_during_exploitation = interventions_df[interventions_df['epsilon'] <= 0.1]
        
        print(f"Interventions during EXPLORATION (Epsilon > 0.1): {len(interventions_during_exploration)}")
        print(f"Interventions during EXPLOITATION (Epsilon <= 0.1): {len(interventions_during_exploitation)}")
        
        if len(interventions_during_exploitation) > 0:
            print("\n>>> FINDING: The 'Intentional' SAI worked! It punished the")
            print("    agent's *policy* (during exploitation), not just random luck.")
        else:
            print("\n>>> FINDING: The SAI *only* punished random exploration.")
            print("    This is the 'Cruel Training' problem again.")
        
    # --- Analysis 2: What was the final outcome? ---
    print("\n" + "="*60)
    print("ANALYSIS: Final Policy Outcomes (The Final Test)")
    print("="*60)
    
    successful_defections = df[df['R_internal'] > 0]
    successful_compliances = df[df['R_social'] > 0] # R_social is 500 or 1000
    
    print(f"Total Successful Defections: {len(successful_defections)}")
    print(f"Total Successful Compliances: {len(successful_compliances)}")
    
    if len(successful_defections) == 0 and len(successful_compliances) > 10:
        print("\n*** 🏆 A+ SUCCESS! The 'Nudge' worked! ***")
        print("    The agent learned the 'Compliance' policy and")
        print("    did NOT learn the 'Defection' policy.")
    elif len(successful_defections) > 0:
        print("\n*** ❌ FAILURE! The agent *still* learned to defect. ***")
        print("    This means the -600 'nudge' was not strong enough.")
    else:
        print("\n*** ❌ FAILURE! The agent learned nothing (Lobotomy). ***")
        print("    This means even the 'nudge' created a 'minefield'.")
        
    return df

def plot_interventions_over_time(df):
    """
    Plots SAI interventions and the CORRECT Epsilon on the same graph.
    """
    print("Generating training analysis plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # --- Plot 1: Interventions (LEFT Y-AXIS) vs. Epsilon (RIGHT Y-AXIS) ---
    ax1 = axes[0]
    interventions_per_ep = df.groupby('episode')['is_conflict'].sum()
    rolling_interventions = interventions_per_ep.rolling(window=50).mean()
    
    color = 'tab:red'
    ax1.set_title('v3.6 "Nudge" SAI Interventions vs. Epsilon Decay')
    ax1.set_ylabel('Avg. Interventions (50-ep window)', color=color)
    ax1.plot(rolling_interventions, label='Avg. "Nudge" Interventions', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='x')

    # --- This will now be the CORRECT Epsilon plot ---
    ax2 = ax1.twinx() 
    epsilon_per_ep = df.groupby('episode')['epsilon'].mean() # Get Epsilon for each episode
    
    color = 'tab:blue'
    ax2.set_ylabel('Epsilon (Exploration Rate)', color=color)
    ax2.plot(epsilon_per_ep, label='Epsilon (decay=0.005)', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # --- Plot 2: Reward per Episode (The "Money" Plot) ---
    ax3 = axes[1]
    reward_per_ep = df.groupby('episode')[['R_social', 'R_internal']].sum()
    rolling_social = reward_per_ep['R_social'].rolling(window=50).mean()
    rolling_internal = reward_per_ep['R_internal'].rolling(window=50).mean()
    
    ax3.plot(rolling_social, label='50-Ep Avg Compliance Reward (R_social)', color='green', linewidth=2)
    ax3.plot(rolling_internal, label='50-Ep Avg Defection Reward (R_internal)', color='red', linewidth=2)
    ax3.set_title('Agent Learning Curve (Policy Outcome)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Avg. Reward')
    ax3.legend()
    ax3.grid(True)

    plt.suptitle('Analysis of v36 "Nudge" Intervention Run', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    root_dir = get_project_root()
    plot_path = os.path.join(root_dir, 'experiments', 'v39_carrot_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Intervention analysis plots saved to: {plot_path}")

def main():
    """Main analysis function"""
    intervention_file = 'clm_data_v39_final.csv'
    
    df = load_data(intervention_file, "v39 Nudge Intervention")
    if df is None:
        return

    df_analyzed = analyze_intervention_patterns(df)
    
    if df_analyzed is not None:
        plot_interventions_over_time(df_analyzed)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()