#!/usr/bin/env python
# coding: utf-8

"""
Plot Partial CMI Heatmap

This script loads available mutual information data and generates a partial heatmap
even when some computations are still missing. 

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def detect_n_t_from_files():
    """Automatically detect n_t from available files"""
    # Look for pattern in existing files
    import glob
    pattern = './results_cmi_padding_k1/Padding_mi_AB_*_in_*.npy'
    files = glob.glob(pattern)
    
    if not files:
        # Default fallback
        print("No mi_AB files found, using default n_t=10")
        return 10
    
    # Extract n_t from filename pattern
    n_t_values = []
    for file in files:
        # Extract the "in_X" part
        try:
            parts = file.split('_in_')
            if len(parts) >= 2:
                n_t_part = parts[1].split('.npy')[0]
                n_t_values.append(int(n_t_part))
        except:
            continue
    
    if n_t_values:
        n_t = max(n_t_values)  # Take the maximum found
        print(f"Auto-detected n_t = {n_t} from existing files")
        return n_t
    else:
        print("Could not detect n_t from files, using default n_t=10")
        return 10

def check_available_data():
    """Check what data files are available and report status"""
    print("=== CHECKING AVAILABLE DATA ===")
    
    # Check mi_ABC_array
    filename_ABC = './results_cmi/mi_ABC_array.npy'
    if os.path.exists(filename_ABC):
        mi_ABC_array = np.load(filename_ABC)
        print(f"✓ Found mi_ABC_array.npy with shape: {mi_ABC_array.shape}")
        non_zero_count = np.count_nonzero(mi_ABC_array)
        total_count = mi_ABC_array.size
        print(f"  → {non_zero_count}/{total_count} values computed")
        
        # Auto-detect n_t from the array shape
        n_t_from_array = mi_ABC_array.shape[0] - 1
        print(f"  → Detected n_t = {n_t_from_array} from mi_ABC_array shape")
    else:
        raise FileNotFoundError(f"Required file {filename_ABC} not found.")
    
    # Auto-detect n_t from files (as backup)
    n_t_from_files = detect_n_t_from_files()
    
    # Use the most reliable source
    n_t = n_t_from_array if 'mi_ABC_array' in locals() else n_t_from_files
    
    # Check mi_AB files
    available_tn = []
    missing_tn = []
    
    for tn in range(n_t + 1):
        filename_AB = f'./results_cmi_padding_k1/Padding_mi_AB_{tn}_in_{n_t}.npy'
        if os.path.exists(filename_AB):
            available_tn.append(tn)
        else:
            missing_tn.append(tn)
    
    print(f"✓ Found {len(available_tn)}/{n_t+1} mi_AB files")
    print(f"  → Available tn values: {available_tn}")
    print(f"  → Missing tn values: {missing_tn}")
    
    print("==============================")
    
    return mi_ABC_array, available_tn, missing_tn, n_t

def plot_partial_heatmap():
    """Generate partial CMI heatmap with available data"""
    
    # Check available data and auto-detect parameters
    mi_ABC_array, available_tn, missing_tn, n_t = check_available_data()
    
    # Calculate other parameters
    Ls = 28  # This is fixed in the original code
    max_r = (Ls - 1) // 2 - 1
    
    tn_list = np.array(range(0, n_t + 1))
    alpha_list = tn_list / n_t
    r_list = np.array(range(1, max_r + 1))
    
    print(f"Parameters: n_t={n_t}, max_r={max_r}")
    print(f"tn_list: {tn_list}")
    print(f"r_list: {r_list}")
    print()
    
    # Load mi_ABC_array
    mi_ABC_list_store = np.max(mi_ABC_array, axis=1)
    print(f"mi_ABC values: {mi_ABC_list_store}")
    print()
    
    # Detect maximum available time step to avoid blank rows
    if not available_tn:
        raise ValueError("No available time steps found!")
    
    max_available_tn = max(available_tn)
    print(f"Maximum available tn: {max_available_tn}")
    print(f"Creating CMI array with shape: ({max_available_tn + 1}, {max_r})")
    
    # Initialize CMI array only up to max available tn (no blank rows)
    cmi_array = np.full((max_available_tn + 1, max_r), np.nan)
    
    # Create mapping from available tn to array indices
    available_tn_array = np.array(sorted(available_tn))
    
    # Load available mi_AB data and compute CMI
    data_loaded = 0
    for tn in available_tn:
        filename_AB = f'./results_cmi_padding_k1/Padding_mi_AB_{tn}_in_{n_t}.npy'
        try:
            mi_AB_r_store = np.load(filename_AB)
            print(f"Loaded tn={tn}: mi_AB values = {mi_AB_r_store}")
            
            # Compute CMI for all r values for this tn
            for r_idx, r in enumerate(r_list):
                cmi_value = mi_ABC_list_store[tn] - mi_AB_r_store[r-1]
                # cmi_array[tn, r_idx] = cmi_value
                cmi_array[tn, r_idx] = max(0, cmi_value)  # Clip negative values
            
            data_loaded += 1
            
        except Exception as e:
            print(f"Error loading {filename_AB}: {e}")
    
    print(f"\nSuccessfully loaded data for {data_loaded} tn values")
    print(f"CMI array shape: {cmi_array.shape}")
    print(f"Non-NaN values: {np.count_nonzero(~np.isnan(cmi_array))}")
    
    # Create output directory
    output_dir = './visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    
    # Normalize time axis: 0 to 1 (where 1 corresponds to max_available_tn/n_t)
    max_normalized_time = max_available_tn / n_t  # Normalize to [0, 1]
    
    # Use a colormap that handles NaN values
    im = plt.imshow(cmi_array, aspect='auto', origin='lower', 
                    extent=[min(r_list)-0.5, max(r_list)+0.5, 0, max_normalized_time],
                    cmap='viridis', interpolation='nearest')
    
    plt.colorbar(im, label='CMI')
    plt.xlabel(r'Neighborhood Radius $r$')
    plt.ylabel(r'Time $t$')
    # plt.title(f'Partial CMI Heatmap ({len(available_tn)}/{n_t+1} time points available)')
    plt.xticks(r_list)
    
    # # Add text annotations to show missing data
    # if missing_tn:
    #     for tn in missing_tn:
    #         t_value = tn / n_t
    #         plt.axhline(y=t_value, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
    
    # # Add legend for missing data
    # if missing_tn:
    #     from matplotlib.lines import Line2D
    #     legend_elements = [
    #         Line2D([0], [0], color='red', linestyle='--', alpha=0.5, 
    #                label=f'Missing data (tn={missing_tn})'),
    #         Line2D([0], [0], color='none', label=f'Available: {len(available_tn)}/{n_t+1} time points')
    #     ]
    #     plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # Save the plot (always save to visualization directory)
    output_file = f'{output_dir}/partial_cmi_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved partial heatmap to: {output_file}")
    
    # Also save as PDF for better quality
    output_file_pdf = f'{output_dir}/partial_cmi_heatmap.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved partial heatmap to: {output_file_pdf}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    valid_cmi = cmi_array[~np.isnan(cmi_array)]
    if len(valid_cmi) > 0:
        print(f"Available CMI values: {len(valid_cmi)}")
        print(f"CMI range: [{np.min(valid_cmi):.4f}, {np.max(valid_cmi):.4f}]")
        print(f"CMI mean: {np.mean(valid_cmi):.4f}")
        print(f"CMI std: {np.std(valid_cmi):.4f}")
    else:
        print("No valid CMI values found!")
    
    # Show completion percentage
    total_possible_for_available = (max_available_tn + 1) * max_r
    computed_values = np.count_nonzero(~np.isnan(cmi_array))
    completion_pct = 100 * computed_values / total_possible_for_available
    print(f"Completion (up to tn={max_available_tn}): {computed_values}/{total_possible_for_available} ({completion_pct:.1f}%)")
    
    return cmi_array, available_tn, missing_tn, n_t, max_available_tn

def plot_available_data_overview():
    """Create a simple overview plot showing which data is available"""
    # Auto-detect n_t
    _, available_tn, missing_tn, n_t = check_available_data()
    
    # Create availability plot
    plt.figure(figsize=(8, 3))
    
    # Create binary availability array
    availability = np.zeros(n_t + 1)
    availability[available_tn] = 1
    
    plt.bar(range(n_t + 1), availability, color=['green' if x == 1 else 'red' for x in availability])
    plt.xlabel('Time step (tn)')
    plt.ylabel('Data Available')
    plt.title(f'Data Availability Status (n_t={n_t})')
    plt.xticks(range(n_t + 1))
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Add text labels
    for tn in range(n_t + 1):
        status = 'Available' if tn in available_tn else 'Missing'
        color = 'white' if tn in available_tn else 'white'
        plt.text(tn, 0.5, status, ha='center', va='center', 
                rotation=90, fontsize=8, color=color, weight='bold')
    
    plt.tight_layout()
    
    # Always save to visualization directory
    output_dir = './visualization'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/data_availability.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved data availability plot to: {output_file}")
    
    plt.show()
    
    return available_tn, missing_tn, n_t

if __name__ == "__main__":
    print("Partial CMI Heatmap Generator")
    print("============================")
    
    try:
        # First show data availability
        available_tn, missing_tn, n_t = plot_available_data_overview()
        print()
        
        # Then generate the partial heatmap
        cmi_array, available_tn, missing_tn, n_t, max_available_tn = plot_partial_heatmap()
        
        print("\n=== NEXT STEPS ===")
        print(f"Heatmap shows data up to tn={max_available_tn} (normalized time = {max_available_tn/n_t:.3f})")
        if missing_tn:
            print(f"To complete the heatmap, you need to finish computing:")
            for tn in missing_tn:
                print(f"  - Padding_mi_AB_{tn}_in_{n_t}.npy")
            print(f"\nYou can continue the main computation by running:")
            print(f"  python mnist_cmi_heatmap.py")
            print(f"  (with CONTINUE=True)")
        else:
            print("All data is available! You can generate the complete heatmap.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have run the main computation script first.")
