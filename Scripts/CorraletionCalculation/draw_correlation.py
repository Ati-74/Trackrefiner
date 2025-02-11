import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    # Read the data
    correlation_df = pd.read_csv("../../12.correlation/correlation.csv")

    # Define dataset names and types
    dataset_names_dict = {
        'unconstrained_1': ['U1', 'agar'],
        'unconstrained_2': ['U2', 'agar'],
        'unconstrained_3': ['U3', 'agar'],
        'unconstrained_4': ['U4', 'agar'],

        'constrained_1': ['C1', 'microfluidic'],
        'constrained_2': ['C2', 'microfluidic'],
        'constrained_3': ['C3', 'microfluidic'],
        'constrained_4': ['C4', 'microfluidic'],
    }

    for feature in correlation_df['Feature'].unique():
        if feature in ['avg_angular_motion', 'avg_speed movement_center', 'pixel density']:
            # Filter rows for the current feature
            sel_rows = correlation_df.loc[correlation_df['Feature'] == feature].copy()

            # Map additional information
            sel_rows["Name"] = sel_rows["Dataset"].map(lambda x: dataset_names_dict[x][0])
            sel_rows["Type"] = sel_rows["Dataset"].map(lambda x: dataset_names_dict[x][1])

            # Convert p-values to -log10 scale
            sel_rows["-log10(p_value)"] = -np.log10(sel_rows["p_value"])

            # Thresholds
            p_value_threshold = 0.05
            correlation_threshold = 0.75

            # Determine significant points based on thresholds
            sel_rows["Significant"] = ((sel_rows["p_value"] < p_value_threshold) &
                                       (sel_rows["Pearson correlation"].abs() >= correlation_threshold))

            # Set scatter point colors based on significance
            sel_rows["ScatterColor"] = sel_rows["Significant"].map({True: '#9467bd', False: 'black'})
            sel_rows["EdgeColor"] = sel_rows["Significant"].map({True: '#9467bd', False: 'black'})

            # Set label colors based on dataset type
            type_colors = {"agar": "blue", "microfluidic": "#ba8e23"}
            sel_rows["LabelColor"] = sel_rows["Type"].map(type_colors)

            # Plot the volcano plot
            # figsize=(10, 8)
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = plt.scatter(
                sel_rows["Pearson correlation"], sel_rows["-log10(p_value)"],
                c=sel_rows["ScatterColor"],
                alpha=1, edgecolors=sel_rows["EdgeColor"]
            )

            # Add threshold lines
            plt.axhline(-np.log10(p_value_threshold), color='red', linestyle='-', linewidth=5.5,
                        label="P-value threshold (0.05)", alpha=0.5)
            plt.axvline(correlation_threshold, color='green', linestyle='-', linewidth=5.5,
                        label="Correlation threshold (0.75)", alpha=0.5)
            plt.axvline(-correlation_threshold, color='green', linestyle='-', linewidth=5.5, alpha=0.5)

            # Annotate significant points
            for i, row in sel_rows.iterrows():
                # Determine dynamic offsets based on point position
                x_offset = 0.02 if row["Pearson correlation"] >= 0 else -0.08  # Offset outward from the point
                y_offset = 0.02  # Consistent upward offset for clarity

                # Calculate annotation position
                annotation_x = row["Pearson correlation"] * 1.01
                annotation_y = row["-log10(p_value)"] * 0.98

                if feature == 'avg_angular_motion':
                    if row["Name"] in ['C2']:
                        annotation_x = row["Pearson correlation"] * 0.97
                        annotation_y = row["-log10(p_value)"] * 0.98

                    if row["Name"] in ['C1']:
                        annotation_x = row["Pearson correlation"] * 1.02
                        annotation_y = row["-log10(p_value)"] * 0.98

                    if row["Name"] in ['C3']:
                        annotation_x = row["Pearson correlation"] * 0.85
                        annotation_y = row["-log10(p_value)"] * 0.98

                    if row["Name"] in ['C4']:
                        annotation_x = row["Pearson correlation"] * 1.08
                        annotation_y = row["-log10(p_value)"] * 1.08

                if feature == 'avg_speed movement_center':

                    if row["Name"] in ['C2']:
                        annotation_x = row["Pearson correlation"] * 0.985
                        # annotation_y = row["-log10(p_value)"] * 0.98

                    if row["Name"] in ['C1', 'U4']:
                        annotation_x = row["Pearson correlation"] * 1.02
                        # annotation_y = row["-log10(p_value)"] * 0.98

                    if row["Name"] in ['C4']:
                        annotation_x = row["Pearson correlation"] * 0.93
                        annotation_y = row["-log10(p_value)"] * 0.95

                    if row["Name"] in ['C3']:
                        annotation_x = row["Pearson correlation"] * 1.005
                        annotation_y = row["-log10(p_value)"] * 1

                if feature == 'pixel density':
                    if row["Name"] in ['C1', 'C3']:
                        annotation_x = row["Pearson correlation"] * 0.98
                        # annotation_y = row["-log10(p_value)"] * 0.98

                    if row["Name"] in ['C4']:
                        annotation_x = row["Pearson correlation"] * 0.97
                        # annotation_y = row["-log10(p_value)"] * 0.98

                    if row["Name"] in ['U2', 'U3']:
                        annotation_x = row["Pearson correlation"] * 1.03
                        # annotation_y = row["-log10(p_value)"] * 0.98

                # if row["Significant"]:
                plt.annotate(
                    row["Name"],  # Use the name instead of the dataset key
                    (row["Pearson correlation"], row["-log10(p_value)"]),
                    xytext=(annotation_x, annotation_y),
                    fontsize=16, fontweight='bold', ha='left', color=row["LabelColor"]  # Annotation color based on type
                )

            # Add labels and title
            plt.xlabel("Pearson correlation coefficient", fontsize=20, labelpad=15, fontweight='bold')
            plt.ylabel(r'$-\log_{10}{(p\mathrm{-value})}$', fontsize=20, labelpad=15, fontweight='bold')
            # plt.title(f"{feature}", fontsize=16)
            plt.title("")
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=18)
            plt.tight_layout()

            # Adjust x-axis range
            plt.xlim(sel_rows["Pearson correlation"].min() - 0.1, sel_rows["Pearson correlation"].max() + 0.1)

            # Save the plot to a file
            plt.savefig(f'../../12.correlation/{feature}.jpg', dpi=600)
            plt.clf()
            plt.close()
