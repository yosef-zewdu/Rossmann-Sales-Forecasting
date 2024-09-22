import matplotlib.pyplot as plt 
import seaborn as sns 




def plot_box_outliers(dff, lower_bounds, upper_bounds, outlier_counts):
    # Create box plots for columns with outliers
    # Prepare to create box plots for columns with outliers
    num_columns = len([column for column, count in outlier_counts.items() if count > 0])
    num_rows = (num_columns + 1) // 2  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 5))  # Create subplots

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Create box plots for columns with outliers
    plot_index = 0
    for column, count in outlier_counts.items():
        if count > 0:  # Only plot columns with outliers
            sns.boxplot(x=dff[column], ax=axes[plot_index], color='skyblue')
            
            # Adding lines for upper and lower bounds
            axes[plot_index].axvline(x=upper_bounds[column], color='r', linestyle='--', label='Upper Bound')
            axes[plot_index].axvline(x=lower_bounds[column], color='g', linestyle='--', label='Lower Bound')
            
            axes[plot_index].set_title(f'Box Plot of {column} with Outlier Bounds')
            axes[plot_index].set_xlabel(column)
            axes[plot_index].legend()
            
            plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()





def plot_scatter_outliers(dff, lower_bounds, upper_bounds, outlier_counts):
    # Prepare to create scatter plots for columns with outliers
    num_columns = len([column for column, count in outlier_counts.items() if count > 0])
    num_rows = (num_columns + 1) // 2  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 5))  # Create subplots

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Create scatter plots for columns with outliers
    plot_index = 0
    for column, count in outlier_counts.items():
        if count > 0:  # Only plot columns with outliers
            axes[plot_index].scatter(dff.index, dff[column], color='skyblue', label='Data Points')
            
            # Adding lines for upper and lower bounds
            axes[plot_index].axhline(y=upper_bounds[column], color='r', linestyle='--', label='Upper Bound')
            axes[plot_index].axhline(y=lower_bounds[column], color='g', linestyle='--', label='Lower Bound')
            
            axes[plot_index].set_title(f'Scatter Plot of {column} with Outlier Bounds')
            axes[plot_index].set_xlabel('Index')
            axes[plot_index].set_ylabel(column)
            axes[plot_index].legend()
            
            plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
