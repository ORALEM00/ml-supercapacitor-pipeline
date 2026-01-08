import matplotlib.pyplot as plt 


def residual_errors(y, y_predictions, filename = None):
    """
    Create scatter plot to show residual errors. 

        Parameters:
            y (pd.Series): Actual values
            y_predictions (pd.Series): Predicted values
            filename (str): Optional filename to save the plot
        
        Return: 
            Scatter Plot
    """
    residuals = y - y_predictions
    # Plot residuals
    plt.scatter(y, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()
    

def histogram_errors(y, y_predictions, filename = None):
    """
    Create histogram error plot. 

        Parameters:
            y (pd.Series): Actual values
            y_predictions (pd.Series): Predicted values
        
        Return: 
            Scatter Plot
    """
    residuals = y - y_predictions
    plt.hist(residuals, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residual Errors")
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_predictions(y, y_predictions, filename = None):
    """
    Create scatter plot to show actual values against predictions.

        Parameters:
            y (pd.Series): Actual values
            y_predictions (pd.Series): Predicted values
        
        Return: 
            Scatter Plot
    """
    plt.scatter(y, y_predictions, edgecolor='k', alpha=0.7)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()