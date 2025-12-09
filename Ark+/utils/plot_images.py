import matplotlib.pyplot as plt
import numpy as np

 

def plot_images_grid(inputs, img_paths, texts, rankings, labels, n_cols=3, figsize_per_row=5):
    """
    Plot images in a grid layout with predictions and ground truth labels.
    
    Parameters:
    -----------
    inputs : dict
        Dictionary containing 'pixel_values' with image data
    img_paths : list
        List of image file paths
    texts : list
        List of text predictions
    bests : list
        List of indices for best predictions
    labels_dict : dict
        Dictionary mapping image identifiers to ground truth labels
    n_cols : int, default=3
        Number of columns in the grid
    figsize_per_row : int, default=3
        Height multiplier for each row in the figure
    """
    # Calculate the number of rows needed
    n_imgs = len(img_paths)
    n_rows = (n_imgs + n_cols - 1) // n_cols
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, figsize_per_row*n_rows))
    
    # Handle the case where we only have one row or one column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    labels_dict = labels.set_index('AccessionNumber_anon').to_dict()['Pathology']
    
    # Plot each image
    for i, p in enumerate(img_paths):
        row = i // n_cols
        col = i % n_cols
        
        img = inputs['pixel_values'][i,:,:,:].numpy()
        img = np.transpose(img, (1,2,0))

        acc_anon = p.split('/')[-2]
        
        pred1 = texts[rankings[i,:][-1]]
        pred2 = texts[rankings[i,:][-2]]
        pred3 = texts[rankings[i,:][-3]]
        gt = labels_dict[acc_anon]
        
        axes[row, col].set_title(f'Top1: {pred1}\n Top2: {pred2}\n Top3: {pred3}\n Label: {gt}')
        axes[row, col].imshow(np.round(255*(img - (-1)) / 2.0).astype(int))
        axes[row, col].axis('off')
    
    # Hide any unused subplots
    for i in range(len(img_paths), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes