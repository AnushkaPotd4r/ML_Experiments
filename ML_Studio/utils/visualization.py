"""
Visualization utilities for ML Studio.
Generates academic-style, publication-quality Matplotlib figures encoded as Base64 strings,
ready for direct embedding into HTML templates.
"""

import io
import base64
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Apply clean academic palette
PALETTE = {
    'primary': '#1d4ed8',     # Royal blue
    'secondary': '#0f172a',   # Navy dark
    'accent': '#0284c7',      # Light blue
    'success': '#16a34a',     # Green
    'danger': '#dc2626',      # Red
    'grid': '#e2e8f0',        # Subtle slate grid
    'bg': '#ffffff',          # Pure white
    'card_bg': '#f8fafc'      # Slate 50
}


def fig_to_base64(fig):
    """Convert Matplotlib figure to a base64 encoded PNG data URI string."""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def plot_slr(X_train, y_train, X_test, y_test, model, user_input=None, user_pred=None):
    """
    Plot Simple Linear Regression: Data points, Fitted Regression Line, and predicted query point.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    # Train points
    ax.scatter(X_train, y_train, color='#94a3b8', alpha=0.6, label='Training Data', s=35, edgecolors='none')
    # Test points
    ax.scatter(X_test, y_test, color='#1d4ed8', alpha=0.85, label='Test Data', s=45, edgecolors='#0f172a', linewidth=0.5)

    # Line of best fit
    x_range = np.linspace(min(X_train.min(), X_test.min()) - 0.2, max(X_train.max(), X_test.max()) + 0.2, 200).reshape(-1, 1)
    y_line = model.predict(x_range)
    ax.plot(x_range, y_line, color='#dc2626', linewidth=2.5, label='Regression Line (Best Fit)')

    # User input query point
    if user_input is not None and user_pred is not None:
        ax.scatter([user_input], [user_pred], color='#16a34a', s=140, zorder=5,
                   label=f'Your Query ({user_input}, {user_pred:.2f} LPA)', marker='*', edgecolors='#0f172a', linewidth=1.2)
        # Drop lines to axes
        ax.vlines(x=user_input, ymin=ax.get_ylim()[0], ymax=user_pred, color='#16a34a', linestyle='--', alpha=0.6)
        ax.hlines(y=user_pred, xmin=ax.get_xlim()[0], xmax=user_input, color='#16a34a', linestyle='--', alpha=0.6)

    ax.set_title('Simple Linear Regression: CGPA vs Package (LPA)', fontsize=13, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel('CGPA (Cumulative Grade Point Average)', fontsize=11, color='#334155', labelpad=8)
    ax.set_ylabel('Package (LPA in ₹ Lakhs)', fontsize=11, color='#334155', labelpad=8)
    ax.grid(True, linestyle='--', alpha=0.5, color=PALETTE['grid'])
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)
    
    return fig_to_base64(fig)


def plot_mlr_performance(y_test, y_pred, feature_names=None, coefs=None):
    """
    Plot Multiple Linear Regression: Actual vs Predicted + Coefficients Bar Chart.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # 1. Actual vs Predicted Scatter
    ax1.scatter(y_test, y_pred, color='#2563eb', alpha=0.75, s=40, edgecolors='#1e293b', linewidth=0.5)
    min_val = min(min(y_test), min(y_pred)) - 1
    max_val = max(max(y_test), max(y_pred)) + 1
    ax1.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Fit Line (y = x)')
    ax1.set_title('Actual vs Predicted Package (LPA)', fontsize=12, fontweight='bold', color='#0f172a', pad=10)
    ax1.set_xlabel('Actual Package (LPA)', fontsize=10.5, color='#334155')
    ax1.set_ylabel('Predicted Package (LPA)', fontsize=10.5, color='#334155')
    ax1.grid(True, linestyle='--', alpha=0.5, color=PALETTE['grid'])
    ax1.legend(frameon=True, fontsize=9)

    # 2. Model Coefficients
    if feature_names and coefs is not None:
        colors = ['#1d4ed8' if c >= 0 else '#dc2626' for c in coefs]
        bars = ax2.barh(feature_names, coefs, color=colors, height=0.5, edgecolor='#0f172a', linewidth=0.5)
        ax2.axvline(0, color='#64748b', linewidth=1)
        ax2.set_title('Feature Coefficients (Impact on Package)', fontsize=12, fontweight='bold', color='#0f172a', pad=10)
        ax2.set_xlabel('Coefficient Weight (β)', fontsize=10.5, color='#334155')
        ax2.grid(True, linestyle='--', alpha=0.5, color=PALETTE['grid'])
        for bar in bars:
            val = bar.get_width()
            ax2.text(val + (0.05 if val >= 0 else -0.2), bar.get_y() + 0.15, f"{val:.3f}", fontsize=9, fontweight='bold', color='#0f172a')

    return fig_to_base64(fig)


def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """Generate a clean academic heatmap for a confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                cbar=False, ax=ax, annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_title(title, fontsize=12, fontweight='bold', color='#0f172a', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=11, color='#334155', labelpad=8)
    ax.set_ylabel('True Label', fontsize=11, color='#334155', labelpad=8)
    return fig_to_base64(fig)


def plot_binary_decision_boundary(model, X, y, scaler, feature_names, title='Logistic Regression Decision Boundary', query_point=None):
    """
    Plot 2D Logistic Regression Decision Boundary with probabilities and optional user query.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Grid for contour
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    grid = np.c_[xx.ravel(), yy.ravel()]
    if scaler:
        grid_scaled = scaler.transform(grid)
    else:
        grid_scaled = grid

    probs = model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

    # Plot probability contour
    contour = ax.contourf(xx, yy, probs, levels=20, cmap='RdYlBu_r', alpha=0.35)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Placement Probability P(Y=1)', fontsize=10, color='#334155')

    # Plot 0.5 decision boundary
    ax.contour(xx, yy, probs, levels=[0.5], colors='#0f172a', linewidths=2.5, linestyles='--')

    # Scatter points
    scatter0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], color='#dc2626', label='Not Placed', s=35, edgecolors='#7f1d1d', alpha=0.8)
    scatter1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], color='#16a34a', label='Placed', s=35, edgecolors='#14532d', alpha=0.8)

    # Query point
    if query_point is not None:
        ax.scatter([query_point[0]], [query_point[1]], color='#f59e0b', s=150, marker='*',
                   label='Your Query Point', edgecolors='#0f172a', linewidth=1.5, zorder=6)

    ax.set_title(title, fontsize=12, fontweight='bold', color='#0f172a', pad=10)
    ax.set_xlabel(feature_names[0], fontsize=10.5, color='#334155')
    ax.set_ylabel(feature_names[1], fontsize=10.5, color='#334155')
    ax.grid(True, linestyle='--', alpha=0.4, color=PALETTE['grid'])
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', loc='upper left', fontsize=9.5)

    return fig_to_base64(fig)


def plot_probabilities_bar(classes, probabilities, title='Class Probability Distribution'):
    """Plot bar chart of predicted probabilities across multiple classes."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ['#dc2626', '#f59e0b', '#16a34a', '#2563eb', '#9333ea'][:len(classes)]
    bars = ax.bar(classes, probabilities * 100, color=colors, width=0.5, edgecolor='#0f172a', linewidth=0.7)

    ax.set_title(title, fontsize=12, fontweight='bold', color='#0f172a', pad=10)
    ax.set_ylabel('Probability (%)', fontsize=10.5, color='#334155')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y', color=PALETTE['grid'])

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    return fig_to_base64(fig)


def plot_decision_tree_structure(model, feature_names, class_names):
    """Generate high-resolution graphical decision tree diagram."""
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=False,
        precision=2,
        fontsize=9,
        ax=ax
    )
    ax.set_title('Trained Decision Tree Architecture', fontsize=14, fontweight='bold', color='#0f172a', pad=12)
    return fig_to_base64(fig)


def plot_feature_importance(feature_names, importances, title='Feature Importance Ranking'):
    """Plot feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]

    bars = ax.barh(sorted_features, sorted_importances, color='#1d4ed8', height=0.55, edgecolor='#0f172a', linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight='bold', color='#0f172a', pad=10)
    ax.set_xlabel('Relative Importance Weight', fontsize=10.5, color='#334155')
    ax.grid(True, linestyle='--', alpha=0.4, axis='x', color=PALETTE['grid'])

    for bar in bars:
        val = bar.get_width()
        ax.text(val + 0.01, bar.get_y() + 0.15, f"{val:.3f}", fontsize=9, fontweight='bold', color='#0f172a')

    return fig_to_base64(fig)


def plot_knn_k_analysis(k_values, accuracies, selected_k):
    """Plot K-value vs Validation Accuracy curve for KNN."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_values, accuracies, marker='o', color='#1d4ed8', linewidth=2.2, markersize=6, label='Cross-Val / Test Accuracy')

    # Highlight chosen k
    if selected_k in k_values:
        idx = k_values.index(selected_k)
        ax.scatter([selected_k], [accuracies[idx]], color='#dc2626', s=130, zorder=5,
                   label=f'Selected K = {selected_k} (Acc: {accuracies[idx]*100:.1f}%)', edgecolors='#0f172a', linewidth=1.2)

    ax.set_title('KNN Performance: K-Value vs Model Accuracy', fontsize=12, fontweight='bold', color='#0f172a', pad=10)
    ax.set_xlabel('Number of Neighbors (K)', fontsize=10.5, color='#334155')
    ax.set_ylabel('Accuracy Score', fontsize=10.5, color='#334155')
    ax.set_xticks(k_values)
    ax.grid(True, linestyle='--', alpha=0.5, color=PALETTE['grid'])
    ax.legend(frameon=True, fontsize=9.5)

    return fig_to_base64(fig)


def plot_svm_decision_boundary(model, X, y, feature_names, title='SVM Decision Boundary & Margins', is_linear=True, query_point=None):
    """
    Plot 2D SVM decision boundary, support vectors, and classification regions.
    Works for both Linear SVM (with explicit margin lines) and Non-Linear SVM (RBF/Poly contours).
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    grid = np.c_[xx.ravel(), yy.ravel()]
    # Decision function value (distance to hyperplane)
    Z = model.decision_function(grid).reshape(xx.shape)

    # Fill regions
    ax.contourf(xx, yy, Z > 0, alpha=0.22, cmap='coolwarm')

    # Plot decision boundary (Z = 0) and margins (Z = -1, +1)
    if is_linear:
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors=['#475569', '#0f172a', '#475569'], linewidths=[1.5, 2.5, 1.5])
    else:
        ax.contour(xx, yy, Z, levels=[0], linestyles=['-'], colors=['#0f172a'], linewidths=[2.5])
        ax.contour(xx, yy, Z, levels=[-1, 1], linestyles=['--'], colors=['#64748b'], linewidths=[1.2])

    # Plot sample data points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color='#dc2626', label='Class 0 / Rejected', s=35, edgecolors='#7f1d1d', alpha=0.85)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='#2563eb', label='Class 1 / Admitted', s=35, edgecolors='#1e3a8a', alpha=0.85)

    # Highlight Support Vectors
    if hasattr(model, 'support_vectors_') and len(model.support_vectors_) > 0:
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=110, facecolors='none', edgecolors='#0f172a', linewidth=1.5,
                   label=f'Support Vectors (N={len(sv)})')

    # Query point
    if query_point is not None:
        ax.scatter([query_point[0]], [query_point[1]], color='#16a34a', s=160, marker='*',
                   label='Query Point', edgecolors='#0f172a', linewidth=1.5, zorder=7)

    ax.set_title(title, fontsize=12, fontweight='bold', color='#0f172a', pad=10)
    ax.set_xlabel(feature_names[0], fontsize=10.5, color='#334155')
    ax.set_ylabel(feature_names[1], fontsize=10.5, color='#334155')
    ax.grid(True, linestyle='--', alpha=0.4, color=PALETTE['grid'])
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', loc='lower right', fontsize=9)

    return fig_to_base64(fig)
