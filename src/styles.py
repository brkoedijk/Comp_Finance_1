import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class FinanceStyles:
    """
    A class providing consistent styling for finance-related visualizations.
    
    This class offers a professional, consistent theme for financial charts with
    methods to apply styling to different chart types commonly used in computational
    finance and quantitative analysis.
    """
    
    def __init__(self, style='modern'):
        """
        Initialize the FinanceStyles class with a specific style theme.
        
        Parameters:
        -----------
        style : str
            The style theme to use ('modern', 'dark', 'classic')
        """
        self.style = style
        
        # Common color palettes for financial charts
        self.colors = {
            'modern': ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'positive': '#2ca02c',  # Green for gains
            'negative': '#d62728',  # Red for losses
            'neutral': '#1f77b4',   # Blue for neutral
            'highlight': '#ff7f0e'  # Orange for highlights
        }
        
        # Custom colormap for heatmaps (red-yellow-green)
        self.cmap_ryg = LinearSegmentedColormap.from_list(
            'finance_ryg', 
            ['#d62728', '#ffed6f', '#2ca02c']
        )
        
        # Set default style
        self.set_style()
    
    def set_style(self):
        """Apply the base style to matplotlib."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Common figure settings
        mpl.rcParams['figure.figsize'] = (10, 6)
        mpl.rcParams['figure.dpi'] = 100
        
        # Font settings
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['axes.labelsize'] = 14
        
        # Grid settings
        mpl.rcParams['grid.alpha'] = 0.3
        mpl.rcParams['grid.linestyle'] = '--'
        
        # Legend settings
        mpl.rcParams['legend.framealpha'] = 0.8
        mpl.rcParams['legend.edgecolor'] = 'lightgray'
        mpl.rcParams['legend.fancybox'] = True
        
        # Tick parameters
        mpl.rcParams['xtick.direction'] = 'out'
        mpl.rcParams['ytick.direction'] = 'out'
        mpl.rcParams['xtick.major.size'] = 5.0
        mpl.rcParams['ytick.major.size'] = 5.0
    
    def apply_chart_style(self, ax=None, title=None, xlabel=None, ylabel=None, 
                          legend=True, grid=True):
        """
        Apply consistent styling to a chart.
        
        Parameters:
        -----------
        ax : matplotlib.axes, optional
            The axes to style, defaults to current axes
        title : str, optional
            Chart title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        legend : bool, optional
            Whether to include a legend
        grid : bool, optional
            Whether to include a grid
        """
        if ax is None:
            ax = plt.gca()
        
        # Apply title and labels if provided
        if title:
            ax.set_title(title, fontweight='bold', pad=15)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        # Style the grid
        if grid:
            ax.grid(alpha=0.3, linestyle='--')
        
        # Style the legend
        if legend and ax.get_legend() is not None:
            ax.legend(frameon=True, facecolor='white', edgecolor='lightgray')
        
        # Style the spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('gray')
            ax.spines[spine].set_linewidth(0.8)
    
    def plot_distribution(self, ax=None, **kwargs):
        """Style for distribution plots (histograms, density plots)."""
        if ax is None:
            ax = plt.gca()
        
        self.apply_chart_style(ax, **kwargs)
        
        # Additional styling specific to distributions
        ax.tick_params(axis='both', which='both', labelsize=10)
    
    def plot_timeseries(self, ax=None, highlight_events=None, **kwargs):
        """
        Style for time series plots.
        
        Parameters:
        -----------
        ax : matplotlib.axes, optional
            The axes to style
        highlight_events : dict, optional
            Dictionary of {time: label} to highlight on the plot
        """
        if ax is None:
            ax = plt.gca()
        
        self.apply_chart_style(ax, **kwargs)
        
        # Additional styling specific to time series
        ax.tick_params(axis='x', rotation=30)
        
        # Highlight specific events if provided
        if highlight_events:
            for time, label in highlight_events.items():
                ax.axvline(x=time, color='gray', linestyle='--', alpha=0.7)
                ax.text(time, ax.get_ylim()[1]*0.95, label, 
                        rotation=90, verticalalignment='top')
    
    def plot_heatmap(self, ax=None, cmap=None, show_values=True, **kwargs):
        """
        Style for heatmaps.
        
        Parameters:
        -----------
        ax : matplotlib.axes, optional
            The axes to style
        cmap : matplotlib colormap, optional
            The colormap to use, defaults to red-yellow-green
        show_values : bool, optional
            Whether to display values in the heatmap cells
        """
        if ax is None:
            ax = plt.gca()
        
        if cmap is None:
            cmap = self.cmap_ryg
        
        self.apply_chart_style(ax, grid=False, **kwargs)
        
        # Additional styling specific to heatmaps
        ax.tick_params(axis='both', which='both', bottom=True, top=False, 
                      labelbottom=True, left=True, right=False, labelleft=True)
        
    def plot_comparison(self, ax=None, colors=None, **kwargs):
        """Style for comparison plots (multiple lines, bars, etc.)."""
        if ax is None:
            ax = plt.gca()
        
        if colors is None:
            colors = self.colors['modern']
        
        self.apply_chart_style(ax, **kwargs)
        
        # Set the color cycle for the plot
        ax.set_prop_cycle(color=colors)
    
    def style_errorbar(self, ax=None, capsize=5, ecolor=None, **kwargs):
        """Style for error bar plots."""
        if ax is None:
            ax = plt.gca()
        
        self.apply_chart_style(ax, **kwargs)
        
        # Get all error bar containers
        for container in ax.containers:
            if isinstance(container, mpl.container.ErrorbarContainer):
                # Style the error bars
                if ecolor:
                    container[2][0].set_color(ecolor)
    
    def annotate_point(self, ax, x, y, text, boxstyle='round,pad=0.5', 
                      fc='white', alpha=0.7, fontsize=10, xytext=(10, -20)):
        """
        Add styled annotation to a specific point on the plot.
        
        Parameters:
        -----------
        ax : matplotlib.axes
            The axes on which to add the annotation
        x, y : float
            Coordinates of the point to annotate
        text : str
            Text for the annotation
        """
        ax.annotate(text, 
                   xy=(x, y),
                   xytext=xytext, 
                   textcoords='offset points',
                   fontsize=fontsize,
                   bbox=dict(boxstyle=boxstyle, fc=fc, alpha=alpha, ec='lightgray'))
    
    def finalize_plot(self, fig=None, tight_layout=True, 
                     suptitle=None, filename=None, dpi=300):
        """
        Finalize the plot with optional title and save functionality.
        
        Parameters:
        -----------
        fig : matplotlib.figure, optional
            The figure to finalize
        tight_layout : bool, optional
            Whether to apply tight layout
        suptitle : str, optional
            Super title for the figure
        filename : str, optional
            If provided, save the figure to this path
        dpi : int, optional
            DPI for saved figure
        """
        if fig is None:
            fig = plt.gcf()
        
        if suptitle:
            fig.suptitle(suptitle, fontsize=18, fontweight='bold', y=0.98)
            fig.subplots_adjust(top=0.9)
        
        if tight_layout:
            fig.tight_layout()
        
        if filename:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')