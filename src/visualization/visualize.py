"""Charts and plots for analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats

class visualization():

    def eda_continous(dataset):
        """Exploratory data analysis for dataset."""
        #Plotting for continous columns
        warnings.filterwarnings('ignore')
        fig,ax = plt.subplots(5,3,figsize=(30,50))
        for index,i in enumerate(dataset.iloc[:, [0, 5, 9, 11, 12]]):
            sns.distplot(dataset[i],ax=ax[index,0])
            sns.boxplot(dataset[i],ax=ax[index,1])
            stats.probplot(dataset[i],plot=ax[index,2])

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        plt.suptitle("Visualizing Continuous Columns",fontsize=40)

    def eda_categorical(dataset):
        fig, axes =plt.subplots(9,1, figsize=(35,120), sharex=False)
        axes = axes.flatten()
        sns.set(font_scale=2)
        columns=list(dataset.select_dtypes('object'))
        for ax, catplot in zip(axes,columns):
            sns.countplot(data=dataset, x=catplot, ax=ax)

        plt.tight_layout()
        fig.subplots_adjust(top=0.99)
        plt.suptitle("Visualizing counts of Categorical Columns",fontsize=40)

    def eda_hist(dataset):
        fig, axes =plt.subplots(8,1, figsize=(35,120), sharex=False)
        axes = axes.flatten()
        sns.set(font_scale=2)
        columns=list(dataset.select_dtypes('object'))
        columns.remove('y')
        for ax, catplot in zip(axes,columns):
            sns.histplot(data=dataset,x=catplot,ax=ax,hue='y',kde=True)

        plt.tight_layout()
        fig.subplots_adjust(top=0.99)
        plt.suptitle("Visualizing hitograms of categorical Columns",fontsize=40)