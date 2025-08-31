# Kabaria_Promoter_Editing_DIAL
Data analysis, figure, and modeling code for Kabaria et al "Programmable Promoter Editing for Precise Control of Transgene Expression"

# Reference Data 
Code in this repository require data from Zenodo (DOI: [10.5281/zenodo.17014280](https://doi.org/10.5281/zenodo.17014280)). 

# Python setup for data analysis and figures
1. Create a virtual environment in the repository directory using python -m venv env
2. Activate the virtual environment using source env/bin/activate (MacOS, Linux) or .\env\Scripts\activate (Windows)
3. Install the current package versions for this project using pip install -r requirements.txt (Note for Figure 6 use requirements_bal.txt, and check the document requirementsold.txt for alternative package versions.) Alternatively, manually install packages as needed using the versions in the requirements documents. Most commonly used packages are in `functions.py`
4. Verify that the correct versions of these packages are installed. Incorrect versions may cause errors in the code. Common packages to check: 
``` 
rushd==0.5.1
scipy==1.7.3
seaborn==0.11.2 # or seaborn==0.12.2
statannot==0.2.3
matplotlib==3.5.2
openpyxl==3.1.5
numpy==1.21.6
pandas==1.3.5
```
These notebooks were coded with Python 3.9.1. 

5. Download the raw and analyzed data from Zenodo, specifically the data.zip file. Unzip this file.
6. Create a file in the root directory of the repo called `datadir.txt` that contains the absolute path to the data directory you just downloaded. This should be a single line.

# Importing data into ipynb 
1. Select your `datadir.txt` at the beginning of the notebook. For example the `datadir_srk.txt` is selected in  `datadir = Path(Path('./datadir_srk.txt').read_text())`
2. Ensure each of the `file_path` and `yaml_path` in each occurrence is modified to be a local reference to your data directory for the downloaded  `attune_data` folder. For example, if `datadir` contained the absolute path for the attune_data folder, the code for importing data would change from: 
```
# Original
# Import Data1
# Import Data -BioRep1
folder = '20240121flow_exp20240118_p2'
file_path = datadir/'instruments'/'data'/'attune'/'Sneha'/folder/'export_singlets' #Assign file paths
yaml_path = datadir/'instruments'/'data'/'attune'/'Sneha'/folder/'well_metadata.yaml' #Assign yaml paths 
data1 = rd.flow.load_csv_with_metadata(file_path, yaml_path) #Pull data
```
and become: 
```
# Your File, with directory datadir for the Data folder
# Import Data1
# Import Data -BioRep1
folder = '20240121flow_exp20240118_p2'
file_path = datadir/'Sneha'/folder/'export_singlets' #Assign file paths
yaml_path = datadir/'Sneha'/folder/'well_metadata.yaml' #Assign yaml paths 
data1 = rd.flow.load_csv_with_metadata(file_path, yaml_path) #Pull data
```
3. The input `columns` in the function `rd.flow.load_csv_with_metadata` may be removed to return all single cell data parameters. The line `#rd.plot.plot_well_metadata(yaml_path)` can be removed as well (this line shows the plate layout.)

# Exporting data 
1. Each file concatenates data across bioreplicates into dataframes labeled `data`, `df`, etc. These dataframes can be exported with a function like `to_excel()` to obtained a merged file containing single cell data with labels of conditions and bioreplicates.
2. Alternatively, the `.ipynb` notebooks specify the folders and `.yaml` plate maps for all data per experiment, and can be used directly with the single cell data exports available at Zenodo.
3. Summary statistic data exports are available in Zenodo. 

# Exporting plots
1. In most notebooks you can set the directory of exported plots. Edit lines such as `figure_folder = './figs_2024_promoter_editing_paper/f_diff-min-prom/'`. Ensure that the directory specified exists. If you do not wish to save exported plots, comment out lines with the `savefig()` function, such as `g.figure.savefig(figure_folder + plottitle + '.svg',dpi=300,bbox_inches='tight')`. 

# MATLAB for modeling 
MATLAB code is included in a folder `MATLAB_modeling_plasmid_titration`. The code enclosed was used for modeling. Relevant experimental data is included in `datafiles`. This data was exported from the experiments in `Fig2_S8_S9A_ZF37plasmidtitration.ipynb` and `FigS8_ZF43plasmidtitration.ipynb`. Shorted notebooks for data export are `modeling_ZF37plasmidtitration.ipynb` and `modeling_ZF43plasmidtitration.ipynb`. The version used was MATLAB 24.1 (2024a). 
