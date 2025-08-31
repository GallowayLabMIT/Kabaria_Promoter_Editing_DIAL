# Kabaria_Promoter_Editing_DIAL
Data analysis, figure, and modeling code for Kabaria et al "Programmable Promoter Editing for Precise Control of Transgene Expression" (2025)

# Python setuppy
1. Create a virtual environment in the repository directory using python -m venv env
2. Activate the virtual environment using source env/bin/activate (MacOS, Linux) or .\env\Scripts\activate (Windows)
3. Install the current package versions for this project using pip install -r requirements.txt (Note for Figure 6 use requirements_bal.txt) Alternatively, if running into errors manually install packages as needed using the versions in the requirments documents. Most used packages are in `functions.py`
4. Verify that the correct versions of these packages are installed. Incorrect versions may cause errors in the code:
``` 
rushd==0.5.1
scipy==1.7.3
seaborn==0.11.2
statannot==0.2.3
matplotlib==3.5.2
openpyxl==3.1.5
numpy==1.21.6
pandas==1.3.5
```
These notebooks were also coded with Python 3.9.1. The input `columns` in the function `rd.flow.load_csv_with_metadata` may be removed if returning errors. 

5. Download the raw and analyzed data from Zenodo (DOI: [10.5281/zenodo.16576807](https://doi.org/10.5281/zenodo.16576807)), specifically the data.zip file. Unzip this file.
6. Create a file in the root directory of the repo called datadir.txt that contains the absolute path to the data directory you just downloaded. This should be a single line.

# Importing data into ipynb 
1. Select your datadir.txt at the beginning of the notebook, for example the `datadir_srk.txt` is selected in  `datadir = Path(Path('./datadir_srk.txt').read_text())`
2. Ensure each of the `file_path` and `yaml_path` in each occurence is modifed to be a local reference to your data directory for the downloaded `data` folder before the `attune` specification. For example, if `datadir` contained the absolute path for the data folder, the code for importing data would change from: 
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
# Your File
# Import Data1
# Import Data -BioRep1
folder = '20240121flow_exp20240118_p2'
file_path = datadir/'attune'/'Sneha'/folder/'export_singlets' #Assign file paths
yaml_path = datadir/'attune'/'Sneha'/folder/'well_metadata.yaml' #Assign yaml paths 
data1 = rd.flow.load_csv_with_metadata(file_path, yaml_path) #Pull data
```
# Set location for saving exported plots
1. In most notebooks you can set the directory of exported plots. Edit lines such as `figure_folder = './figs_2024_promoter_editing_paper/f_diff-min-prom/'`. If you drun into errors, check that that the directory specified exists. If you do not wish to save exported plots, comment out lines with the `savefig()` function, such as `g.figure.savefig(figure_folder + plottitle + '.svg',dpi=300,bbox_inches='tight')`. 
