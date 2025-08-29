# import importlib.resources as pkg_resources
from importlib import resources
import pandas as pd

# gvpList = pkg_resources.open_text('data', "GVP_Volcano_List.csv")

def get_gvp_list():

    gvpList = resources.files(__package__).joinpath("GVP_Volcano_List.csv").open('r')

    gvp_list = pd.read_csv(gvpList)
    gvp_list['Search_Name'] = gvp_list['Volcano_Name']
    gvp_list['Primary_Name'] = ''
    gvp_list['Primary_indx'] = None
    gvp_list['Primary_Volcano'] = False
    gvp_list.loc[(~pd.isna(gvp_list['Latitude'])) & (~pd.isna(gvp_list['Longitude'])), 'Primary_Volcano'] = True
    for j, row in gvp_list.iterrows():
        if not row.Primary_Volcano:
            volc_num = row.Volcano_Number
            primary_indx = gvp_list.loc[(gvp_list.Volcano_Number==volc_num) & (gvp_list.Primary_Volcano==True)].index[0]
            primary_volc = gvp_list.loc[primary_indx,'Volcano_Name']
            gvp_list.loc[j,'Primary_Name'] = primary_volc
            gvp_list.loc[j,'Search_Name'] += f" ({primary_volc})"
            gvp_list.loc[j,'Primary_indx'] = primary_indx
        else:
            gvp_list.loc[j,'Primary_Name'] = row.Volcano_Name
            gvp_list.loc[j,'Primary_indx'] = j

    return gvp_list

def get_gvp_primary_list() -> pd.DataFrame:
    gvp_list = get_gvp_list()
    gvp_primary_list = gvp_list.dropna(subset=['Latitude','Longitude'])

    return gvp_primary_list