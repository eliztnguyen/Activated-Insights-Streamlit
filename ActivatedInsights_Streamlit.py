# import requirements
import streamlit as st
import pandas as pd
from copy import copy
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import os
import json
import pickle
import uuid
import re


# functions
#@st.cache(hash_funcs={pandas.core.frame.DataFrame: my_hash_func})
def readinresults(data):
    '''simpler readin function for excel file'''
    df = pd.read_excel(data)
    return df

@st.cache
def readinFile(filepath, extension):
    '''function to read in file with the following options;
    :param filepath: written as "str";
    :param extension: "csv", "xlsx", "xls";
    :return: data as dataframe
    '''

    tempdf = pd.DataFrame()
    if datatype == "dataframe":
        if extension == "csv":
            tempdf = pd.read_csv(filepath)
        elif extension == "xlsx" or extension == "xls":
            tempdf = pd.read_excel(filepath)
        return tempdf


@st.cache
def setupResults(results):
    '''
    set up results for processing in Accurate Insight project
    :param results: results dataframe for one_organization-one_survey
    :param mycolnames: colnames to be associated with dataframe
    :return: results dataframe with column names, new columns, and only if "submitted"==True
    '''

    # subset data to only submitted surveys
    results = results[results["submitted"] == True]
    ### NOTE: these include submitted surveys that have ZEROS, which means NO ANSWER

    # add new "Location - Department" column
    locdepCol = results["Location Name"] + " - " + results["Department Name"]
    results["Location - Department"] = locdepCol

    return results


@st.cache
def getDFqcolnames(results):
    '''
    gets column names for all questions in results DF
    :param results: results dataframe
    :return: list of column names
    '''

    return list(results.columns[5:65])


@st.cache
def getCategories(statKey):
    '''
    gets list of all categories from statKey DF
    :param statKey: DF with questions and their associated categories
    :return: list of categories
    '''

    # make categories list from all unique categories
    categories = list(statKey["Category 1"].unique())

    # remove 'nan' indexed at position 1
    categories.pop(1)

    return categories


@st.cache
def createStatDict(sourceDF, categories):
    '''
    create dictionary with question number and categories associated with question)
    :param sourceDF: statKey DF with questions and their associated categories
    :param categories: list of categories
    :return: dictionary with (keys = question number) and (values = categories associated with question)
    '''
    tempDict = {}
    tempKey = 1  # dictionary key
    for i in range(len(sourceDF)):  # define length of dataframe iterating over
        row = sourceDF.iloc[i]  # identify dataframe row of interest
        tempList = [] # empty list to be dictionary value
        for cat in categories:  # categories in colnames
            for value in row[4:10]:  # category columns across row
                if value == cat:
                    tempList.append(cat)
        tempDict[tempKey] = tempList
        tempKey += 1

    return tempDict


@st.cache
def createNewCatQDF(sourceDF, categories):
    '''
    create new dataframe for cataloguing question numbers by their category
    :param sourceDF: statKey DF with questions and their associated categories
    :param categories: list of categories
    :return: DF with columns = categories; index = question number
    '''
    newDF = pd.DataFrame(columns=categories,
                             index=sourceDF["Stmt_survey_num"])

    for i in range(len(sourceDF)):  # define length of dataframe iterating over
        row = sourceDF.iloc[i]  # identify dataframe row of interest
        templist = [0] * 9  # create placeholder empty list
        j = 0  # position tracker for categories in colnames
        for cat in categories:  # categories in colnames
            for value in row[4:10]:  # category columns across row
                if value == cat:
                    templist[j] += 1
            j += 1  # increment cat position by 1 after looping through row searching for cat
        newDF.iloc[i] = templist  # add list to emptyDF

    return newDF


@st.cache
def createCatbyQDict(sourceDF, categories):
    """
    function to create diciontary with (key = categories) and (values = all associated Q's)
    :param sourceDF: DF with categories as index and q# as columns;
    :param categories: = list of categories
    :return: dictionary with (key = categories) and (values = all associated Q's)
    """

    # transpose dataframe
    catbyQ = sourceDF.T

    tempDict = {}
    # sourceDFrow = sourceDF.index.values
    catpos = 0  # keeps track of category position in dataframe
    for cat in categories:
        tempq = 1  # keep track of Q number
        tempqlist = []  # empty list to keep track of all associated questions for a given category
        rowdata = list(catbyQ.iloc[catpos])
        for j in range(60):  # for each question position
            yesnoq = rowdata[j]  # yes/no binary to if question in category
            if yesnoq == 1:  # if question is associated with category
                tempqlist.append(tempq)  # append question number
            tempq += 1  # increment question number by tracker by 1, regardless
        tempDict[cat] = tempqlist  # set key to list of associated questions
        catpos += 1

    return tempDict


@st.cache
def getDepCat_unhappyCount(source_df, categories, questions, statbyCat_dict):
    '''
    create dataframe of low score counts by department
    :param source_df: dataframe of results of interest
    :param categories: list of all categories of interest
    :param questions: list of all questions as labelled in the survey dataframe
    :param statbyCat_dict: dictionary with (keys = question number) and (values = associated categories)
    :return: Dataframe of low scores for each category for every location-dept
    '''
    # create list of unique locations-departments
    depts = list(source_df["Location - Department"].unique())

    # create new DF to populate
    dep_catunhappyQ = pd.DataFrame(columns=categories,
                                   index=depts)  # dataframe with 9 categories, across all departments for said location

    dep_pos = 0  # keeps track of where we are in new dataframe location-department
    person_row = source_df.index.values  # gives all row label for all rows associated with dataframe
    for dept in depts:  # evaluate each location-department separately
        scorelist = [0] * 9  # keeps track of number of people with low scores per category
        for row in person_row:  # for each row aka person
            rowdept = source_df.loc[row, "Location - Department"]  # gets location-department for said person
            if rowdept == dept:  # evaluate if person is in said location-department
                qnum = 0
                for q in questions:
                    qnum += 1  # increment the question number count by 1 to get to the correct question number
                    qscore = source_df.loc[row, q]  # get score for said question
                    qcat = statbyCat_dict[qnum]  # get categories associated with said question
                    cat_pos = 0  # to keep track of category position
                    if qscore > 0:  # scores of ZERO means no entry was made, and data point ignored
                        if qscore < 4:  # want only low score counts
                            for cat in categories:  # evaluate for each category
                                if cat in qcat:
                                    scorelist[cat_pos] += 1  # updates score of said position
                                cat_pos += 1  # increments category position by 1

        dep_catunhappyQ.iloc[dep_pos] = scorelist  # update normalized score for said location/department
        dep_pos += 1

    dep_catunhappyQ.insert(0, 'Location - Department', depts)  # add new column with department names
    return dep_catunhappyQ


@st.cache
def cleanCountDFforRecs(dataframe, categories):
    '''
    add necessary new columns to and restructure low score count DF
    :param dataframe: dataframe with low score counts by "Location - Department"
    :param categories: list of categories of interest (columns in dataframe)
    :return: dataframe with Location-Department-Category recommendations ranked by low score count
    '''

    # split "Location - Department"
    dataframe['Location'], dataframe['Department'] = dataframe['Location - Department'].str.split(' - ', 1).str

    # reshape dataframe
    dataframe = pd.melt(dataframe,
                                   id_vars=["Location - Department", "Location", "Department"],
                                   value_vars=categories,
                                   var_name="Category",
                                   value_name='Scores with Improvement Potential')

    # sort Descending for low scores
    sorted_dataframe = dataframe.sort_values(by=['Scores with Improvement Potential'], ascending=False)

    # remove location - departments with zero unhappy scores
    sorted_dataframe = sorted_dataframe[sorted_dataframe['Scores with Improvement Potential'] != 0]

    return sorted_dataframe


@st.cache
def makerecs(dataframe):
    '''
    make dataframe of recommendations (3 cat per dep, 3 dep per loc)
    :param dataframe: sorted low count DF
    :return: subset of dataframe with recommendations only
    '''

    # only want top 3 rows of each group
    # THEREFORE
    # only want top 3 recommendations for each location-department; aka top 3 categories for each department of a given location
    # only want top 9 recommendations for each location, aka 3 dept
    top3recs_fordept = dataframe.groupby("Location - Department").head(3)
    top9recs_forloc = top3recs_fordept.groupby("Location").head(9)

    # sort by location
    top9recs_byloc = top9recs_forloc.sort_values(by = ["Location", "Scores with Improvement Potential"],
                                                 ascending = [True, False])

    # drop the "location - Department" category
    clean_top9recs = top9recs_byloc.drop(columns = "Location - Department")

    return clean_top9recs


@st.cache
def locRec_dict(dataDF, locColName):
    '''
    function to turn recommendation dataframe into dictionary by location;
    :param dataDF: dataframe of recommendations
    :param locColName: column name that will be reference of dictionary key
    :return:
    '''
    temp_dict = {}
    locList = dataDF[locColName].unique()  # list of all locations

    rowIndex = dataDF.index.values  # gives all row label(index) for all rows associated with dataframe

    for loc in locList:
        recs = []  # list of recommendations for location
        for row in rowIndex:
            rowloc = dataDF.loc[row, "Location"]
            if rowloc == loc:
                rec = list(dataDF.loc[row, "Department":"Scores with Improvement Potential"])
                recs.append(rec)
        recsDF = pd.DataFrame(recs, columns=['Department', 'Category', "Scores with Improvement Potential"])
        temp_dict[loc] = recsDF

    return temp_dict

@st.cache
def qResponseDF(results):
    ''''
    create dataframe with all question responses in one column
    :param results: results dataframe
    :return: dataframe with all question responses in on column and id-Location-Department-q# data
    '''

    # get list of numbers 1-60 for 60 questions
    qnum_list = list(range(61))
    qnum_list.remove(0)

    # rename question columns in results dataframe
    results.columns.values[5:65] = qnum_list

    # get all Q columns with responses
    response_allQ = results[qnum_list]

    # attach Location and Department information
    response_allQ["Location Name"] = results["Location Name"]
    response_allQ["Department Name"] = results["Department Name"]
    response_allQ["id"] = results["id"]

    # melt Q's
    allResponse_melt = pd.melt(response_allQ,
                               id_vars=["id", "Location Name", "Department Name"],
                               value_vars=qnum_list,
                               var_name="Question",
                               value_name='Value')

    # create/add location-department column
    allResponse_melt['Location - Department'] = allResponse_melt["Location Name"] + " - " + allResponse_melt[
        "Department Name"]

    return allResponse_melt

@st.cache
def getLowResponseQ(sourceDF):
    '''
    subset dataframe with all Q responses to only include low scores
    :param sourceDF: dataframe with all question responses in on column
    :return: dataframe with only Low Score question responses
    '''

    low_allResponse_melt = sourceDF[sourceDF["Value"] < 4]
    low_allResponse_melt = low_allResponse_melt[low_allResponse_melt["Value"] != 0]

    return low_allResponse_melt

@st.cache
def makehistboxcombo(data, xlabel, ylabel, quant, quantLab):
    '''
    makes combo histogram and boxplot for data with labled quantile information
    :param data: series data
    :param xlabel: desired "x-label" as string
    :param ylabel: desired "y-label" as string
    :param quant: desired quantile cut-off as number
    :param quantLab: desired "quantile label" as string; ex: "First Quantile"
    :return: Labeled combo histogram boxplot graph
    '''
    sns.set(style="ticks")

    x = data

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                        gridspec_kw={"height_ratios": (.15, .85)}, )

    # ax = sns.boxplot(x, ax=ax_box).set(xlabel = None, title = "High Scores Across the Organization")
    ax = sns.boxplot(x, ax=ax_box).set(xlabel=None)
    ax = sns.distplot(x, ax=ax_hist, kde=False)

    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)

    # ax.set_title("High Scores Across the Organization", fontsize = 20)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # plot median and quantile lines
    ax.axvline(x.quantile(quant), color='black', linestyle='dashed', linewidth=1)

    # add label for median and quantile lines
    min_ylim, max_ylim = plt.ylim()
    ax.text(x.quantile(quant) * 1, max_ylim * 1.01, quantLab + ': {:.2f}%'.format(x.quantile(quant)))


@st.cache
def getHigh_lowScoreCount_locdep(series):
    '''
    get array with the Location-Department with the highest count of low scores
    :param series: series with all low scores and their associated Location-Deparment
    :return: array with Location-Department with the highest count of low scores
    '''

    # get third quantile value of series
    quantile_third = series.quantile(.75)

    highcount_series = series[series > quantile_third]

    highcount_locdep_names = highcount_series.index.values

    return highcount_locdep_names


@st.cache
def getCountResults_locdep(sourceDF, locdeplist):
    '''
    create new DF with only location-department of interest
    :param sourceDF: dataframe with total low counts for all Location-Departments
    :param locdeplist: series with Location-Departments of interest
    :return: DF with only location-department of interest
    '''

    subset = sourceDF.loc[sourceDF["Location - Department"].isin(locdeplist)]

    return subset

@st.cache
def createNaNQDict(sourceDF, categories):
    '''
    fuction to find questions associated with no categories
    :param sourceDF: original 'statKey' dataframe
    :param categories: list of all categories
    :return: dictionary with key="nan" and value=(list of all Q's associated with no categories)
    '''
    tempDict = {}
    tempList = [] # empty list to be dictionary value
    qNum = 1  # keep track of qNum
    for i in range(len(sourceDF)):  # define length of dataframe iterating over
        row = sourceDF.iloc[i]  # identify dataframe row of interest
        value = row[4]  # set value to be first category
        if value not in categories:
            tempList.append(qNum)
        qNum += 1

    tempDict["nan"] = tempList
    return tempDict


@st.cache
def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }}
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


#################################
# taking care of things that can be done without the results file
#################################

# define relative file path
filepath1 = "data/statement_key.csv"  # file for statkey only
filepath2 = "data/results_colnames.csv"  # file for colnames only


# Web App Intro text
st.title('Activated Insights Area of Improvement Identifier')

st.write('Upload your provided "Great Place to Work" survey results. '
         'We will provide the recommended areas for targeted improvement.')

# disable FileUploaderEncodingWarning:
st.set_option('deprecation.showfileUploaderEncoding', False)


# read in files
statKey = readinFile(filepath1, "csv")
mycolnames = readinFile(filepath2, "csv")

# create a list of question columns as they appear in the data
questions = mycolnames[5:65]

# create list of categories
categories = getCategories(statKey)

# create dictionary with (key = question number) and (value = categories associated with question)
statbyCat_dict = createStatDict(statKey, categories)

# create new dataframe for cataloguing question numbers by their category
statbyCat = createNewCatQDF(statKey, categories)


#################################
# read in results file and run analysis
#################################

# give option to upload data file
uploaded_file = st.file_uploader("Choose your Excel data file", type="xlsx")

# if file is uploaded, run analysis
if uploaded_file is not None:
    # read in and set uploaded file to results
    #results = readinFile(uploaded_file, "dataframe", "xlsx")
    results = readinresults(uploaded_file)
    st.dataframe(results)


    # clean results DF for processing
    results_new = setupResults(results)

    # get DF with count of low scores for each category for every Location-Department
    results_unhappyCount = getDepCat_unhappyCount(results_new, categories, questions, statbyCat_dict)

    # sort recommendations by Location-Department-Category
    sorted_lowCounts = cleanCountDFforRecs(copy(results_unhappyCount), categories)

    # make recommendations based on overall low score count across Location-Department-Categories
    recommendations = makerecs(sorted_lowCounts)

    # create dictionary of recommendations by location
    rec_dict = locRec_dict(recommendations, "Location")

    # create download button
    rec_download_button = download_button(recommendations, "Recommendations.csv", "Download All Recommendations")
    st.markdown(rec_download_button, unsafe_allow_html=True)


#####################################################
##### Interactive Recommendations
#####################################################


# option to show all areas of recommendation by location
if st.checkbox('Show All Recommendations for Improvement:'):

    location_selected = st.selectbox(
    'Which Location do you want to look at?',
    list(rec_dict.keys()),
    key = 1)

     #recsDF['Location/Department'])
    ###### test with dictionary
    st.write(rec_dict[location_selected])

#####################################################
###### option to show areas of recommendation for loc-dep with highest count low scores
#####################################################

if st.checkbox('Calculate Recommendations for Worse Performing Departments:'):

    # Description
    st.write('These departments have the highest number of low scores (top 25%) '
             'for questions associated with the recommended improvement categories.')

    # get dataframe of all question responses in one column
    allResponse_melt = qResponseDF(results_new)

    # get dictionary of all questions associated with no categories ("nan"
    nanDict = createNaNQDict(statKey, categories)

    # subset allResponse_melt to only include questions associated with categories
    allResponse_melt_noNan = allResponse_melt[~allResponse_melt["Question"].isin(nanDict["nan"])]

    # get dataframe of Qresponses with low score responses only
    low_allResponse_melt = getLowResponseQ(allResponse_melt_noNan)

    # get series with all low scores by Location/Department
    lowscore_locdep = low_allResponse_melt.groupby("Location - Department")["Value"].count()

    # make labeled histogram/boxplot combo for Location/Department with high number low scores
    lowScore_comboHistBox = makehistboxcombo(lowscore_locdep,
                                             "Location/Department Low Score Count",
                                             "Frequency",
                                             .75,
                                             "Third Quantile")

    # get array with Location/Department with low score count greater than 3rd quantile
    highcount_locdep_array = getHigh_lowScoreCount_locdep(lowscore_locdep)

    # get dataframe of low score counts ONLY for Location-Deparments with highest counts
    highScore_locdep_sorted_lowCounts = getCountResults_locdep(sorted_lowCounts, highcount_locdep_array)

    # make recommendations based on overall low score count ONLY for Location-Department with highest lowscore counts
    lowscore_locdep_recs = makerecs(highScore_locdep_sorted_lowCounts)

    # export recommendations to CSV file
    #lowscore_locdep_recs.to_csv(r'data/recs_lowscoreDepartments.csv', index=False)

    # create dictionary of recommendations by location
    lowscore_locdep_rec_dict = locRec_dict(lowscore_locdep_recs, "Location")

    # create download button
    rec_lowscore_download_button = download_button(lowscore_locdep_recs, "recs_lowscoreDepartments.csv", "Download Subset Recommendations")
    st.markdown(rec_lowscore_download_button, unsafe_allow_html=True)


#####################################################
##### Interactive Recommendations
#####################################################

# option to show all areas of recommendation by location
if st.checkbox('Show Recommendations for Worse Performing Departments:'):

    location_selected = st.selectbox(
    'Which Location do you want to look at?',
    list(lowscore_locdep_rec_dict.keys()),
    key = 2)

     #recsDF['Location/Department'])
    ###### test with dictionary
    st.write(lowscore_locdep_rec_dict[location_selected])






