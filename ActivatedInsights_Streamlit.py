# import requirements
import streamlit as st
import pandas as pd
from copy import copy
import base64
import json
import pickle
import uuid
import re


####### Functions that don't require results file

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
        tempList = []  # empty list to be dictionary value
        for cat in categories:  # categories in colnames
            for value in row[4:10]:  # category columns across row
                if value == cat:
                    tempList.append(cat)
        tempDict[tempKey] = tempList
        tempKey += 1

    return tempDict


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


###### Functions that require results file

def setupResults(results):
    '''
    set up results for processing in Accurate Insight project
    :param results: results dataframe for one_organization-one_survey
    :return: results dataframe with new 'Location - Deparment' column, and only if "submitted"==True
    '''

    # subset data to only submitted surveys
    results = results[results["submitted"] == True]
    ### NOTE: these include submitted surveys that have ZEROS, which means NO ANSWER

    # add new "Location - Department" column
    locdepCol = results["Location Name"] + " - " + results["Department Name"]
    results["Location - Department"] = locdepCol

    return results


def getDepCat_unhappyCount(source_df, categories, questions, statbyCat_dict):
    '''
    create dataframe of low score counts by department and category
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
    # dep_allpplcount = []  # list to keep track of how many people are in each department
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


def getDepCat_unhappyCount_ppl(source_df, categories, questions, statbyCat_dict):
    '''
    get NUMBER/COUNT "unhappy" people for each category for each location-department
    :param source_df: dataframe of results of interest
    :param categories: list of all categories of interest
    :param questions: list of all questions as labelled in the survey dataframe
    :param statbyCat_dict: dictionary with (keys = question number) and (values = associated categories)
    :return: df with count of people with low scores for each category for each location-department
    '''
    # create list of unique locations-departments
    depts = list(source_df["Location - Department"].unique())

    # create new DF to populate
    dep_catunhappyQ = pd.DataFrame(columns=categories,
                                   index=depts)  # dataframe with 9 categories, across all departments for said location

    dep_pos = 0  # keeps track of where we are in new dataframe location-department
    person_row = source_df.index.values  # gives all row label for all rows associated with dataframe
    # dep_allpplcount = []  # list to keep track of how many people are in each department
    for dept in depts:  # evaluate each location-department separately
        scorelist = [0] * 9  # keeps track of number of people with low scores per category
        for row in person_row:  # for each row aka person
            catset = set()  # set to keep track of all unique low score categories associated with person
            rowdept = source_df.loc[row, "Location - Department"]  # gets location-department for said person
            if rowdept == dept:  # evaluate if person is in said location-department
                qnum = 0
                for q in questions:
                    qnum += 1  # increment the question number count by 1 to get to the correct question number
                    qscore = source_df.loc[row, q]  # get score for said question
                    qcat = statbyCat_dict[qnum]  # get categories associated with said question
                    if qscore > 0:  # scores of ZERO means no entry was made, and data point ignored
                        if qscore < 4:  # want only low score counts
                            for cat in categories:  # evaluate for each category
                                if cat in qcat:
                                    catset.add(cat)  # updates catset with cat

            # add person count for every category associated with person
            cat_pos = 0
            for cat in categories:
                if cat in catset:
                    scorelist[cat_pos] += 1
                cat_pos += 1

        dep_catunhappyQ.iloc[dep_pos] = scorelist  # add category ppl count to said location/department
        dep_pos += 1

    dep_catunhappyQ.insert(0, 'Location - Department', depts)  # add new column with department names
    return dep_catunhappyQ


def getDepCat_allCount_ppl(source_df, categories, questions, statbyCat_dict):
    '''
    get NUMBER/COUNT all people for each category for each location-department
    :param source_df: dataframe of results of interest
    :param categories: list of all categories of interest
    :param questions: list of all questions as labelled in the survey dataframe
    :param statbyCat_dict: dictionary with (keys = question number) and (values = associated categories)
    :return: df with count of people with low scores for each category for each location-department
    '''
    # create list of unique locations-departments
    depts = list(source_df["Location - Department"].unique())

    # create new DF to populate
    dep_catAllQ = pd.DataFrame(columns=categories,
                               index=depts)  # dataframe with 9 categories, across all departments for said location

    dep_pos = 0  # keeps track of where we are in new dataframe location-department
    person_row = source_df.index.values  # gives all row label for all rows associated with dataframe
    # dep_allpplcount = []  # list to keep track of how many people are in each department
    for dept in depts:  # evaluate each location-department separately
        scorelist = [0] * 9  # keeps track of number of people with low scores per category
        for row in person_row:  # for each row aka person
            catset = set()  # set to keep track of all unique low score categories associated with person
            rowdept = source_df.loc[row, "Location - Department"]  # gets location-department for said person
            if rowdept == dept:  # evaluate if person is in said location-department
                qnum = 0
                for q in questions:
                    qnum += 1  # increment the question number count by 1 to get to the correct question number
                    qscore = source_df.loc[row, q]  # get score for said question
                    qcat = statbyCat_dict[qnum]  # get categories associated with said question
                    if qscore > 0:  # scores of ZERO means no entry was made, and data point ignored
                        for cat in categories:  # evaluate for each category
                            if cat in qcat:
                                catset.add(cat)  # updates catset with cat

            # add person count for every category associated with person
            cat_pos = 0
            for cat in categories:
                if cat in catset:
                    scorelist[cat_pos] += 1
                cat_pos += 1

        dep_catAllQ.iloc[dep_pos] = scorelist  # add category ppl count to said location/department
        dep_pos += 1

    dep_catAllQ.insert(0, 'Location - Department', depts)  # add new column with department names
    return dep_catAllQ


def restructureDFforRecs(dataframe, categories, valueName):
    '''
    add necessary new columns to and restructure low score count DF
    :param dataframe: dataframe with low score counts by "Location - Department"
    :param categories: list of categories of interest (columns in dataframe)
    :param valueName: name to be assigned to 'values' column as string
    :return: dataframe with Location-Department-Category recommendations ranked by low score count
    '''

    # split "Location - Department"
    dataframe['Location'], dataframe['Department'] = dataframe['Location - Department'].str.split(' - ', 1).str

    # reshape dataframe
    dataframe = pd.melt(dataframe,
                        id_vars=["Location - Department", "Location", "Department"],
                        value_vars=categories,
                        var_name="Category",
                        value_name=valueName)

    # sort Location-Department-Category
    sorted_dataframe = dataframe.sort_values(by=['Location', 'Department', 'Category'])

    # remove location - departments with zero unhappy scores
    sorted_dataframe = sorted_dataframe[sorted_dataframe[valueName] != 0]

    return sorted_dataframe


def top3cat(dataframe, refcolumn):
    '''
    make dataframe of top 3 cats per refcolumn
    :param dataframe: DF with low counts per Location-Departemnt-Category
    :param refcolumn: column for which you want the top 3 categories for
    :return: subset of dataframe with top3cat recommendations only for refcolumn
    '''

    # sort Descending for low scores
    sortedScores = dataframe.sort_values(by=['Scores with Improvement Potential'], ascending=False)

    # only want top 3 of each category for a given Location-Department
    top3cat = sortedScores.groupby("Location - Department").head(3)

    # only want top 3 rows of each reference group
    top3cat_1 = top3cat.groupby(refcolumn).head(3)

    # sort by 'Location' and "Scores with Improvement Potential"
    top3cat_2 = top3cat_1.sort_values(by=["Location", "Scores with Improvement Potential"],
                                      ascending=[True, False])

    return top3cat_2


def catRank_loc(dataframe, refcolumn):
    '''
     make dataframe of top 3 cats per refcolumn
    :param dataframe: DF with low counts per Location-Category
    :param refcolumn: column for which you want the top 3 categories for
    :return: subset of dataframe with top3cat recommendations only for refcolumn
    '''

    # sort Descending for low scores
    sortedScores = dataframe.sort_values(by=['Scores with Improvement Potential'], ascending=False)

    # only want top 3 rows of each group
    top3cat = sortedScores.groupby(refcolumn).head(3)

    # sort by 'Location' and "Scores with Improvement Potential"
    top3cat_1 = top3cat.sort_values(by=["Location", "Scores with Improvement Potential"],
                                    ascending=[True, False])

    return top3cat_1


def catRank_org(dataframe):
    '''
    make dataframe of top 3 cats for organization
    :param dataframe: DF with low counts per Location-Departemnt-Category
    :param refcolumn: column for which you want the top 3 categories for
    :return: subset of dataframe with top3cat recommendations only for refcolumn
    '''

    # sort Descending for low scores
    sortedScores = dataframe.sort_values(by=['Scores with Improvement Potential'], ascending=False)

    return sortedScores


def cleanRecs(recs_df):
    # drop the "location - Department" category
    recs_clean = recs_df.drop(columns="Location - Department")

    return recs_clean


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
                rec = list(dataDF.loc[row, "Department":"Percent Department"])
                recs.append(rec)
        recsDF = pd.DataFrame(recs, columns=['Department', 'Category', "Scores with Improvement Potential",
                                             'People Affected', 'Percent Department'])
        temp_dict[loc] = recsDF

    return temp_dict


def locRec_cat_dict(dataDF, locColName):
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
                rec = list(dataDF.loc[row, "Category":"Percent Location"])
                recs.append(rec)
        recsDF = pd.DataFrame(recs, columns=['Category', "Scores with Improvement Potential",
                                             'People Affected', 'Percent Location'])
        temp_dict[loc] = recsDF

    return temp_dict


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


def getLowResponseQ(sourceDF):
    '''
    subset dataframe with all Q responses to only include low scores
    :param sourceDF: dataframe with all question responses in on column
    :return: dataframe with only Low Score question responses
    '''

    low_allResponse_melt = sourceDF[sourceDF["Value"] < 4]
    low_allResponse_melt = low_allResponse_melt[low_allResponse_melt["Value"] != 0]

    return low_allResponse_melt


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


def getCountResults_locdep(sourceDF, locdeplist):
    '''
    create new DF with only location-department of interest
    :param sourceDF: dataframe with total low counts for all Location-Departments
    :param locdeplist: series with Location-Departments of interest
    :return: DF with only location-department of interest
    '''

    subset = sourceDF.loc[sourceDF["Location - Department"].isin(locdeplist)]

    return subset


#@st.cache
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
filepath1 = "statement_key.csv"  # file for statkey only


# Web App Intro text
st.title('Activated Insights Area of Improvement Identifier')

st.write('Upload your provided "Great Place to Work" survey results. '
         'We will provide the recommended areas for targeted improvement.')

# disable FileUploaderEncodingWarning:
st.set_option('deprecation.showfileUploaderEncoding', False)


# read in files
statKey = pd.read_csv(filepath1)

# list of question columns as they appear in the data
questions = statKey["Question"].tolist()

# list of categories
categories = getCategories(statKey)

# dictionary with (key = question number) and (value = categories associated with question)
statbyCat_dict = createStatDict(statKey, categories)

# new dataframe for cataloguing question numbers by their category
statbyCat_df = createNewCatQDF(statKey, categories)

# dictionary with (key = category) and (values = questions-associated)
catbyQ_dict = createCatbyQDict(statbyCat_df, categories)


#################################
# read in results file and run analysis
#################################

# give option to upload data file
uploaded_file = st.file_uploader("Choose your Excel data file", type="xlsx")

#define relative file path
filepath2 = "ExampleResultsFile.xlsx"  # file for statkey only

# if not file uploaded, display ExampleResultsFile & run analysis on example
if uploaded_file is None:
    st.write('An example results file is currently preloaded.')
    results = pd.read_excel(filepath2)
# if file is uploaded, run analysis
elif uploaded_file is not None:
    # read in and set uploaded file to results
    results = pd.read_excel(uploaded_file)
    st.write('Your results file has been uploaded.')

# display results table analyzed
st.dataframe(results)

# clean results DF for processing
results_new = setupResults(results)

# DF with count of low scores for each category for every Location-Department
results_unhappyCount = getDepCat_unhappyCount(results_new, categories, questions, statbyCat_dict)

# low score count by Location-Department-Category
sorted_lowCounts = restructureDFforRecs(copy(results_unhappyCount), categories, "Scores with Improvement Potential")

# DF with count of ppl with low score for each category for every Location-Department
results_unhappyCount_ppl = getDepCat_unhappyCount_ppl(results_new, categories, questions, statbyCat_dict)

# low score ppl count by Location-Department-Category
sorted_lowCounts_ppl = restructureDFforRecs(copy(results_unhappyCount_ppl), categories, "People Affected")

# DF with count of ALL ppl for each category for every Location-Department
results_allCount_ppl = getDepCat_allCount_ppl(results_new, categories, questions, statbyCat_dict)

# all ppl count by Location-Department-Category
sorted_allCounts_ppl = restructureDFforRecs(copy(results_allCount_ppl), categories, "People Responding")

# get only count of ppl for Location-Department-Category with low scores
sorted_allCounts_ppl_sub = sorted_allCounts_ppl.loc[sorted_lowCounts_ppl.index.values]

# check that row items match up
# all(sorted_allCounts_ppl_sub["Location - Department"] == sorted_lowCounts_ppl["Location - Department"])  # True
# all(sorted_allCounts_ppl_sub["Category"] == sorted_lowCounts_ppl["Category"])  # True

# new column for 'sorted_lowCounts_ppl' with percent people with low scores for Department
sorted_lowCounts_ppl["Percent Department"] = (
        sorted_lowCounts_ppl["People Affected"] / sorted_allCounts_ppl_sub["People Responding"] * 100).astype(
    int)

# check that indexes match between DF
# all(sorted_lowCounts.index.values == sorted_lowCounts_ppl.index.values)  # True

# add affected people count & percent department to sorted_lowCounts
sorted_lowCounts["People Affected"] = sorted_lowCounts_ppl["People Affected"]
sorted_lowCounts["Percent Department"] = sorted_lowCounts_ppl["Percent Department"]


############################################################
############# Recommendations for ANY department with improvement potential
############################################################

# make recommendations for ALL departments with ANY category for improvement
allDep_rec = top3cat(sorted_lowCounts, "Location - Department")

# clean and write recs to CSV
allDep_rec1 = cleanRecs(allDep_rec)

# dictionary of recommendations by location
allDep_rec1_dict = locRec_dict(allDep_rec1, "Location")

# make only 3 recommendations total per location with improvement potential
loc3_rec = top3cat(sorted_lowCounts, "Location")

# clean and write recs to CSV
loc3_rec1 = cleanRecs(loc3_rec)

# dictionary of recommendations by location
loc3_rec1_dict = locRec_dict(loc3_rec1, "Location")

############################################################
############# look into only subset of recommendations
############################################################

####### Based on Location-Departments with MOST low scores

# dataframe of all question responses in one column
allResponse_melt = qResponseDF(results_new)

# dataframe of Qresponses with low score responses only
low_allResponse_melt = getLowResponseQ(allResponse_melt)

# series with all low scores by Location/Department
lowscore_locdep = low_allResponse_melt.groupby("Location - Department")["Value"].count()

# array with Location/Department with low score count greater than 3rd quantile
highcount_locdep_array = getHigh_lowScoreCount_locdep(lowscore_locdep)

# dataframe of low score counts ONLY for Location-Deparments with highest counts
highScore_locdep_sorted_lowCounts = getCountResults_locdep(sorted_lowCounts, highcount_locdep_array)

# make recommendations for WORSE departments
worseDep_rec = top3cat(highScore_locdep_sorted_lowCounts, "Location - Department")

# clean and write recs to CSV
worseDep_rec1 = cleanRecs(worseDep_rec)

# dictionary of recommendations by location
worseDep_rec1_dict = locRec_dict(worseDep_rec1, "Location")

############################################################
############# Category Recommendations per Location
############################################################

# Scores with Improvement Potential across each category for every location
loc_cat_score = sorted_lowCounts.groupby(["Location", "Category"])[
    "Scores with Improvement Potential"].sum().reset_index(name="Scores with Improvement Potential")

# People Affected across each category for every location
loc_cat_affected = sorted_lowCounts.groupby(["Location", "Category"])["People Affected"].sum().reset_index(
    name="People Affected")

# check that row items match up
# all(loc_cat_score["Location"] == loc_cat_affected["Location"])  # True
# all(loc_cat_score["Category"] == loc_cat_affected["Category"])  # True

# Total People across each category for every location
loc_cat_responding = sorted_allCounts_ppl_sub.groupby(["Location", "Category"])[
    "People Responding"].sum().reset_index(name="People Responding")

# check that row items match up
# all(loc_cat_responding["Location"] == loc_cat_affected["Location"])  # True
# all(loc_cat_responding["Category"] == loc_cat_affected["Category"])  # True

# new column for 'loc_cat_affected' with percent people with low scores each location-category
loc_cat_affected["Percent Location"] = (
        loc_cat_affected["People Affected"] / loc_cat_responding["People Responding"] * 100).astype(int)

# add 'People Affected' and 'Percent Location' to 'loc_cat_score'
loc_cat_score["People Affected"] = loc_cat_affected["People Affected"]
loc_cat_score["Percent Location"] = loc_cat_affected["Percent Location"]

# make 3 category recommendations per location
loc_cat_rec = catRank_loc(loc_cat_score, "Location")

# dictionary of recommendations by location
loc_cat_rec_dict = locRec_cat_dict(loc_cat_rec, "Location")

############################################################
############# Category Recommendations for Organization
############################################################

# Scores with Improvement Potential across each category for organization
org_cat_score = sorted_lowCounts.groupby("Category")["Scores with Improvement Potential"].sum().reset_index(
    name="Scores with Improvement Potential")

# People Affected across each category for organization
org_cat_affected = sorted_lowCounts.groupby("Category")["People Affected"].sum().reset_index(name="People Affected")

# check that row items match up
# all(org_cat_score["Category"] == org_cat_affected["Category"])  # True

# Total People across each category for organization
org_cat_responding = sorted_allCounts_ppl_sub.groupby("Category")["People Responding"].sum().reset_index(
    name="People Responding")

# check that row items match up
# all(org_cat_responding["Category"] == org_cat_affected["Category"])  # True

# new column for 'org_cat_affected' with percent people with low scores each location-category
org_cat_affected["Percent Organization"] = (
        org_cat_affected["People Affected"] / org_cat_responding["People Responding"] * 100).astype(int)

# add 'People Affected' and 'Percent Location' to 'org_cat_score'
org_cat_score["People Affected"] = org_cat_affected["People Affected"]
org_cat_score["Percent Organization"] = org_cat_affected["Percent Organization"]

# show category details for organization
org_cat_detail = catRank_org(org_cat_score)

# category recommendations for organization
org_cat_rec = org_cat_detail.head(3)



#####################################################
##### Interactive Recommendations - Organization Categories
#####################################################

# option to show recommendation
if st.checkbox('Recommendation: Organization Categories'):

    # create download button
    rec_download_button1 = download_button(org_cat_rec, "organization_top3categories.csv", "Download Organization Categories")
    st.markdown(rec_download_button1, unsafe_allow_html=True)

    # write recs
    st.dataframe(org_cat_rec)



#####################################################
##### Interactive Recommendations - Location Categories
#####################################################

# option to show recommendation
if st.checkbox('Recommendation: Location Categories'):

    # create download button
    rec_download_button2 = download_button(loc_cat_rec, "all_location_top3categories.csv", "Download Location Categories")
    st.markdown(rec_download_button2, unsafe_allow_html=True)

    # option to show recommendation by location
    location_selected = st.selectbox(
        'Which Location do you want to look at?',
        list(loc_cat_rec_dict.keys()),
        key=2)

    # write recommendation for location
    st.write(loc_cat_rec_dict[location_selected])


#####################################################
##### Interactive Recommendations - Location - 3 Recommendations
#####################################################

# option to show recommendation
if st.checkbox('Recommendation: Location Dept Categories'):

    # create download button
    rec_download_button3 = download_button(loc3_rec1, "all_location_top3recommendations.csv", "Download Location Dept Categories")
    st.markdown(rec_download_button3, unsafe_allow_html=True)

    # option to show recommendation by location
    location_selected = st.selectbox(
        'Which Location do you want to look at?',
        list(loc3_rec1_dict.keys()),
        key=3)

    # write recommendation for location
    st.write(loc3_rec1_dict[location_selected])


#####################################################
##### Interactive Recommendations - Worse Departments - 3 Recommendations
#####################################################

# option to show recommendation
if st.checkbox('Recommendation: Worse Dept Categories'):

    # Description
    st.write('These departments have the highest number of low scores (top 25%) across the organization.')

    # create download button
    rec_download_button4 = download_button(worseDep_rec1, "worse_department_top3categories.csv", "Download Worse Dept Categories")
    st.markdown(rec_download_button4, unsafe_allow_html=True)

    # option to show recommendation by location
    location_selected = st.selectbox(
        'Which Location do you want to look at?',
        list(worseDep_rec1_dict.keys()),
        key=4)

    # write recommendation for location
    st.write(worseDep_rec1_dict[location_selected])







