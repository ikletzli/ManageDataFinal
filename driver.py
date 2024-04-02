import pandas as pd
import numpy as np
import re
import time


# Data Loading Methods =================================================================================================

def LoadData():
    author_data = LoadAuthors()
    address_data = LoadAddresses()

    return author_data, address_data


def LoadAddresses():
    address_df = pd.read_csv('data/address.csv')
    address_df = address_df[['EIN', 'Street Address 1']]

    address_df = address_df.drop(address_df[address_df['Street Address 1'].isnull()].index)
    address_df = address_df.drop(address_df[address_df['EIN'].isnull()].index)

    address_df['EIN'] = address_df['EIN'].astype('string')
    address_df["EIN"] = address_df["EIN"].map(lambda x: re.sub(r'\D', '', x))
    address_df['EIN'] = address_df['EIN'].astype('int64')

    address_df = address_df.rename(columns={'EIN': 'ein', 'Street Address 1': 'address'})

    address_df = address_df.fillna('')
    address_df = address_df.groupby('ein')['address'].agg(set= lambda x: set(x))
    return address_df


def LoadAuthors():
    book_df = pd.read_csv('data/book.txt', sep="\t", header=None, names=['publisher', 'isbn', 'title', 'author'])
    book_df = book_df.drop('publisher', axis=1)
    book_df = book_df.drop('title', axis=1)

    book_df = book_df.fillna('')

    book_df = book_df.groupby('isbn')['author'].agg(set= lambda x: set(x))

    return book_df


def GeneratingCandidateReplacements(author_data, address_data):

    # Paper says maybe we shouldn't include identical values in the candidate replacements,
    #   so maybe add
    #     if author_list[i] != author_list[j]
    #   above line 56
    #   and the respective line for adresses above line 64

    author_candidates = []
    for index in range(len(author_data)):
        author_list = list(author_data.iloc[index, 0])
        for i in range(len(author_list) - 1):
            for j in range(i+1, len(author_list)):
                author_candidates.append((author_list[i], author_list[j]))

    address_candidates = []
    for index in range(len(address_data)):
        address_list = list(address_data.iloc[index, 0])
        for i in range(len(address_list) - 1):
            for j in range(i+1, len(address_list)):
                address_candidates.append((address_list[i], address_list[j]))
            
    return author_candidates, address_candidates

def BuildTransformationGraphs(author_candidates, address_candidates):
    author_graphs = BuildTransformationGraph(author_candidates)
    address_graphs = BuildTransformationGraph(address_candidates)

def BuildTransformationGraph(candidates):
    graphs = []
    for candidate in candidates:
        pre_defined_regex = ["[A-Z]+", "[a-z]+", "\s+", "[0-9]+"]
        matches_0 = {"[A-Z]+": [], "[a-z]+": [], "\s+": [], "[0-9]+": []}
        matches_1 = {"[A-Z]+": [], "[a-z]+": [], "\s+": [], "[0-9]+": []}
        for regex in pre_defined_regex:
            for match in re.finditer(regex, candidate[0]):
                matches_0[regex] += [match]
            for match in re.finditer(regex, candidate[1]):
                matches_1[regex] += [match]

        P_0 = {}
        P_1 = {}
        for match_dict, P in [(matches_0, P_0), (matches_1, P_1)]:
            for regex, match_list in match_dict.items():
                for i in range(len(match_list)):
                    match = match_list[i]

                    x_str_one = "match" + regex + str(i+1) + "B"
                    x_str_two = "match" + regex + str(i-len(match_list)) + "B"

                    y_str_one = "match" + regex + str(i+1) + "E"
                    y_str_two = "match" + regex + str(i-len(match_list)) + "E"

                    if match.start() not in P:
                        P[match.start()] = [x_str_one, x_str_two]
                    else:
                        P[match.start()] += [x_str_one, x_str_two]

                    if match.end() not in P:
                        P[match.end()] = [y_str_one, y_str_two]
                    else:
                        P[match.end()] += [y_str_one, y_str_two]

        for index, P in [(0, P_0), (1, P_1)]:
            for i in range(len(candidate[index]) + 1):
                k = i+1
                str_one = "constpos" + str(k)
                str_two = "constpos" + str(k - len(candidate[index]) - 2)

                if k not in P:
                    P[k] = [str_one, str_two]
                else:
                    P[k] += [str_one, str_two]

        graph_0 = {}
        graph_1 = {}

        for i in range(len(candidate[0])):
            for j in range(i+1, len(candidate[0]) + 1):
                sub = candidate[0][i:j]
                graph_0[(i,j)] = ["conststr" + sub]

        for i in range(len(candidate[1]) - 1):
            for j in range(i+1, len(candidate[1])):
                sub = candidate[1][i:j]
                graph_1[(i,j)] = ["conststr" + sub]
                for match in re.finditer(re.escape(sub), candidate[0]):
                    for f in P_0[match.start()]:
                        for g in P_0[match.end()]:
                            graph_1[(i,j)] += ["sub" + f + g]
                    for f in P_1[i]:
                        for g in P_1[j]:
                            graph_0[(match.start(),match.end())] += ["sub" + f + g]

        graphs += [graph_0, graph_1]
    
    return graphs


# Incremental Grouping Methods =========================================================================================

def IncrementalGrouping():
    '''
    Algorithm 5
    Place holder method containing the function calls required to implement incremental grouping in the main algorithm
    '''
    # replace line 4 of Algorithm 1 (GoldenRecordCreation)
    G = Preprocessing()

    # replace line 6 of Algorithm 1
    sigma = GenerateNextLargestGroup(G)

    # replace line 9 of Algorithm 1
    G = UpdateGraph(G, sigma)

    return


def Preprocessing(phi):
    '''
    Algorithm 6
    Generates the set of graphs G corresponding to the set of candidate replacements phi

    arguments:
    return value:
    '''
    return


def GenerateNextLargestGroup(G):
    '''
    Algorithm 7
    Desc

    arguments:
    return value:
    '''
    return


def UpdateGraph(G, sigma):
    '''
    Desc

    arguments:
    return value:
    '''
    return


# Driver ===============================================================================================================

if __name__ == "__main__":
    author_data, address_data = LoadData()

    author_candidates, address_candidates = GeneratingCandidateReplacements(author_data, address_data)

    BuildTransformationGraphs(author_candidates, address_candidates)